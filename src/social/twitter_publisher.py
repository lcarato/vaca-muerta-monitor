#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter Publisher

This module handles publishing well detection and methane analysis
results to Twitter (X).
"""

import os
from pathlib import Path
from datetime import datetime
import tempfile
from typing import Dict, List, Tuple, Optional, Union, Any

import tweepy
from shapely.geometry import shape
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.plot import show
from PIL import Image, ImageDraw, ImageFont
import cv2

from src.utils.logger import get_logger
from src.utils.localization import translate as _, get_language


class TwitterPublisher:
    """Class for publishing results to Twitter."""
    
    def __init__(self, config: Dict):
        """Initialize the publisher with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directory for generated images
        self.output_dir = Path(config['general']['data_directory']) / 'social'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check if social media posting is enabled
        self.enabled = config['social']['enabled']
        if not self.enabled:
            self.logger.info(_("Social media publishing is disabled in configuration"))
            return
        
        # Initialize Twitter API client
        try:
            # Get Twitter API credentials from environment variables
            api_key = os.environ.get('TWITTER_API_KEY')
            api_secret = os.environ.get('TWITTER_API_SECRET')
            access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.environ.get('TWITTER_ACCESS_TOKEN_SECRET')
            bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
            
            # Check if all credentials are available
            if not all([api_key, api_secret, access_token, access_token_secret]):
                self.logger.warning(
                    _("Twitter API credentials not found in environment variables. Social media posting disabled.")
                )
                self.enabled = False
                self.client = None
                return
            
            # Initialize Twitter API v2 client
            self.client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            
            # For media uploads, we need v1.1 API
            auth = tweepy.OAuth1UserHandler(
                api_key, api_secret, access_token, access_token_secret
            )
            self.api = tweepy.API(auth)
            
            self.logger.info(_("Successfully initialized Twitter API client"))
            
        except Exception as e:
            self.logger.error(_("Failed to initialize Twitter API client: {0}").format(str(e)))
            self.enabled = False
            self.client = None
    
    def post_detection(self, development: Dict, map_path: Optional[str] = None) -> bool:
        """Post a new development detection to Twitter.
        
        Args:
            development: Dictionary with development information
            map_path: Optional path to a map image to include
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            self.logger.info(_("Social media posting is disabled"))
            return False
        
        try:
            # Get current language
            lang = get_language()
            
            # Create message based on development type and language
            if 'wellpad' in development['id'].lower():
                development_type = _("well pad")
            else:
                development_type = _("well")
            
            # Format the location information
            geometry = shape(development['geometry'])
            centroid = geometry.centroid
            lat, lon = centroid.y, centroid.x
            
            # Get concession information if available
            concession_info = development.get('concession', '')
            if concession_info:
                concession_text = _(" in the {0} concession").format(concession_info)
            else:
                concession_text = ""
            
            # Create message
            messages = {
                'en': f"NEW DETECTION: A new {development_type} has been detected{concession_info} in the Vaca Muerta Basin, Argentina. Location: {lat:.6f}, {lon:.6f}",
                'es': f"NUEVA DETECCIÓN: Se ha detectado un nuevo {development_type}{concession_text} en la Cuenca de Vaca Muerta, Argentina. Ubicación: {lat:.6f}, {lon:.6f}",
                'pt': f"NOVA DETECÇÃO: Um novo {development_type} foi detectado{concession_info} na Bacia de Vaca Muerta, Argentina. Localização: {lat:.6f}, {lon:.6f}"
            }
            
            # Get message in current language (default to English if not available)
            message = messages.get(lang, messages['en'])
            
            # Add hashtags
            hashtags = ' '.join(self.config['social']['platforms']['twitter']['hashtags'])
            message = f"{message}\n\n{hashtags}"
            
            # Create and include an image if no map is provided
            media_ids = []
            if map_path and os.path.exists(map_path):
                # Use the provided map
                media = self.api.media_upload(filename=map_path)
                media_ids.append(media.media_id)
            elif self.config['social']['platforms']['twitter']['include_map']:
                # Generate a simple visualization
                img_path = self._create_detection_image(development)
                if img_path and os.path.exists(img_path):
                    media = self.api.media_upload(filename=img_path)
                    media_ids.append(media.media_id)
            
            # Post to Twitter
            if media_ids:
                response = self.client.create_tweet(text=message, media_ids=media_ids)
            else:
                response = self.client.create_tweet(text=message)
            
            self.logger.info(_("Successfully posted detection to Twitter"))
            return True
            
        except Exception as e:
            self.logger.error(_("Failed to post to Twitter: {0}").format(str(e)))
            return False
    
    def post_methane_alert(self, emission_result: Dict, map_path: Optional[str] = None) -> bool:
        """Post a methane emission alert to Twitter.
        
        Args:
            emission_result: Dictionary with emission analysis information
            map_path: Optional path to a map image to include
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            self.logger.info(_("Social media posting is disabled"))
            return False
        
        # Only post if there's a significant anomaly
        if not emission_result.get('has_anomaly', False):
            self.logger.info(_("No significant methane anomaly to report"))
            return False
        
        try:
            # Get current language
            lang = get_language()
            
            # Format the anomaly information
            anomaly_pct = emission_result.get('latest_anomaly_pct', 0)
            development_id = emission_result.get('development_id', '')
            
            # Create message
            messages = {
                'en': f"METHANE ALERT: Detected elevated methane levels ({anomaly_pct:.1f}% above background) at {development_id} in the Vaca Muerta Basin, Argentina.",
                'es': f"ALERTA DE METANO: Detectados niveles elevados de metano ({anomaly_pct:.1f}% por encima del fondo) en {development_id} en la Cuenca de Vaca Muerta, Argentina.",
                'pt': f"ALERTA DE METANO: Detectados níveis elevados de metano ({anomaly_pct:.1f}% acima do background) em {development_id} na Bacia de Vaca Muerta, Argentina."
            }
            
            # Get message in current language (default to English if not available)
            message = messages.get(lang, messages['en'])
            
            # Add hashtags
            hashtags = ' '.join(self.config['social']['platforms']['twitter']['hashtags'])
            message = f"{message}\n\n{hashtags} #MethaneEmissions"
            
            # Create and include an image if no map is provided
            media_ids = []
            if map_path and os.path.exists(map_path):
                # Use the provided map
                media = self.api.media_upload(filename=map_path)
                media_ids.append(media.media_id)
            elif self.config['social']['platforms']['twitter']['include_map']:
                # Generate a simple visualization
                img_path = self._create_methane_image(emission_result)
                if img_path and os.path.exists(img_path):
                    media = self.api.media_upload(filename=img_path)
                    media_ids.append(media.media_id)
            
            # Post to Twitter
            if media_ids:
                response = self.client.create_tweet(text=message, media_ids=media_ids)
            else:
                response = self.client.create_tweet(text=message)
            
            self.logger.info(_("Successfully posted methane alert to Twitter"))
            return True
            
        except Exception as e:
            self.logger.error(_("Failed to post methane alert to Twitter: {0}").format(str(e)))
            return False
    
    def _create_detection_image(self, development: Dict) -> Optional[str]:
        """Create an image for the development detection.
        
        Args:
            development: Dictionary with development information
            
        Returns:
            Path to the created image file, or None if failed
        """
        try:
            # Generate a simple map showing the development location
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon as MplPolygon
            import contextily as ctx
            from shapely.geometry import shape
            
            # Create a GeoDataFrame with the development
            geometry = shape(development['geometry'])
            gdf = gpd.GeoDataFrame([{'geometry': geometry}], crs='EPSG:4326')
            
            # Buffer the geometry to create a suitable map extent
            buffer_size = 0.01  # degrees (approximate)
            buffered = gdf.buffer(buffer_size)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Plot the buffered area for extent
            buffered.plot(ax=ax, alpha=0)
            
            # Plot the development
            gdf.plot(ax=ax, facecolor='red', edgecolor='black', alpha=0.7)
            
            # Add basemap
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs=gdf.crs
                )
            except Exception as e:
                self.logger.warning(_("Failed to add basemap: {0}").format(str(e)))
            
            # Add title and attribution
            plot_title = _("New Detection in Vaca Muerta")
            plt.title(plot_title, fontsize=16)
            plt.figtext(
                0.99, 0.01, 
                "Vaca Muerta Monitor", 
                ha='right', fontsize=8, style='italic'
            )
            
            # Remove axis labels
            ax.set_axis_off()
            
            # Save to temporary file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"detection_{timestamp}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(_("Failed to create detection image: {0}").format(str(e)))
            return None
    
    def _create_methane_image(self, emission_result: Dict) -> Optional[str]:
        """Create an image for the methane emission alert.
        
        Args:
            emission_result: Dictionary with emission analysis information
            
        Returns:
            Path to the created image file, or None if failed
        """
        try:
            # Generate a visualization of the methane anomaly
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            import contextily as ctx
            from shapely.geometry import shape
            
            # Create figure for time series plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Check if we have time series data
            if 'time_series' in emission_result and emission_result['time_series']:
                # Plot the methane time series
                time_series = emission_result['time_series']
                
                # Create a pandas DataFrame
                import pandas as pd
                df = pd.DataFrame(time_series)
                
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Plot methane and background
                ax.plot(df['date'], df['methane'], 'r-', label=_('Site'))
                ax.plot(df['date'], df['background'], 'b-', label=_('Background'))
                
                # Plot anomaly percentage on second y-axis
                ax2 = ax.twinx()
                ax2.plot(df['date'], df['anomaly_pct'], 'g--', label=_('Anomaly (%)'))
                
                # Add threshold line
                threshold = self.config['methane']['analysis']['detection_threshold'] * 100
                ax2.axhline(y=threshold, color='k', linestyle=':', 
                           label=_('Threshold ({0}%)').format(threshold))
                
                # Add labels
                ax.set_xlabel(_('Date'))
                ax.set_ylabel(_('Methane Concentration'))
                ax2.set_ylabel(_('Anomaly (%)'))
                
                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
            else:
                # Create a simple bar chart showing current anomaly
                categories = [_('Background'), _('Site'), _('Anomaly %')]
                values = [
                    emission_result.get('latest_background', 0),
                    emission_result.get('latest_methane', 0),
                    emission_result.get('latest_anomaly_pct', 0)
                ]
                
                colors = ['blue', 'red', 'green']
                
                ax.bar(categories[:2], values[:2], color=colors[:2])
                
                # Add anomaly percentage on second y-axis
                ax2 = ax.twinx()
                ax2.bar(categories[2], values[2], color=colors[2])
                
                # Add threshold line
                threshold = self.config['methane']['analysis']['detection_threshold'] * 100
                ax2.axhline(y=threshold, color='black', linestyle=':', 
                           label=_('Threshold ({0}%)').format(threshold))
                
                # Add labels
                ax.set_ylabel(_('Methane Concentration'))
                ax2.set_ylabel(_('Anomaly (%)'))
            
            # Add title
            detection_id = emission_result.get('development_id', '')
            plot_title = _("Methane Anomaly at {0}").format(detection_id)
            plt.title(plot_title, fontsize=14)
            
            # Add timestamp and attribution
            plt.figtext(
                0.99, 0.01, 
                "Vaca Muerta Monitor", 
                ha='right', fontsize=8, style='italic'
            )
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"methane_{timestamp}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(_("Failed to create methane image: {0}").format(str(e)))
            return None
