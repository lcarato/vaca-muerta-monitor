#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methane Analyzer

This module analyzes methane emissions from detected well pads
using Sentinel-5P TROPOMI data.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape, mapping, Point, box
from pyproj import Transformer
import cv2
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class MethaneAnalyzer:
    """Class for analyzing methane emissions from oil and gas well pads."""
    
    def __init__(self, config: Dict):
        """Initialize the analyzer with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directory
        self.output_dir = Path(config['general']['data_directory']) / 'emissions'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load analysis parameters
        self.enabled = config['methane']['enabled']
        self.background_window = config['methane']['analysis']['background_window']  # km
        self.detection_threshold = config['methane']['analysis']['detection_threshold']
        self.temporal_aggregation = config['methane']['analysis']['temporal_aggregation']
        self.min_quality = config['methane']['min_quality']
    
    def analyze(
        self, 
        developments: List[Dict], 
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Analyze methane emissions for new well developments.
        
        Args:
            developments: List of dictionaries with well/wellpad information
            start_date: Start date for methane data
            end_date: End date for methane data
            
        Returns:
            List of dictionaries with emissions information
        """
        if not self.enabled:
            self.logger.info(_("Methane analysis disabled in configuration"))
            return []
        
        if not developments:
            self.logger.info(_("No developments to analyze for methane emissions"))
            return []
        
        self.logger.info(_("Analyzing methane emissions for {0} developments").format(len(developments)))
        
        # Check if we have Sentinel-5P data available
        s5p_dir = Path(self.config['general']['data_directory']) / 'raw' / 'SENTINEL-5P'
        processed_dir = Path(self.config['general']['data_directory']) / 'processed' / 'SENTINEL-5P'
        
        if not s5p_dir.exists() and not processed_dir.exists():
            self.logger.warning(_("No Sentinel-5P data available for methane analysis"))
            
            # Download Sentinel-5P data if not available
            self.logger.info(_("Downloading Sentinel-5P data for methane analysis"))
            # Call sentinel downloader directly for S5P data
            from src.satellite.sentinel_downloader import SentinelDownloader
            
            # Extract AOI from developments
            development_gdf = gpd.GeoDataFrame([
                {'geometry': shape(dev['geometry'])} for dev in developments
            ])
            bounds = development_gdf.total_bounds
            
            # Add buffer around bounds
            buffer_km = max(50, self.background_window * 1.5)  # km
            # Approximate degrees per km at this latitude (rough estimate)
            deg_per_km = 1 / 111  # ~111 km per degree at equator
            buffer_deg = buffer_km * deg_per_km
            
            aoi_bbox = {
                'min_lon': bounds[0] - buffer_deg,
                'min_lat': bounds[1] - buffer_deg,
                'max_lon': bounds[2] + buffer_deg,
                'max_lat': bounds[3] + buffer_deg
            }
            
            downloader = SentinelDownloader(self.config)
            s5p_imagery = downloader.download(
                start_date, 
                end_date, 
                aoi_bbox, 
                None, 
                collection_filter='SENTINEL-5P'
            )
            
            # Check if download was successful
            if not s5p_imagery or 'SENTINEL-5P' not in s5p_imagery or not s5p_imagery['SENTINEL-5P']:
                self.logger.warning(_("Failed to download Sentinel-5P data"))
                return []
            
            # Process the downloaded data
            from src.satellite.image_processor import ImageProcessor
            processor = ImageProcessor(self.config)
            processed = processor.process({'SENTINEL-5P': s5p_imagery['SENTINEL-5P']})
            
            if not processed or 'SENTINEL-5P' not in processed or not processed['SENTINEL-5P']:
                self.logger.warning(_("Failed to process Sentinel-5P data"))
                return []
        
        # Find all available Sentinel-5P processed data
        s5p_files = []
        if processed_dir.exists():
            s5p_files = list(processed_dir.glob('*.tif'))
        
        if not s5p_files:
            self.logger.warning(_("No processed Sentinel-5P data available for analysis"))
            return []
        
        emissions_results = []
        
        # Create GeoDataFrame from developments
        developments_gdf = gpd.GeoDataFrame([
            {
                'id': dev['id'],
                'geometry': shape(dev['geometry']),
                'type': 'wellpad' if 'wellpad' in dev['id'] else 'well'
            }
            for dev in developments
        ])
        developments_gdf.crs = 'EPSG:4326'
        
        # Process each development
        for _, dev in tqdm(developments_gdf.iterrows(), total=len(developments_gdf),
                          desc="Analyzing methane emissions"):
            
            # Calculate buffer for background calculation
            # Convert km to degrees (approximately)
            buffer_size_deg = self.background_window / 111  # ~111 km per degree at equator
            
            # Create buffered polygon for background calculation
            dev_centroid = dev.geometry.centroid
            background_box = box(
                dev_centroid.x - buffer_size_deg,
                dev_centroid.y - buffer_size_deg,
                dev_centroid.x + buffer_size_deg,
                dev_centroid.y + buffer_size_deg
            )
            
            # Process each methane file
            methane_values = []
            background_values = []
            dates = []
            
            for s5p_file in s5p_files:
                try:
                    with rasterio.open(s5p_file) as src:
                        # Check if development is within raster bounds
                        if not rasterio.coords.disjoint_bounds(
                            dev.geometry.bounds, src.bounds
                        ):
                            # Get timestamp from filename
                            file_date = self._extract_date_from_filename(s5p_file.name)
                            if file_date:
                                dates.append(file_date)
                            
                            # Mask raster by development geometry
                            dev_data, dev_transform = mask(
                                src, [dev.geometry], crop=True, all_touched=True
                            )
                            
                            # Handle no-data values
                            dev_data = np.ma.masked_array(
                                dev_data, mask=(dev_data == src.nodata)
                            )
                            
                            # Get mean value for development area
                            if not dev_data.mask.all():
                                methane_value = float(np.ma.mean(dev_data))
                                methane_values.append(methane_value)
                            else:
                                methane_values.append(np.nan)
                            
                            # Mask raster by background area
                            bg_data, bg_transform = mask(
                                src, [background_box], crop=True, all_touched=True
                            )
                            
                            # Handle no-data values
                            bg_data = np.ma.masked_array(
                                bg_data, mask=(bg_data == src.nodata)
                            )
                            
                            # Get mean value for background area
                            if not bg_data.mask.all():
                                bg_value = float(np.ma.mean(bg_data))
                                background_values.append(bg_value)
                            else:
                                background_values.append(np.nan)
                
                except Exception as e:
                    self.logger.error(
                        _("Error processing methane data {0} for development {1}: {2}").format(
                            s5p_file, dev.id, str(e)
                        )
                    )
            
            # Process the time series based on temporal aggregation
            if methane_values and background_values and dates:
                # Create DataFrame with time series
                df = pd.DataFrame({
                    'date': dates,
                    'methane': methane_values,
                    'background': background_values
                })
                
                # Remove NaN values
                df = df.dropna()
                
                if len(df) > 0:
                    # Calculate anomaly (development vs background)
                    df['anomaly'] = df['methane'] - df['background']
                    df['anomaly_pct'] = (df['anomaly'] / df['background']) * 100
                    
                    # Aggregate based on configuration
                    if self.temporal_aggregation == 'daily':
                        df['date'] = pd.to_datetime(df['date']).dt.date
                        agg_df = df.groupby('date').mean()
                    elif self.temporal_aggregation == 'weekly':
                        df['week'] = pd.to_datetime(df['date']).dt.to_period('W')
                        agg_df = df.groupby('week').mean()
                    elif self.temporal_aggregation == 'monthly':
                        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
                        agg_df = df.groupby('month').mean()
                    else:
                        # No aggregation, use all data points
                        agg_df = df
                    
                    # Determine if there's a significant methane anomaly
                    # Check if any anomaly percentage is above threshold
                    has_anomaly = any(agg_df['anomaly_pct'] > (self.detection_threshold * 100))
                    
                    # Calculate mean, max, and latest values
                    mean_anomaly = float(agg_df['anomaly'].mean())
                    max_anomaly = float(agg_df['anomaly'].max())
                    max_anomaly_pct = float(agg_df['anomaly_pct'].max())
                    
                    # Get latest values
                    latest_date = max(agg_df.index)
                    latest_values = agg_df.loc[latest_date]
                    
                    latest_methane = float(latest_values['methane'])
                    latest_background = float(latest_values['background'])
                    latest_anomaly = float(latest_values['anomaly'])
                    latest_anomaly_pct = float(latest_values['anomaly_pct'])
                    
                    # Create emissions result entry
                    result = {
                        'development_id': dev.id,
                        'development_type': dev.type,
                        'geometry': mapping(dev.geometry),
                        'has_anomaly': has_anomaly,
                        'mean_methane': float(agg_df['methane'].mean()),
                        'mean_background': float(agg_df['background'].mean()),
                        'mean_anomaly': mean_anomaly,
                        'mean_anomaly_pct': float(agg_df['anomaly_pct'].mean()),
                        'max_anomaly': max_anomaly,
                        'max_anomaly_pct': max_anomaly_pct,
                        'latest_date': str(latest_date),
                        'latest_methane': latest_methane,
                        'latest_background': latest_background,
                        'latest_anomaly': latest_anomaly,
                        'latest_anomaly_pct': latest_anomaly_pct,
                        'data_points': len(df),
                        'time_series': df.to_dict(orient='records')
                    }
                    
                    emissions_results.append(result)
                else:
                    self.logger.warning(
                        _("No valid methane data available for development {0}").format(dev.id)
                    )
            else:
                self.logger.warning(
                    _("Insufficient methane data for development {0}").format(dev.id)
                )
        
        # Save results
        if emissions_results:
            self._save_results(emissions_results)
        
        return emissions_results
    
    def _extract_date_from_filename(self, filename: str) -> Optional[str]:
        """Extract date from Sentinel-5P filename.
        
        Args:
            filename: Sentinel-5P filename
            
        Returns:
            Date string in YYYY-MM-DD format, or None if can't be extracted
        """
        try:
            # Try to find date pattern in filename
            import re
            
            # Look for patterns like 20210315 or 2021-03-15
            date_patterns = [
                r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
                r'(\d{4})[_-](\d{2})[_-](\d{2})'  # YYYY_MM_DD or YYYY-MM-DD
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    year, month, day = match.groups()
                    return f"{year}-{month}-{day}"
            
            return None
        except Exception:
            return None
    
    def _save_results(self, emissions_results: List[Dict]) -> None:
        """Save emissions analysis results to file.
        
        Args:
            emissions_results: List of dictionaries with emissions information
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as GeoJSON
        features = []
        for result in emissions_results:
            properties = {k: v for k, v in result.items() if k != 'geometry' and k != 'time_series'}
            
            # Convert time series to separate JSON file to keep GeoJSON smaller
            time_series = result.get('time_series', [])
            if time_series:
                ts_filename = f"time_series_{result['development_id']}.json"
                ts_path = self.output_dir / ts_filename
                
                with open(ts_path, 'w') as f:
                    import json
                    json.dump(time_series, f, indent=2)
                
                properties['time_series_file'] = ts_filename
            
            features.append({
                'type': 'Feature',
                'id': result['development_id'],
                'geometry': result['geometry'],
                'properties': properties
            })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        # Save GeoJSON
        output_path = self.output_dir / f"methane_analysis_{timestamp}.geojson"
        
        with open(output_path, 'w') as f:
            import json
            json.dump(geojson, f, indent=2)
        
        self.logger.info(_("Saved methane analysis results to {0}").format(output_path))
        
        # Generate additional visualizations
        self._generate_visualizations(emissions_results, timestamp)
    
    def _generate_visualizations(self, emissions_results: List[Dict], timestamp: str) -> None:
        """Generate visualization files for methane emissions.
        
        Args:
            emissions_results: List of dictionaries with emissions information
            timestamp: Timestamp string for filenames
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            
            # Create GeoDataFrame for mapping
            gdf = gpd.GeoDataFrame([
                {
                    'id': result['development_id'],
                    'geometry': shape(result['geometry']),
                    'anomaly': result['mean_anomaly'],
                    'anomaly_pct': result['mean_anomaly_pct'],
                    'has_anomaly': result['has_anomaly']
                }
                for result in emissions_results
            ])
            gdf.crs = 'EPSG:4326'
            
            # Set up colormap
            norm = Normalize(vmin=0, vmax=max(gdf['anomaly_pct'].max(), self.detection_threshold * 100))
            cmap = cm.YlOrRd
            
            # Create directory for visualizations
            vis_dir = self.output_dir / 'visualizations'
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate map
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            gdf.plot(
                column='anomaly_pct',
                cmap=cmap,
                norm=norm,
                legend=True,
                ax=ax
            )
            
            # Highlight significant anomalies
            if any(gdf['has_anomaly']):
                gdf[gdf['has_anomaly']].plot(
                    color='none',
                    edgecolor='red',
                    linewidth=2,
                    ax=ax
                )
            
            plt.title(_('Methane Anomalies (%) Relative to Background'))
            plt.savefig(vis_dir / f"methane_map_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Generate time series plots for each development
            for result in emissions_results:
                if 'time_series' in result and result['time_series']:
                    # Convert to DataFrame
                    ts_df = pd.DataFrame(result['time_series'])
                    
                    # Create plot
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    
                    # Plot methane and background
                    ax.plot(ts_df['date'], ts_df['methane'], 'r-', label=_('Development Area'))
                    ax.plot(ts_df['date'], ts_df['background'], 'b-', label=_('Background'))
                    
                    # Plot anomaly
                    ax2 = ax.twinx()
                    ax2.plot(ts_df['date'], ts_df['anomaly_pct'], 'g--', label=_('Anomaly (%)'))
                    
                    # Add threshold line
                    ax2.axhline(y=self.detection_threshold * 100, color='k', linestyle=':', 
                               label=_('Threshold ({0}%)').format(self.detection_threshold * 100))
                    
                    # Set labels and title
                    ax.set_xlabel(_('Date'))
                    ax.set_ylabel(_('Methane Concentration'))
                    ax2.set_ylabel(_('Anomaly (%)'))
                    
                    plt.title(_('Methane Time Series for {0}').format(result['development_id']))
                    
                    # Add legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                    
                    # Save plot
                    plt.savefig(
                        vis_dir / f"methane_timeseries_{result['development_id']}_{timestamp}.png", 
                        dpi=300, 
                        bbox_inches='tight'
                    )
                    plt.close()
            
            self.logger.info(_("Generated methane visualization files in {0}").format(vis_dir))
        
        except Exception as e:
            self.logger.error(_("Error generating methane visualizations: {0}").format(str(e)))
