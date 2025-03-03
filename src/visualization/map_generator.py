#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map Generator

This module generates various visualizations and maps for well pads, 
wells, concessions, and methane emissions in the Vaca Muerta Basin.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import MarkerCluster
import contextily as ctx
from shapely.geometry import shape, Point

from src.utils.logger import get_logger
from src.utils.localization import translate as _
from src.data.database import WellDatabase


class MapGenerator:
    """Class for generating various maps and visualizations."""
    
    def __init__(self, config: Dict, db: WellDatabase):
        """Initialize the map generator.
        
        Args:
            config: Configuration dictionary
            db: Database connection
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directories
        self.output_dir = Path(config['visualization']['output_directory'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Database connection
        self.db = db
        
        # Color configuration
        self.colors = config['visualization']['colors']
        
        # Basemap choice
        self.basemap_provider = {
            'OpenStreetMap': ctx.providers.OpenStreetMap.Mapnik,
            'ESRI': ctx.providers.Esri.WorldImagery,
            'Stamen': ctx.providers.Stamen.Terrain
        }.get(config['visualization']['basemap'], ctx.providers.OpenStreetMap.Mapnik)
    
    def generate_detection_maps(self, developments: List[Dict]) -> List[str]:
        """Generate maps for new detections.
        
        Args:
            developments: List of newly detected developments
            
        Returns:
            List of paths to generated map files
        """
        map_paths = []
        
        # Static detection maps
        map_paths.extend(self._generate_static_detection_maps(developments))
        
        # Interactive detection maps
        map_paths.extend(self._generate_interactive_detection_maps(developments))
        
        return map_paths
    
    def _generate_static_detection_maps(self, developments: List[Dict]) -> List[str]:
        """Generate static maps for new detections.
        
        Args:
            developments: List of newly detected developments
            
        Returns:
            List of paths to generated static map files
        """
        map_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Convert developments to GeoDataFrame
            gdf = gpd.GeoDataFrame([
                {
                    'id': dev['id'], 
                    'geometry': shape(dev['geometry']),
                    'type': 'wellpad' if 'wellpad' in dev['id'].lower() else 'well'
                } 
                for dev in developments
            ], crs="EPSG:4326")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot all development areas
            gdf[gdf['type'] == 'wellpad'].plot(
                ax=ax, 
                color=self.colors['well_pad'], 
                alpha=0.7, 
                label=_('Well Pads')
            )
            
            gdf[gdf['type'] == 'well'].plot(
                ax=ax, 
                color=self.colors['well'], 
                marker='o', 
                markersize=50, 
                alpha=0.7, 
                label=_('Wells')
            )
            
            # Add title and legend
            plt.title(_('New Developments in Vaca Muerta'), fontsize=16)
            plt.legend()
            
            # Add basemap
            try:
                ctx.add_basemap(
                    ax, 
                    source=self.basemap_provider,
                    crs=gdf.crs
                )
            except Exception as e:
                self.logger.warning(_("Could not add basemap: {0}").format(str(e)))
            
            # Remove axes
            ax.set_axis_off()
            
            # Save map
            output_path = self.output_dir / f"new_detections_static_{timestamp}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            map_paths.append(str(output_path))
            
        except Exception as e:
            self.logger.error(_("Error generating static detection map: {0}").format(str(e)))
        
        return map_paths
    
    def _generate_interactive_detection_maps(self, developments: List[Dict]) -> List[str]:
        """Generate interactive maps for new detections.
        
        Args:
            developments: List of newly detected developments
            
        Returns:
            List of paths to generated interactive map files
        """
        map_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Create map centered on the developments
            gdf = gpd.GeoDataFrame([
                {
                    'id': dev['id'], 
                    'geometry': shape(dev['geometry']),
                    'type': 'wellpad' if 'wellpad' in dev['id'].lower() else 'well'
                } 
                for dev in developments
            ], crs="EPSG:4326")
            
            # Get map center
            center_lat = gdf.geometry.centroid.y.mean()
            center_lon = gdf.geometry.centroid.x.mean()
            
            # Create Folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add detection layers
            # Well Pads
            wellpads = gdf[gdf['type'] == 'wellpad']
            if not wellpads.empty:
                folium.GeoJson(
                    wellpads,
                    name=_('Well Pads'),
                    style_function=lambda x: {
                        'fillColor': self.colors['well_pad'],
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['id'],
                        aliases=[_('ID:')],
                        localize=True
                    )
                ).add_to(m)
            
            # Wells
            wells = gdf[gdf['type'] == 'well']
            if not wells.empty:
                wells_cluster = MarkerCluster(name=_('Wells')).add_to(m)
                
                for _, well in wells.iterrows():
                    folium.Marker(
                        location=[well.geometry.y, well.geometry.x],
                        popup=well['id'],
                        icon=folium.Icon(
                            color='red', 
                            icon='oil-well', 
                            prefix='fa'
                        )
                    ).add_to(wells_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            output_path = self.output_dir / f"new_detections_interactive_{timestamp}.html"
            m.save(str(output_path))
            
            map_paths.append(str(output_path))
            
        except Exception as e:
            self.logger.error(_("Error generating interactive detection map: {0}").format(str(e)))
        
        return map_paths
    
    def generate_all_maps(self):
        """Generate comprehensive maps of the entire system."""
        # Generate maps of all official wells, concessions, and detected wells
        self._generate_comprehensive_system_map()
        
        # Generate individual maps for each concession
        self._generate_individual_concession_maps()
        
        # Generate methane emissions overview map
        self._generate_methane_emissions_map()
    
    def _generate_comprehensive_system_map(self):
        """Generate a comprehensive map of the entire monitoring system."""
        try:
            # Load wells and concessions
            from src.data.concession_mapper import ConcessionMapper
            concession_mapper = ConcessionMapper(self.db)
            
            wells_gdf = concession_mapper._load_wells_geodataframe()
            concessions_gdf = concession_mapper._load_concessions_geodataframe()
            
            # Create map
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Plot concessions
            concessions_gdf.plot(
                ax=ax, 
                color='lightblue', 
                edgecolor='black', 
                alpha=0.5, 
                label=_('Concessions')
            )
            
            # Plot wells
            if 'source' in wells_gdf.columns:
                official_wells = wells_gdf[wells_gdf['source'] == 'official']
                detected_wells = wells_gdf[wells_gdf['source'] == 'detected']
                
                official_wells.plot(
                    ax=ax, 
                    color='blue', 
                    marker='o', 
                    markersize=30, 
                    alpha=0.7, 
                    label=_('Official Wells')
                )
                
                detected_wells.plot(
                    ax=ax, 
                    color='red', 
                    marker='x', 
                    markersize=30, 
                    alpha=0.7, 
                    label=_('Detected Wells')
                )
            
            # Add title and legend
            plt.title(_('Vaca Muerta Monitoring System Overview'), fontsize=16)
            plt.legend()
            
            # Remove axes
            ax.set_axis_off()
            
            # Add basemap
            try:
                ctx.add_basemap(
                    ax, 
                    source=self.basemap_provider,
                    crs=wells_gdf.crs
                )
            except Exception as e:
                self.logger.warning(_("Could not add basemap: {0}").format(str(e)))
            
            # Save map
            output_path = self.output_dir / "system_overview_map.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(_("Generated comprehensive system map"))
            
        except Exception as e:
            self.logger.error(_("Error generating comprehensive system map: {0}").format(str(e)))
    
    def _generate_individual_concession_maps(self):
        """Generate individual maps for each concession."""
        from src.data.concession_mapper import ConcessionMapper
        concession_mapper = ConcessionMapper(self.db)
        concession_mapper.generate_concession_maps()
    
    def _generate_methane_emissions_map(self):
        """Generate map showing methane emissions across all developments."""
        try:
            # Load methane emissions data
            from pathlib import Path
            emissions_dir = Path(self.config['general']['data_directory']) / 'emissions'
            
            # Find the latest emissions GeoJSON
            emissions_files = list(emissions_dir.glob('methane_analysis_*.geojson'))
            if not emissions_files:
                self.logger.warning(_("No methane emissions data found"))
                return
            
            latest_file = max(emissions_files, key=os.path.getctime)
            
            # Load emissions as GeoDataFrame
            emissions_gdf = gpd.read_file(latest_file)
            
            # Create map
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Color map based on anomaly percentage
            norm = plt.Normalize(
                0, 
                max(emissions_gdf['mean_anomaly_pct'], default=self.config['methane']['analysis']['detection_threshold'] * 100)
            )
            
            emissions_gdf.plot(
                column='mean_anomaly_pct', 
                cmap='YlOrRd', 
                norm=norm,
                alpha=0.7, 
                edgecolor='black',
                linewidth=0.5,
                ax=ax,
                legend=True,
                legend_kwds={'label': _('Methane Anomaly (%)')}
            )
            
            # Highlight significant anomalies
            high_anomalies = emissions_gdf[emissions_gdf['has_anomaly']]
            if not high_anomalies.empty:
                high_anomalies.plot(
                    ax=ax,
                    color='none',
                    edgecolor='red',
                    linewidth=2,
                    label=_('Significant Anomalies')
                )
            
            # Add title
            plt.title(_('Methane Emissions in Vaca Muerta'), fontsize=16)
            
            # Remove axes
            ax.set_axis_off()
            
            # Add basemap
            try:
                ctx.add_basemap(
                    ax, 
                    source=self.basemap_provider,
                    crs=emissions_gdf.crs
                )
            except Exception as e:
                self.logger.warning(_("Could not add basemap: {0}").format(str(e)))
            
            # Save map
            output_path = self.output_dir / "methane_emissions_map.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also generate an interactive map
            center_lat = emissions_gdf.geometry.centroid.y.mean()
            center_lon = emissions_gdf.geometry.centroid.x.mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            # Add emissions layer
            folium.Choropleth(
                geo_data=emissions_gdf,
                name=_('Methane Anomalies'),
                data=emissions_gdf,
                columns=['development_id', 'mean_anomaly_pct'],
                key_on='feature.id',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=_('Methane Anomaly (%)')
            ).add_to(m)
            
            # Markers for high anomaly sites
            high_anomalies_cluster = MarkerCluster(name=_('High Anomaly Sites')).add_to(m)
            
            for _, row in high_anomalies.iterrows():
                popup_content = f"""
                <b>{_('Development ID')}:</b> {row['development_id']}<br>
                <b>{_('Mean Anomaly %')}:</b> {row['mean_anomaly_pct']:.2f}%<br>
                <b>{_('Significant Anomaly')}:</b> {row['has_anomaly']}
                """
                
                folium.Marker(
                    location=[row.geometry.centroid.y, row.geometry.centroid.x],
                    popup=popup_content,
                    icon=folium.Icon(color='red', icon='warning-sign', prefix='glyphicon')
                ).add_to(high_anomalies_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save interactive map
            interactive_path = self.output_dir / "methane_emissions_map.html"
            m.save(str(interactive_path))
            
            self.logger.info(_("Generated methane emissions maps"))
            
        except Exception as e:
            self.logger.error(_("Error generating methane emissions map: {0}").format(str(e)))


def generate_map_from_config(config_path: str):
    """
    Utility function to generate maps based on configuration.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    import yaml
    from src.data.database import initialize_db, WellDatabase
    from src.utils.config_loader import load_config

    # Load configuration
    config = load_config(config_path)
    
    # Initialize database
    db_path = initialize_db(config)
    db = WellDatabase(db_path)
    
    # Create map generator
    map_generator = MapGenerator(config, db)
    
    # Generate all maps
    map_generator.generate_all_maps()


if __name__ == "__main__":
    # Allow direct execution for map generation
    import sys
    
    if len(sys.argv) > 1:
        generate_map_from_config(sys.argv[1])
    else:
        print("Please provide a path to the configuration file.")
        sys.exit(1)
