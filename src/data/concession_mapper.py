#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concession Mapper

This module maps well pads and wells to their respective concessions
and generates visualization maps of the concessions.
"""

import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from folium.plugins import MarkerCluster
import contextily as ctx

from src.utils.logger import get_logger
from src.utils.localization import translate as _
from src.data.database import WellDatabase
from src.data.odata_client import ODataClient


class ConcessionMapper:
    """Class for mapping wells to concessions and generating visualizations."""
    
    def __init__(self, db: WellDatabase):
        """Initialize the concession mapper.
        
        Args:
            db: Database connection
        """
        self.logger = get_logger()
        self.db = db
        
        # Set up output directory
        self.output_dir = Path('data') / 'concessions'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Visualizations directory
        self.vis_dir = self.output_dir / 'visualizations'
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def map_developments_to_concessions(self, developments: List[Dict]) -> List[Dict]:
        """Map developments (well pads and wells) to their respective concessions.
        
        Args:
            developments: List of dictionaries with development information
            
        Returns:
            Updated developments with concession information
        """
        # Load concessions as GeoDataFrame
        concessions_gdf = self._load_concessions_geodataframe()
        
        if concessions_gdf.empty:
            self.logger.warning(_("No concessions data available for mapping"))
            return developments
        
        # Convert developments to GeoDataFrame
        developments_gdf = gpd.GeoDataFrame([
            {
                'id': dev['id'],
                'geometry': shape(dev['geometry']),
                'type': 'wellpad' if 'wellpad' in dev['id'].lower() else 'well'
            }
            for dev in developments
        ], crs="EPSG:4326")
        
        # Perform spatial join
        joined = gpd.sjoin(developments_gdf, concessions_gdf, how='left', predicate='within')
        
        # Update developments with concession info
        for i, dev in enumerate(developments):
            try:
                dev_id = dev['id']
                
                # Find this development in the joined GeoDataFrame
                match = joined[joined['id'] == dev_id]
                
                if not match.empty:
                    # Get the first match (there should be only one)
                    concession_info = match.iloc[0]
                    
                    # Add concession information to the development
                    dev['concession_id'] = concession_info.get('index_right')
                    dev['concession_name'] = concession_info.get('name')
                    dev['concession_company'] = concession_info.get('company')
                    dev['concession'] = concession_info.get('name')  # Convenience field
                
            except Exception as e:
                self.logger.warning(_("Error mapping development {0} to concession: {1}").format(dev['id'], str(e)))
        
        self.logger.info(_("Mapped {0} developments to concessions").format(len(developments)))
        return developments
    
    def generate_concession_maps(self) -> None:
        """Generate visualization maps of concessions with wells."""
        # Load concessions as GeoDataFrame
        concessions_gdf = self._load_concessions_geodataframe()
        
        if concessions_gdf.empty:
            self.logger.warning(_("No concessions data available for mapping"))
            return
        
        # Load wells as GeoDataFrame
        wells_gdf = self._load_wells_geodataframe()
        
        # Generate static map with matplotlib
        self._generate_static_map(concessions_gdf, wells_gdf)
        
        # Generate interactive map with folium
        self._generate_interactive_map(concessions_gdf, wells_gdf)
        
        # Generate individual concession maps
        self._generate_individual_concession_maps(concessions_gdf, wells_gdf)
    
    def _load_concessions_geodataframe(self) -> gpd.GeoDataFrame:
        """Load concessions data as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with concession data
        """
        try:
            # Try to load from local GeoJSON first
            concessions_file = self.output_dir / 'concessions.geojson'
            
            if concessions_file.exists():
                concessions_gdf = gpd.read_file(concessions_file)
                
                if not concessions_gdf.empty:
                    self.logger.info(_("Loaded concessions from local GeoJSON file"))
                    return concessions_gdf
            
            # If local file doesn't exist or is empty, query the database
            session = self.db.Session()
            from src.data.database import Concession
            
            # Query all concessions
            concessions = session.query(Concession).all()
            
            if not concessions:
                self.logger.warning(_("No concessions found in database"))
                return gpd.GeoDataFrame()
            
            # Convert to GeoDataFrame
            concessions_data = []
            for concession in concessions:
                try:
                    # Convert GeoAlchemy geometry to shapely
                    geometry = shape(concession.geometry.data)
                    
                    # Add to list
                    concessions_data.append({
                        'id': concession.concession_id,
                        'name': concession.name,
                        'company': concession.company,
                        'type': concession.type,
                        'geometry': geometry
                    })
                except Exception as e:
                    self.logger.warning(_("Error processing concession {0}: {1}").format(
                        concession.concession_id, str(e)
                    ))
            
            # Create GeoDataFrame
            concessions_gdf = gpd.GeoDataFrame(concessions_data, crs="EPSG:4326")
            
            # Save to GeoJSON for future use
            if not concessions_gdf.empty:
                concessions_gdf.to_file(concessions_file, driver="GeoJSON")
                self.logger.info(_("Saved concessions to local GeoJSON file"))
            
            return concessions_gdf
        
        except Exception as e:
            self.logger.error(_("Error loading concessions: {0}").format(str(e)))
            return gpd.GeoDataFrame()
        
        finally:
            # Ensure session is closed
            if 'session' in locals():
                session.close()
    
    def _load_wells_geodataframe(self) -> gpd.GeoDataFrame:
        """Load wells data (official and detected) as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with well data
        """
        try:
            # Try to load from local GeoJSON first
            wells_file = self.output_dir / 'wells.geojson'
            
            if wells_file.exists():
                wells_gdf = gpd.read_file(wells_file)
                
                if not wells_gdf.empty:
                    self.logger.info(_("Loaded wells from local GeoJSON file"))
                    return wells_gdf
            
            # If local file doesn't exist or is empty, query the database
            session = self.db.Session()
            from src.data.database import Well, OfficialWell
            
            # Query official wells
            official_wells = session.query(OfficialWell).all()
            
            # Query detected wells
            detected_wells = session.query(Well).all()
            
            # Combine into a single list
            wells_data = []
            
            # Process official wells
            for well in official_wells:
                try:
                    # Convert GeoAlchemy geometry to shapely
                    geometry = shape(well.geometry.data)
                    
                    # Add to list
                    wells_data.append({
                        'id': well.official_id,
                        'name': well.name,
                        'company': well.company,
                        'concession': well.concession,
                        'status': well.status,
                        'type': well.type,
                        'source': 'official',
                        'geometry': geometry
                    })
                except Exception as e:
                    self.logger.warning(_("Error processing official well {0}: {1}").format(
                        well.official_id, str(e)
                    ))
            
            # Process detected wells
            for well in detected_wells:
                try:
                    # Convert GeoAlchemy geometry to shapely
                    geometry = shape(well.geometry.data)
                    
                    # Add to list
                    wells_data.append({
                        'id': well.well_id,
                        'name': well.well_id,
                        'official_id': well.official_id,
                        'wellpad_id': well.wellpad_id,
                        'confidence': well.confidence,
                        'source': 'detected',
                        'geometry': geometry
                    })
                except Exception as e:
                    self.logger.warning(_("Error processing detected well {0}: {1}").format(
                        well.well_id, str(e)
                    ))
            
            # Create GeoDataFrame
            wells_gdf = gpd.GeoDataFrame(wells_data, crs="EPSG:4326")
            
            # Save to GeoJSON for future use
            if not wells_gdf.empty:
                wells_gdf.to_file(wells_file, driver="GeoJSON")
                self.logger.info(_("Saved wells to local GeoJSON file"))
            
            return wells_gdf
        
        except Exception as e:
            self.logger.error(_("Error loading wells: {0}").format(str(e)))
            return gpd.GeoDataFrame()
        
        finally:
            # Ensure session is closed
            if 'session' in locals():
                session.close()
    
    def _generate_static_map(self, concessions_gdf: gpd.GeoDataFrame, wells_gdf: gpd.GeoDataFrame) -> None:
        """Generate a static map of concessions and wells.
        
        Args:
            concessions_gdf: GeoDataFrame with concession data
            wells_gdf: GeoDataFrame with well data
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 15))
            
            # Plot concessions with random colors
            concessions_gdf.plot(
                ax=ax,
                column='company',
                categorical=True,
                legend=True,
                alpha=0.6,
                edgecolor='black',
                linewidth=0.5,
                cmap='tab20'
            )
            
            # Plot official wells
            if 'source' in wells_gdf.columns:
                official_wells = wells_gdf[wells_gdf['source'] == 'official']
                detected_wells = wells_gdf[wells_gdf['source'] == 'detected']
                
                if not official_wells.empty:
                    official_wells.plot(
                        ax=ax,
                        color='blue',
                        marker='o',
                        markersize=15,
                        alpha=0.7,
                        label='Official Wells'
                    )
                
                if not detected_wells.empty:
                    detected_wells.plot(
                        ax=ax,
                        color='red',
                        marker='x',
                        markersize=15,
                        alpha=0.7,
                        label='Detected Wells'
                    )
            else:
                # Plot all wells
                wells_gdf.plot(
                    ax=ax,
                    color='blue',
                    marker='o',
                    markersize=15,
                    alpha=0.7,
                    label='Wells'
                )
            
            # Add legend and title
            plt.legend(fontsize=12)
            plt.title(_('Vaca Muerta Concessions and Wells'), fontsize=16)
            
            # Remove axes
            ax.set_axis_off()
            
            # Add basemap
            try:
                ctx.add_basemap(
                    ax, 
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    crs=concessions_gdf.crs
                )
            except Exception as e:
                self.logger.warning(_("Could not add basemap: {0}").format(str(e)))
            
            # Save figure
            output_path = self.vis_dir / 'concessions_map.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(_("Generated static concessions map: {0}").format(output_path))
            
        except Exception as e:
            self.logger.error(_("Error generating static concessions map: {0}").format(str(e)))
    
    def _generate_interactive_map(self, concessions_gdf: gpd.GeoDataFrame, wells_gdf: gpd.GeoDataFrame) -> None:
        """Generate an interactive map of concessions and wells using folium.
        
        Args:
            concessions_gdf: GeoDataFrame with concession data
            wells_gdf: GeoDataFrame with well data
        """
        try:
            # Get center of the map
            center_lat, center_lon = concessions_gdf.geometry.unary_union.centroid.y, concessions_gdf.geometry.unary_union.centroid.x
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            # Add concessions layer
            folium.GeoJson(
                concessions_gdf,
                name='Concessions',
                style_function=lambda x: {
                    'fillColor': self._get_random_color(x['properties']['name']),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'company', 'type'],
                    aliases=['Name:', 'Company:', 'Type:'],
                    localize=True
                )
            ).add_to(m)
            
            # Add wells layer
            if not wells_gdf.empty:
                # Create marker cluster
                marker_cluster = MarkerCluster(name='Wells').add_to(m)
                
                # Add markers for wells
                for idx, row in wells_gdf.iterrows():
                    # Determine icon based on source
                    if 'source' in row and row['source'] == 'detected':
                        icon = folium.Icon(color='red', icon='x', prefix='fa')
                    else:
                        icon = folium.Icon(color='blue', icon='oil-well', prefix='fa')
                    
                    # Create popup content
                    popup_content = f"""
                    <b>ID:</b> {row.get('id', 'Unknown')}<br>
                    <b>Name:</b> {row.get('name', 'Unknown')}<br>
                    """
                    
                    if 'company' in row and row['company']:
                        popup_content += f"<b>Company:</b> {row['company']}<br>"
                    
                    if 'concession' in row and row['concession']:
                        popup_content += f"<b>Concession:</b> {row['concession']}<br>"
                    
                    if 'status' in row and row['status']:
                        popup_content += f"<b>Status:</b> {row['status']}<br>"
                    
                    if 'type' in row and row['type']:
                        popup_content += f"<b>Type:</b> {row['type']}<br>"
                    
                    if 'source' in row and row['source']:
                        popup_content += f"<b>Source:</b> {row['source'].capitalize()}<br>"
                    
                    # Add marker
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=row.get('name', row.get('id', 'Well')),
                        icon=icon
                    ).add_to(marker_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            output_path = self.vis_dir / 'concessions_map.html'
            m.save(str(output_path))
            
            self.logger.info(_("Generated interactive concessions map: {0}").format(output_path))
            
        except Exception as e:
            self.logger.error(_("Error generating interactive concessions map: {0}").format(str(e)))
    
    def _generate_individual_concession_maps(self, concessions_gdf: gpd.GeoDataFrame, wells_gdf: gpd.GeoDataFrame) -> None:
        """Generate individual maps for each concession.
        
        Args:
            concessions_gdf: GeoDataFrame with concession data
            wells_gdf: GeoDataFrame with well data
        """
        try:
            for idx, concession in concessions_gdf.iterrows():
                try:
                    # Extract concession details
                    concession_id = concession['id']
                    concession_name = concession['name']
                    concession_geometry = concession['geometry']
                    
                    # Filter wells for this concession
                    if 'concession' in wells_gdf.columns:
                        # Using concession name from wells data
                        wells_in_concession = wells_gdf[wells_gdf['concession'] == concession_name]
                    else:
                        # Spatial filter if concession name not available
                        wells_in_concession = wells_gdf[wells_gdf.geometry.within(concession_geometry)]
                    
                    # Skip if no wells in this concession
                    if wells_in_concession.empty:
                        continue
                    
                    # Create GeoDataFrame with just this concession
                    concession_df = gpd.GeoDataFrame([concession], crs=concessions_gdf.crs)
                    
                    # Generate static map
                    fig, ax = plt.subplots(figsize=(12, 12))
                    
                    # Plot concession
                    concession_df.plot(
                        ax=ax,
                        color='lightblue',
                        edgecolor='black',
                        alpha=0.6
                    )
                    
                    # Plot wells
                    if 'source' in wells_in_concession.columns:
                        official_wells = wells_in_concession[wells_in_concession['source'] == 'official']
                        detected_wells = wells_in_concession[wells_in_concession['source'] == 'detected']
                        
                        if not official_wells.empty:
                            official_wells.plot(
                                ax=ax,
                                color='blue',
                                marker='o',
                                markersize=20,
                                alpha=0.7,
                                label='Official Wells'
                            )
                        
                        if not detected_wells.empty:
                            detected_wells.plot(
                                ax=ax,
                                color='red',
                                marker='x',
                                markersize=20,
                                alpha=0.7,
                                label='Detected Wells'
                            )
                    else:
                        # Plot all wells
                        wells_in_concession.plot(
                            ax=ax,
                            color='blue',
                            marker='o',
                            markersize=20,
                            alpha=0.7,
                            label='Wells'
                        )
                    
                    # Add title and legend
                    plt.title(f"{concession_name} - {len(wells_in_concession)} Wells", fontsize=16)
                    plt.legend(fontsize=12)
                    
                    # Remove axes
                    ax.set_axis_off()
                    
                    # Add basemap
                    try:
                        ctx.add_basemap(
                            ax, 
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            crs=concessions_gdf.crs
                        )
                    except Exception as e:
                        self.logger.warning(_("Could not add basemap for {0}: {1}").format(concession_name, str(e)))
                    
                    # Save figure
                    concession_name_safe = concession_name.replace('/', '_').replace(' ', '_')
                    output_path = self.vis_dir / f"concession_{concession_name_safe}.png"
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    self.logger.info(_("Generated map for concession {0}").format(concession_name))
                    
                    # Generate interactive map
                    self._generate_interactive_concession_map(concession, wells_in_concession)
                
                except Exception as e:
                    self.logger.warning(_("Error generating map for concession {0}: {1}").format(
                        concession.get('name', concession.get('id', 'Unknown')), str(e)
                    ))
            
        except Exception as e:
            self.logger.error(_("Error generating individual concession maps: {0}").format(str(e)))
    
    def _generate_interactive_concession_map(self, concession: pd.Series, wells_gdf: gpd.GeoDataFrame) -> None:
        """Generate an interactive map for a single concession.
        
        Args:
            concession: Series with concession data
            wells_gdf: GeoDataFrame with wells in this concession
        """
        try:
            # Extract concession details
            concession_name = concession['name']
            concession_geometry = concession['geometry']
            
            # Get center of the concession
            center_lat, center_lon = concession_geometry.centroid.y, concession_geometry.centroid.x
            
            # Create map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add concession boundary
            concession_df = gpd.GeoDataFrame([concession], crs="EPSG:4326")
            
            folium.GeoJson(
                concession_df,
                name='Concession',
                style_function=lambda x: {
                    'fillColor': 'lightblue',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'company', 'type'],
                    aliases=['Name:', 'Company:', 'Type:'],
                    localize=True
                )
            ).add_to(m)
            
            # Add wells
            if not wells_gdf.empty:
                # Create marker cluster
                marker_cluster = MarkerCluster(name='Wells').add_to(m)
                
                # Add markers for wells
                for idx, row in wells_gdf.iterrows():
                    # Determine icon based on source
                    if 'source' in row and row['source'] == 'detected':
                        icon = folium.Icon(color='red', icon='x', prefix='fa')
                    else:
                        icon = folium.Icon(color='blue', icon='oil-well', prefix='fa')
                    
                    # Create popup content
                    popup_content = f"""
                    <b>ID:</b> {row.get('id', 'Unknown')}<br>
                    <b>Name:</b> {row.get('name', 'Unknown')}<br>
                    """
                    
                    if 'company' in row and row['company']:
                        popup_content += f"<b>Company:</b> {row['company']}<br>"
                    
                    if 'status' in row and row['status']:
                        popup_content += f"<b>Status:</b> {row['status']}<br>"
                    
                    if 'type' in row and row['type']:
                        popup_content += f"<b>Type:</b> {row['type']}<br>"
                    
                    if 'source' in row and row['source']:
                        popup_content += f"<b>Source:</b> {row['source'].capitalize()}<br>"
                    
                    # Add marker
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=row.get('name', row.get('id', 'Well')),
                        icon=icon
                    ).add_to(marker_cluster)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            concession_name_safe = concession_name.replace('/', '_').replace(' ', '_')
            output_path = self.vis_dir / f"concession_{concession_name_safe}.html"
            m.save(str(output_path))
            
        except Exception as e:
            self.logger.warning(_("Error generating interactive map for concession {0}: {1}").format(
                concession.get('name', 'Unknown'), str(e)
            ))
    
    def _get_random_color(self, seed_string: str) -> str:
        """Generate a random color based on a seed string.
        
        Args:
            seed_string: String to seed the random color
            
        Returns:
            Hex color code
        """
        # Create a hash of the string
        import hashlib
        hash_value = int(hashlib.md5(seed_string.encode()).hexdigest(), 16)
        
        # Get a color from tab20 colormap based on the hash
        colors = plt.cm.tab20.colors
        color = colors[hash_value % len(colors)]
        
        # Convert to hex
        hex_color = mcolors.rgb2hex(color)
        
        return hex_color
