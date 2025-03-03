#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OData Client for Argentina Oil & Gas Data

This module interfaces with Argentina's energy data APIs,
specifically the well and concession data from datos.energia.gob.ar.
"""

import os
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Union, Any

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point, MultiPolygon, Polygon
import json

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class ODataClient:
    """Client for accessing Argentina's energy data via OData APIs."""
    
    def __init__(self, wells_url: str, concessions_url: str):
        """Initialize the OData client.
        
        Args:
            wells_url: URL for the wells OData API
            concessions_url: URL for the concessions OData API
        """
        self.logger = get_logger()
        self.wells_url = wells_url
        self.concessions_url = concessions_url
        
        # Set up cache directory
        self.cache_dir = Path('data') / 'cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize session
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'VacaMuertaMonitor/1.0'
        })
    
    def get_wells(self, use_cache: bool = True, cache_expiry: int = 86400) -> List[Dict]:
        """Get well data from the API.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_expiry: Cache expiry time in seconds (default: 1 day)
            
        Returns:
            List of dictionaries with well information
        """
        cache_file = self.cache_dir / 'wells_data.json'
        
        # Check if cache file exists and is not expired
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if cache_age < cache_expiry:
                self.logger.info(_("Using cached wells data"))
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(_("Failed to load wells data from cache: {0}").format(str(e)))
        
        # If cache is not used or not available, fetch from API
        self.logger.info(_("Fetching wells data from API"))
        
        try:
            wells_data = []
            
            # OData URLs often support pagination
            skip = 0
            page_size = 1000
            has_more = True
            
            while has_more:
                # Build the query URL with pagination
                query_url = f"{self.wells_url}?$skip={skip}&$top={page_size}&$format=json"
                
                response = self.session.get(query_url, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract wells from the response
                page_wells = data.get('value', [])
                wells_data.extend(page_wells)
                
                # Check if we have more data to fetch
                if len(page_wells) < page_size:
                    has_more = False
                else:
                    skip += page_size
                
                self.logger.info(_("Fetched {0} wells (total: {1})").format(
                    len(page_wells), len(wells_data)
                ))
            
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(wells_data, f, ensure_ascii=False, indent=2)
            
            return wells_data
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(_("HTTP error fetching wells data: {0}").format(str(e)))
            
            # If we have a cache file, use it as fallback even if expired
            if cache_file.exists():
                self.logger.warning(_("Using expired cache as fallback for wells data"))
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return []
            
        except Exception as e:
            self.logger.error(_("Error fetching wells data: {0}").format(str(e)))
            
            # If we have a cache file, use it as fallback even if expired
            if cache_file.exists():
                self.logger.warning(_("Using expired cache as fallback for wells data"))
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return []
    
    def get_concessions(self, use_cache: bool = True, cache_expiry: int = 86400) -> List[Dict]:
        """Get concession data from the API.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_expiry: Cache expiry time in seconds (default: 1 day)
            
        Returns:
            List of dictionaries with concession information
        """
        cache_file = self.cache_dir / 'concessions_data.json'
        
        # Check if cache file exists and is not expired
        if use_cache and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            
            if cache_age < cache_expiry:
                self.logger.info(_("Using cached concessions data"))
                
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    self.logger.warning(_("Failed to load concessions data from cache: {0}").format(str(e)))
        
        # If cache is not used or not available, fetch from API
        self.logger.info(_("Fetching concessions data from API"))
        
        try:
            concessions_data = []
            
            # OData URLs often support pagination
            skip = 0
            page_size = 1000
            has_more = True
            
            while has_more:
                # Build the query URL with pagination
                query_url = f"{self.concessions_url}?$skip={skip}&$top={page_size}&$format=json"
                
                response = self.session.get(query_url, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract concessions from the response
                page_concessions = data.get('value', [])
                concessions_data.extend(page_concessions)
                
                # Check if we have more data to fetch
                if len(page_concessions) < page_size:
                    has_more = False
                else:
                    skip += page_size
                
                self.logger.info(_("Fetched {0} concessions (total: {1})").format(
                    len(page_concessions), len(concessions_data)
                ))
            
            # Process concessions to extract geometries
            processed_concessions = self._process_concessions_geometries(concessions_data)
            
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(processed_concessions, f, ensure_ascii=False, indent=2)
            
            return processed_concessions
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(_("HTTP error fetching concessions data: {0}").format(str(e)))
            
            # If we have a cache file, use it as fallback even if expired
            if cache_file.exists():
                self.logger.warning(_("Using expired cache as fallback for concessions data"))
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return []
            
        except Exception as e:
            self.logger.error(_("Error fetching concessions data: {0}").format(str(e)))
            
            # If we have a cache file, use it as fallback even if expired
            if cache_file.exists():
                self.logger.warning(_("Using expired cache as fallback for concessions data"))
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            return []
    
    def _process_concessions_geometries(self, concessions_data: List[Dict]) -> List[Dict]:
        """Process concession data to extract and standardize geometries.
        
        Args:
            concessions_data: List of dictionaries with concession information
            
        Returns:
            Processed concessions with standardized geometries
        """
        processed_concessions = []
        
        for concession in concessions_data:
            try:
                # Check for different potential geometry fields
                geometry = None
                
                # Look for GeoJSON in 'GEOMETRIA' or 'geometry' field
                if 'GEOMETRIA' in concession and concession['GEOMETRIA']:
                    try:
                        geom_data = json.loads(concession['GEOMETRIA'])
                        geometry = geom_data
                    except (json.JSONDecodeError, TypeError):
                        # Try WKT format
                        pass
                
                # If geometry is still None, check for WKT in 'WKT' field
                if geometry is None and 'WKT' in concession and concession['WKT']:
                    concession['wkt'] = concession['WKT']
                
                # Add the processed geometry to the concession data
                processed_concession = concession.copy()
                
                if geometry:
                    processed_concession['geometry'] = geometry
                
                processed_concessions.append(processed_concession)
                
            except Exception as e:
                self.logger.warning(_("Error processing concession geometry: {0}").format(str(e)))
                # Still add the concession without geometry
                processed_concessions.append(concession)
        
        return processed_concessions
    
    def search_wells(self, query: str) -> List[Dict]:
        """Search for wells by name, ID, or other criteria.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching wells
        """
        # Get all wells data
        wells_data = self.get_wells()
        
        # Filter wells based on the query
        query = query.lower()
        
        matching_wells = []
        
        for well in wells_data:
            # Check various fields for matches
            if any(query in str(well.get(field, '')).lower() for field in 
                  ['POZO', 'POZO_ID', 'EMPRESA', 'CONCESION', 'YACIMIENTO']):
                matching_wells.append(well)
        
        return matching_wells
    
    def search_concessions(self, query: str) -> List[Dict]:
        """Search for concessions by name, ID, or other criteria.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching concessions
        """
        # Get all concessions data
        concessions_data = self.get_concessions()
        
        # Filter concessions based on the query
        query = query.lower()
        
        matching_concessions = []
        
        for concession in concessions_data:
            # Check various fields for matches
            if any(query in str(concession.get(field, '')).lower() for field in 
                  ['NOMBRE', 'ID', 'EMPRESA', 'PROVINCIA']):
                matching_concessions.append(concession)
        
        return matching_concessions
    
    def get_wells_by_concession(self, concession_id: str) -> List[Dict]:
        """Get all wells within a specific concession.
        
        Args:
            concession_id: ID of the concession
            
        Returns:
            List of wells in the concession
        """
        # Get all wells data
        wells_data = self.get_wells()
        
        # Filter wells by concession ID
        wells_in_concession = [
            well for well in wells_data
            if well.get('CONCESION') == concession_id or well.get('concession') == concession_id
        ]
        
        return wells_in_concession
    
    def get_well_by_id(self, well_id: str) -> Optional[Dict]:
        """Get a specific well by ID.
        
        Args:
            well_id: ID of the well
            
        Returns:
            Well information or None if not found
        """
        # Get all wells data
        wells_data = self.get_wells()
        
        # Find the well with the given ID
        for well in wells_data:
            if (well.get('POZO_ID') == well_id or 
                well.get('id') == well_id or 
                well.get('ID') == well_id):
                return well
        
        return None
    
    def get_concession_by_id(self, concession_id: str) -> Optional[Dict]:
        """Get a specific concession by ID.
        
        Args:
            concession_id: ID of the concession
            
        Returns:
            Concession information or None if not found
        """
        # Get all concessions data
        concessions_data = self.get_concessions()
        
        # Find the concession with the given ID
        for concession in concessions_data:
            if (concession.get('ID') == concession_id or 
                concession.get('id') == concession_id):
                return concession
        
        return None
    
    def get_wells_dataframe(self) -> pd.DataFrame:
        """Get wells data as a pandas DataFrame.
        
        Returns:
            DataFrame with wells data
        """
        wells_data = self.get_wells()
        
        if not wells_data:
            return pd.DataFrame()
        
        return pd.DataFrame(wells_data)
    
    def get_concessions_geodataframe(self) -> gpd.GeoDataFrame:
        """Get concessions data as a GeoPandas GeoDataFrame.
        
        Returns:
            GeoDataFrame with concessions data and geometries
        """
        concessions_data = self.get_concessions()
        
        if not concessions_data:
            return gpd.GeoDataFrame()
        
        # Create a list to store processed concessions
        processed_concessions = []
        
        for concession in concessions_data:
            try:
                # Extract basic properties
                properties = {
                    'id': concession.get('ID') or concession.get('id', ''),
                    'name': concession.get('NOMBRE') or concession.get('name', ''),
                    'company': concession.get('EMPRESA') or concession.get('company', ''),
                    'province': concession.get('PROVINCIA') or concession.get('province', ''),
                    'type': concession.get('TIPO') or concession.get('type', '')
                }
                
                # Extract geometry
                if 'geometry' in concession and concession['geometry']:
                    # Create shapely geometry from GeoJSON
                    geometry = shape(concession['geometry'])
                    
                    # Add to processed list
                    processed_concessions.append({
                        **properties,
                        'geometry': geometry
                    })
                elif 'wkt' in concession and concession['wkt']:
                    # Try to parse WKT format
                    from shapely import wkt
                    try:
                        geometry = wkt.loads(concession['wkt'])
                        
                        # Add to processed list
                        processed_concessions.append({
                            **properties,
                            'geometry': geometry
                        })
                    except Exception as e:
                        self.logger.warning(_("Error parsing WKT for concession {0}: {1}").format(
                            properties['id'], str(e)
                        ))
                
            except Exception as e:
                self.logger.warning(_("Error processing concession for GeoDataFrame: {0}").format(str(e)))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(processed_concessions, crs="EPSG:4326")
        
        return gdf
    
    def get_wells_geodataframe(self) -> gpd.GeoDataFrame:
        """Get wells data as a GeoPandas GeoDataFrame.
        
        Returns:
            GeoDataFrame with wells data and geometries
        """
        wells_data = self.get_wells()
        
        if not wells_data:
            return gpd.GeoDataFrame()
        
        # Create a list to store processed wells
        processed_wells = []
        
        for well in wells_data:
            try:
                # Extract coordinates
                lat = float(well.get('LATITUD') or well.get('latitude', 0))
                lon = float(well.get('LONGITUD') or well.get('longitude', 0))
                
                # Skip wells without valid coordinates
                if lat == 0 or lon == 0:
                    continue
                
                # Create point geometry
                geometry = Point(lon, lat)
                
                # Extract basic properties
                properties = {
                    'id': well.get('POZO_ID') or well.get('id', ''),
                    'name': well.get('POZO') or well.get('name', ''),
                    'company': well.get('EMPRESA') or well.get('company', ''),
                    'concession': well.get('CONCESION') or well.get('concession', ''),
                    'status': well.get('ESTADO') or well.get('status', ''),
                    'type': well.get('TIPO') or well.get('type', ''),
                    'formation': well.get('FORMACION') or well.get('formation', ''),
                    'latitude': lat,
                    'longitude': lon
                }
                
                # Add to processed list
                processed_wells.append({
                    **properties,
                    'geometry': geometry
                })
                
            except Exception as e:
                self.logger.warning(_("Error processing well for GeoDataFrame: {0}").format(str(e)))
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(processed_wells, crs="EPSG:4326")
        
        return gdf
