#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel Satellite Data Downloader

This module handles the downloading of Sentinel satellite imagery
from the Copernicus Open Access Hub and Sentinel Hub services.
"""

import os
import tempfile
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple, Union, Optional

import requests
import numpy as np
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from sentinelhub import (
    SHConfig, 
    BBox, 
    CRS, 
    DataCollection, 
    SentinelHubRequest, 
    MimeType, 
    bbox_to_dimensions
)
from shapely.geometry import box, shape
import geopandas as gpd
from osgeo import gdal
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class SentinelDownloader:
    """Class for downloading Sentinel satellite imagery."""
    
    def __init__(self, config: Dict):
        """Initialize the downloader with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directory
        self.output_dir = Path(config['general']['data_directory']) / 'raw'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Sentinel Hub configuration
        self.sh_config = SHConfig()
        self.sh_config.instance_id = os.environ.get(
            config['sentinel']['instance_id_env_var'],
            ''
        )
        self.sh_config.sh_client_id = os.environ.get('SENTINEL_HUB_CLIENT_ID', '')
        self.sh_config.sh_client_secret = os.environ.get('SENTINEL_HUB_CLIENT_SECRET', '')
        
        # Initialize Sentinel API for direct access to Copernicus Hub
        self.sentinel_api = SentinelAPI(
            os.environ.get('COPERNICUS_USERNAME', ''),
            os.environ.get('COPERNICUS_PASSWORD', ''),
            'https://scihub.copernicus.eu/dhus'
        )
        
        # Create collection mappings
        self.collection_map = {
            'SENTINEL-2-L2A': DataCollection.SENTINEL2_L2A,
            'SENTINEL-1-GRD': DataCollection.SENTINEL1_IW,
            'SENTINEL-5P': DataCollection.SENTINEL5P
        }
    
    def download(
        self, 
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        bbox: Dict = None,
        geojson_path: str = None
    ) -> Dict[str, List[str]]:
        """Download Sentinel imagery for the specified parameters.
        
        Args:
            start_date: Start date for image collection
            end_date: End date for image collection
            bbox: Dictionary with min_lon, min_lat, max_lon, max_lat
            geojson_path: Path to GeoJSON file with area of interest
        
        Returns:
            Dictionary mapping collection names to lists of downloaded file paths
        """
        # Validate dates
        if isinstance(start_date, str) and start_date != 'now':
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str) and end_date != 'now':
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        if end_date == 'now':
            end_date = datetime.now()
        
        # Define geometry from either bbox or geojson
        if geojson_path and os.path.exists(geojson_path):
            self.logger.info(_("Using area of interest from GeoJSON: {0}").format(geojson_path))
            footprint = geojson_to_wkt(read_geojson(geojson_path))
            # Also create a Sentinel Hub compatible bbox
            geojson_data = json.load(open(geojson_path))
            gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
            bounds = gdf.total_bounds
            sh_bbox = BBox(bbox=[bounds[0], bounds[1], bounds[2], bounds[3]], crs=CRS.WGS84)
        elif bbox:
            self.logger.info(_("Using bounding box area of interest"))
            footprint = geojson_to_wkt(box(
                bbox['min_lon'], bbox['min_lat'], 
                bbox['max_lon'], bbox['max_lat']
            ).buffer(0).__geo_interface__)
            sh_bbox = BBox(
                bbox=[bbox['min_lon'], bbox['min_lat'], 
                      bbox['max_lon'], bbox['max_lat']], 
                crs=CRS.WGS84
            )
        else:
            raise ValueError(_("Either bbox or geojson_path must be provided"))
        
        # Prepare result container
        result = {}
        
        # Download data for each configured collection
        for collection_config in self.config['sentinel']['collections']:
            collection_name = collection_config['name']
            self.logger.info(_("Downloading {0} imagery").format(collection_name))
            
            # Create output directory for this collection
            collection_dir = self.output_dir / collection_name
            os.makedirs(collection_dir, exist_ok=True)
            
            # Use appropriate method based on collection
            if collection_name.startswith('SENTINEL-2') or collection_name.startswith('SENTINEL-1'):
                file_paths = self._download_optical_or_sar(
                    collection_name, 
                    collection_config,
                    start_date,
                    end_date,
                    footprint,
                    sh_bbox,
                    collection_dir
                )
            elif collection_name.startswith('SENTINEL-5P'):
                file_paths = self._download_sentinel5p(
                    collection_config,
                    start_date,
                    end_date,
                    sh_bbox,
                    collection_dir
                )
            else:
                self.logger.warning(_("Unsupported collection: {0}").format(collection_name))
                continue
            
            result[collection_name] = file_paths
        
        return result
    
    def _download_optical_or_sar(
        self,
        collection_name: str,
        collection_config: Dict,
        start_date: datetime,
        end_date: datetime,
        footprint: str,
        sh_bbox: BBox,
        output_dir: Path
    ) -> List[str]:
        """Download Sentinel-1 or Sentinel-2 imagery.
        
        Args:
            collection_name: Name of the collection ('SENTINEL-2-L2A' or 'SENTINEL-1-GRD')
            collection_config: Configuration for this collection
            start_date: Start date for download
            end_date: End date for download
            footprint: WKT representation of area of interest
            sh_bbox: Sentinel Hub BBox object
            output_dir: Directory to save files
            
        Returns:
            List of paths to downloaded files
        """
        # First try direct download from Copernicus Open Access Hub
        try:
            file_paths = self._download_from_copernicus(
                collection_name, 
                collection_config,
                start_date,
                end_date,
                footprint,
                output_dir
            )
            if file_paths:
                return file_paths
        except Exception as e:
            self.logger.warning(
                _("Failed to download from Copernicus: {0}. Trying Sentinel Hub.").format(str(e))
            )
        
        # If Copernicus fails or returns no data, try Sentinel Hub
        return self._download_from_sentinelhub(
            collection_name,
            collection_config,
            start_date,
            end_date,
            sh_bbox,
            output_dir
        )
    
    def _download_from_copernicus(
        self,
        collection_name: str,
        collection_config: Dict,
        start_date: datetime,
        end_date: datetime,
        footprint: str,
        output_dir: Path
    ) -> List[str]:
        """Download imagery directly from the Copernicus Open Access Hub.
        
        Args:
            collection_name: Name of the collection
            collection_config: Configuration for this collection
            start_date: Start date for download
            end_date: End date for download
            footprint: WKT representation of area of interest
            output_dir: Directory to save files
            
        Returns:
            List of paths to downloaded files
        """
        # Map collection name to Copernicus product type
        product_map = {
            'SENTINEL-2-L2A': 'S2MSI2A',
            'SENTINEL-1-GRD': 'GRD'
        }
        
        platform_map = {
            'SENTINEL-2-L2A': 'Sentinel-2',
            'SENTINEL-1-GRD': 'Sentinel-1'
        }
        
        product_type = product_map.get(collection_name)
        platform = platform_map.get(collection_name)
        
        if not product_type or not platform:
            self.logger.warning(_("Unsupported collection for Copernicus: {0}").format(collection_name))
            return []
        
        # Prepare query parameters
        query_kwargs = {
            'platformname': platform,
            'producttype': product_type,
            'date': (start_date, end_date),
            'footprint': f"intersects({footprint})"
        }
        
        # Add collection-specific parameters
        if collection_name == 'SENTINEL-2-L2A':
            query_kwargs['cloudcoverpercentage'] = (0, collection_config.get('max_cloud_coverage', 20))
        
        if collection_name == 'SENTINEL-1-GRD' and 'polarization' in collection_config:
            query_kwargs['polarisationmode'] = collection_config['polarization'].replace(',', ' ')
        
        if collection_name == 'SENTINEL-1-GRD' and 'orbit_direction' in collection_config:
            if collection_config['orbit_direction'] != 'BOTH':
                query_kwargs['orbitdirection'] = collection_config['orbit_direction']
        
        # Query products
        products = self.sentinel_api.query(**query_kwargs)
        
        if not products:
            self.logger.warning(_("No products found for {0} from {1} to {2}").format(
                collection_name, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            ))
            return []
        
        # Download products
        file_paths = []
        for product_id, product_info in tqdm(products.items(), desc=f"Downloading {collection_name}"):
            try:
                # Check if already downloaded
                title = product_info['title']
                output_file = output_dir / f"{title}.zip"
                
                if output_file.exists():
                    self.logger.info(_("Product already downloaded: {0}").format(title))
                    file_paths.append(str(output_file))
                    continue
                
                # Download product
                self.sentinel_api.download(product_id, directory_path=str(output_dir))
                file_paths.append(str(output_file))
                
            except Exception as e:
                self.logger.error(_("Failed to download product {0}: {1}").format(product_id, str(e)))
        
        return file_paths
    
    def _download_from_sentinelhub(
        self,
        collection_name: str,
        collection_config: Dict,
        start_date: datetime,
        end_date: datetime,
        bbox: BBox,
        output_dir: Path
    ) -> List[str]:
        """Download imagery from Sentinel Hub.
        
        Args:
            collection_name: Name of the collection
            collection_config: Configuration for this collection
            start_date: Start date for download
            end_date: End date for download
            bbox: Sentinel Hub BBox object
            output_dir: Directory to save files
            
        Returns:
            List of paths to downloaded files
        """
        # Get collection from mapping
        data_collection = self.collection_map.get(collection_name)
        if not data_collection:
            self.logger.warning(_("Unsupported collection for Sentinel Hub: {0}").format(collection_name))
            return []
        
        # Calculate resolution and dimensions
        resolution = self.config['sentinel']['processing']['resolution']
        size = bbox_to_dimensions(bbox, resolution=resolution)
        
        # Prepare time intervals based on temporal resolution
        temporal_resolution = collection_config.get('temporal_resolution', 5)  # days
        
        time_intervals = []
        current_date = start_date
        while current_date < end_date:
            interval_end = current_date + timedelta(days=temporal_resolution)
            if interval_end > end_date:
                interval_end = end_date
            
            time_intervals.append((current_date, interval_end))
            current_date = interval_end
        
        # Download for each time interval
        file_paths = []
        for interval_start, interval_end in time_intervals:
            # Format date strings
            time_interval = f"{interval_start.strftime('%Y-%m-%d')}T00:00:00Z/{interval_end.strftime('%Y-%m-%d')}T23:59:59Z"
            
            # Set up request configuration
            evalscript = self._get_evalscript_for_collection(collection_name, collection_config)
            
            request_config = {
                'dataCollection': data_collection,
                'evalscript': evalscript,
                'input': {
                    'bounds': {
                        'bbox': bbox.get_lower_left() + bbox.get_upper_right(),
                        'properties': {
                            'crs': bbox.crs.opengis_string
                        }
                    },
                    'data': [{
                        'dataFilter': {
                            'timeRange': time_interval
                        }
                    }]
                },
                'output': {
                    'width': size[0],
                    'height': size[1],
                    'responses': [{
                        'identifier': 'default',
                        'format': {
                            'type': MimeType.TIFF
                        }
                    }]
                }
            }
            
            # Add collection-specific parameters
            if collection_name == 'SENTINEL-2-L2A':
                request_config['input']['data'][0]['dataFilter']['maxCloudCoverage'] = collection_config.get('max_cloud_coverage', 20)
            
            # Create request
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=request_config['input']['data'],
                responses=request_config['output']['responses'],
                bbox=bbox,
                size=size,
                config=self.sh_config
            )
            
            # Execute request and save data
            try:
                data = request.get_data()
                
                if data:
                    # Generate output filename
                    filename = f"{collection_name}_{interval_start.strftime('%Y%m%d')}_{interval_end.strftime('%Y%m%d')}.tif"
                    output_file = output_dir / filename
                    
                    # Save GeoTIFF
                    with open(output_file, 'wb') as f:
                        f.write(data[0])
                    
                    file_paths.append(str(output_file))
            except Exception as e:
                self.logger.error(_("Failed to download from Sentinel Hub: {0}").format(str(e)))
        
        return file_paths
    
    def _download_sentinel5p(
        self,
        collection_config: Dict,
        start_date: datetime,
        end_date: datetime,
        bbox: BBox,
        output_dir: Path
    ) -> List[str]:
        """Download Sentinel-5P data.
        
        Args:
            collection_config: Configuration for this collection
            start_date: Start date for download
            end_date: End date for download
            bbox: Sentinel Hub BBox object
            output_dir: Directory to save files
            
        Returns:
            List of paths to downloaded files
        """
        # Sentinel-5P products are not available through sentinelsat API
        # We need to use Sentinel Hub for this
        data_collection = DataCollection.SENTINEL5P
        
        # Get product (defaulting to methane)
        product = collection_config.get('product', 'L2__CH4___')
        
        # Calculate resolution and dimensions
        resolution = self.config['sentinel']['processing']['resolution']
        size = bbox_to_dimensions(bbox, resolution=resolution)
        
        # Prepare time intervals (usually larger for S5P)
        temporal_resolution = collection_config.get('temporal_resolution', 30)  # days
        
        time_intervals = []
        current_date = start_date
        while current_date < end_date:
            interval_end = current_date + timedelta(days=temporal_resolution)
            if interval_end > end_date:
                interval_end = end_date
            
            time_intervals.append((current_date, interval_end))
            current_date = interval_end
        
        file_paths = []
        for interval_start, interval_end in time_intervals:
            # Format date strings
            time_interval = f"{interval_start.strftime('%Y-%m-%d')}T00:00:00Z/{interval_end.strftime('%Y-%m-%d')}T23:59:59Z"
            
            # Set up evalscript for methane
            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["CH4"],
                        units: "STANDARD"
                    }],
                    output: {
                        bands: 1,
                        sampleType: "FLOAT32"
                    }
                };
            }

            function evaluatePixel(sample) {
                return [sample.CH4];
            }
            """
            
            if product != 'L2__CH4___':
                # For other products, use a generic evalscript
                evalscript = f"""
                //VERSION=3
                function setup() {{
                    return {{
                        input: [{{
                            bands: ["{product}"],
                            units: "STANDARD"
                        }}],
                        output: {{
                            bands: 1,
                            sampleType: "FLOAT32"
                        }}
                    }};
                }}

                function evaluatePixel(sample) {{
                    return [sample.{product}];
                }}
                """
            
            # Create request
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[{
                    "dataFilter": {
                        "timeRange": time_interval,
                        "productType": product
                    },
                    "type": "sentinel-5p"
                }],
                responses=[{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }],
                bbox=bbox,
                size=size,
                config=self.sh_config
            )
            
            # Execute request and save data
            try:
                data = request.get_data()
                
                if data:
                    # Generate output filename
                    filename = f"SENTINEL-5P_{product}_{interval_start.strftime('%Y%m%d')}_{interval_end.strftime('%Y%m%d')}.tif"
                    output_file = output_dir / filename
                    
                    # Save GeoTIFF
                    with open(output_file, 'wb') as f:
                        f.write(data[0])
                    
                    file_paths.append(str(output_file))
            except Exception as e:
                self.logger.error(_("Failed to download Sentinel-5P data: {0}").format(str(e)))
        
        return file_paths
    
    def _get_evalscript_for_collection(self, collection_name: str, collection_config: Dict) -> str:
        """Get the appropriate evalscript for a collection.
        
        Args:
            collection_name: Name of the collection
            collection_config: Configuration for this collection
            
        Returns:
            Evalscript string
        """
        if collection_name == 'SENTINEL-2-L2A':
            # Get bands from config
            bands = collection_config.get('bands', ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
            bands_str = ', '.join([f'"{band}"' for band in bands])
            
            return f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: [{bands_str}],
                        units: "DN"
                    }}],
                    output: {{
                        bands: {len(bands)},
                        sampleType: "FLOAT32"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'sample.{band}' for band in bands])}];
            }}
            """
        
        elif collection_name == 'SENTINEL-1-GRD':
            polarizations = collection_config.get('polarization', 'VV,VH').split(',')
            polarizations_str = ', '.join([f'"{pol}"' for pol in polarizations])
            
            return f"""
            //VERSION=3
            function setup() {{
                return {{
                    input: [{{
                        bands: [{polarizations_str}],
                        units: "DB"
                    }}],
                    output: {{
                        bands: {len(polarizations)},
                        sampleType: "FLOAT32"
                    }}
                }};
            }}

            function evaluatePixel(sample) {{
                return [{', '.join([f'sample.{pol}' for pol in polarizations])}];
            }}
            """
        
        # Default script for other collections
        return """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01", "B02", "B03"],
                    units: "DN"
                }],
                output: {
                    bands: 3,
                    sampleType: "FLOAT32"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01, sample.B02, sample.B03];
        }
        """
