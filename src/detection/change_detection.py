#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite Image Processor

This module processes raw Sentinel satellite imagery for well detection,
including radiometric calibration, atmospheric correction, tiling,
and preprocessing for machine learning models.
"""

import os
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from rasterio.plot import reshape_as_image
from osgeo import gdal
from scipy.ndimage import median_filter
import cv2
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box, mapping

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class ImageProcessor:
    """Class for processing satellite imagery for well pad detection."""
    
    def __init__(self, config: Dict):
        """Initialize the processor with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directory
        self.output_dir = Path(config['general']['data_directory']) / 'processed'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get processing parameters
        self.resolution = config['sentinel']['processing']['resolution']
        self.tile_size = config['sentinel']['processing']['tile_size']
        self.tile_overlap = config['sentinel']['processing']['tile_overlap']
    
    def process(self, imagery: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """Process satellite imagery for detection tasks.
        
        Args:
            imagery: Dictionary mapping collection names to lists of image paths
            
        Returns:
            Dictionary mapping collection names to lists of processed tile information
        """
        result = {}
        
        for collection_name, image_paths in imagery.items():
            self.logger.info(_("Processing {0} imagery").format(collection_name))
            
            collection_dir = self.output_dir / collection_name
            os.makedirs(collection_dir, exist_ok=True)
            
            processed_tiles = []
            
            for image_path in tqdm(image_paths, desc=f"Processing {collection_name} images"):
                # Select appropriate processing method based on collection
                if collection_name.startswith('SENTINEL-2'):
                    tiles = self._process_optical(image_path, collection_dir)
                elif collection_name.startswith('SENTINEL-1'):
                    tiles = self._process_sar(image_path, collection_dir)
                elif collection_name.startswith('SENTINEL-5P'):
                    tiles = self._process_sentinel5p(image_path, collection_dir)
                else:
                    self.logger.warning(_("Unsupported collection for processing: {0}").format(collection_name))
                    continue
                
                processed_tiles.extend(tiles)
            
            result[collection_name] = processed_tiles
        
        return result
    
    def _process_optical(self, image_path: str, output_dir: Path) -> List[Dict]:
        """Process optical imagery (Sentinel-2).
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save processed tiles
            
        Returns:
            List of dictionaries with processed tile information
        """
        tiles = []
        image_filename = Path(image_path).stem
        
        try:
            with rasterio.open(image_path) as src:
                # Check if image needs to be reprojected
                dst_crs = 'EPSG:4326'  # WGS84
                
                # If source CRS is different from target CRS, reproject
                if src.crs != dst_crs:
                    reprojected_path = output_dir / f"{image_filename}_reprojected.tif"
                    
                    # Calculate the transformation parameters
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create the reprojected raster
                    with rasterio.open(
                        reprojected_path,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=dst_crs,
                        transform=transform,
                        nodata=src.nodata
                    ) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest
                            )
                    
                    # Use the reprojected image for further processing
                    src_path = reprojected_path
                else:
                    src_path = image_path
                
                # Now process the correctly projected image
                with rasterio.open(src_path) as src:
                    # Read metadata
                    meta = src.meta.copy()
                    
                    # Ensure all bands have the same dimensions
                    min_shape = min([src.shape for i in range(1, src.count + 1)])
                    
                    # Tile the image
                    tile_size = self.tile_size
                    overlap = self.tile_overlap
                    
                    # Calculate effective step size
                    step_size = tile_size - overlap
                    
                    # Calculate number of tiles in each dimension
                    n_tiles_y = (min_shape[0] - overlap) // step_size
                    n_tiles_x = (min_shape[1] - overlap) // step_size
                    
                    for i in range(n_tiles_y):
                        for j in range(n_tiles_x):
                            # Calculate tile coordinates
                            y_start = i * step_size
                            x_start = j * step_size
                            
                            # Create window for this tile
                            window = Window(x_start, y_start, tile_size, tile_size)
                            
                            # Read the tile data for all bands
                            tile_data = src.read(window=window)
                            
                            # Skip tiles with too many no-data values
                            if np.sum(tile_data == src.nodata) / tile_data.size > 0.5:
                                continue
                            
                            # Calculate geolocation of tile
                            tile_transform = src.window_transform(window)
                            tile_bounds = rasterio.windows.bounds(window, src.transform)
                            
                            # Generate tile name
                            tile_name = f"{image_filename}_tile_{i}_{j}.tif"
                            tile_path = output_dir / tile_name
                            
                            # Create GeoTIFF for this tile
                            tile_meta = meta.copy()
                            tile_meta.update({
                                'height': tile_size,
                                'width': tile_size,
                                'transform': tile_transform
                            })
                            
                            with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                                dst.write(tile_data)
                            
                            # Create a false color RGB image for visualization
                            rgb_path = output_dir / f"{image_filename}_tile_{i}_{j}_rgb.tif"
                            
                            # For Sentinel-2, typically use bands 4,3,2 (R,G,B) for natural color
                            # or 8,4,3 (NIR,R,G) for false color
                            if tile_data.shape[0] >= 8:  # Ensure we have enough bands
                                rgb_bands = [3, 2, 1]  # 0-indexed, so bands 4,3,2
                                rgb_data = tile_data[rgb_bands, :, :]
                                
                                # Normalize values for visualization
                                for b in range(3):
                                    band_data = rgb_data[b, :, :]
                                    p2 = np.percentile(band_data[band_data > 0], 2)
                                    p98 = np.percentile(band_data[band_data > 0], 98)
                                    rgb_data[b, :, :] = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                                
                                rgb_meta = tile_meta.copy()
                                rgb_meta.update({
                                    'count': 3,
                                    'dtype': 'uint8'
                                })
                                
                                with rasterio.open(rgb_path, 'w', **rgb_meta) as dst:
                                    dst.write(rgb_data.astype('uint8'))
                            
                            # Add tile info to result
                            tiles.append({
                                'path': str(tile_path),
                                'rgb_path': str(rgb_path) if tile_data.shape[0] >= 8 else None,
                                'bounds': tile_bounds,
                                'timestamp': image_filename.split('_')[2:4],  # Extract date info from filename
                                'collection': 'SENTINEL-2'
                            })
                        
        except Exception as e:
            self.logger.error(_("Error processing optical image {0}: {1}").format(image_path, str(e)))
        
        return tiles
    
    def _process_sentinel5p(self, image_path: str, output_dir: Path) -> List[Dict]:
        """Process Sentinel-5P imagery (methane data).
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save processed tiles
            
        Returns:
            List of dictionaries with processed tile information
        """
        tiles = []
        image_filename = Path(image_path).stem
        
        try:
            with rasterio.open(image_path) as src:
                # Read metadata
                meta = src.meta.copy()
                
                # Read data
                data = src.read()
                
                # For Sentinel-5P, we may not need to tile as resolution is usually coarser
                # But still divide into manageable regions if very large
                if src.height > 5000 or src.width > 5000:
                    # Use larger tiles for S5P
                    tile_size = 2 * self.tile_size
                    overlap = self.tile_overlap
                    
                    # Calculate effective step size
                    step_size = tile_size - overlap
                    
                    # Calculate number of tiles in each dimension
                    n_tiles_y = max(1, (src.height - overlap) // step_size)
                    n_tiles_x = max(1, (src.width - overlap) // step_size)
                    
                    for i in range(n_tiles_y):
                        for j in range(n_tiles_x):
                            # Calculate tile coordinates
                            y_start = i * step_size
                            x_start = j * step_size
                            
                            # Adjust for image boundaries
                            y_end = min(y_start + tile_size, src.height)
                            x_end = min(x_start + tile_size, src.width)
                            actual_height = y_end - y_start
                            actual_width = x_end - x_start
                            
                            # Skip if tile is too small
                            if actual_height < tile_size / 2 or actual_width < tile_size / 2:
                                continue
                            
                            # Create window
                            window = Window(x_start, y_start, actual_width, actual_height)
                            
                            # Read the tile data
                            tile_data = src.read(window=window)
                            
                            # Skip tiles with too many no-data or invalid values
                            if (
                                (src.nodata is not None and np.sum(tile_data == src.nodata) / tile_data.size > 0.7) or
                                np.sum(np.isnan(tile_data)) / tile_data.size > 0.7
                            ):
                                continue
                            
                            # Calculate geolocation of tile
                            tile_transform = src.window_transform(window)
                            tile_bounds = rasterio.windows.bounds(window, src.transform)
                            
                            # Generate tile name
                            tile_name = f"{image_filename}_tile_{i}_{j}.tif"
                            tile_path = output_dir / tile_name
                            
                            # Create GeoTIFF for this tile
                            tile_meta = meta.copy()
                            tile_meta.update({
                                'height': actual_height,
                                'width': actual_width,
                                'transform': tile_transform
                            })
                            
                            with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                                dst.write(tile_data)
                            
                            # Create a visualization image
                            vis_path = output_dir / f"{image_filename}_tile_{i}_{j}_vis.tif"
                            
                            # For methane, create a single-band visualization with colormap
                            vis_data = tile_data.copy()
                            
                            # Replace NaN with 0
                            vis_data = np.nan_to_num(vis_data, nan=0)
                            
                            # Normalize data for visualization
                            for b in range(vis_data.shape[0]):
                                band_data = vis_data[b]
                                valid_data = band_data[band_data > 0]
                                if valid_data.size > 0:
                                    p2 = np.percentile(valid_data, 2)
                                    p98 = np.percentile(valid_data, 98)
                                    vis_data[b] = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                            
                            vis_meta = tile_meta.copy()
                            vis_meta.update({'dtype': 'uint8'})
                            
                            with rasterio.open(vis_path, 'w', **vis_meta) as dst:
                                dst.write(vis_data.astype('uint8'))
                            
                            # Add tile info to result
                            tiles.append({
                                'path': str(tile_path),
                                'vis_path': str(vis_path),
                                'bounds': tile_bounds,
                                'timestamp': image_filename.split('_')[2:4],  # Extract date info from filename
                                'collection': 'SENTINEL-5P'
                            })
                else:
                    # For smaller images, just save the whole thing without tiling
                    # Generate output filename
                    out_name = f"{image_filename}_processed.tif"
                    out_path = output_dir / out_name
                    
                    # Save processed file
                    with rasterio.open(out_path, 'w', **meta) as dst:
                        dst.write(data)
                    
                    # Create visualization
                    vis_path = output_dir / f"{image_filename}_vis.tif"
                    
                    # Normalize data for visualization
                    vis_data = data.copy()
                    vis_data = np.nan_to_num(vis_data, nan=0)
                    
                    for b in range(vis_data.shape[0]):
                        band_data = vis_data[b]
                        valid_data = band_data[band_data > 0]
                        if valid_data.size > 0:
                            p2 = np.percentile(valid_data, 2)
                            p98 = np.percentile(valid_data, 98)
                            vis_data[b] = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                    
                    vis_meta = meta.copy()
                    vis_meta.update({'dtype': 'uint8'})
                    
                    with rasterio.open(vis_path, 'w', **vis_meta) as dst:
                        dst.write(vis_data.astype('uint8'))
                    
                    # Add to result
                    tiles.append({
                        'path': str(out_path),
                        'vis_path': str(vis_path),
                        'bounds': src.bounds,
                        'timestamp': image_filename.split('_')[2:4],  # Extract date info from filename
                        'collection': 'SENTINEL-5P'
                    })
                    
        except Exception as e:
            self.logger.error(_("Error processing Sentinel-5P image {0}: {1}").format(image_path, str(e)))
        
        return tiles
    
    def _process_sar(self, image_path: str, output_dir: Path) -> List[Dict]:
        """Process SAR imagery (Sentinel-1).
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save processed tiles
            
        Returns:
            List of dictionaries with processed tile information
        """
        tiles = []
        image_filename = Path(image_path).stem
        
        try:
            with rasterio.open(image_path) as src:
                # Check if image needs to be reprojected
                dst_crs = 'EPSG:4326'  # WGS84
                
                # If source CRS is different from target CRS, reproject
                if src.crs != dst_crs:
                    reprojected_path = output_dir / f"{image_filename}_reprojected.tif"
                    
                    # Calculate the transformation parameters
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create the reprojected raster
                    with rasterio.open(
                        reprojected_path,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=dst_crs,
                        transform=transform,
                        nodata=src.nodata
                    ) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest
                            )
                    
                    # Use the reprojected image for further processing
                    src_path = reprojected_path
                else:
                    src_path = image_path
                
                # Now process the correctly projected image
                with rasterio.open(src_path) as src:
                    # Read metadata
                    meta = src.meta.copy()
                    
                    # Apply SAR-specific preprocessing
                    # For Sentinel-1, apply speckle filtering
                    data = src.read()
                    
                    # Apply median filter for speckle reduction
                    filtered_data = np.zeros_like(data)
                    for band in range(data.shape[0]):
                        filtered_data[band] = median_filter(data[band], size=5)
                    
                    # Tile the filtered image
                    tile_size = self.tile_size
                    overlap = self.tile_overlap
                    
                    # Calculate effective step size
                    step_size = tile_size - overlap
                    
                    # Calculate number of tiles in each dimension
                    n_tiles_y = (src.height - overlap) // step_size
                    n_tiles_x = (src.width - overlap) // step_size
                    
                    for i in range(n_tiles_y):
                        for j in range(n_tiles_x):
                            # Calculate tile coordinates
                            y_start = i * step_size
                            x_start = j * step_size
                            
                            # Ensure we don't go beyond image bounds
                            if y_start + tile_size > src.height or x_start + tile_size > src.width:
                                continue
                            
                            # Extract tile data
                            tile_data = filtered_data[:, y_start:y_start+tile_size, x_start:x_start+tile_size]
                            
                            # Skip tiles with too many no-data values
                            if src.nodata is not None and np.sum(tile_data == src.nodata) / tile_data.size > 0.5:
                                continue
                            
                            # Calculate geolocation of tile
                            tile_transform = src.window_transform(Window(x_start, y_start, tile_size, tile_size))
                            tile_bounds = rasterio.windows.bounds(
                                Window(x_start, y_start, tile_size, tile_size), 
                                src.transform
                            )
                            
                            # Generate tile name
                            tile_name = f"{image_filename}_tile_{i}_{j}.tif"
                            tile_path = output_dir / tile_name
                            
                            # Create GeoTIFF for this tile
                            tile_meta = meta.copy()
                            tile_meta.update({
                                'height': tile_size,
                                'width': tile_size,
                                'transform': tile_transform
                            })
                            
                            with rasterio.open(tile_path, 'w', **tile_meta) as dst:
                                dst.write(tile_data)
                            
                            # For SAR, create a visualization image (VV,VH,VV/VH)
                            vis_path = output_dir / f"{image_filename}_tile_{i}_{j}_vis.tif"
                            
                            if tile_data.shape[0] >= 2:  # If we have both VV and VH polarizations
                                # Create a 3-band visualization
                                vis_data = np.zeros((3, tile_size, tile_size), dtype=np.float32)
                                
                                # Use VV for first band
                                vis_data[0] = tile_data[0]
                                
                                # Use VH for second band
                                vis_data[1] = tile_data[1]
                                
                                # Use VV/VH ratio for third band (with protection against division by zero)
                                vv = tile_data[0].copy()
                                vh = tile_data[1].copy()
                                vh[vh == 0] = 1e-6  # Prevent division by zero
                                vis_data[2] = vv / vh
                                
                                # Clip and normalize each band for visualization
                                for b in range(3):
                                    band_data = vis_data[b]
                                    valid_data = band_data[~np.isnan(band_data) & (band_data != 0)]
                                    if valid_data.size > 0:
                                        p2 = np.percentile(valid_data, 2)
                                        p98 = np.percentile(valid_data, 98)
                                        vis_data[b] = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                                    else:
                                        vis_data[b] = np.zeros_like(band_data)
                                
                                vis_meta = tile_meta.copy()
                                vis_meta.update({
                                    'count': 3,
                                    'dtype': 'uint8'
                                })
                                
                                with rasterio.open(vis_path, 'w', **vis_meta) as dst:
                                    dst.write(vis_data.astype('uint8'))
                            
                            # Add tile info to result
                            tiles.append({
                                'path': str(tile_path),
                                'vis_path': str(vis_path) if tile_data.shape[0] >= 2 else None,
                                'bounds': tile_bounds,
                                'timestamp': image_filename.split('_')[2:4],  # Extract date info from filename
                                'collection': 'SENTINEL-1'
                            })
