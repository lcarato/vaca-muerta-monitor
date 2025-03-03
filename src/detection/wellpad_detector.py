#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well Pad Detector

This module implements well pad detection using a combination of 
deep learning and computer vision techniques to identify well pads
in satellite imagery.
"""

import os
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
import rasterio
from rasterio.features import shapes
from sklearn.cluster import DBSCAN
import cv2
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import shapely
from osgeo import gdal
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class WellPadDetector:
    """Class for detecting well pads in satellite imagery."""
    
    def __init__(self, config: Dict):
        """Initialize the detector with configuration settings.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Set up output directory
        self.output_dir = Path(config['general']['data_directory']) / 'detections'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load detection parameters
        self.confidence_threshold = config['detection']['well_pad']['confidence_threshold']
        self.min_size = config['detection']['well_pad']['min_size']  # in square meters
        self.max_size = config['detection']['well_pad']['max_size']  # in square meters
        
        # Load model if specified
        self.model_path = config['detection']['well_pad']['model_path']
        self.model = None
        
        # Initialize model if path exists
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            self.logger.warning(_("Well pad detection model not found at {0}").format(self.model_path))
            self.logger.info(_("Will use conventional image processing instead of deep learning"))
    
    def _load_model(self):
        """Load the TensorFlow model for well pad detection."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info(_("Successfully loaded well pad detection model from {0}").format(self.model_path))
        except Exception as e:
            self.logger.error(_("Failed to load model: {0}").format(str(e)))
            self.model = None
    
    def detect(self, processed_images: Dict[str, List[Dict]]) -> List[Dict]:
        """Detect well pads in the processed satellite imagery.
        
        Args:
            processed_images: Dictionary mapping collection names to lists of processed tile info
            
        Returns:
            List of dictionaries with well pad information
        """
        self.logger.info(_("Starting well pad detection"))
        
        wellpads = []
        
        # Prioritize Sentinel-2 images for well pad detection (if available)
        if 'SENTINEL-2-L2A' in processed_images and processed_images['SENTINEL-2-L2A']:
            self.logger.info(_("Using Sentinel-2 imagery for well pad detection"))
            collection_images = processed_images['SENTINEL-2-L2A']
        elif 'SENTINEL-2' in processed_images and processed_images['SENTINEL-2']:
            self.logger.info(_("Using Sentinel-2 imagery for well pad detection"))
            collection_images = processed_images['SENTINEL-2']
        elif 'SENTINEL-1-GRD' in processed_images and processed_images['SENTINEL-1-GRD']:
            self.logger.info(_("Using Sentinel-1 imagery for well pad detection"))
            collection_images = processed_images['SENTINEL-1-GRD']
        elif 'SENTINEL-1' in processed_images and processed_images['SENTINEL-1']:
            self.logger.info(_("Using Sentinel-1 imagery for well pad detection"))
            collection_images = processed_images['SENTINEL-1']
        else:
            self.logger.error(_("No suitable imagery found for well pad detection"))
            return []
        
        # Process each tile
        for tile_info in tqdm(collection_images, desc="Detecting well pads"):
            # Select appropriate detection method based on the collection
            if 'SENTINEL-2' in tile_info['collection']:
                detected_pads = self._detect_in_optical(tile_info)
            elif 'SENTINEL-1' in tile_info['collection']:
                detected_pads = self._detect_in_sar(tile_info)
            else:
                continue
            
            wellpads.extend(detected_pads)
        
        # Post-process to merge duplicates and filter results
        wellpads = self._post_process(wellpads)
        
        self.logger.info(_("Detected {0} well pads").format(len(wellpads)))
        
        # Save results to GeoJSON
        if wellpads:
            self._save_results(wellpads)
        
        return wellpads
    
    def _detect_in_optical(self, tile_info: Dict) -> List[Dict]:
        """Detect well pads in optical (Sentinel-2) imagery.
        
        Args:
            tile_info: Dictionary with tile information
            
        Returns:
            List of dictionaries with well pad information
        """
        wellpads = []
        
        # If we have a model, use it
        if self.model is not None:
            wellpads = self._detect_with_model(tile_info)
        else:
            # Otherwise, use conventional image processing
            wellpads = self._detect_with_indices(tile_info)
        
        return wellpads
    
    def _detect_with_model(self, tile_info: Dict) -> List[Dict]:
        """Detect well pads using the deep learning model.
        
        Args:
            tile_info: Dictionary with tile information
            
        Returns:
            List of dictionaries with well pad information
        """
        wellpads = []
        
        try:
            # Open the image
            with rasterio.open(tile_info['path']) as src:
                # Read image data and prepare for model
                img = src.read()
                
                # Transpose to format expected by model (H,W,C)
                img = np.transpose(img, (1, 2, 0))
                
                # Normalize image
                img = img.astype(np.float32)
                for i in range(img.shape[2]):
                    band = img[:, :, i]
                    min_val = np.percentile(band[band > 0], 2)
                    max_val = np.percentile(band[band > 0], 98)
                    img[:, :, i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
                
                # Resize if needed to match model input shape
                model_input_shape = self.model.input_shape[1:3]  # (height, width)
                if img.shape[0] != model_input_shape[0] or img.shape[1] != model_input_shape[1]:
                    img = cv2.resize(img, (model_input_shape[1], model_input_shape[0]))
                
                # Ensure we have the right number of channels
                expected_channels = self.model.input_shape[3]
                if img.shape[2] != expected_channels:
                    if img.shape[2] > expected_channels:
                        img = img[:, :, :expected_channels]
                    else:
                        # Pad with zeros
                        padded = np.zeros((img.shape[0], img.shape[1], expected_channels), dtype=np.float32)
                        padded[:, :, :img.shape[2]] = img
                        img = padded
                
                # Add batch dimension
                img = np.expand_dims(img, axis=0)
                
                # Run inference
                prediction = self.model.predict(img)
                
                # Process prediction (assuming segmentation output)
                if prediction.shape[1:3] != (src.height, src.width):
                    prediction = cv2.resize(prediction[0], (src.width, src.height))
                    prediction = np.expand_dims(prediction, axis=0)
                
                # Binarize the prediction
                binary_mask = (prediction[0] > self.confidence_threshold).astype(np.uint8)
                
                # Apply morphological operations to clean up the mask
                kernel = np.ones((5, 5), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                
                # Extract polygons from the mask
                pad_polygons = []
                for geom, value in shapes(binary_mask, transform=src.transform):
                    if value == 1:  # Only consider areas predicted as well pads
                        polygon = shape(geom)
                        
                        # Filter by size
                        area_m2 = polygon.area  # Area in square meters (assuming projection in meters)
                        if self.min_size <= area_m2 <= self.max_size:
                            pad_polygons.append(polygon)
                
                # Create well pad entries
                for i, polygon in enumerate(pad_polygons):
                    wellpads.append({
                        'id': f"wellpad_{Path(tile_info['path']).stem}_{i}",
                        'geometry': mapping(polygon),
                        'confidence': float(np.mean(prediction[0][binary_mask == 1])),
                        'area': polygon.area,
                        'source_image': tile_info['path'],
                        'timestamp': tile_info['timestamp'],
                        'bounds': tile_info['bounds'],
                    })
        
        except Exception as e:
            self.logger.error(_("Error detecting well pads with model in {0}: {1}").format(tile_info['path'], str(e)))
        
        return wellpads
    
    def _detect_with_indices(self, tile_info: Dict) -> List[Dict]:
        """Detect well pads using spectral indices and image processing.
        
        Args:
            tile_info: Dictionary with tile information
            
        Returns:
            List of dictionaries with well pad information
        """
        wellpads = []
        
        try:
            # Open the image
            with rasterio.open(tile_info['path']) as src:
                # Read image data
                img = src.read()
                
                # For Sentinel-2, compute indices useful for bare soil detection
                if img.shape[0] >= 8:  # Ensure we have enough bands
                    # Calculate NDBI (Normalized Difference Built-up Index)
                    # For Sentinel-2: (B11 - B08) / (B11 + B08)
                    # Band indices are 0-based, so B11 is 10, B08 is 7
                    swir = img[10].astype(np.float32)
                    nir = img[7].astype(np.float32)
                    
                    # Handle division by zero
                    denominator = swir + nir
                    ndbi = np.zeros_like(denominator)
                    valid = denominator > 0
                    ndbi[valid] = (swir[valid] - nir[valid]) / denominator[valid]
                    
                    # Calculate BSI (Bare Soil Index)
                    # For Sentinel-2: ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
                    # B04 is 3, B02 is 1
                    red = img[3].astype(np.float32)
                    blue = img[1].astype(np.float32)
                    
                    numerator = (swir + red) - (nir + blue)
                    denominator = (swir + red) + (nir + blue)
                    bsi = np.zeros_like(denominator)
                    valid = denominator > 0
                    bsi[valid] = numerator[valid] / denominator[valid]
                    
                    # Combine indices to enhance well pad detection
                    combined = (ndbi + 1) * 0.5 + (bsi + 1) * 0.5  # Rescale from [-1,1] to [0,1]
                    
                    # Threshold the combined index
                    threshold = np.percentile(combined[combined > 0], 90)  # Adaptive threshold
                    binary_mask = (combined > threshold).astype(np.uint8)
                    
                    # Apply morphological operations
                    kernel = np.ones((5, 5), np.uint8)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Extract polygons from the mask
                    pad_polygons = []
                    for geom, value in shapes(binary_mask, transform=src.transform):
                        if value == 1:
                            polygon = shape(geom)
                            
                            # Filter by size
                            area_m2 = polygon.area
                            if self.min_size <= area_m2 <= self.max_size:
                                pad_polygons.append(polygon)
                    
                    # Create well pad entries
                    for i, polygon in enumerate(pad_polygons):
                        wellpads.append({
                            'id': f"wellpad_{Path(tile_info['path']).stem}_{i}",
                            'geometry': mapping(polygon),
                            'confidence': 0.7,  # Arbitrary confidence for index-based detection
                            'area': polygon.area,
                            'source_image': tile_info['path'],
                            'timestamp': tile_info['timestamp'],
                            'bounds': tile_info['bounds'],
                        })
                
                else:
                    self.logger.warning(
                        _("Not enough bands in {0} for spectral index calculation").format(tile_info['path'])
                    )
        
        except Exception as e:
            self.logger.error(
                _("Error detecting well pads with indices in {0}: {1}").format(tile_info['path'], str(e))
            )
        
        return wellpads
    
    def _detect_in_sar(self, tile_info: Dict) -> List[Dict]:
        """Detect well pads in SAR (Sentinel-1) imagery.
        
        Args:
            tile_info: Dictionary with tile information
            
        Returns:
            List of dictionaries with well pad information
        """
        wellpads = []
        
        try:
            # Open the image
            with rasterio.open(tile_info['path']) as src:
                # Read image data
                img = src.read()
                
                # For Sentinel-1, use VV/VH ratio to enhance man-made structures
                if img.shape[0] >= 2:  # Ensure we have both VV and VH polarizations
                    # Extract VV and VH bands
                    vv = img[0].astype(np.float32)
                    vh = img[1].astype(np.float32)
                    
                    # Calculate VV/VH ratio (with protection against division by zero)
                    vh_safe = vh.copy()
                    vh_safe[vh_safe <= 0] = 1e-6
                    ratio = vv / vh_safe
                    
                    # Apply threshold to the ratio image
                    # Well pads often have high VV/VH ratio due to corner reflectors
                    threshold = np.percentile(ratio[~np.isnan(ratio) & (ratio > 0)], 95)
                    binary_mask = (ratio > threshold).astype(np.uint8)
                    
                    # Apply morphological operations
                    kernel = np.ones((5, 5), np.uint8)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                    
                    # Extract polygons from the mask
                    pad_polygons = []
                    for geom, value in shapes(binary_mask, transform=src.transform):
                        if value == 1:
                            polygon = shape(geom)
                            
                            # Filter by size
                            area_m2 = polygon.area
                            if self.min_size <= area_m2 <= self.max_size:
                                pad_polygons.append(polygon)
                    
                    # Create well pad entries
                    for i, polygon in enumerate(pad_polygons):
                        wellpads.append({
                            'id': f"wellpad_{Path(tile_info['path']).stem}_{i}",
                            'geometry': mapping(polygon),
                            'confidence': 0.6,  # Lower confidence for SAR-based detection
                            'area': polygon.area,
                            'source_image': tile_info['path'],
                            'timestamp': tile_info['timestamp'],
                            'bounds': tile_info['bounds'],
                        })
                
                else:
                    self.logger.warning(
                        _("Not enough polarizations in {0} for SAR-based detection").format(tile_info['path'])
                    )
        
        except Exception as e:
            self.logger.error(_("Error detecting well pads in SAR image {0}: {1}").format(tile_info['path'], str(e)))
        
        return wellpads
    
    def _post_process(self, wellpads: List[Dict]) -> List[Dict]:
        """Post-process detected well pads to merge duplicates and filter results.
        
        Args:
            wellpads: List of dictionaries with well pad information
            
        Returns:
            Filtered and merged list of well pad dictionaries
        """
        if not wellpads:
            return []
        
        # Convert to GeoDataFrame for spatial operations
        gdf = gpd.GeoDataFrame([
            {
                'id': pad['id'],
                'geometry': shape(pad['geometry']),
                'confidence': pad['confidence'],
                'area': pad['area'],
                'source_image': pad['source_image'],
                'timestamp': pad['timestamp'],
                'bounds': pad['bounds'],
            }
            for pad in wellpads
        ])
        
        # Set CRS to WGS 84
        gdf.crs = 'EPSG:4326'
        
        # Identify overlapping geometries
        def merge_overlaps(group):
            # If only one geometry in group, return it
            if len(group) == 1:
                return group.iloc[0]
            
            # Otherwise, merge overlapping geometries
            union_geom = group.geometry.unary_union
            
            # If the result is a MultiPolygon, take the largest polygon
            if isinstance(union_geom, MultiPolygon):
                largest = max(union_geom.geoms, key=lambda g: g.area)
                union_geom = largest
            
            # Create merged entry
            highest_conf_idx = group['confidence'].idxmax()
            
            return gpd.GeoSeries({
                'id': group.iloc[0]['id'],
                'geometry': union_geom,
                'confidence': group.loc[highest_conf_idx, 'confidence'],
                'area': union_geom.area,
                'source_image': group.loc[highest_conf_idx, 'source_image'],
                'timestamp': group.loc[highest_conf_idx, 'timestamp'],
                'bounds': group.loc[highest_conf_idx, 'bounds'],
            })
        
        # Find overlapping geometries using spatial index
        sindex = gdf.sindex
        merged_pads = []
        processed = set()
        
        for idx, geom in enumerate(gdf.geometry):
            if idx in processed:
                continue
                
            # Find geometries that potentially intersect
            candidates_idx = list(sindex.intersection(geom.bounds))
            candidates = gdf.iloc[candidates_idx]
            
            # Find actual intersections
            intersects = candidates[candidates.intersects(geom)]
            
            # Add to processed set
            processed.update(intersects.index)
            
            # Merge overlapping geometries
            merged = merge_overlaps(intersects)
            merged_pads.append(merged)
        
        # Create new GeoDataFrame with merged geometries
        if merged_pads:
            merged_gdf = gpd.GeoDataFrame(merged_pads)
            
            # Filter again by size
            merged_gdf = merged_gdf[
                (merged_gdf.area >= self.min_size) & 
                (merged_gdf.area <= self.max_size)
            ]
            
            # Convert back to list of dictionaries
            result = merged_gdf.to_dict('records')
            
            # Convert geometries back to GeoJSON format
            for pad in result:
                pad['geometry'] = mapping(pad['geometry'])
            
            return result
        
        return wellpads
    
    def _save_results(self, wellpads: List[Dict]) -> None:
        """Save detected well pads to GeoJSON file.
        
        Args:
            wellpads: List of dictionaries with well pad information
        """
        # Create GeoJSON feature collection
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": pad['id'],
                    "geometry": pad['geometry'],
                    "properties": {
                        "confidence": pad['confidence'],
                        "area": pad['area'],
                        "source_image": os.path.basename(pad['source_image']),
                        "timestamp": '_'.join(pad['timestamp']),
                    }
                }
                for pad in wellpads
            ]
        }
        
        # Save to file with timestamp
        timestamp = wellpads[0]['timestamp'][0]
        output_path = self.output_dir / f"wellpads_{timestamp}.geojson"
        
        with open(output_path, 'w') as f:
            import json
            json.dump(feature_collection, f, indent=2)
        
        self.logger.info(_("Saved {0} detected well pads to {1}").format(len(wellpads), output_path))
