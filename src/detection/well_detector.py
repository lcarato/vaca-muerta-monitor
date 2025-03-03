#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well Detector

This module implements detection of individual wells within well pads
using template matching and image processing techniques.
"""

import os
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping, Point
import cv2
import geopandas as gpd
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.localization import translate as _

class WellDetector:
    """Class for detecting individual wells within well pads."""
    
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
        self.detection_method = config['detection']['well']['detection_method']
        self.confidence_threshold = config['detection']['well']['confidence_threshold']
        
        # Load templates if using template matching
        if self.detection_method == 'template_matching':
            self.template_dir = Path(config['detection']['well']['template_directory'])
            self.templates = self._load_templates()
    
    def _load_templates(self) -> List[Dict]:
        """Load well templates for template matching.
        
        Returns:
            List of dictionaries with template information
        """
        templates = []
        
        # Create template directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Check for template files
        template_files = list(self.template_dir.glob('*.png')) + \
                         list(self.template_dir.glob('*.jpg')) + \
                         list(self.template_dir.glob('*.tif'))
        
        if not template_files:
            self.logger.warning(_("No template files found in {0}").format(self.template_dir))
            # Create default templates if none exist
            self._create_default_templates()
            template_files = list(self.template_dir.glob('*.png'))
        
        # Load templates
        for template_file in template_files:
            template = cv2.imread(str(template_file))
            if template is not None:
                templates.append({
                    'image': template,
                    'name': template_file.stem
                })
        
        self.logger.info(_("Loaded {0} well templates for detection").format(len(templates)))
        return templates
    
    def _create_default_templates(self) -> None:
        """Create default well templates for detection."""
        # Create simple circular template
        circle_template = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(circle_template, (16, 16), 12, 255, -1)
        cv2.imwrite(str(self.template_dir / 'circle_well.png'), circle_template)
        
        # Create square template (well pad equipment)
        square_template = np.zeros((32, 32), dtype=np.uint8)
        cv2.rectangle(square_template, (8, 8), (24, 24), 255, -1)
        cv2.imwrite(str(self.template_dir / 'square_equipment.png'), square_template)
        
        # Create plus shape template (typical well pattern)
        plus_template = np.zeros((32, 32), dtype=np.uint8)
        cv2.rectangle(plus_template, (14, 8), (18, 24), 255, -1)
        cv2.rectangle(plus_template, (8, 14), (24, 18), 255, -1)
        cv2.imwrite(str(self.template_dir / 'plus_well.png'), plus_template)
        
        self.logger.info(_("Created default well templates"))
    
    def detect(self, processed_images: Dict[str, List[Dict]], wellpads: List[Dict]) -> List[Dict]:
        """Detect individual wells within the identified well pads.
        
        Args:
            processed_images: Dictionary mapping collection names to lists of processed tile info
            wellpads: List of dictionaries with well pad information
            
        Returns:
            List of dictionaries with well information
        """
        self.logger.info(_("Starting individual well detection"))
        
        wells = []
        
        # Group wellpads by source image for efficient processing
        wellpads_by_image = {}
        for pad in wellpads:
            if pad['source_image'] not in wellpads_by_image:
                wellpads_by_image[pad['source_image']] = []
            wellpads_by_image[pad['source_image']].append(pad)
        
        # Find all processed images
        all_images = []
        for collection, images in processed_images.items():
            all_images.extend(images)
        
        # Create a map of image paths to RGB/vis paths for visualization
        vis_path_map = {}
        for img in all_images:
            if 'rgb_path' in img and img['rgb_path']:
                vis_path_map[img['path']] = img['rgb_path']
            elif 'vis_path' in img and img['vis_path']:
                vis_path_map[img['path']] = img['vis_path']
        
        # Process each image that contains wellpads
        for image_path, pads in wellpads_by_image.items():
            # Find the corresponding visualization image
            vis_path = vis_path_map.get(image_path, image_path)
            
            # Select appropriate detection method
            if self.detection_method == 'template_matching':
                detected_wells = self._detect_with_templates(image_path, vis_path, pads)
            else:
                detected_wells = self._detect_with_image_processing(image_path, vis_path, pads)
            
            wells.extend(detected_wells)
        
        self.logger.info(_("Detected {0} individual wells").format(len(wells)))
        
        # Save results to GeoJSON
        if wells:
            self._save_results(wells)
        
        return wells
    
    def _detect_with_templates(
        self, 
        image_path: str, 
        vis_path: str, 
        wellpads: List[Dict]
    ) -> List[Dict]:
        """Detect wells using template matching.
        
        Args:
            image_path: Path to the source image
            vis_path: Path to the visualization image
            wellpads: List of well pad dictionaries located in this image
            
        Returns:
            List of dictionaries with well information
        """
        wells = []
        
        try:
            # Open the visualization image for template matching
            # (Usually better to use the RGB/visualization image than raw bands)
            vis_img = cv2.imread(vis_path)
            if vis_img is None:
                # If reading fails, try with rasterio and convert
                with rasterio.open(vis_path) as src:
                    vis_data = src.read()
                    vis_img = np.transpose(vis_data, (1, 2, 0))
                    
                    # If more than 3 bands, select first 3
                    if vis_img.shape[2] > 3:
                        vis_img = vis_img[:, :, :3]
                    
                    # If single band, duplicate to 3 channels
                    if vis_img.shape[2] == 1:
                        vis_img = np.repeat(vis_img, 3, axis=2)
                    
                    # Normalize to 0-255 range
                    vis_img = (vis_img * 255 / vis_img.max()).astype(np.uint8)
            
            # Convert to grayscale for template matching
            vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
            
            # Open the source image to get geotransform
            with rasterio.open(image_path) as src:
                transform = src.transform
                
                # Process each well pad
                for pad_idx, pad in enumerate(wellpads):
                    # Get the geometry as shapely object
                    pad_geom = shape(pad['geometry'])
                    
                    # Get bounding box of the pad
                    minx, miny, maxx, maxy = pad_geom.bounds
                    
                    # Convert to pixel coordinates
                    row_min, col_min = ~transform * (minx, maxy)
                    row_max, col_max = ~transform * (maxx, miny)
                    
                    # Ensure coordinates are within image bounds
                    row_min = max(0, int(row_min))
                    col_min = max(0, int(col_min))
                    row_max = min(vis_gray.shape[0], int(row_max))
                    col_max = min(vis_gray.shape[1], int(col_max))
                    
                    # Extract the well pad region
                    pad_img = vis_gray[row_min:row_max, col_min:col_max]
                    
                    # Skip if the pad region is too small
                    if pad_img.shape[0] < 20 or pad_img.shape[1] < 20:
                        continue
                    
                    # Apply template matching for each template
                    well_locations = []
                    
                    for template in self.templates:
                        # Convert template to grayscale
                        tpl_gray = cv2.cvtColor(template['image'], cv2.COLOR_BGR2GRAY) \
                            if len(template['image'].shape) == 3 else template['image']
                        
                        # Resize template if needed
                        if tpl_gray.shape[0] > pad_img.shape[0] // 2 or tpl_gray.shape[1] > pad_img.shape[1] // 2:
                            scale = min(pad_img.shape[0] / (tpl_gray.shape[0] * 3), 
                                      pad_img.shape[1] / (tpl_gray.shape[1] * 3))
                            new_size = (int(tpl_gray.shape[1] * scale), int(tpl_gray.shape[0] * scale))
                            tpl_gray = cv2.resize(tpl_gray, new_size, interpolation=cv2.INTER_AREA)
                        
                        # Apply template matching
                        try:
                            result = cv2.matchTemplate(pad_img, tpl_gray, cv2.TM_CCOEFF_NORMED)
                            
                            # Find locations above threshold
                            loc = np.where(result >= self.confidence_threshold)
                            for pt in zip(*loc[::-1]):  # Switch back to (x,y) format
                                # Calculate center point of the template
                                center_x = pt[0] + tpl_gray.shape[1] // 2
                                center_y = pt[1] + tpl_gray.shape[0] // 2
                                
                                # Add to locations with confidence score
                                well_locations.append({
                                    'col': center_x + col_min,  # Add offset to get global image coordinates
                                    'row': center_y + row_min,
                                    'confidence': float(result[pt[1], pt[0]]),
                                    'template': template['name']
                                })
                        except cv2.error:
                            self.logger.warning(
                                _("Template matching failed for template {0} in well pad {1}").format(
                                    template['name'], pad['id']
                                )
                            )
                    
                    # Apply non-maximum suppression to remove duplicates
                    well_locations = self._non_maximum_suppression(well_locations)
                    
                    # Convert pixel coordinates to geographic coordinates
                    for loc in well_locations:
                        # Convert pixel to geographic coordinates
                        lon, lat = transform * (loc['col'], loc['row'])
                        
                        # Create point geometry
                        point = Point(lon, lat)
                        
                        # Skip points outside the well pad
                        if not point.within(pad_geom):
                            continue
                        
                        # Add to wells list
                        wells.append({
                            'id': f"well_{Path(image_path).stem}_{pad_idx}_{len(wells)}",
                            'geometry': mapping(point),
                            'confidence': loc['confidence'],
                            'template': loc['template'],
                            'wellpad_id': pad['id'],
                            'source_image': image_path,
                            'timestamp': pad['timestamp'],
                        })
        
        except Exception as e:
            self.logger.error(_("Error detecting wells in {0}: {1}").format(image_path, str(e)))
        
        return wells
    
    def _non_maximum_suppression(self, detections: List[Dict], threshold: int = 10) -> List[Dict]:
        """Apply non-maximum suppression to remove duplicate detections.
        
        Args:
            detections: List of dictionaries with detection information
            threshold: Distance threshold for considering detections as duplicates
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by confidence (higher first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize list of indices to keep
        keep = []
        
        for i in range(len(detections)):
            # If this detection has been marked for removal, skip it
            if i not in keep:
                keep.append(i)
                
                # Check all remaining detections
                for j in range(i + 1, len(detections)):
                    # Calculate distance between detections
                    dist = np.sqrt((detections[i]['col'] - detections[j]['col']) ** 2 + 
                                   (detections[i]['row'] - detections[j]['row']) ** 2)
                    
                    # If too close, mark for removal
                    if dist < threshold:
                        # Do not add j to keep list
                        pass
                    else:
                        if j not in keep:
                            keep.append(j)
        
        # Return filtered list
        return [detections[i] for i in keep]
    
    def _detect_with_image_processing(
        self, 
        image_path: str, 
        vis_path: str, 
        wellpads: List[Dict]
    ) -> List[Dict]:
        """Detect wells using image processing techniques.
        
        Args:
            image_path: Path to the source image
            vis_path: Path to the visualization image
            wellpads: List of well pad dictionaries located in this image
            
        Returns:
            List of dictionaries with well information
        """
        wells = []
        
        try:
            # Open the visualization image
            vis_img = cv2.imread(vis_path)
            if vis_img is None:
                # If reading fails, try with rasterio and convert
                with rasterio.open(vis_path) as src:
                    vis_data = src.read()
                    vis_img = np.transpose(vis_data, (1, 2, 0))
                    
                    # If more than 3 bands, select first 3
                    if vis_img.shape[2] > 3:
                        vis_img = vis_img[:, :, :3]
                    
                    # If single band, duplicate to 3 channels
                    if vis_img.shape[2] == 1:
                        vis_img = np.repeat(vis_img, 3, axis=2)
                    
                    # Normalize to 0-255 range
                    vis_img = (vis_img * 255 / vis_img.max()).astype(np.uint8)
            
            # Convert to grayscale
            vis_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
            
            # Open the source image to get geotransform
            with rasterio.open(image_path) as src:
                transform = src.transform
                
                # Process each well pad
                for pad_idx, pad in enumerate(wellpads):
                    # Get the geometry as shapely object
                    pad_geom = shape(pad['geometry'])
                    
                    # Get bounding box of the pad
                    minx, miny, maxx, maxy = pad_geom.bounds
                    
                    # Convert to pixel coordinates
                    row_min, col_min = ~transform * (minx, maxy)
                    row_max, col_max = ~transform * (maxx, miny)
                    
                    # Ensure coordinates are within image bounds
                    row_min = max(0, int(row_min))
                    col_min = max(0, int(col_min))
                    row_max = min(vis_gray.shape[0], int(row_max))
                    col_max = min(vis_gray.shape[1], int(col_max))
                    
                    # Extract the well pad region
                    pad_img = vis_gray[row_min:row_max, col_min:col_max]
                    
                    # Skip if the pad region is too small
                    if pad_img.shape[0] < 20 or pad_img.shape[1] < 20:
                        continue
                    
                    # Apply Gaussian blur to reduce noise
                    blurred = cv2.GaussianBlur(pad_img, (5, 5), 0)
                    
                    # Apply adaptive thresholding
                    binary = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY_INV, 11, 2
                    )
                    
                    # Apply morphological operations
                    kernel = np.ones((3, 3), np.uint8)
                    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
                    
                    # Find contours
                    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by size and shape
                    well_locations = []
                    for contour in contours:
                        # Get area and perimeter
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Skip if too small
                        if area < 10:
                            continue
                        
                        # Calculate circularity (4π × area / perimeter²)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        # Wells are often circular-ish
                        if circularity > 0.5:
                            # Get centroid
                            M = cv2.moments(contour)
                            if M["m00"] > 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                # Add to locations
                                well_locations.append({
                                    'col': cx + col_min,  # Add offset to get global image coordinates
                                    'row': cy + row_min,
                                    'confidence': circularity,  # Use circularity as confidence
                                    'template': 'image_processing'
                                })
                    
                    # Apply non-maximum suppression
                    well_locations = self._non_maximum_suppression(well_locations)
                    
                    # Convert pixel coordinates to geographic coordinates
                    for loc in well_locations:
                        # Convert pixel to geographic coordinates
                        lon, lat = transform * (loc['col'], loc['row'])
                        
                        # Create point geometry
                        point = Point(lon, lat)
                        
                        # Skip points outside the well pad
                        if not point.within(pad_geom):
                            continue
                        
                        # Add to wells list
                        wells.append({
                            'id': f"well_{Path(image_path).stem}_{pad_idx}_{len(wells)}",
                            'geometry': mapping(point),
                            'confidence': loc['confidence'],
                            'template': loc['template'],
                            'wellpad_id': pad['id'],
                            'source_image': image_path,
                            'timestamp': pad['timestamp'],
                        })
        
        except Exception as e:
            self.logger.error(_("Error detecting wells with image processing in {0}: {1}").format(image_path, str(e)))
        
        return wells
    
    def _save_results(self, wells: List[Dict]) -> None:
        """Save detected wells to GeoJSON file.
        
        Args:
            wells: List of dictionaries with well information
        """
        # Create GeoJSON feature collection
        feature_collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": well['id'],
                    "geometry": well['geometry'],
                    "properties": {
                        "confidence": well['confidence'],
                        "template": well['template'],
                        "wellpad_id": well['wellpad_id'],
                        "source_image": os.path.basename(well['source_image']),
                        "timestamp": '_'.join(well['timestamp']),
                    }
                }
                for well in wells
            ]
        }
        
        # Save to file with timestamp
        timestamp = wells[0]['timestamp'][0]
        output_path = self.output_dir / f"wells_{timestamp}.geojson"
        
        with open(output_path, 'w') as f:
            import json
            json.dump(feature_collection, f, indent=2)
        
        self.logger.info(_("Saved {0} detected wells to {1}").format(len(wells), output_path))
