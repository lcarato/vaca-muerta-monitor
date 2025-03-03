#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Module

This module handles database operations for storing well pad and well detections,
emissions data, and official well information.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import sqlite3
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, Point, Polygon
from shapely.errors import WKTReadingError
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
import geoalchemy2 as ga
from geoalchemy2 import Geometry

from src.utils.logger import get_logger
from src.utils.localization import translate as _

# Create Base class for SQLAlchemy models
Base = declarative_base()

# Define SQLAlchemy models
class WellPad(Base):
    """Well pad table model."""
    __tablename__ = 'wellpads'
    
    id = Column(Integer, primary_key=True)
    wellpad_id = Column(String(100), unique=True, nullable=False)
    geometry = Column(Geometry('POLYGON'))
    area = Column(Float)
    confidence = Column(Float)
    detection_date = Column(DateTime)
    source_image = Column(String(255))
    properties = Column(JSON)
    
    # Relationships
    wells = relationship("Well", back_populates="wellpad")
    emissions = relationship("EmissionsMeasurement", back_populates="wellpad")
    
    def __repr__(self):
        return f"<WellPad(id={self.id}, wellpad_id='{self.wellpad_id}')>"


class Well(Base):
    """Individual well table model."""
    __tablename__ = 'wells'
    
    id = Column(Integer, primary_key=True)
    well_id = Column(String(100), unique=True, nullable=False)
    geometry = Column(Geometry('POINT'))
    confidence = Column(Float)
    detection_date = Column(DateTime)
    wellpad_id = Column(Integer, ForeignKey('wellpads.id'))
    official_id = Column(String(100), nullable=True)
    properties = Column(JSON)
    
    # Relationships
    wellpad = relationship("WellPad", back_populates="wells")
    
    def __repr__(self):
        return f"<Well(id={self.id}, well_id='{self.well_id}')>"


class OfficialWell(Base):
    """Official wells from government data."""
    __tablename__ = 'official_wells'
    
    id = Column(Integer, primary_key=True)
    official_id = Column(String(100), unique=True, nullable=False)
    geometry = Column(Geometry('POINT'))
    name = Column(String(255))
    company = Column(String(255))
    concession = Column(String(255))
    status = Column(String(100))
    type = Column(String(100))
    formation = Column(String(255))
    completion_date = Column(DateTime, nullable=True)
    properties = Column(JSON)
    
    def __repr__(self):
        return f"<OfficialWell(id={self.id}, official_id='{self.official_id}')>"


class Concession(Base):
    """Oil and gas concessions."""
    __tablename__ = 'concessions'
    
    id = Column(Integer, primary_key=True)
    concession_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255))
    geometry = Column(Geometry('POLYGON'))
    company = Column(String(255))
    type = Column(String(100))  # e.g., exploration, exploitation
    properties = Column(JSON)
    
    def __repr__(self):
        return f"<Concession(id={self.id}, name='{self.name}')>"


class EmissionsMeasurement(Base):
    """Methane emissions measurements."""
    __tablename__ = 'emissions'
    
    id = Column(Integer, primary_key=True)
    wellpad_id = Column(Integer, ForeignKey('wellpads.id'))
    measurement_date = Column(DateTime)
    methane_value = Column(Float)
    background_value = Column(Float)
    anomaly = Column(Float)
    anomaly_percent = Column(Float)
    has_significant_anomaly = Column(Boolean, default=False)
    source = Column(String(100))  # e.g., 'sentinel-5p'
    properties = Column(JSON)
    
    # Relationships
    wellpad = relationship("WellPad", back_populates="emissions")
    
    def __repr__(self):
        return f"<EmissionsMeasurement(id={self.id}, wellpad_id={self.wellpad_id}, anomaly_percent={self.anomaly_percent})>"


def initialize_db(config: Dict) -> str:
    """Initialize the database based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Database connection string
    """
    logger = get_logger()
    
    # Determine database type from config
    db_type = config['data_integration']['database']['type']
    
    if db_type == 'sqlite':
        # Set up SQLite database
        db_path = Path(config['general']['data_directory']) / 'vaca_muerta_monitor.db'
        os.makedirs(db_path.parent, exist_ok=True)
        
        db_url = f"sqlite:///{db_path}"
        logger.info(_("Initializing SQLite database at {0}").format(db_path))
        
    elif db_type == 'postgresql':
        # Set up PostgreSQL database from environment variables
        db_host = os.environ.get('DB_HOST', 'localhost')
        db_port = os.environ.get('DB_PORT', '5432')
        db_name = os.environ.get('DB_NAME', 'vaca_muerta_monitor')
        db_user = os.environ.get('DB_USER', 'postgres')
        db_password = os.environ.get('DB_PASSWORD', '')
        
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logger.info(_("Initializing PostgreSQL database at {0}:{1}/{2}").format(db_host, db_port, db_name))
    
    else:
        raise ValueError(_("Unsupported database type: {0}").format(db_type))
    
    # Create engine and tables
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    
    return db_url


class WellDatabase:
    """Database interface for well detection and monitoring."""
    
    def __init__(self, db_url: str):
        """Initialize database connection.
        
        Args:
            db_url: Database connection URL
        """
        self.logger = get_logger()
        self.engine = create_engine(db_url)
        
        # Create session factory
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
    
    def add_detections(self, developments: List[Dict]) -> None:
        """Add detected well pads and wells to the database.
        
        Args:
            developments: List of dictionaries with development information
        """
        session = self.Session()
        
        try:
            for dev in developments:
                # Determine if this is a wellpad or well
                is_wellpad = 'wellpad' in dev['id'].lower()
                
                if is_wellpad:
                    # Create WellPad entry
                    geom = shape(dev['geometry'])
                    
                    wellpad = WellPad(
                        wellpad_id=dev['id'],
                        geometry=ga.shape.from_shape(geom),
                        area=dev.get('area', 0.0),
                        confidence=dev.get('confidence', 0.0),
                        detection_date=datetime.now(),
                        source_image=dev.get('source_image', ''),
                        properties=dev
                    )
                    
                    session.add(wellpad)
                
                else:
                    # Create Well entry
                    geom = shape(dev['geometry'])
                    
                    # Try to find associated wellpad
                    wellpad_id = dev.get('wellpad_id')
                    wellpad = None
                    
                    if wellpad_id:
                        wellpad = session.query(WellPad).filter_by(wellpad_id=wellpad_id).first()
                    
                    well = Well(
                        well_id=dev['id'],
                        geometry=ga.shape.from_shape(geom),
                        confidence=dev.get('confidence', 0.0),
                        detection_date=datetime.now(),
                        wellpad_id=wellpad.id if wellpad else None,
                        properties=dev
                    )
                    
                    session.add(well)
            
            session.commit()
            self.logger.info(_("Added {0} new developments to database").format(len(developments)))
        
        except Exception as e:
            session.rollback()
            self.logger.error(_("Error adding developments to database: {0}").format(str(e)))
        
        finally:
            session.close()
    
    def add_emissions_data(self, emissions_results: List[Dict]) -> None:
        """Add methane emissions measurements to the database.
        
        Args:
            emissions_results: List of dictionaries with emissions information
        """
        session = self.Session()
        
        try:
            for result in emissions_results:
                # Find the associated wellpad
                development_id = result.get('development_id')
                
                wellpad = None
                if development_id:
                    wellpad = session.query(WellPad).filter_by(wellpad_id=development_id).first()
                
                if not wellpad:
                    self.logger.warning(_("Could not find wellpad {0} for emissions data").format(development_id))
                    continue
                
                # Create emissions measurement entry
                measurement = EmissionsMeasurement(
                    wellpad_id=wellpad.id,
                    measurement_date=datetime.now(),
                    methane_value=result.get('latest_methane', 0.0),
                    background_value=result.get('latest_background', 0.0),
                    anomaly=result.get('latest_anomaly', 0.0),
                    anomaly_percent=result.get('latest_anomaly_pct', 0.0),
                    has_significant_anomaly=result.get('has_anomaly', False),
                    source='sentinel-5p',
                    properties={
                        'mean_methane': result.get('mean_methane', 0.0),
                        'mean_background': result.get('mean_background', 0.0),
                        'mean_anomaly': result.get('mean_anomaly', 0.0),
                        'mean_anomaly_pct': result.get('mean_anomaly_pct', 0.0),
                        'max_anomaly': result.get('max_anomaly', 0.0),
                        'max_anomaly_pct': result.get('max_anomaly_pct', 0.0),
                        'data_points': result.get('data_points', 0)
                    }
                )
                
                session.add(measurement)
            
            session.commit()
            self.logger.info(_("Added {0} emissions measurements to database").format(len(emissions_results)))
        
        except Exception as e:
            session.rollback()
            self.logger.error(_("Error adding emissions data to database: {0}").format(str(e)))
        
        finally:
            session.close()
    
    def update_official_wells(self, wells_data: List[Dict]) -> None:
        """Update the official wells table with data from government sources.
        
        Args:
            wells_data: List of dictionaries with official well information
        """
        session = self.Session()
        
        try:
            # Track existing wells to identify which ones are new
            existing_ids = {well.official_id for well in session.query(OfficialWell.official_id).all()}
            added_count = 0
            updated_count = 0
            
            for well_data in wells_data:
                official_id = well_data.get('POZO_ID') or well_data.get('id')
                
                if not official_id:
                    continue
                
                # Extract coordinates
                try:
                    lat = float(well_data.get('LATITUD') or well_data.get('latitude', 0))
                    lon = float(well_data.get('LONGITUD') or well_data.get('longitude', 0))
                    
                    # Skip wells without valid coordinates
                    if lat == 0 or lon == 0:
                        continue
                    
                    point = Point(lon, lat)
                    
                    # Extract other properties
                    name = well_data.get('POZO') or well_data.get('name', '')
                    company = well_data.get('EMPRESA') or well_data.get('company', '')
                    concession = well_data.get('CONCESION') or well_data.get('concession', '')
                    status = well_data.get('ESTADO') or well_data.get('status', '')
                    well_type = well_data.get('TIPO') or well_data.get('type', '')
                    formation = well_data.get('FORMACION') or well_data.get('formation', '')
                    
                    # Parse completion date if available
                    completion_date = None
                    completion_date_str = well_data.get('FECHA_TERM') or well_data.get('completion_date')
                    
                    if completion_date_str:
                        try:
                            completion_date = datetime.strptime(completion_date_str, '%Y-%m-%d')
                        except ValueError:
                            # Try alternative date formats
                            try:
                                completion_date = datetime.strptime(completion_date_str, '%d/%m/%Y')
                            except ValueError:
                                pass
                    
                    # Check if well already exists in the database
                    if official_id in existing_ids:
                        # Update existing well
                        well = session.query(OfficialWell).filter_by(official_id=official_id).first()
                        
                        well.geometry = ga.shape.from_shape(point)
                        well.name = name
                        well.company = company
                        well.concession = concession
                        well.status = status
                        well.type = well_type
                        well.formation = formation
                        
                        if completion_date:
                            well.completion_date = completion_date
                        
                        well.properties = well_data
                        
                        updated_count += 1
                    
                    else:
                        # Add new well
                        well = OfficialWell(
                            official_id=official_id,
                            geometry=ga.shape.from_shape(point),
                            name=name,
                            company=company,
                            concession=concession,
                            status=status,
                            type=well_type,
                            formation=formation,
                            completion_date=completion_date,
                            properties=well_data
                        )
                        
                        session.add(well)
                        existing_ids.add(official_id)
                        added_count += 1
                
                except Exception as e:
                    self.logger.warning(_("Error processing official well {0}: {1}").format(official_id, str(e)))
                    continue
            
            session.commit()
            self.logger.info(_("Updated official wells: {0} added, {1} updated").format(added_count, updated_count))
        
        except Exception as e:
            session.rollback()
            self.logger.error(_("Error updating official wells: {0}").format(str(e)))
        
        finally:
            session.close()
    
    def update_concessions(self, concessions_data: List[Dict]) -> None:
        """Update the concessions table with data from government sources.
        
        Args:
            concessions_data: List of dictionaries with concession information
        """
        session = self.Session()
        
        try:
            # Track existing concessions to identify which ones are new
            existing_ids = {c.concession_id for c in session.query(Concession.concession_id).all()}
            added_count = 0
            updated_count = 0
            
            for concession_data in concessions_data:
                concession_id = concession_data.get('ID') or concession_data.get('id')
                
                if not concession_id:
                    continue
                
                try:
                    # Extract properties
                    name = concession_data.get('NOMBRE') or concession_data.get('name', '')
                    company = concession_data.get('EMPRESA') or concession_data.get('company', '')
                    concession_type = concession_data.get('TIPO') or concession_data.get('type', '')
                    
                    # Extract geometry
                    geometry_data = concession_data.get('geometry')
                    if not geometry_data:
                        continue
                    
                    try:
                        geom = shape(geometry_data)
                    except Exception:
                        # Try WKT format
                        wkt = concession_data.get('WKT') or concession_data.get('wkt')
                        if wkt:
                            try:
                                from shapely import wkt as shapely_wkt
                                geom = shapely_wkt.loads(wkt)
                            except WKTReadingError:
                                self.logger.warning(_("Could not parse concession geometry for {0}").format(concession_id))
                                continue
                        else:
                            continue
                    
                    # Check if concession already exists in the database
                    if concession_id in existing_ids:
                        # Update existing concession
                        concession = session.query(Concession).filter_by(concession_id=concession_id).first()
                        
                        concession.name = name
                        concession.geometry = ga.shape.from_shape(geom)
                        concession.company = company
                        concession.type = concession_type
                        concession.properties = concession_data
                        
                        updated_count += 1
                    
                    else:
                        # Add new concession
                        concession = Concession(
                            concession_id=concession_id,
                            name=name,
                            geometry=ga.shape.from_shape(geom),
                            company=company,
                            type=concession_type,
                            properties=concession_data
                        )
                        
                        session.add(concession)
                        existing_ids.add(concession_id)
                        added_count += 1
                
                except Exception as e:
                    self.logger.warning(_("Error processing concession {0}: {1}").format(concession_id, str(e)))
                    continue
            
            session.commit()
            self.logger.info(_("Updated concessions: {0} added, {1} updated").format(added_count, updated_count))
        
        except Exception as e:
            session.rollback()
            self.logger.error(_("Error updating concessions: {0}").format(str(e)))
        
        finally:
            session.close()
    
    def get_developments_by_date_range(self, start_date: datetime, end_date: datetime) -> Tuple[List[WellPad], List[Well]]:
        """Get well pads and wells detected within a date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Tuple of (wellpads, wells) lists
        """
        session = self.Session()
        
        try:
            wellpads = session.query(WellPad).filter(
                WellPad.detection_date >= start_date,
                WellPad.detection_date <= end_date
            ).all()
            
            wells = session.query(Well).filter(
                Well.detection_date >= start_date,
                Well.detection_date <= end_date
            ).all()
            
            return wellpads, wells
        
        finally:
            session.close()
    
    def get_emissions_by_development(self, development_id: str) -> List[EmissionsMeasurement]:
        """Get emissions measurements for a specific development.
        
        Args:
            development_id: ID of the wellpad or well
            
        Returns:
            List of emissions measurements
        """
        session = self.Session()
        
        try:
            # Find the wellpad
            wellpad = session.query(WellPad).filter_by(wellpad_id=development_id).first()
            
            if not wellpad:
                return []
            
            # Get associated emissions measurements
            emissions = session.query(EmissionsMeasurement).filter_by(wellpad_id=wellpad.id).all()
            
            return emissions
        
        finally:
            session.close()
    
    def match_with_official_wells(self, detected_wells: List[Dict], max_distance: float = 100.0) -> List[Dict]:
        """Match detected wells with official wells from the database.
        
        Args:
            detected_wells: List of dictionaries with detected well information
            max_distance: Maximum distance in meters for matching
            
        Returns:
            Updated list of dictionaries with official well information added
        """
        session = self.Session()
        
        try:
            # Load all official wells
            official_wells = session.query(OfficialWell).all()
            
            if not official_wells:
                self.logger.warning(_("No official wells in database for matching"))
                return detected_wells
            
            # Convert to GeoDataFrame for spatial operations
            official_gdf = gpd.GeoDataFrame([
                {
                    'official_id': well.official_id,
                    'name': well.name,
                    'company': well.company,
                    'concession': well.concession,
                    'status': well.status,
                    'type': well.type,
                    'geometry': shape(well.geometry.data)  # Convert from GeoAlchemy to Shapely
                }
                for well in official_wells
            ])
            
            # Set CRS
            official_gdf.crs = 'EPSG:4326'
            
            # Convert to projected CRS for distance calculations in meters
            official_gdf = official_gdf.to_crs('EPSG:32720')  # UTM zone 20S for Argentina
            
            # Create GeoDataFrame for detected wells
            detected_gdf = gpd.GeoDataFrame([
                {
                    'detected_id': well['id'],
                    'confidence': well.get('confidence', 0.0),
                    'geometry': shape(well['geometry'])
                }
                for well in detected_wells
            ])
            
            # Set CRS and project
            detected_gdf.crs = 'EPSG:4326'
            detected_gdf = detected_gdf.to_crs('EPSG:32720')  # UTM zone 20S
            
            # For each detected well, find the nearest official well
            for i, detected_well in detected_gdf.iterrows():
                # Calculate distances to all official wells
                distances = official_gdf.geometry.distance(detected_well.geometry)
                
                # Find the index of the nearest well
                if distances.min() <= max_distance:
                    nearest_idx = distances.idxmin()
                    nearest_well = official_gdf.iloc[nearest_idx]
                    
                    # Add official well info to the detected well
                    detected_wells[i].update({
                        'official_id': nearest_well['official_id'],
                        'official_name': nearest_well['name'],
                        'company': nearest_well['company'],
                        'concession': nearest_well['concession'],
                        'status': nearest_well['status'],
                        'well_type': nearest_well['type'],
                        'match_distance': distances.min()
                    })
            
            return detected_wells
        
        finally:
            session.close()
    
    def get_development_within_concession(self, development: Dict) -> Optional[Dict]:
        """Find which concession a development is located within.
        
        Args:
            development: Dictionary with development information including geometry
            
        Returns:
            Concession information dictionary or None if not found
        """
        session = self.Session()
        
        try:
            # Convert development geometry to shapely
            dev_geom = shape(development['geometry'])
            
            # Query for concession that contains this point
            stmt = session.query(Concession).filter(
                ga.functions.ST_Contains(Concession.geometry, ga.shape.from_shape(dev_geom))
            )
            
            concession = stmt.first()
            
            if concession:
                return {
                    'concession_id': concession.concession_id,
                    'name': concession.name,
                    'company': concession.company,
                    'type': concession.type,
                    'properties': concession.properties
                }
            
            return None
        
        finally:
            session.close()
    
    def export_to_geojson(self, output_path: str) -> bool:
        """Export detected wellpads and wells to GeoJSON.
        
        Args:
            output_path: Path to save the GeoJSON file
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        
        try:
            # Get all wellpads and wells
            wellpads = session.query(WellPad).all()
            wells = session.query(Well).all()
            
            # Create features for wellpads
            wellpad_features = []
            for pad in wellpads:
                geom = shape(pad.geometry.data)
                
                wellpad_features.append({
                    'type': 'Feature',
                    'id': pad.wellpad_id,
                    'geometry': mapping(geom),
                    'properties': {
                        'id': pad.wellpad_id,
                        'area': pad.area,
                        'confidence': pad.confidence,
                        'detection_date': pad.detection_date.isoformat() if pad.detection_date else None,
                        'type': 'wellpad'
                    }
                })
            
            # Create features for wells
            well_features = []
            for well in wells:
                geom = shape(well.geometry.data)
                
                well_features.append({
                    'type': 'Feature',
                    'id': well.well_id,
                    'geometry': mapping(geom),
                    'properties': {
                        'id': well.well_id,
                        'confidence': well.confidence,
                        'detection_date': well.detection_date.isoformat() if well.detection_date else None,
                        'wellpad_id': well.wellpad_id,
                        'official_id': well.official_id,
                        'type': 'well'
                    }
                })
            
            # Combine all features
            feature_collection = {
                'type': 'FeatureCollection',
                'features': wellpad_features + well_features
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f, indent=2)
            
            self.logger.info(_("Exported {0} wellpads and {1} wells to {2}").format(
                len(wellpad_features), len(well_features), output_path
            ))
            
            return True
        
        except Exception as e:
            self.logger.error(_("Error exporting to GeoJSON: {0}").format(str(e)))
            return False
        
        finally:
            session.close()
