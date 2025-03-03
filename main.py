#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the Vaca Muerta Satellite Monitoring System.
This script orchestrates the pipeline for downloading satellite imagery,
detecting well pads and wells, analyzing methane emissions, and publishing
results to social media.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import time

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.utils.localization import set_language, translate as _
from src.satellite.sentinel_downloader import SentinelDownloader
from src.satellite.image_processor import ImageProcessor
from src.detection.wellpad_detector import WellPadDetector
from src.detection.well_detector import WellDetector
from src.detection.change_detection import ChangeDetector
from src.data.database import initialize_db, WellDatabase
from src.data.odata_client import ODataClient
from src.data.concession_mapper import ConcessionMapper
from src.emissions.methane_analyzer import MethaneAnalyzer
from src.social.twitter_publisher import TwitterPublisher
from src.visualization.map_generator import MapGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=_('Vaca Muerta Satellite Monitoring System'))
    
    parser.add_argument('--schedule', choices=['daily', 'weekly', 'monthly', 'once'],
                        help=_('Scheduling option for automated runs'))
    
    parser.add_argument('--no-methane', action='store_true',
                        help=_('Disable methane emissions analysis'))
    
    parser.add_argument('--no-social', action='store_true',
                        help=_('Disable social media posting'))
    
    parser.add_argument('--language', choices=['en', 'es', 'pt'], default=None,
                        help=_('Set output language (overrides config)'))
    
    parser.add_argument('--aoi', type=str, 
                        help=_('Path to custom area of interest GeoJSON file'))
    
    parser.add_argument('--start-date', type=str, 
                        help=_('Start date for data processing (YYYY-MM-DD)'))
    
    parser.add_argument('--end-date', type=str, default='now',
                        help=_('End date for data processing (YYYY-MM-DD or "now")'))
    
    parser.add_argument('--visualize-only', action='store_true',
                        help=_('Only generate visualizations from existing data'))
    
    parser.add_argument('--update-db-only', action='store_true',
                        help=_('Only update the database from official sources'))
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help=_('Path to configuration file'))
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup language
    language = args.language or config['general']['default_language']
    set_language(language)
    
    # Setup logger
    logger = setup_logger(config['general']['log_level'])
    logger.info(_('Starting Vaca Muerta Satellite Monitoring System'))
    
    # Initialize database
    db_path = initialize_db(config)
    db = WellDatabase(db_path)
    
    # Update official data if requested
    if args.update_db_only:
        logger.info(_('Updating database from official sources'))
        odata_client = ODataClient(
            config['data_integration']['official_wells_url'],
            config['data_integration']['concessions_url']
        )
        wells_data = odata_client.get_wells()
        concessions_data = odata_client.get_concessions()
        
        db.update_official_wells(wells_data)
        db.update_concessions(concessions_data)
        
        concession_mapper = ConcessionMapper(db)
        concession_mapper.generate_concession_maps()
        logger.info(_('Database update completed'))
        return
    
    # If visualization only, generate maps and exit
    if args.visualize_only:
        logger.info(_('Generating visualizations only'))
        map_generator = MapGenerator(config, db)
        map_generator.generate_all_maps()
        logger.info(_('Visualization completed'))
        return
    
    # Setup processing dates
    start_date = args.start_date or config['sentinel']['time_interval']['start_date']
    if start_date != 'now' and not isinstance(start_date, datetime):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    end_date = args.end_date or config['sentinel']['time_interval']['end_date']
    if end_date == 'now':
        end_date = datetime.now()
    elif not isinstance(end_date, datetime):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Get area of interest
    aoi_path = args.aoi or config['area_of_interest'].get('geojson_path')
    aoi_bbox = config['area_of_interest']['bbox']
    
    # Download satellite imagery
    logger.info(_('Downloading Sentinel imagery for specified time period'))
    downloader = SentinelDownloader(config)
    imagery = downloader.download(start_date, end_date, aoi_bbox, aoi_path)
    
    # Process imagery
    logger.info(_('Processing satellite imagery'))
    processor = ImageProcessor(config)
    processed_images = processor.process(imagery)
    
    # Detect well pads
    logger.info(_('Detecting well pads'))
    wellpad_detector = WellPadDetector(config)
    wellpads = wellpad_detector.detect(processed_images)
    
    # Detect individual wells
    logger.info(_('Detecting individual wells'))
    well_detector = WellDetector(config)
    wells = well_detector.detect(processed_images, wellpads)
    
    # Detect changes from previous data
    logger.info(_('Performing change detection'))
    change_detector = ChangeDetector(config, db)
    new_developments = change_detector.detect_changes(wellpads, wells)
    
    # If no new developments found, log and continue
    if not new_developments:
        logger.info(_('No new well developments detected'))
    else:
        logger.info(_('Detected {0} new well developments').format(len(new_developments)))
        
        # Store new detections in database
        db.add_detections(new_developments)
        
        # Analyze methane emissions if enabled
        if config['methane']['enabled'] and not args.no_methane:
            logger.info(_('Analyzing methane emissions for new developments'))
            methane_analyzer = MethaneAnalyzer(config)
            emissions_data = methane_analyzer.analyze(new_developments, start_date, end_date)
            db.add_emissions_data(emissions_data)
        
        # Generate visualization maps
        logger.info(_('Generating visualization maps'))
        map_generator = MapGenerator(config, db)
        map_paths = map_generator.generate_detection_maps(new_developments)
        
        # Post to social media if enabled
        if config['social']['enabled'] and not args.no_social:
            logger.info(_('Publishing results to social media'))
            twitter_publisher = TwitterPublisher(config)
            for development, map_path in zip(new_developments, map_paths):
                twitter_publisher.post_detection(development, map_path)
    
    # Schedule next run if requested
    if args.schedule:
        schedule_next_run(args.schedule)
    
    logger.info(_('Processing completed successfully'))


def schedule_next_run(schedule_type):
    """Schedule the next run based on the specified schedule type."""
    if schedule_type == 'daily':
        # Schedule for tomorrow at 1:00 AM
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        tomorrow = tomorrow.replace(hour=1, minute=0, second=0, microsecond=0)
        sleep_seconds = (tomorrow - now).total_seconds()
    elif schedule_type == 'weekly':
        # Schedule for next Monday at 1:00 AM
        now = datetime.now()
        days_ahead = 7 - now.weekday()
        next_monday = now + timedelta(days=days_ahead)
        next_monday = next_monday.replace(hour=1, minute=0, second=0, microsecond=0)
        sleep_seconds = (next_monday - now).total_seconds()
    elif schedule_type == 'monthly':
        # Schedule for the 1st of next month at 1:00 AM
        now = datetime.now()
        if now.month == 12:
            next_month = now.replace(year=now.year+1, month=1, day=1, 
                                     hour=1, minute=0, second=0, microsecond=0)
        else:
            next_month = now.replace(month=now.month+1, day=1, 
                                     hour=1, minute=0, second=0, microsecond=0)
        sleep_seconds = (next_month - now).total_seconds()
    else:
        return
    
    print(f"Sleeping for {sleep_seconds} seconds until next scheduled run")
    time.sleep(sleep_seconds)
    
    # Restart the script with the same arguments
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    main()
