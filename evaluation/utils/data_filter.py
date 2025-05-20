import pandas as pd
import numpy as np

from evaluation.utils.processing import DataProcessor

class DataFilter:
    """Handles filtering of tail posture data based on quality metrics."""
    
    def __init__(self, config, logger=None):
        """Initialize with configuration and optional logger."""
        self.config = config
        self.logger = logger
        
        # Initialize tracking structures for exclusions
        self.excluded_elements = {
            'consecutive_missing': [],
            'missing_percentage': [],
            'other_reasons': []
        }
        
        # Track exclusions by analysis type
        self.exclusion_by_analysis = {
            'tail_biting_analysis': {
                'consecutive_missing': [],
                'missing_percentage': [],
                'other_reasons': []
            },
            'control_analysis': {
                'consecutive_missing': [],
                'missing_percentage': [],
                'other_reasons': []
            }
        }
        
        # Load quality thresholds from config
        self.max_missing_threshold = self.config.get('max_allowed_consecutive_missing_days', 3)
        self.max_missing_pct_threshold = self.config.get('max_allowed_missing_days_pct', 50.0)
        
    def log_exclusion(self, exclusion_type, camera, date_span, pen_type, value, threshold=None, message=None, analysis_type="unknown"):
        """Log an exclusion with enhanced tracking of which analysis excluded it."""
        if message and self.logger:
            self.logger.info(message)
        
        # Track in the original structure
        if exclusion_type in self.excluded_elements:
            if exclusion_type in ['consecutive_missing', 'missing_percentage']:
                self.excluded_elements[exclusion_type].append((camera, date_span, pen_type, value, threshold))
            else:
                self.excluded_elements[exclusion_type].append((camera, date_span, pen_type, value))
        
        # Also track in the analysis-specific structure
        if analysis_type in self.exclusion_by_analysis:
            if exclusion_type in self.exclusion_by_analysis[analysis_type]:
                if exclusion_type in ['consecutive_missing', 'missing_percentage']:
                    self.exclusion_by_analysis[analysis_type][exclusion_type].append(
                        (camera, date_span, pen_type, value, threshold)
                    )
                else:
                    self.exclusion_by_analysis[analysis_type][exclusion_type].append(
                        (camera, date_span, pen_type, value)
                    )
        
        return 1
    
    def filter_by_quality_metrics(self, processed_results, pen_info_func, json_data, analysis_type="unknown"):
        """
        Filter processed results based on quality metrics.
        
        Args:
            processed_results: List of processed data dictionaries
            pen_info_func: Function to get pen type information
            json_data: JSON data containing pen information
            analysis_type: Type of analysis for tracking exclusions
            
        Returns:
            List of filtered processed data dictionaries
        """
        if not processed_results:
            if self.logger:
                self.logger.error("No processed data available for filtering.")
            return []
        
        filtered_results = []
        excluded_count = 0
        excluded_pct_count = 0
        
        for processed_data in processed_results:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            quality_metrics = processed_data.get('quality_metrics', {})
            
            # Get pen type for tracking
            pen_type, culprit_removal, datespan_gt = pen_info_func(camera, date_span, json_data)
            
            # Check for consecutive missing days threshold
            consecutive_missing = quality_metrics.get('max_consecutive_missing_resampled', 0)
            if consecutive_missing > self.max_missing_threshold:
                message = (
                    f"Excluding {camera}/{date_span} from analysis due to {consecutive_missing} consecutive missing periods "
                    f"in resampled data (threshold: {self.max_missing_threshold}). Raw missing days: {quality_metrics.get('missing_days_detected', 'unknown')}"
                )
                excluded_count += self.log_exclusion(
                    'consecutive_missing', camera, date_span, pen_type, consecutive_missing, 
                    self.max_missing_threshold, message, analysis_type=analysis_type
                )
                continue
            
            # Check for percentage of missing days
            missing_days = quality_metrics.get('missing_days_detected', 0)
            total_days = quality_metrics.get('total_expected_days', 0)
            missing_pct = (missing_days / total_days * 100) if total_days > 0 else 0
            
            if missing_pct > self.max_missing_pct_threshold:
                message = (
                    f"Excluding {camera}/{date_span} from analysis due to excessive missing days "
                    f"({missing_days}/{total_days} = {missing_pct:.1f}% > {self.max_missing_pct_threshold}%)"
                )
                excluded_pct_count += self.log_exclusion(
                    'missing_percentage', camera, date_span, pen_type, missing_pct, 
                    self.max_missing_pct_threshold, message, analysis_type=analysis_type
                )
                continue
            
            # If passed all quality checks, include in filtered results
            filtered_results.append(processed_data)
        
        # Report on filtering results
        if self.logger:
            self.logger.info(f"Quality filtering: {len(filtered_results)}/{len(processed_results)} datasets passed.")
            self.logger.info(f"Excluded {excluded_count} datasets due to >{self.max_missing_threshold} consecutive missing days.")
            self.logger.info(f"Excluded {excluded_pct_count} datasets due to excessive missing days (>{self.max_missing_pct_threshold}%).")
        
        return filtered_results, excluded_count, excluded_pct_count
    
    def filter_tail_biting_events(self, processed_results, pen_info_func, json_data):
        """
        Filter for valid tail biting events with proper removal dates.
        
        Args:
            processed_results: List of processed data dictionaries (already quality filtered)
            pen_info_func: Function to get pen type information
            json_data: JSON data containing pen information
            
        Returns:
            List of filtered processed data with only valid tail biting events
        """
        filtered_events = []
        excluded_event_count = 0
        
        for processed_data in processed_results:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            
            # Check if it's a tail biting event with valid culprit removal info
            pen_type, culprit_removal, datespan_gt = pen_info_func(camera, date_span, json_data)
            
            if pen_type != "tail biting" or culprit_removal is None or culprit_removal == "Unknown" or culprit_removal == []:
                reason = "Not a tail biting event or missing culprit removal info"
                self.log_exclusion('other_reasons', camera, date_span, pen_type, reason, 
                                None, f"Excluding {camera}/{date_span}: {reason}", 
                                analysis_type="tail_biting_analysis")
                excluded_event_count += 1
                continue
            
            # Check if interpolated data is available
            interpolated_data = processed_data.get('interpolated_data')
            if interpolated_data is None or interpolated_data.empty:
                reason = "Empty interpolated data"
                self.log_exclusion('other_reasons', camera, date_span, pen_type, reason,
                              None, f"Excluding {camera}/{date_span}: {reason}",
                              analysis_type="tail_biting_analysis")
                excluded_event_count += 1
                continue
            
            # Include in valid events
            filtered_events.append(processed_data)
        
        if self.logger:
            self.logger.info(f"Event filtering: {len(filtered_events)}/{len(processed_results)} events are valid tail biting events.")
            self.logger.info(f"Excluded {excluded_event_count} events due to invalid event type or missing data.")
        
        return filtered_events, excluded_event_count
    
    def filter_control_pens(self, processed_results, pen_info_func, json_data):
        """
        Filter for valid control pens.
        
        Args:
            processed_results: List of processed data dictionaries (already quality filtered)
            pen_info_func: Function to get pen type information
            json_data: JSON data containing pen information
            
        Returns:
            List of filtered processed data with only valid control pens
        """
        control_pens = []
        excluded_count = 0
        
        for processed_data in processed_results:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            
            # Check if it's a control pen
            pen_type, _, _ = pen_info_func(camera, date_span, json_data)
            
            if pen_type != "control":
                excluded_count += 1
                continue
            
            # Check if interpolated data is available
            interpolated_data = processed_data.get('interpolated_data')
            if interpolated_data is None or interpolated_data.empty:
                reason = "Empty interpolated data"
                self.log_exclusion('other_reasons', camera, date_span, pen_type, reason,
                              None, f"Excluding control pen {camera}/{date_span}: {reason}",
                              analysis_type="control_analysis")
                excluded_count += 1
                continue
            
            # Include in valid control pens
            control_pens.append(processed_data)
        
        if self.logger:
            self.logger.info(f"Control pen filtering: {len(control_pens)}/{len(processed_results)} pens are valid control pens.")
        
        return control_pens
    
    def get_summary_statistics(self):
        """
        Get summary statistics of the filtering process.
        
        Returns:
            Dictionary with filtering statistics
        """
        summary = {
            'total_excluded_consecutive_missing': len(self.excluded_elements['consecutive_missing']),
            'total_excluded_missing_percentage': len(self.excluded_elements['missing_percentage']),
            'total_excluded_other_reasons': len(self.excluded_elements['other_reasons']),
            'by_analysis': {
                analysis_type: {
                    'consecutive_missing': len(exclusions['consecutive_missing']),
                    'missing_percentage': len(exclusions['missing_percentage']),
                    'other_reasons': len(exclusions['other_reasons']),
                    'total': len(exclusions['consecutive_missing']) + 
                            len(exclusions['missing_percentage']) + 
                            len(exclusions['other_reasons'])
                }
                for analysis_type, exclusions in self.exclusion_by_analysis.items()
            }
        }
        
        summary['total_excluded'] = (summary['total_excluded_consecutive_missing'] + 
                                   summary['total_excluded_missing_percentage'] + 
                                   summary['total_excluded_other_reasons'])
        
        return summary