import os
import numpy as np
import pandas as pd
from datetime import timedelta
from pipeline.utils.general import load_json_data

class EarlyWarningAnalyzer:
    """Class for evaluating early warning thresholds on tail posture data."""
    
    def __init__(self, analyzer, logger=None, config=None):
        """
        Initialize the evaluator with a TailPostureAnalyzer instance.
        
        Args:
            analyzer: An instance of TailPostureAnalyzer containing processed data
            logger: Optional logger instance. If None, uses analyzer's logger
            config: Optional config dict. If None, uses analyzer's config with early warning defaults
        """
        self.analyzer = analyzer
        self.logger = logger if logger is not None else analyzer.logger
        self.config = config
        
        # Ensure the path manager is available
        self.path_manager = analyzer.path_manager
        
        # Set random seed for reproducibility
        np.random.seed(self.config.get('random_seed', 42))
        
        self.logger.info("EarlyWarningAnalyzer initialized with configuration:")
        for key, value in self.config.items():
            if key != "default_threshold_sets":
                self.logger.info(f"  {key}: {value}")
        
    def evaluate_thresholds(self, thresholds=None, include_levels=None, ignore_first_percent=20, max_percent_before=60):
        """
        Evaluate early warning system thresholds.
        
        Args:
            thresholds (dict, optional): Dictionary of thresholds to use. If None, uses config thresholds.
            include_levels (list, optional): List of levels to include in evaluation.
                E.g., ['attention', 'critical'] would exclude 'alert' level.
                If None, includes all levels: ['attention', 'alert', 'critical'].
            ignore_first_percent (int, optional): Percentage of the beginning of each run to ignore.
                Default is 20%, as no culprit removals occurred in the first 20% of runs.
            max_percent_before (int, optional): Maximum percentage of the run to look back before outbreak.
                Default is 60%, which means the analysis will look at data from the end,
                going backward 60% of the total run duration.
                
        Returns:
            dict: Dictionary of evaluation metrics and results
        """
        self.logger.info("Evaluating early warning thresholds...")
        self.logger.info(f"Ignoring first {ignore_first_percent}% of each run")
        self.logger.info(f"Using maximum lookback of {max_percent_before}% of each run's duration (from the end)")
        
        # Use thresholds from config if not provided
        if thresholds is None:
            thresholds = self.config.get('thresholds', {
                'attention': {'posture_diff': 0.6},
                'alert': {'posture_diff': 0.5},
                'critical': {'posture_diff': 0.2}
            })
        
        # Define which levels to include in evaluation
        all_levels = ['attention', 'alert', 'critical']
        if include_levels is None:
            include_levels = all_levels
        else:
            # Validate levels
            for level in include_levels:
                if level not in all_levels:
                    self.logger.warning(f"Invalid level '{level}' specified. Valid levels are: {all_levels}")
            include_levels = [level for level in include_levels if level in all_levels]
            
        self.logger.info(f"Including the following threshold levels in evaluation: {include_levels}")
        
        # Load JSON data
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        
        # Initialize result counters - only for included levels
        results = {
            'total_pens_analyzed': 0,
            'total_outbreak_pens': 0,
            'total_control_pens': 0,
            'include_levels': include_levels,
            'pen_details': []
        }
        
        # Initialize metrics for each included level
        for level in include_levels:
            level_key = f'{level}_level'
            results[level_key] = {
                'true_positives': 0,   # Outbreak pen correctly receives attention
                'false_positives': 0,  # Control pen incorrectly receives attention
                'true_negatives': 0,   # Control pen correctly receives no attention
                'false_negatives': 0,  # Outbreak pen incorrectly receives no attention
                'time_before_outbreak': [],  # Time before outbreak when attention was triggered
            }
        
        # Process both outbreak and control pens
        pens_analyzed = 0
        
        # Get max_percent_before from config or use default (60%)
        max_percent_before = self.config.get('max_percent_before', max_percent_before)
        self.logger.info(f"Using max_percent_before: {max_percent_before}% of each run's duration (from the end)")
        
        use_interpolated = self.config.get('use_interpolated_data', True)
        
        # Set data field name based on configuration
        data_field = 'interpolated_data' if use_interpolated else 'resampled_data'
        self.logger.info(f"Using {data_field} for threshold evaluation")
        
        # Determine time resolution
        is_hourly = use_interpolated
        time_unit = 'hours' if is_hourly else 'days'
        results['time_unit'] = time_unit
        
        # Process outbreak pens
        if hasattr(self.analyzer, 'pre_outbreak_stats') and not self.analyzer.pre_outbreak_stats.empty:
            results['total_outbreak_pens'] = len(self.analyzer.pre_outbreak_stats)
            
            # Group by pen and datespan to process each unique pen event
            for (pen, datespan), group in self.analyzer.pre_outbreak_stats.groupby(['pen', 'datespan']):
                pens_analyzed += 1
                
                # Get the processed data for this pen
                pen_data = None
                for processed_data in self.analyzer.processed_results:
                    camera_label = processed_data['camera'].replace("Kamera", "Pen ")
                    if camera_label == pen and processed_data['date_span'] == datespan:
                        pen_data = processed_data
                        break
                
                if pen_data is None or data_field not in pen_data or pen_data[data_field].empty:
                    self.logger.warning(f"No {data_field} found for outbreak pen {pen} / {datespan}. Skipping.")
                    continue
                
                # Get removal date
                removal_date = pd.to_datetime(group['culprit_removal_date'].iloc[0])
                
                # Analyze timepoints before removal
                data = pen_data[data_field].copy()
                
                # Ensure datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if 'datetime' in data.columns:
                        data = data.set_index('datetime')
                    else:
                        self.logger.warning(f"Cannot set datetime index for {pen} / {datespan}. Skipping.")
                        continue
                
                # Calculate the index position to start from (ignore first X% of the run)
                ignore_points = int(len(data) * ignore_first_percent / 100)
                min_valid_index = ignore_points if ignore_points < len(data) else 0
                
                # Get the timestamp corresponding to the min_valid_index
                if min_valid_index > 0 and min_valid_index < len(data):
                    min_valid_timestamp = data.index[min_valid_index]
                    self.logger.info(f"For pen {pen} / {datespan}, ignoring first {ignore_first_percent}% " 
                                    f"({min_valid_index} points). Analysis starts from: {min_valid_timestamp}")
                else:
                    min_valid_timestamp = data.index[0]
                
                # Calculate maximum lookback period as percentage of run duration FROM THE END
                total_run_duration = (removal_date - data.index[0]).total_seconds()
                
                # Convert percentage to time-based units
                if is_hourly:
                    max_lookback_seconds = total_run_duration * max_percent_before / 100
                    max_time_before = int(max_lookback_seconds / 3600)  # Convert seconds to hours
                else:
                    max_lookback_seconds = total_run_duration * max_percent_before / 100
                    max_time_before = int(max_lookback_seconds / 86400)  # Convert seconds to days
                
                earliest_analysis_date = data.index[min_valid_index]
                max_possible_time = removal_date - earliest_analysis_date
                if is_hourly:
                    max_possible_hours = int(max_possible_time.total_seconds() / 3600)
                    max_time_before = min(max_time_before, max_possible_hours)
                else:
                    max_possible_days = int(max_possible_time.total_seconds() / 86400)
                    max_time_before = min(max_time_before, max_possible_days)
                
                self.logger.info(f"For pen {pen} / {datespan}, max lookback period is {max_time_before} {time_unit} " 
                               f"({max_percent_before}% of total duration from the end)")
                
                # Initialize alert flags - only for included levels
                triggered = {level: False for level in include_levels}
                trigger_times = {level: None for level in include_levels}
                
                # Analyze data time point by time point from max_time_before down to 0
                for time_before in range(max_time_before, -1, -1):
                    # Calculate target date based on time unit
                    if is_hourly:
                        target_date = removal_date - timedelta(hours=time_before)
                    else:
                        target_date = removal_date - timedelta(days=time_before)
                    
                    # Get data up to this time point
                    data_up_to_time = data[(data.index <= target_date) & (data.index >= min_valid_timestamp)]
                    
                    if data_up_to_time.empty:
                        continue
                    
                    # Get the last data point for this time point
                    current_point = data_up_to_time.iloc[-1]
                    
                    # Get current posture_diff value
                    current_posture_diff = current_point.get('posture_diff', np.nan)
                    
                    if pd.isna(current_posture_diff):
                        continue
                    
                    # Check thresholds only for included levels
                    for level in include_levels:
                        if not triggered[level] and current_posture_diff < thresholds[level]['posture_diff']:
                            triggered[level] = True
                            trigger_times[level] = time_before
                            results[f'{level}_level']['time_before_outbreak'].append(time_before)
                
                # Record the results for this outbreak pen - only for included levels
                for level in include_levels:
                    level_key = f'{level}_level'
                    if triggered[level]:
                        results[level_key]['true_positives'] += 1
                    else:
                        results[level_key]['false_negatives'] += 1
                
                # Format readable times
                readable_times = {
                    level: self._format_time_value(trigger_times[level], is_hourly) 
                    for level in include_levels
                }
                
                # Add detailed results - include only selected levels
                pen_detail = {
                    'pen': pen,
                    'datespan': datespan,
                    'type': 'outbreak'
                }
                
                # Add level-specific fields only for included levels
                for level in include_levels:
                    pen_detail[f'{level}_triggered'] = triggered[level]
                    pen_detail[f'{level}_time_before'] = trigger_times[level]
                    pen_detail[f'{level}_time_before_readable'] = readable_times[level]
                
                results['pen_details'].append(pen_detail)
        
        # Process control pens
        if hasattr(self.analyzer, 'control_stats') and not self.analyzer.control_stats.empty:
            # Count unique pens (not reference points)
            unique_control_pens = self.analyzer.control_stats[['pen', 'datespan']].drop_duplicates()
            results['total_control_pens'] = len(unique_control_pens)
            
            # Group by pen and datespan
            for (pen, datespan), group in self.analyzer.control_stats.groupby(['pen', 'datespan']):
                pens_analyzed += 1
                
                # Get the processed data for this pen
                pen_data = None
                for processed_data in self.analyzer.processed_results:
                    camera_label = processed_data['camera'].replace("Kamera", "Pen ")
                    if camera_label == pen and processed_data['date_span'] == datespan:
                        pen_data = processed_data
                        break
                
                if pen_data is None or data_field not in pen_data or pen_data[data_field].empty:
                    self.logger.warning(f"No {data_field} found for control pen {pen} / {datespan}. Skipping.")
                    continue
                
                # Analyze all available data for this control pen
                data = pen_data[data_field].copy()
                
                # Ensure datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if 'datetime' in data.columns:
                        data = data.set_index('datetime')
                    else:
                        self.logger.warning(f"Cannot set datetime index for {pen} / {datespan}. Skipping.")
                        continue
                
                # Calculate the index position to start from (ignore first X% of the run)
                ignore_points = int(len(data) * ignore_first_percent / 100)
                min_valid_index = ignore_points if ignore_points < len(data) else 0
                
                # Get the timestamp corresponding to the min_valid_index
                if min_valid_index > 0 and min_valid_index < len(data):
                    min_valid_timestamp = data.index[min_valid_index]
                    self.logger.info(f"For pen {pen} / {datespan}, ignoring first {ignore_first_percent}% " 
                                    f"({min_valid_index} points). Analysis starts from: {min_valid_timestamp}")
                else:
                    min_valid_timestamp = data.index[0]
                
                # CHANGED: Calculate the last X% of the run duration
                last_date = data.index[-1]
                total_run_duration = (last_date - data.index[0]).total_seconds()
                lookback_seconds = total_run_duration * max_percent_before / 100
                
                # Calculate the earliest date within the lookback period
                if is_hourly:
                    lookback_hours = int(lookback_seconds / 3600)
                    lookback_start = last_date - timedelta(hours=lookback_hours)
                else:
                    lookback_days = int(lookback_seconds / 86400)
                    lookback_start = last_date - timedelta(days=lookback_days)
                
                # Ensure we don't look before the valid start time
                lookback_start = max(lookback_start, min_valid_timestamp)
                
                # Only analyze data within the lookback period and after the ignore zone
                data = data[(data.index >= lookback_start)]
                
                self.logger.info(f"For control pen {pen} / {datespan}, analyzing {max_percent_before}% of run from the end " 
                               f"(from {lookback_start} to {last_date})")
                
                # Initialize alert flags - only for included levels
                triggered = {level: False for level in include_levels}
                
                # Analyze the entire dataset for this control pen
                for i in range(len(data)):
                    # Get the current data point
                    current_point = data.iloc[i]
                    
                    # Get current posture_diff value
                    current_posture_diff = current_point.get('posture_diff', np.nan)
                    
                    # Skip if there is no key metric
                    if pd.isna(current_posture_diff):
                        continue
                    
                    # Check thresholds only for included levels
                    for level in include_levels:
                        if not triggered[level] and current_posture_diff < thresholds[level]['posture_diff']:
                            triggered[level] = True
                    
                    # If all alerts triggered, no need to continue checking
                    if all(triggered.values()):
                        break
                
                # Record the results for this control pen - only for included levels
                for level in include_levels:
                    level_key = f'{level}_level'
                    if triggered[level]:
                        results[level_key]['false_positives'] += 1
                    else:
                        results[level_key]['true_negatives'] += 1
                
                # Add detailed results for this control pen - include only selected levels
                pen_detail = {
                    'pen': pen,
                    'datespan': datespan,
                    'type': 'control'
                }
                
                # Add level-specific fields only for included levels
                for level in include_levels:
                    pen_detail[f'{level}_triggered'] = triggered[level]
                
                results['pen_details'].append(pen_detail)
        
        results['total_pens_analyzed'] = pens_analyzed
        
        # Calculate performance metrics - only for included levels
        for level in include_levels:
            level_key = f'{level}_level'
            
            # Get counts
            tp = results[level_key]['true_positives']
            fp = results[level_key]['false_positives']
            tn = results[level_key]['true_negatives']
            fn = results[level_key]['false_negatives']
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Store metrics
            results[level_key]['metrics'] = {
                'sensitivity': sensitivity,  # TPR: TP / (TP + FN)
                'specificity': specificity,  # TNR: TN / (TN + FP)
                'precision': precision,      # PPV: TP / (TP + FP)
                'npv': npv,                  # NPV: TN / (TN + FN)
                'accuracy': accuracy,        # (TP + TN) / (TP + FP + TN + FN)
                'f1_score': f1_score         # 2 * (PPV * TPR) / (PPV + TPR)
            }
            
            # Calculate average time before outbreak for warnings
            time_before = results[level_key]['time_before_outbreak']
            
            # Keep original time unit metrics
            results[level_key]['avg_time_before_outbreak'] = np.mean(time_before) if time_before else np.nan
            results[level_key]['median_time_before_outbreak'] = np.median(time_before) if time_before else np.nan
            results[level_key]['min_time_before_outbreak'] = min(time_before) if time_before else np.nan
            results[level_key]['max_time_before_outbreak'] = max(time_before) if time_before else np.nan
            
            # Convert time to days if hours for better interpretation
            if is_hourly and time_before:
                time_before_days = [t/24.0 for t in time_before]
                results[level_key]['avg_days_before_outbreak'] = np.mean(time_before_days)
                results[level_key]['median_days_before_outbreak'] = np.median(time_before_days)
                results[level_key]['min_days_before_outbreak'] = min(time_before_days)
                results[level_key]['max_days_before_outbreak'] = max(time_before_days)
            else:
                results[level_key]['avg_days_before_outbreak'] = results[level_key]['avg_time_before_outbreak']
                results[level_key]['median_days_before_outbreak'] = results[level_key]['median_time_before_outbreak']
                results[level_key]['min_days_before_outbreak'] = results[level_key]['min_time_before_outbreak']
                results[level_key]['max_days_before_outbreak'] = results[level_key]['max_time_before_outbreak']
        
        # Create a summary results dataframe - only for included levels
        summary_data = {
            'Threshold Level': [],
            'True Positives': [],
            'False Positives': [],
            'True Negatives': [],
            'False Negatives': [],
            'Sensitivity': [],
            'Specificity': [],
            'Precision': [],
            'F1 Score': [],
            'Average Days Warning': []
        }
        
        level_display_names = {
            'attention': 'Attention Level',
            'alert': 'Alert Level',
            'critical': 'Critical Level'
        }
        
        for level in include_levels:
            level_key = f'{level}_level'
            summary_data['Threshold Level'].append(level_display_names[level])
            summary_data['True Positives'].append(results[level_key]['true_positives'])
            summary_data['False Positives'].append(results[level_key]['false_positives'])
            summary_data['True Negatives'].append(results[level_key]['true_negatives'])
            summary_data['False Negatives'].append(results[level_key]['false_negatives'])
            summary_data['Sensitivity'].append(results[level_key]['metrics']['sensitivity'])
            summary_data['Specificity'].append(results[level_key]['metrics']['specificity'])
            summary_data['Precision'].append(results[level_key]['metrics']['precision'])
            summary_data['F1 Score'].append(results[level_key]['metrics']['f1_score'])
            summary_data['Average Days Warning'].append(results[level_key]['avg_days_before_outbreak'])
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary results
        output_dir = self.config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        # Add included levels and ignore_percent to filenames
        levels_suffix = '_'.join(include_levels)
        ignore_suffix = f"_ignore{ignore_first_percent}pct"
        summary_filename = self.config.get('summary_filename', f'early_warning_evaluation_{levels_suffix}{ignore_suffix}.csv')
        output_path = os.path.join(output_dir, summary_filename)
        summary_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved early warning evaluation to {output_path}")
        
        # Also save detailed pen results
        details_df = pd.DataFrame(results['pen_details'])
        details_filename = self.config.get('details_filename', f'early_warning_pen_details_{levels_suffix}{ignore_suffix}.csv')
        details_path = os.path.join(output_dir, details_filename)
        details_df.to_csv(details_path, index=False)
        
        # Log summary results
        self.logger.info("\nEarly Warning System Evaluation:")
        self.logger.info(f"Levels included: {include_levels}")
        self.logger.info(f"Ignoring first {ignore_first_percent}% of each run")
        self.logger.info(f"Total pens analyzed: {results['total_pens_analyzed']} ({results['total_outbreak_pens']} outbreak, {results['total_control_pens']} control)")
        
        # Log metrics for each included level
        for level in include_levels:
            level_key = f'{level}_level'
            self.logger.info(f"\n{level_display_names[level]} Metrics:")
            self.logger.info(f"Sensitivity: {results[level_key]['metrics']['sensitivity']:.2f}")
            self.logger.info(f"Specificity: {results[level_key]['metrics']['specificity']:.2f}")
            
            # Log time units appropriately
            if is_hourly:
                avg_time = results[level_key]['avg_time_before_outbreak']
                avg_days = results[level_key]['avg_days_before_outbreak']
                if not np.isnan(avg_time) and not np.isnan(avg_days):
                    self.logger.info(f"Average warning time: {avg_time:.1f} hours ({avg_days:.1f} days)")
                else:
                    self.logger.info("Average warning time: N/A")
            else:
                avg_days = results[level_key]['avg_days_before_outbreak']
                if not np.isnan(avg_days):
                    self.logger.info(f"Average warning time: {avg_days:.1f} days")
                else:
                    self.logger.info("Average warning time: N/A")
        
        return results
    
    def _format_time_value(self, time_value, is_hourly):
        """
        Format a time value into a human-readable string.
        
        Args:
            time_value: Time value in hours or days
            is_hourly: Whether the time value is in hours
            
        Returns:
            str: Formatted time string
        """
        if time_value is None:
            return "N/A"
        
        if is_hourly:
            days = time_value // 24
            hours = time_value % 24
            if days > 0:
                return f"{days}d {hours}h"
            else:
                return f"{hours}h"
        else:
            return f"{time_value}d"
    
    def print_threshold_analysis_results(self, results):
        """
        Prints a formatted summary of the threshold analysis results.
        
        Args:
            results (dict): The results dictionary returned by evaluate_thresholds method
        """
        print("\n=== EARLY WARNING SYSTEM EVALUATION SUMMARY ===")
        print(f"Total pens analyzed: {results['total_pens_analyzed']} ({results['total_outbreak_pens']} outbreak, {results['total_control_pens']} control)")
        
        # Get included levels from results
        include_levels = results.get('include_levels', ['attention', 'alert', 'critical'])
        print(f"Threshold levels included: {', '.join(include_levels)}")
        
        # Map base levels to their result keys and display names
        level_keys = [f'{level}_level' for level in include_levels]
        level_display_names = {
            'attention_level': 'Attention',
            'alert_level': 'Alert',
            'critical_level': 'Critical'
        }
        
        # Get time unit
        time_unit = results.get('time_unit', 'days')
        is_hourly = time_unit == 'hours'
        
        print("\n{:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<15} | {:<15}".format(
            "Level", "Sens.", "Spec.", "Prec.", "F1", f"Avg Warning", f"Med Warning"))
        print("-" * 90)
        
        for level in level_keys:
            if level not in results:
                continue
                
            metrics = results[level]['metrics']
            
            # Display time in appropriate format
            if is_hourly:
                avg_time = results[level].get('avg_time_before_outbreak', float('nan'))
                med_time = results[level].get('median_time_before_outbreak', float('nan'))
                
                if not np.isnan(avg_time):
                    avg_days = avg_time / 24.0
                    avg_hours = avg_time % 24
                    avg_str = f"{int(avg_days)}d {avg_hours:.1f}h"
                else:
                    avg_str = "N/A"
                    
                if not np.isnan(med_time):
                    med_days = med_time / 24.0
                    med_hours = med_time % 24
                    med_str = f"{int(med_days)}d {med_hours:.1f}h"
                else:
                    med_str = "N/A"
            else:
                avg_days = results[level].get('avg_days_before_outbreak', float('nan'))
                med_days = results[level].get('median_days_before_outbreak', float('nan'))
                avg_str = f"{avg_days:.1f}d" if not np.isnan(avg_days) else "N/A"
                med_str = f"{med_days:.1f}d" if not np.isnan(med_days) else "N/A"
            
            print("{:<10} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<10.2f} | {:<15} | {:<15}".format(
                level_display_names[level],
                metrics['sensitivity'],
                metrics['specificity'],
                metrics['precision'],
                metrics['f1_score'],
                avg_str,
                med_str
            ))
        
        print("\n=== CONFUSION MATRIX INFORMATION ===")
        for level in level_keys:
            if level not in results:
                continue
                
            tp = results[level]['true_positives']
            fp = results[level]['false_positives']
            tn = results[level]['true_negatives']
            fn = results[level]['false_negatives']
            
            print(f"\n{level_display_names[level]} Level:")
            print(f"  True Positives:  {tp} (Outbreak pen correctly flagged)")
            print(f"  False Positives: {fp} (Control pen incorrectly flagged)")
            print(f"  True Negatives:  {tn} (Control pen correctly not flagged)")
            print(f"  False Negatives: {fn} (Outbreak pen incorrectly not flagged)")
        
        # Print warning time distribution if available for any included level
        if any('time_before_outbreak' in results.get(level, {}) for level in level_keys):
            print("\n=== WARNING TIME DISTRIBUTION ===")
            for level in level_keys:
                if level not in results:
                    continue
                    
                time_values = results[level].get('time_before_outbreak', [])
                if time_values:
                    print(f"\n{level_display_names[level]} Level:")
                    
                    if is_hourly:
                        min_time = results[level].get('min_time_before_outbreak', float('nan'))
                        max_time = results[level].get('max_time_before_outbreak', float('nan'))
                        avg_time = results[level].get('avg_time_before_outbreak', float('nan'))
                        med_time = results[level].get('median_time_before_outbreak', float('nan'))
                        
                        # Check for NaN values before calculations
                        if not np.isnan(min_time):
                            min_days = min_time // 24
                            min_hours = min_time % 24
                            print(f"  Min time: {int(min_days)}d {min_hours:.1f}h ({min_time:.1f}h total)")
                        else:
                            print("  Min time: N/A")
                            
                        if not np.isnan(max_time):
                            max_days = max_time // 24
                            max_hours = max_time % 24
                            print(f"  Max time: {int(max_days)}d {max_hours:.1f}h ({max_time:.1f}h total)")
                        else:
                            print("  Max time: N/A")
                            
                        if not np.isnan(avg_time):
                            avg_days = avg_time / 24
                            print(f"  Avg time: {avg_days:.1f}d ({avg_time:.1f}h total)")
                        else:
                            print("  Avg time: N/A")
                            
                        if not np.isnan(med_time):
                            med_days = med_time / 24
                            print(f"  Med time: {med_days:.1f}d ({med_time:.1f}h total)")
                        else:
                            print("  Med time: N/A")
                    else:
                        min_days = results[level].get('min_days_before_outbreak', float('nan'))
                        max_days = results[level].get('max_days_before_outbreak', float('nan'))
                        avg_days = results[level].get('avg_days_before_outbreak', float('nan'))
                        med_days = results[level].get('median_days_before_outbreak', float('nan'))
                        
                        print(f"  Min days: {min_days:.1f}" if not np.isnan(min_days) else "  Min days: N/A")
                        print(f"  Max days: {max_days:.1f}" if not np.isnan(max_days) else "  Max days: N/A")
                        print(f"  Avg days: {avg_days:.1f}" if not np.isnan(avg_days) else "  Avg days: N/A")
                        print(f"  Med days: {med_days:.1f}" if not np.isnan(med_days) else "  Med days: N/A")

    # def optimize_thresholds(self, threshold_combinations=None):
    #     """
    #     Evaluate multiple sets of thresholds to find optimal values.
        
    #     Args:
    #         threshold_combinations (list, optional): List of threshold dictionaries to evaluate.
    #             If None, uses default combinations from config.
            
    #     Returns:
    #         pd.DataFrame: DataFrame with evaluation results for each threshold set
    #     """
    #     if threshold_combinations is None:
    #         threshold_combinations = self.config.get('default_threshold_sets', [
    #             {
    #                 'name': 'Default',
    #                 'thresholds': {
    #                     'attention': {'posture_diff': 0.6, 'abs_change_3d': -0.15, 'slope_3d': -0.04},
    #                     'alert': {'posture_diff': 0.5, 'abs_change_3d': -0.2, 'slope_3d': -0.05},
    #                     'critical': {'posture_diff': 0.2, 'abs_change_1d': -0.1, 'consecutive_neg_slope': 3}
    #                 }
    #             },
    #             {
    #                 'name': 'More Sensitive',
    #                 'thresholds': {
    #                     'attention': {'posture_diff': 0.7, 'abs_change_3d': -0.1, 'slope_3d': -0.03},
    #                     'alert': {'posture_diff': 0.6, 'abs_change_3d': -0.15, 'slope_3d': -0.04},
    #                     'critical': {'posture_diff': 0.3, 'abs_change_1d': -0.08, 'consecutive_neg_slope': 2}
    #                 }
    #             },
    #             {
    #                 'name': 'More Specific',
    #                 'thresholds': {
    #                     'attention': {'posture_diff': 0.55, 'abs_change_3d': -0.2, 'slope_3d': -0.05},
    #                     'alert': {'posture_diff': 0.4, 'abs_change_3d': -0.25, 'slope_3d': -0.07},
    #                     'critical': {'posture_diff': 0.15, 'abs_change_1d': -0.12, 'consecutive_neg_slope': 3}
    #                 }
    #             }
    #         ])
        
    #     self.logger.info(f"Starting threshold optimization with {len(threshold_combinations)} threshold combinations")
        
    #     # Test all combinations and collect results
    #     optimization_results = []
    #     for combo in threshold_combinations:
    #         self.logger.info(f"Evaluating threshold set: {combo['name']}")
    #         results = self.evaluate_thresholds(thresholds=combo['thresholds'])
            
    #         # Determine time units
    #         is_hourly = results.get('time_unit', 'days') == 'hours'
            
    #         # Extract key metrics
    #         result_entry = {
    #             'Threshold Set': combo['name'],
    #             'Attention Sensitivity': results['attention_level']['metrics']['sensitivity'],
    #             'Attention Specificity': results['attention_level']['metrics']['specificity'],
    #             'Attention F1 Score': results['attention_level']['metrics']['f1_score'],
    #             'Alert Sensitivity': results['alert_level']['metrics']['sensitivity'],
    #             'Alert Specificity': results['alert_level']['metrics']['specificity'],
    #             'Alert F1 Score': results['alert_level']['metrics']['f1_score'],
    #             'Critical Sensitivity': results['critical_level']['metrics']['sensitivity'],
    #             'Critical Specificity': results['critical_level']['metrics']['specificity'],
    #             'Critical F1 Score': results['critical_level']['metrics']['f1_score'],
    #         }
            
    #         # Add warning time metrics in appropriate units
    #         if is_hourly:
    #             result_entry.update({
    #                 'Attention Warning Hours': results['attention_level']['avg_time_before_outbreak'],
    #                 'Attention Warning Days': results['attention_level']['avg_days_before_outbreak'],
    #                 'Alert Warning Hours': results['alert_level']['avg_time_before_outbreak'],
    #                 'Alert Warning Days': results['alert_level']['avg_days_before_outbreak'],
    #                 'Critical Warning Hours': results['critical_level']['avg_time_before_outbreak'],
    #                 'Critical Warning Days': results['critical_level']['avg_days_before_outbreak']
    #             })
    #         else:
    #             result_entry.update({
    #                 'Attention Warning Days': results['attention_level']['avg_days_before_outbreak'],
    #                 'Alert Warning Days': results['alert_level']['avg_days_before_outbreak'],
    #                 'Critical Warning Days': results['critical_level']['avg_days_before_outbreak']
    #             })
            
    #         optimization_results.append(result_entry)

    #     # Convert to DataFrame and save
    #     optimization_df = pd.DataFrame(optimization_results)
    #     output_dir = self.config.get('output_dir', '.')
    #     os.makedirs(output_dir, exist_ok=True)
    #     optimization_filename = self.config.get('optimization_filename', 'threshold_optimization_results.csv')
    #     output_path = os.path.join(output_dir, optimization_filename)
    #     optimization_df.to_csv(output_path, index=False)
    #     self.logger.info(f"Saved threshold optimization results to {output_path}")
        
    #     return optimization_df