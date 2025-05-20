import numpy as np
import pandas as pd
from scipy import stats
import os
from datetime import timedelta

from evaluation.utils.processing import DataProcessor
from evaluation.utils.data_filter import DataFilter
from evaluation.utils.utils import save_filtered_dataframes
from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info


class PigBehaviorAnalyzer(DataProcessor):
    """Methods for analyzing pig behavior data (lying, not lying, and activity)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the data filter
        self.data_filter = DataFilter(self.config, self.logger)
        
        # Define the metrics to be analyzed
        self.behavior_metrics = ['num_pigs_lying', 'num_pigs_notLying', 'activity']
        
        # Store analysis results for each metric
        self.pre_outbreak_stats = {}
        self.control_stats = {}
        self.comparison_results = {}
        
        # Initialize processed_results
        self.processed_results = []

    def preprocess_monitoring_results(self):
        """Preprocess the monitoring results data for analysis.
        
        Returns:
            list: The processed results ready for further analysis.
        """
        self.processed_results = []
        self.excluded_events_count = 0
        self.logger.info("Starting preprocessing...")
        
        if not hasattr(self, 'monitoring_results') or not self.monitoring_results:
            self.logger.error("No monitoring results available. Please load data first.")
            return None
        
        for i, result in enumerate(self.monitoring_results):
            self.logger.info(f"Preprocessing result {i+1}/{len(self.monitoring_results)}: {result.get('camera','?')}/{result.get('date_span','?')}")
            processed_data = self.preprocess_data(result)
            if processed_data:
                self.processed_results.append(processed_data)
            else:
                self.logger.error(f"Preprocessing failed for {result.get('camera','?')}/{result.get('date_span','?')}. Skipping.")
        
        self.logger.info(f"Total processed results after preprocessing: {len(self.processed_results)}")
        return self.processed_results
        
    def analyze_pre_outbreak_statistics(self):
        """Analyze pre-outbreak statistics for behavior metrics using earliest removal date per event."""
        self.logger.info("Analyzing pre-outbreak behavior statistics (using earliest removal date per event)...")
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        results = {metric: [] for metric in self.behavior_metrics}
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            for metric in self.behavior_metrics:
                self.pre_outbreak_stats[metric] = pd.DataFrame()
            return self.pre_outbreak_stats
        
        # First, filter by quality metrics
        quality_filtered_results, excluded_count, excluded_pct_count = self.data_filter.filter_by_quality_metrics(
            self.processed_results, get_pen_info, json_data, analysis_type="behavior_analysis"
        )
        
        # Then filter for valid tail biting events
        filtered_events, excluded_event_count = self.data_filter.filter_tail_biting_events(
            quality_filtered_results, get_pen_info, json_data
        )
        
        if self.config.get('interpolate_resampled_data', False):
            # interpolate filtered, resampled data to avoid NaN values
            filtered_events = DataProcessor().interpolate_resampled_data(filtered_events)
            
        if self.config.get('save_preprocessed_data', False):
            # Save filtered DataFrames before processing
            if filtered_events:
                save_filtered_dataframes(
                    filtered_results=filtered_events,
                    config=self.config,
                    logger=self.logger,
                    path_manager=self.path_manager
                )
                
        # Store exclusion counts for reporting
        self.excluded_events_count = excluded_count
        self.excluded_events_missing_pct_count = excluded_pct_count

        for processed_data in filtered_events:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            pen_type, culprit_removal, datespan_gt = get_pen_info(camera, date_span, json_data)
            camera_label = camera.replace("Kamera", "Pen ")
            
            self.logger.debug(f"Processing behavior data for tail biting event: {camera_label} / {date_span}")
            interpolated_data = processed_data.get('interpolated_data')
            
            if interpolated_data is None or interpolated_data.empty:
                reason = "Empty interpolated data"
                self.data_filter.log_exclusion(
                    'other_reasons', camera, date_span, pen_type, reason, 
                    None, f"Skipping {camera_label} / {date_span} due to empty interpolated data.",
                    analysis_type="behavior_analysis"
                )
                continue
                
            # Check if any of our behavior metrics are in the data
            metrics_present = [metric for metric in self.behavior_metrics if metric in interpolated_data.columns]
            if not metrics_present:
                reason = "No behavior metrics found in data"
                self.data_filter.log_exclusion(
                    'other_reasons', camera, date_span, pen_type, reason,
                    None, f"Skipping {camera_label} / {date_span} due to missing behavior metrics.",
                    analysis_type="behavior_analysis"
                )
                continue
                
            if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                if 'datetime' in interpolated_data.columns: 
                    interpolated_data = interpolated_data.set_index('datetime')
                elif isinstance(interpolated_data.index, pd.RangeIndex) and 'datetime' in interpolated_data.index.name: 
                    interpolated_data.index = pd.to_datetime(interpolated_data.index)
                else: 
                    self.logger.error(f"Cannot set datetime index for {camera_label} / {date_span}"); 
                    continue

            culprit_removal_dates_str = culprit_removal if isinstance(culprit_removal, list) else [culprit_removal]
            valid_removal_dts = []
            for date_str in culprit_removal_dates_str:
                try:
                    dt = pd.to_datetime(date_str)
                    if not interpolated_data.empty:
                        # If the removal date is after the available data, use the last available date
                        if dt > interpolated_data.index.max():
                            self.logger.info(f"Removal date {date_str} is after last data point for {camera_label}. Using last available data point.")
                            valid_removal_dts.append(interpolated_data.index.max())
                        elif dt >= interpolated_data.index.min():
                            valid_removal_dts.append(dt)
                        else: 
                            self.logger.debug(f"Removal date {date_str} before first data point for {camera_label}. Skipping.")
                except (ValueError, TypeError) as e: 
                    self.logger.warning(f"Invalid date format '{date_str}' for {camera_label}. Error: {e}")
                    
            if not valid_removal_dts: 
                reason = "No valid removal dates in data range (all before first data point)"
                self.data_filter.log_exclusion(
                    'other_reasons', camera, date_span, pen_type, reason, 
                    None, f"No valid removal dates in range for {camera_label}. Skipping.",
                    analysis_type="behavior_analysis"
                )
                self.logger.warning(f"No valid removal dates in range for {camera_label}. Skipping.")
                continue
                
            earliest_removal_dt = min(valid_removal_dts)
            self.logger.debug(f"Using earliest removal date: {earliest_removal_dt.strftime('%Y-%m-%d')} for {camera_label}")
            removal_datetime = earliest_removal_dt
            
            # Process each behavior metric separately
            for metric in metrics_present:
                # Skip if the metric doesn't exist in the data
                if metric not in interpolated_data.columns:
                    continue
                    
                removal_data_points = interpolated_data.loc[interpolated_data.index <= removal_datetime]
                if removal_data_points.empty: 
                    self.logger.warning(f"No data on or before removal date {removal_datetime} for {camera_label}. Skipping.")
                    continue
                    
                removal_value = removal_data_points.iloc[-1].get(metric, np.nan)
                if pd.isna(removal_value): 
                    self.logger.warning(f"{metric} is NaN at removal date {removal_datetime} for {camera_label}. Skipping.")
                    continue

                day_statistics = {}
                days_list = self.config.get('days_before_list', [1, 3, 7])
                for days in days_list:
                    before_date = removal_datetime - timedelta(days=days)
                    before_value = np.nan
                    if before_date >= interpolated_data.index.min():
                        before_data_points = interpolated_data.loc[interpolated_data.index <= before_date]
                        if not before_data_points.empty: 
                            before_value = before_data_points.iloc[-1].get(metric, np.nan)
                    
                    # Initialize variables to NaN by default
                    abs_change, sym_pct_change, effect_size = np.nan, np.nan, np.nan
                    
                    # Only perform calculations if both values are valid
                    if pd.notna(removal_value) and pd.notna(before_value):
                        abs_change = removal_value - before_value
                        
                        # Calculate symmetric percentage change instead of traditional percentage change
                        # This approach handles near-zero values gracefully and is bounded between -200% and +200%
                        denominator = np.abs(removal_value) + np.abs(before_value)
                        if denominator > 1e-10:  # Small threshold to avoid division by absolute zero
                            sym_pct_change = 200 * abs_change / denominator
                        else:
                            # Both values are effectively zero
                            sym_pct_change = 0.0  # No change if both values are essentially zero
                        
                        # Calculate effect size (Cohen's d)
                        if hasattr(before_data_points, 'std') and hasattr(removal_data_points, 'std'):
                            # Get scalar standard deviation for the specific column
                            before_std = before_data_points[metric].std() if metric in before_data_points.columns else 0
                            removal_std = removal_data_points[metric].std() if metric in removal_data_points.columns else 0
                            
                            pooled_std = np.sqrt((before_std**2 + removal_std**2) / 2)
                            if pooled_std > 0:
                                cohens_d = abs_change / pooled_std
                                day_statistics[f'effect_size_{days}d'] = cohens_d
                        
                    day_statistics[f'value_{days}d_before'] = before_value
                    day_statistics[f'abs_change_{days}d'] = abs_change
                    day_statistics[f'pct_change_{days}d'] = sym_pct_change

                window_stats = {}
                analysis_windows = self.config.get('analysis_window_days', [3, 7])
                for window_days in analysis_windows:
                    window_end = removal_datetime
                    window_start = window_end - timedelta(days=window_days)
                    window_data_all_cols = interpolated_data[(interpolated_data.index > window_start) & 
                                                          (interpolated_data.index <= window_end)]
                    
                    # Initialize all statistics to NaN
                    stat_keys = ['avg', 'min', 'max', 'std', 'ci_lower', 'ci_upper', 'slope', 'slope_p_value', 
                                'slope_r_squared', 'slope_std_err']
                    for key in stat_keys:
                        window_stats[f'{window_days}d_window_{key}'] = np.nan
                        
                    # Only proceed if data is available
                    if not window_data_all_cols.empty and metric in window_data_all_cols.columns:
                        window_data = window_data_all_cols[metric].dropna()
                        
                        if window_data.empty:
                            continue
                            
                        n_points = len(window_data)
                        
                        # Basic statistics are safe with any number of points
                        window_stats[f'{window_days}d_window_avg'] = window_data.mean()
                        window_stats[f'{window_days}d_window_min'] = window_data.min()
                        window_stats[f'{window_days}d_window_max'] = window_data.max()
                        
                        # Statistics requiring at least 2 data points
                        if n_points >= 2:
                            window_stats[f'{window_days}d_window_std'] = window_data.std(ddof=1)
                            
                            if n_points > 2:  # Need more than 2 points for meaningful CIs
                                ci = self.config.get('confidence_level', 0.95)
                                sem = window_stats[f'{window_days}d_window_std'] / np.sqrt(n_points)
                                df_ci = n_points - 1
                                if df_ci > 0:
                                    t_critical = stats.t.ppf((1 + ci) / 2, df_ci)
                                    margin_of_error = t_critical * sem
                                    window_stats[f'{window_days}d_window_ci_lower'] = window_stats[f'{window_days}d_window_avg'] - margin_of_error
                                    window_stats[f'{window_days}d_window_ci_upper'] = window_stats[f'{window_days}d_window_avg'] + margin_of_error
                                    
                            # Linear regression requires at least 2 points
                            time_index = window_data_all_cols.loc[window_data.index].index
                            x = (time_index - window_end).total_seconds() / (24 * 3600)
                            y = window_data.values
                            
                            if len(x) >= 2 and len(y) >= 2 and not np.isnan(x).any() and not np.isnan(y).any():
                                try:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                    window_stats[f'{window_days}d_window_slope'] = slope
                                    window_stats[f'{window_days}d_window_slope_p_value'] = p_value
                                    window_stats[f'{window_days}d_window_slope_r_squared'] = r_value**2
                                    window_stats[f'{window_days}d_window_slope_std_err'] = std_err
                                except ValueError as e:
                                    self.logger.warning(f"Linregress failed {window_days}d in {camera_label}: {e}")
                        elif n_points == 1:
                            # Handle the case of a single data point
                            window_stats[f'{window_days}d_window_std'] = 0.0
                            window_stats[f'{window_days}d_window_ci_lower'] = window_stats[f'{window_days}d_window_avg']
                            window_stats[f'{window_days}d_window_ci_upper'] = window_stats[f'{window_days}d_window_avg']
                            # Cannot calculate slope with a single point

                result_entry = {
                    'pen': camera_label,
                    'datespan': date_span,
                    'datespan_gt': datespan_gt if datespan_gt != "Unknown" else date_span,
                    'culprit_removal_date': earliest_removal_dt.strftime('%Y-%m-%d'),
                    'value_at_removal': removal_value,
                    'metric': metric,
                    **day_statistics,
                    **window_stats
                }
                results[metric].append(result_entry)

        # Create DataFrames for each metric and save results
        for metric in self.behavior_metrics:
            if metric in results and results[metric]:
                self.pre_outbreak_stats[metric] = pd.DataFrame(results[metric])
                
                # Save each metric to a separate file
                filename = self.config.get(f'pre_outbreak_stats_{metric}_filename', 
                                           f'pre_outbreak_statistics_{metric}_filtered.csv')
                output_path = os.path.join(self.config['output_dir'], filename)
                self.pre_outbreak_stats[metric].to_csv(output_path, index=False)
                self.logger.info(f"Saved descriptive pre-outbreak statistics for {metric} to {output_path}")
            else:
                self.pre_outbreak_stats[metric] = pd.DataFrame()
                self.logger.warning(f"No pre-outbreak statistics generated for {metric} after filtering.")
        
        # Calculate statistics about the analysis
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        num_total_biting = sum(1 for p in self.processed_results if get_pen_info(p['camera'], p['date_span'], json_data)[0] == "tail biting")
        
        # Get filtering summary
        filter_summary = self.data_filter.get_summary_statistics()
        
        # Count successful analyses across all metrics
        num_analyzed = sum(1 for m in self.behavior_metrics if m in self.pre_outbreak_stats and not self.pre_outbreak_stats[m].empty)
        
        self.logger.info(f"Attempted to analyze {num_total_biting} potential tail biting events for behavior metrics.")
        self.logger.info(f"Successfully analyzed at least one metric for {num_analyzed} events.")
        self.logger.info(f"Filtering summary: {filter_summary}")
        
        # Perform a more detailed count of each metric
        for metric in self.behavior_metrics:
            metric_count = len(self.pre_outbreak_stats.get(metric, pd.DataFrame()))
            self.logger.info(f"  - {metric}: {metric_count} valid analyses")

        return self.pre_outbreak_stats
        
    def analyze_control_pen_statistics(self):
        """Analyze control pen data for behavior metrics to establish baseline variability."""
        self.logger.info("Analyzing control pen statistics for behavior metrics comparison...")
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        results = {metric: [] for metric in self.behavior_metrics}
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            for metric in self.behavior_metrics:
                self.control_stats[metric] = pd.DataFrame()
            return self.control_stats
            
        # First filter by quality metrics
        quality_filtered_results, excluded_count, excluded_pct_count = self.data_filter.filter_by_quality_metrics(
            self.processed_results, get_pen_info, json_data, analysis_type="control_analysis"
        )
        
        # Then filter for control pens
        control_pens = self.data_filter.filter_control_pens(
            quality_filtered_results, get_pen_info, json_data
        )
        
        if self.config.get('interpolate_resampled_data', False):
            # interpolate filtered, resampled data to avoid NaN values
            control_pens = DataProcessor().interpolate_resampled_data(control_pens)
         
        if self.config.get('save_preprocessed_data', False):
            # Save filtered DataFrames before processing
            if control_pens:
                save_filtered_dataframes(
                    filtered_results=control_pens,
                    config=self.config,
                    logger=self.logger,
                    path_manager=self.path_manager,
                    pen_type_override="control"
                )
        
        # Store exclusion counts for reporting
        self.excluded_controls_count = excluded_count
        self.excluded_controls_missing_pct_count = excluded_pct_count

        self.logger.info(f"Found {len(control_pens)} control pen datasets for behavior analysis")
        
        for processed_data in control_pens:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            camera_label = camera.replace("Kamera", "Pen ")
            interpolated_data = processed_data.get('interpolated_data')
            
            # Check if any of our behavior metrics are in the data
            metrics_present = [metric for metric in self.behavior_metrics if metric in interpolated_data.columns]
            if not metrics_present:
                reason = "No behavior metrics found in data"
                self.data_filter.log_exclusion(
                    'other_reasons', camera, date_span, "control", reason,
                    None, f"Skipping control pen {camera_label} / {date_span} due to missing behavior metrics.",
                    analysis_type="control_analysis"
                )
                continue
                
            if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                if 'datetime' in interpolated_data.columns:
                    interpolated_data = interpolated_data.set_index('datetime')
                elif isinstance(interpolated_data.index, pd.RangeIndex) and 'datetime' in interpolated_data.index.name:
                    interpolated_data.index = pd.to_datetime(interpolated_data.index)
                else:
                    self.logger.error(f"Cannot set datetime index for control pen {camera_label} / {date_span}")
                    continue
            
            # Sample random timepoints with sufficient data for analysis
            valid_dates = np.unique(interpolated_data.index.date)
            
            min_dates = self.config.get('min_control_dates_threshold', 10)
            if len(valid_dates) < min_dates:
                self.logger.warning(f"Control pen {camera_label} has too few dates ({len(valid_dates)}). Skipping.")
                continue
                
            # Remove edge dates to ensure there is enough data for windows
            margin = self.config.get('control_date_margin', 7)
            analysis_dates = valid_dates[margin:-margin] if len(valid_dates) > 2 * margin else []
            min_analysis_d = self.config.get('min_control_analysis_dates', 3)
            if len(analysis_dates) < min_analysis_d:
                self.logger.warning(f"Control pen {camera_label} has insufficient dates after margin removal. Skipping.")
                continue
                
            # Set random seed for reproducibility but unique per pen
            np.random.seed(self.config['random_seed'] + hash(camera + date_span) % 1000)
            
            # Sample up to X dates (or fewer if limited data)
            n_samples = self.config.get('control_samples_per_pen', 3)
            sample_count = min(n_samples, len(analysis_dates))

            sampled_dates = np.random.choice(analysis_dates, size=sample_count, replace=False)
            
            for sample_idx, sample_date in enumerate(sampled_dates):
                # Convert to datetime
                reference_datetime = pd.to_datetime(sample_date)
                sample_id = f"{sample_idx+1}"
                
                # Process each behavior metric
                for metric in metrics_present:
                    # Skip if the metric doesn't exist in the data
                    if metric not in interpolated_data.columns:
                        continue
                        
                    # Get value at reference point
                    reference_data_points = interpolated_data.loc[interpolated_data.index <= reference_datetime]
                    if reference_data_points.empty:
                        continue
                        
                    reference_value = reference_data_points.iloc[-1].get(metric, np.nan)
                    if pd.isna(reference_value):
                        continue
                        
                    # Calculate same statistics as for outbreaks
                    day_statistics = {}
                    days_list = self.config.get('days_before_list', [1, 3, 7])
                    for days in days_list:
                        before_date = reference_datetime - timedelta(days=days)
                        before_value = np.nan
                        if before_date >= interpolated_data.index.min():
                            before_data_points = interpolated_data.loc[interpolated_data.index <= before_date]
                            if not before_data_points.empty:
                                before_value = before_data_points.iloc[-1].get(metric, np.nan)
                                
                        # Initialize with NaN
                        abs_change, pct_change, ttest_pvalue = np.nan, np.nan, np.nan
                        
                        # Only calculate if both values exist
                        if pd.notna(reference_value) and pd.notna(before_value):
                            abs_change = reference_value - before_value
                            
                            min_threshold = 1e-6
                            if abs(before_value) > min_threshold:
                                pct_change = (abs_change / np.abs(before_value) * 100)
                            else:
                                # Division by zero/small value case
                                pct_change = np.nan
                                
                            if reference_value != before_value:
                                try:
                                    ttest_pvalue = 1.0 
                                except ValueError:
                                    ttest_pvalue = 1.0
                            else:
                                ttest_pvalue = 1.0
                                
                        day_statistics[f'value_{days}d_before'] = before_value
                        day_statistics[f'abs_change_{days}d'] = abs_change
                        day_statistics[f'pct_change_{days}d'] = pct_change
                        day_statistics[f'ttest_pvalue_{days}d'] = ttest_pvalue
                        
                    window_stats = {}
                    analysis_windows = self.config.get('analysis_window_days', [3, 7])
                    for window_days in analysis_windows:
                        window_end = reference_datetime
                        window_start = window_end - timedelta(days=window_days)
                        window_data_all_cols = interpolated_data[(interpolated_data.index > window_start) & 
                                                            (interpolated_data.index <= window_end)]
                                                            
                        # Initialize all statistics to NaN
                        stat_keys = ['avg', 'min', 'max', 'std', 'ci_lower', 'ci_upper', 
                                'slope', 'slope_p_value', 'slope_r_squared', 'slope_std_err']
                        for key in stat_keys:
                            window_stats[f'{window_days}d_window_{key}'] = np.nan
                            
                        # Only proceed if there is data to analyze
                        if not window_data_all_cols.empty and metric in window_data_all_cols.columns:
                            window_data = window_data_all_cols[metric].dropna()
                            
                            if window_data.empty:
                                continue
                                
                            n_points = len(window_data)
                            # Basic statistics that work with any number of points
                            window_stats[f'{window_days}d_window_avg'] = window_data.mean()
                            window_stats[f'{window_days}d_window_min'] = window_data.min()
                            window_stats[f'{window_days}d_window_max'] = window_data.max()
                            
                            # Statistics requiring at least 2 data points
                            if n_points >= 2:
                                window_stats[f'{window_days}d_window_std'] = window_data.std(ddof=1)
                                
                                if n_points > 2:  # Need more than 2 points for meaningful CIs
                                    ci = self.config.get('confidence_level', 0.95)
                                    sem = window_stats[f'{window_days}d_window_std'] / np.sqrt(n_points)
                                    df_ci = n_points - 1
                                    
                                    if df_ci > 0:
                                        t_critical = stats.t.ppf((1 + ci) / 2, df_ci)
                                        margin_of_error = t_critical * sem
                                        window_stats[f'{window_days}d_window_ci_lower'] = window_stats[f'{window_days}d_window_avg'] - margin_of_error
                                        window_stats[f'{window_days}d_window_ci_upper'] = window_stats[f'{window_days}d_window_avg'] + margin_of_error
                                        
                                # Linear regression needs at least 2 points
                                time_index = window_data_all_cols.loc[window_data.index].index
                                x = (time_index - window_end).total_seconds() / (24 * 3600)
                                y = window_data.values
                                
                                if len(x) >= 2 and len(y) >= 2 and not np.isnan(x).any() and not np.isnan(y).any():
                                    try:
                                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                                        window_stats[f'{window_days}d_window_slope'] = slope
                                        window_stats[f'{window_days}d_window_slope_p_value'] = p_value
                                        window_stats[f'{window_days}d_window_slope_r_squared'] = r_value**2
                                        window_stats[f'{window_days}d_window_slope_std_err'] = std_err
                                    except ValueError as e:
                                        self.logger.warning(f"Linregress failed {window_days}d in control {camera_label}: {e}")
                            elif n_points == 1:
                                # Handle single point case more explicitly
                                window_stats[f'{window_days}d_window_std'] = 0.0
                                window_stats[f'{window_days}d_window_ci_lower'] = window_stats[f'{window_days}d_window_avg']
                                window_stats[f'{window_days}d_window_ci_upper'] = window_stats[f'{window_days}d_window_avg']
                                # Cannot calculate slope with one point
                                    
                    result_entry = {
                        'pen': camera_label,
                        'datespan': date_span,
                        'reference_date': reference_datetime.strftime('%Y-%m-%d'),
                        'sample_id': sample_id,
                        'value_at_reference': reference_value,
                        'metric': metric,
                        **day_statistics,
                        **window_stats
                    }
                    results[metric].append(result_entry)
                
        # Create and save DataFrames for each metric
        for metric in self.behavior_metrics:
            if metric in results and results[metric]:
                self.control_stats[metric] = pd.DataFrame(results[metric])
                
                # Save each metric to a separate file
                filename = self.config.get(f'control_stats_{metric}_filename', 
                                          f'control_statistics_{metric}.csv')
                output_path = os.path.join(self.config['output_dir'], filename)
                self.control_stats[metric].to_csv(output_path, index=False)
                self.logger.info(f"Saved control pen statistics for {metric} to {output_path}")
            else:
                self.control_stats[metric] = pd.DataFrame()
                self.logger.warning(f"No control pen statistics generated for {metric} after filtering.")
            
        # Get filtering summary
        filter_summary = self.data_filter.get_summary_statistics()
        num_total_control = sum(1 for p in self.processed_results if get_pen_info(p['camera'], p['date_span'], json_data)[0] == "control")
        
        self.logger.info(f"Analyzed {num_total_control} control pens for behavior metrics.")
        self.logger.info(f"Filtering summary: {filter_summary}")
        
        for metric in self.behavior_metrics:
            if metric in self.control_stats and not self.control_stats[metric].empty:
                num_analyzed = len(self.control_stats[metric][['pen', 'datespan']].drop_duplicates())
                num_reference_points = len(self.control_stats[metric])
                self.logger.info(f"  - {metric}: {num_analyzed} control pen datasets with {num_reference_points} reference points")
        
        return self.control_stats
        
    def compare_outbreak_vs_control_statistics(self):
        """Compare statistics between outbreak and control pens for behavior metrics."""
        self.logger.info("Comparing outbreak vs control pen statistics for behavior metrics...")

        # For each metric, check if there is data for both outbreak and control
        for metric in self.behavior_metrics:
            if metric not in self.pre_outbreak_stats or self.pre_outbreak_stats[metric] is None or self.pre_outbreak_stats[metric].empty:
                self.logger.error(f"No pre-outbreak statistics available for {metric}. Cannot compare with controls.")
                continue

            if metric not in self.control_stats or self.control_stats[metric] is None or self.control_stats[metric].empty:
                self.logger.error(f"No control pen statistics available for {metric}. Cannot compare with outbreaks.")
                continue
                
            self.logger.info(f"Comparing {metric} between outbreak and control pens...")

            comparison_results = {
                'metrics': [],
                'outbreak_stats': [],
                'control_stats': [],
                'p_values': [],
                'is_significant': [],
                'effect_size': []
            }

            # Get metrics from config
            metrics_to_compare = self.config.get('comparison_metrics', [
                'value_at_removal', '3d_window_avg', '7d_window_avg', 
                '3d_window_slope', '7d_window_slope',
                'abs_change_1d', 'abs_change_3d', 'abs_change_7d'
            ])

            # Build dictionaries dynamically
            outbreak_cols = {col: col for col in metrics_to_compare}
            control_cols = {col: col for col in metrics_to_compare}
            # Special case for value_at_removal
            if 'value_at_removal' in control_cols:
                control_cols['value_at_removal'] = 'value_at_reference'

            # Create a dictionary to store results
            comparison_dict = {}

            for compare_metric in metrics_to_compare:
                outbreak_col = outbreak_cols[compare_metric]
                control_col = control_cols[compare_metric]

                # Skip if column doesn't exist in either dataset
                if outbreak_col not in self.pre_outbreak_stats[metric].columns or control_col not in self.control_stats[metric].columns:
                    self.logger.warning(f"Skipping comparison for {metric} - {compare_metric}: columns not found.")
                    continue

                # Get values for comparison
                outbreak_values = self.pre_outbreak_stats[metric][outbreak_col].dropna()
                control_values = self.control_stats[metric][control_col].dropna()

                if len(outbreak_values) < self.config.get('min_sample_values_comparison', 2) or len(control_values) < self.config.get('min_sample_values_comparison', 2):
                    self.logger.warning(f"Insufficient data for {metric} - {compare_metric} comparison.")
                    continue

                # Calculate statistics
                outbreak_mean = outbreak_values.mean()
                outbreak_std = outbreak_values.std()
                control_mean = control_values.mean()
                control_std = control_values.std()

                # Perform Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_value = stats.mannwhitneyu(outbreak_values, control_values, alternative='two-sided')
                    alpha = self.config.get('significance_level', 0.05)
                    is_significant = p_value < alpha

                    # Calculate effect size (Cohen's d) - approximate for non-parametric
                    n1, n2 = len(outbreak_values), len(control_values)
                    pooled_std = np.sqrt(((n1 - 1) * outbreak_std**2 + (n2 - 1) * control_std**2) / (n1 + n2 - 2))
                    effect_size = abs(outbreak_mean - control_mean) / pooled_std if pooled_std > 0 else np.nan

                    # Store in results
                    comparison_results['metrics'].append(f"{metric}_{compare_metric}")
                    comparison_results['outbreak_stats'].append((outbreak_mean, outbreak_std))
                    comparison_results['control_stats'].append((control_mean, control_std))
                    comparison_results['p_values'].append(p_value)
                    comparison_results['is_significant'].append(is_significant)
                    comparison_results['effect_size'].append(effect_size)

                    # Add to dictionary
                    comparison_dict[f"{metric}_{compare_metric}"] = {
                        'outbreak_mean': outbreak_mean,
                        'outbreak_std': outbreak_std,
                        'control_mean': control_mean,
                        'control_std': control_std,
                        'p_value': p_value,
                        'is_significant': is_significant,
                        'effect_size': effect_size
                    }

                    self.logger.info(f"Comparison for {metric}_{compare_metric}: Outbreak={outbreak_mean:.3f}±{outbreak_std:.3f}, " +
                            f"Control={control_mean:.3f}±{control_std:.3f}, p={p_value:.4f}, " +
                            f"Significant={is_significant}, Effect Size={effect_size:.2f}")

                except Exception as e:
                    self.logger.error(f"Error comparing {metric}_{compare_metric}: {e}")

            # Save comparison results for this metric
            if comparison_results['metrics']:
                # Save the comparison results per metric
                self.comparison_results[metric] = comparison_dict
                
                # Save to CSV file
                filename = self.config.get(f'comparison_stats_{metric}_filename', 
                                         f'outbreak_vs_control_comparison_{metric}.csv')
                # Ensure output_dir exists
                output_dir = self.config.get('output_dir', '.')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)

                # Create a dataframe for saving
                comparison_df = pd.DataFrame({
                    'Metric': comparison_results['metrics'],
                    'Outbreak_Mean': [stats[0] for stats in comparison_results['outbreak_stats']],
                    'Outbreak_Std': [stats[1] for stats in comparison_results['outbreak_stats']],
                    'Control_Mean': [stats[0] for stats in comparison_results['control_stats']],
                    'Control_Std': [stats[1] for stats in comparison_results['control_stats']],
                    'P_Value': comparison_results['p_values'],
                    'Is_Significant': comparison_results['is_significant'],
                    'Effect_Size': comparison_results['effect_size']
                })

                try:
                    comparison_df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved outbreak vs control comparison for {metric} to {output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save comparison CSV for {metric}: {e}")
            else:
                self.logger.warning(f"No valid comparisons could be made for {metric}.")

        return self.comparison_results
        
    def process_all_data(self):
        """Process all monitoring results and store them for analysis."""
        return self.preprocess_monitoring_results()
    
    def analyze_activity_components(self):
        """Analyze lying and not-lying pig components for both outbreak and control pens."""
        self.logger.info("Analyzing activity components (lying vs not-lying pigs)...")
        
        required_metrics = ['num_pigs_lying', 'num_pigs_notLying']
        missing_metrics = []
        
        for metric in required_metrics:
            if metric not in self.pre_outbreak_stats or self.pre_outbreak_stats[metric].empty:
                missing_metrics.append(metric)
        
        if missing_metrics:
            self.logger.error(f"Missing required metrics for component analysis: {missing_metrics}")
            return None
        
        # Initialize results
        results = {
            'outbreak_trajectories': pd.DataFrame(),
            'control_trajectories': pd.DataFrame(),
            'component_statistics': pd.DataFrame(),
            'comparison_results': pd.DataFrame()
        }
        
        # --- 1. Analyze Outbreak Component Trajectories ---
        self.logger.info("Analyzing outbreak component trajectories...")
        
        outbreak_lying = self.pre_outbreak_stats['num_pigs_lying'].copy()
        outbreak_not_lying = self.pre_outbreak_stats['num_pigs_notLying'].copy()
        
        # Extract trajectories for outbreak pens
        days_before_list = self.config.get('days_before_list', [7, 3, 1])
        days_list_with_removal = days_before_list + [0]
        
        outbreak_trajectory_data = []
        
        for idx, row in outbreak_lying.iterrows():
            pen = row['pen']
            datespan = row['datespan']
            
            # Get corresponding not lying data
            not_lying_row = outbreak_not_lying[
                (outbreak_not_lying['pen'] == pen) & 
                (outbreak_not_lying['datespan'] == datespan)
            ]
            
            if not_lying_row.empty:
                continue
                
            not_lying_row = not_lying_row.iloc[0]
            
            # Extract trajectories (including removal day as day 0)
            for d in days_list_with_removal:
                if d == 0:
                    lying_val = row.get('value_at_removal', np.nan)
                    not_lying_val = not_lying_row.get('value_at_removal', np.nan)
                else:
                    lying_val = row.get(f'value_{d}d_before', np.nan)
                    not_lying_val = not_lying_row.get(f'value_{d}d_before', np.nan)
                
                if pd.notna(lying_val) and pd.notna(not_lying_val):
                    outbreak_trajectory_data.append({
                        'pen': pen,
                        'datespan': datespan,
                        'days_before_removal': -d if d > 0 else 0,
                        'lying_pigs': lying_val,
                        'not_lying_pigs': not_lying_val,
                        'group': 'outbreak'
                    })
        
        if outbreak_trajectory_data:
            results['outbreak_trajectories'] = pd.DataFrame(outbreak_trajectory_data)
        
        # --- 2. Analyze Control Component Trajectories ---
        self.logger.info("Analyzing control component trajectories...")
        
        if ('num_pigs_lying' in self.control_stats and 'num_pigs_notLying' in self.control_stats and
            not self.control_stats['num_pigs_lying'].empty and not self.control_stats['num_pigs_notLying'].empty):
            
            control_lying = self.control_stats['num_pigs_lying'].copy()
            control_not_lying = self.control_stats['num_pigs_notLying'].copy()
            
            control_trajectory_data = []
            
            for idx, row in control_lying.iterrows():
                pen = row['pen']
                datespan = row['datespan']
                
                not_lying_row = control_not_lying[
                    (control_not_lying['pen'] == pen) & 
                    (control_not_lying['datespan'] == datespan)
                ]
                
                if not_lying_row.empty:
                    continue
                    
                not_lying_row = not_lying_row.iloc[0]
                
                # Extract trajectories (including reference day as day 0)
                for d in days_list_with_removal:
                    if d == 0:
                        lying_val = row.get('value_at_reference', np.nan)
                        not_lying_val = not_lying_row.get('value_at_reference', np.nan)
                    else:
                        lying_val = row.get(f'value_{d}d_before', np.nan)
                        not_lying_val = not_lying_row.get(f'value_{d}d_before', np.nan)
                    
                    if pd.notna(lying_val) and pd.notna(not_lying_val):
                        control_trajectory_data.append({
                            'pen': pen,
                            'datespan': datespan,
                            'days_before_removal': -d if d > 0 else 0,
                            'lying_pigs': lying_val,
                            'not_lying_pigs': not_lying_val,
                            'group': 'control'
                        })
            
            if control_trajectory_data:
                results['control_trajectories'] = pd.DataFrame(control_trajectory_data)
        
        # --- 3. Calculate Component Statistics by Day ---
        self.logger.info("Calculating component statistics...")
        
        if not results['outbreak_trajectories'].empty:
            # Calculate statistics by day for outbreak
            outbreak_stats = results['outbreak_trajectories'].groupby('days_before_removal').agg({
                'lying_pigs': ['mean', 'std', 'count', 'min', 'max'],
                'not_lying_pigs': ['mean', 'std', 'count', 'min', 'max']
            }).reset_index()
            outbreak_stats['group'] = 'outbreak'
            
            # Rename columns for clarity
            outbreak_stats.columns = [
                'days_before_removal',
                'lying_mean', 'lying_std', 'lying_count', 'lying_min', 'lying_max',
                'not_lying_mean', 'not_lying_std', 'not_lying_count', 'not_lying_min', 'not_lying_max',
                'group'
            ]
            
            # Calculate total pigs (lying + not lying)
            outbreak_stats['total_pigs_mean'] = outbreak_stats['lying_mean'] + outbreak_stats['not_lying_mean']
            outbreak_stats['lying_percentage'] = (outbreak_stats['lying_mean'] / outbreak_stats['total_pigs_mean']) * 100
            
            if not results['control_trajectories'].empty:
                # Calculate statistics by day for control
                control_stats = results['control_trajectories'].groupby('days_before_removal').agg({
                    'lying_pigs': ['mean', 'std', 'count', 'min', 'max'],
                    'not_lying_pigs': ['mean', 'std', 'count', 'min', 'max']
                }).reset_index()
                control_stats['group'] = 'control'
                
                # Rename columns for clarity
                control_stats.columns = [
                    'days_before_removal',
                    'lying_mean', 'lying_std', 'lying_count', 'lying_min', 'lying_max',
                    'not_lying_mean', 'not_lying_std', 'not_lying_count', 'not_lying_min', 'not_lying_max',
                    'group'
                ]
                
                # Calculate total pigs
                control_stats['total_pigs_mean'] = control_stats['lying_mean'] + control_stats['not_lying_mean']
                control_stats['lying_percentage'] = (control_stats['lying_mean'] / control_stats['total_pigs_mean']) * 100
                
                # Combine outbreak and control statistics
                results['component_statistics'] = pd.concat([outbreak_stats, control_stats], ignore_index=True)
            else:
                results['component_statistics'] = outbreak_stats
        
        # --- 4. Perform Comparison Analysis ---
        self.logger.info("Performing component comparison analysis...")
        
        if not results['outbreak_trajectories'].empty and not results['control_trajectories'].empty:
            # Find common days for comparison
            outbreak_days = set(results['outbreak_trajectories']['days_before_removal'].unique())
            control_days = set(results['control_trajectories']['days_before_removal'].unique())
            common_days = sorted(list(outbreak_days & control_days))
            
            comparison_results = []
            
            for day in common_days:
                # Get data for this specific day
                outbreak_day_data = results['outbreak_trajectories'][
                    results['outbreak_trajectories']['days_before_removal'] == day
                ]
                control_day_data = results['control_trajectories'][
                    results['control_trajectories']['days_before_removal'] == day
                ]
                
                # Compare lying pigs
                outbreak_lying_values = outbreak_day_data['lying_pigs'].dropna()
                control_lying_values = control_day_data['lying_pigs'].dropna()
                
                if len(outbreak_lying_values) >= 2 and len(control_lying_values) >= 2:
                    u_stat_lying, p_value_lying = stats.mannwhitneyu(
                        outbreak_lying_values, control_lying_values, alternative='two-sided'
                    )
                    
                    # Calculate effect size for lying
                    n1, n2 = len(outbreak_lying_values), len(control_lying_values)
                    pooled_std = np.sqrt(((n1 - 1) * outbreak_lying_values.std()**2 + 
                                        (n2 - 1) * control_lying_values.std()**2) / (n1 + n2 - 2))
                    if pooled_std > 0:
                        effect_size_lying = abs(outbreak_lying_values.mean() - control_lying_values.mean()) / pooled_std
                    else:
                        effect_size_lying = np.nan
                else:
                    p_value_lying = np.nan
                    effect_size_lying = np.nan
                
                # Compare not lying pigs
                outbreak_not_lying_values = outbreak_day_data['not_lying_pigs'].dropna()
                control_not_lying_values = control_day_data['not_lying_pigs'].dropna()
                
                if len(outbreak_not_lying_values) >= 2 and len(control_not_lying_values) >= 2:
                    u_stat_not_lying, p_value_not_lying = stats.mannwhitneyu(
                        outbreak_not_lying_values, control_not_lying_values, alternative='two-sided'
                    )
                    
                    # Calculate effect size for not lying
                    n1, n2 = len(outbreak_not_lying_values), len(control_not_lying_values)
                    pooled_std = np.sqrt(((n1 - 1) * outbreak_not_lying_values.std()**2 + 
                                        (n2 - 1) * control_not_lying_values.std()**2) / (n1 + n2 - 2))
                    if pooled_std > 0:
                        effect_size_not_lying = abs(outbreak_not_lying_values.mean() - control_not_lying_values.mean()) / pooled_std
                    else:
                        effect_size_not_lying = np.nan
                else:
                    p_value_not_lying = np.nan
                    effect_size_not_lying = np.nan
                
                comparison_results.append({
                    'days_before_removal': day,
                    'lying_p_value': p_value_lying,
                    'lying_effect_size': effect_size_lying,
                    'not_lying_p_value': p_value_not_lying,
                    'not_lying_effect_size': effect_size_not_lying,
                    'lying_outbreak_mean': outbreak_lying_values.mean() if len(outbreak_lying_values) > 0 else np.nan,
                    'lying_control_mean': control_lying_values.mean() if len(control_lying_values) > 0 else np.nan,
                    'not_lying_outbreak_mean': outbreak_not_lying_values.mean() if len(outbreak_not_lying_values) > 0 else np.nan,
                    'not_lying_control_mean': control_not_lying_values.mean() if len(control_not_lying_values) > 0 else np.nan
                })
            
            if comparison_results:
                results['comparison_results'] = pd.DataFrame(comparison_results)
        
        # --- 5. Save Results to CSV Files ---
        output_dir = self.config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save outbreak trajectories
        if not results['outbreak_trajectories'].empty:
            outbreak_traj_file = os.path.join(output_dir, 'activity_components_outbreak_trajectories.csv')
            results['outbreak_trajectories'].to_csv(outbreak_traj_file, index=False)
            self.logger.info(f"Saved outbreak component trajectories to {outbreak_traj_file}")
        
        # Save control trajectories
        if not results['control_trajectories'].empty:
            control_traj_file = os.path.join(output_dir, 'activity_components_control_trajectories.csv')
            results['control_trajectories'].to_csv(control_traj_file, index=False)
            self.logger.info(f"Saved control component trajectories to {control_traj_file}")
        
        # Save component statistics
        if not results['component_statistics'].empty:
            stats_file = os.path.join(output_dir, 'activity_components_statistics.csv')
            results['component_statistics'].to_csv(stats_file, index=False)
            self.logger.info(f"Saved component statistics to {stats_file}")
        
        # Save comparison results
        if not results['comparison_results'].empty:
            comparison_file = os.path.join(output_dir, 'activity_components_comparison.csv')
            results['comparison_results'].to_csv(comparison_file, index=False)
            self.logger.info(f"Saved component comparison results to {comparison_file}")
        
        return results
        
    def run_complete_behavior_analysis(self):
        """Run all analysis steps for behavior metrics and return combined results."""
        results = {}
        
        # Process all data first
        if not hasattr(self, 'processed_results') or not self.processed_results:
            self.process_all_data()
            
        if not self.processed_results:
            self.logger.error("No processed results available for analysis.")
            return {"error": "No processed data available"}
        
        # Step 1: Analyze pre-outbreak statistics for behavior metrics
        pre_outbreak_stats = self.analyze_pre_outbreak_statistics()
        results['pre_outbreak_stats'] = {metric: not df.empty for metric, df in pre_outbreak_stats.items()}
        
        # Step 2: Analyze control pen statistics for behavior metrics
        control_stats = self.analyze_control_pen_statistics()
        results['control_stats'] = {metric: not df.empty for metric, df in control_stats.items()}
        
        # Step 3: Compare outbreak and control statistics for behavior metrics
        comparison_results = self.compare_outbreak_vs_control_statistics()
        results['comparison_results'] = {metric: bool(comparison) for metric, comparison in comparison_results.items()}
        
        # Step 4: Analyze activity components (lying vs not-lying)
        component_results = self.analyze_activity_components()
        if component_results:
            results['component_analysis'] = {
                'outbreak_trajectories': not component_results['outbreak_trajectories'].empty,
                'control_trajectories': not component_results['control_trajectories'].empty,
                'component_statistics': not component_results['component_statistics'].empty,
                'comparison_results': not component_results['comparison_results'].empty
            }
        else:
            results['component_analysis'] = {'error': 'No component analysis performed'}
        
        self.logger.info("Completed behavior metrics analysis.")
        
        # Log summary of results
        self.logger.info("=== Behavior Analysis Summary ===")
        for metric in self.behavior_metrics:
            outbreak_count = len(self.pre_outbreak_stats.get(metric, pd.DataFrame()))
            control_count = len(self.control_stats.get(metric, pd.DataFrame()))
            comparison_count = len(self.comparison_results.get(metric, {}))
            
            self.logger.info(f"{metric}: {outbreak_count} outbreaks, {control_count} controls, {comparison_count} comparisons")
        
        # Log component analysis summary
        if 'component_analysis' in results and 'error' not in results['component_analysis']:
            self.logger.info("\n=== Component Analysis Summary ===")
            self.logger.info(f"Outbreak trajectories available: {results['component_analysis']['outbreak_trajectories']}")
            self.logger.info(f"Control trajectories available: {results['component_analysis']['control_trajectories']}")
            self.logger.info(f"Component statistics available: {results['component_analysis']['component_statistics']}")
            self.logger.info(f"Comparison results available: {results['component_analysis']['comparison_results']}")
        
        return results