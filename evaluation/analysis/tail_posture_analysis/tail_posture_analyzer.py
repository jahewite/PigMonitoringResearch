import os
import numpy as np
import pandas as pd

from scipy import stats
from datetime import timedelta

from evaluation.utils.processing import DataProcessor
from evaluation.utils.utils import save_filtered_dataframes
from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info
from evaluation.utils.data_filter import DataFilter


class TailPostureAnalyzer(DataProcessor):
    """Methods for analyzing tail posture data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_filter = DataFilter(self.config, self.logger)
        
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
        """Analyze pre-outbreak statistics using earliest removal date per event."""
        self.logger.info("Analyzing pre-outbreak statistics (using earliest removal date per event)...")
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        results = []
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            self.pre_outbreak_stats = pd.DataFrame()
            return self.pre_outbreak_stats
        
        # First, filter by quality metrics
        quality_filtered_results, excluded_count, excluded_pct_count = self.data_filter.filter_by_quality_metrics(
            self.processed_results, get_pen_info, json_data, analysis_type="tail_biting_analysis"
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
            
            self.logger.debug(f"Processing tail biting event: {camera_label} / {date_span}")
            interpolated_data = processed_data.get('interpolated_data')
            
            # Ensure datetime index is properly set
            if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                if 'datetime' in interpolated_data.columns: 
                    interpolated_data = interpolated_data.set_index('datetime')
                elif isinstance(interpolated_data.index, pd.RangeIndex) and 'datetime' in interpolated_data.index.name: 
                    interpolated_data.index = pd.to_datetime(interpolated_data.index)
                else: 
                    self.logger.error(f"Cannot set datetime index for {camera_label} / {date_span}")
                    continue

            # Process removal dates
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
                self.data_filter.log_exclusion('other_reasons', camera, date_span, pen_type, reason,
                                None, f"No valid removal dates in range for {camera_label}. Skipping.",
                                analysis_type="tail_biting_analysis")
                self.logger.warning(f"No valid removal dates in range for {camera_label}. Skipping.")
                continue
                
            earliest_removal_dt = min(valid_removal_dts)
            self.logger.debug(f"Using earliest removal date: {earliest_removal_dt.strftime('%Y-%m-%d')} for {camera_label}")
            removal_datetime = earliest_removal_dt
            removal_data_points = interpolated_data.loc[interpolated_data.index <= removal_datetime]
            
            if removal_data_points.empty: 
                self.logger.warning(f"No data on or before removal date {removal_datetime} for {camera_label}. Skipping.")
                continue
                
            removal_value = removal_data_points.iloc[-1].get('posture_diff', np.nan)
            if pd.isna(removal_value): 
                self.logger.warning(f"Posture diff is NaN at removal date {removal_datetime} for {camera_label}. Skipping.")
                continue

            day_statistics = {}
            days_list = self.config.get('days_before_list', [1, 3, 7])
            for days in days_list:
                before_date = removal_datetime - timedelta(days=days)
                before_value = np.nan
                if before_date >= interpolated_data.index.min():
                    before_data_points = interpolated_data.loc[interpolated_data.index <= before_date]
                    if not before_data_points.empty: 
                        before_value = before_data_points.iloc[-1].get('posture_diff', np.nan)
                
                # Initialize variables to NaN by default
                abs_change, sym_pct_change, ttest_pvalue = np.nan, np.nan, np.nan
                
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
                        before_std = before_data_points['posture_diff'].std() if 'posture_diff' in before_data_points.columns else 0
                        removal_std = removal_data_points['posture_diff'].std() if 'posture_diff' in removal_data_points.columns else 0
                        
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
                window_data_all_cols = interpolated_data[(interpolated_data.index > window_start) & (interpolated_data.index <= window_end)]
                
                # Initialize all statistics to NaN
                stat_keys = ['avg', 'min', 'max', 'std', 'ci_lower', 'ci_upper', 'slope', 'slope_p_value', 'slope_r_squared', 'slope_std_err']
                for key in stat_keys:
                    window_stats[f'{window_days}d_window_{key}'] = np.nan
                    
                # Only proceed if data is available
                if not window_data_all_cols.empty and 'posture_diff' in window_data_all_cols.columns:
                    window_data = window_data_all_cols['posture_diff'].dropna()
                    
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

            result_entry = {
                'pen': camera_label,
                'datespan': date_span,
                'datespan_gt': datespan_gt if datespan_gt != "Unknown" else date_span,
                'culprit_removal_date': earliest_removal_dt.strftime('%Y-%m-%d'),
                'value_at_removal': removal_value,
                **day_statistics,
                **window_stats
            }
            results.append(result_entry)

        # Create DataFrame and save results
        if results:
            self.pre_outbreak_stats = pd.DataFrame(results)
        else:
            self.pre_outbreak_stats = pd.DataFrame()
            
        # Calculate statistics about the analysis
        num_total_biting = sum(1 for p in self.processed_results if get_pen_info(p['camera'], p['date_span'], json_data)[0] == "tail biting")
        num_analyzed = len(self.pre_outbreak_stats)
        total_excluded = self.excluded_events_count + self.excluded_events_missing_pct_count + excluded_event_count
        
        # Get filtering summary
        filter_summary = self.data_filter.get_summary_statistics()
        
        self.logger.info(f"Attempted to analyze {num_total_biting} potential tail biting events.")
        self.logger.info(f"Successfully analyzed {num_analyzed} events.")
        self.logger.info(f"Filtering summary: {filter_summary}")
        
        if not self.pre_outbreak_stats.empty:
            filename = self.config.get('pre_outbreak_stats_filename', 'pre_outbreak_statistics_filtered.csv')
            output_path = os.path.join(self.config['output_dir'], filename)
            self.pre_outbreak_stats.to_csv(output_path, index=False)
            self.logger.info(f"Saved descriptive pre-outbreak statistics to {output_path}")
        else:
            self.logger.warning("No pre-outbreak statistics generated after filtering.")
            
        return self.pre_outbreak_stats
        
    def analyze_control_pen_statistics(self):
        """Analyze control pen data to establish baseline variability."""
        self.logger.info("Analyzing control pen statistics for comparison...")
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        results = []
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            self.control_stats = pd.DataFrame()
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
        
        # Continue with analysis of filtered control pens
        for processed_data in control_pens:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            camera_label = camera.replace("Kamera", "Pen ")
            interpolated_data = processed_data.get('interpolated_data')
            
            # Ensure datetime index is properly set
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
                
                # Get value at reference point
                reference_data_points = interpolated_data.loc[interpolated_data.index <= reference_datetime]
                if reference_data_points.empty:
                    continue
                    
                reference_value = reference_data_points.iloc[-1].get('posture_diff', np.nan)
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
                            before_value = before_data_points.iloc[-1].get('posture_diff', np.nan)
                            
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
                        
                    # Only proceed if there is enough data to analyse
                    if not window_data_all_cols.empty and 'posture_diff' in window_data_all_cols.columns:
                        window_data = window_data_all_cols['posture_diff'].dropna()
                        
                        if window_data.empty:
                            continue
                            
                        n_points = len(window_data)
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
                                
                result_entry = {
                    'pen': camera_label,
                    'datespan': date_span,
                    'reference_date': reference_datetime.strftime('%Y-%m-%d'),
                    'sample_id': sample_id,
                    'value_at_reference': reference_value,
                    **day_statistics,
                    **window_stats
                }
                results.append(result_entry)
                
        if results:
            self.control_stats = pd.DataFrame(results)
        else:
            self.control_stats = pd.DataFrame()
            
        # Calculate statistics about the analysis
        num_total_control = sum(1 for p in self.processed_results if get_pen_info(p['camera'], p['date_span'], json_data)[0] == "control")
        num_analyzed = len(self.control_stats[['pen', 'datespan']].drop_duplicates()) if not self.control_stats.empty else 0
        num_reference_points = len(self.control_stats)
        
        # Get filtering summary
        filter_summary = self.data_filter.get_summary_statistics()
        
        self.logger.info(f"Analyzed {num_total_control} potential control pens.")
        self.logger.info(f"Successfully analyzed {num_analyzed} control pen datasets (from {len(self.control_stats['pen'].unique())} unique pens) with {num_reference_points} reference points.")
        self.logger.info(f"Filtering summary: {filter_summary}")
        
        if not self.control_stats.empty:
            filename = self.config.get('control_stats_filename', 'control_statistics.csv')
            output_path = os.path.join(self.config['output_dir'], filename)
            self.control_stats.to_csv(output_path, index=False)
            self.logger.info(f"Saved control pen statistics to {output_path}")
        else:
            self.logger.warning("No control pen statistics generated after filtering.")
            
        return self.control_stats
        
    def compare_outbreak_vs_control_statistics(self):
        """Compare statistics between outbreak and control pens."""
        self.logger.info("Comparing outbreak vs control pen statistics...")

        if not hasattr(self, 'pre_outbreak_stats') or self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot compare with controls.")
            return None

        if not hasattr(self, 'control_stats') or self.control_stats is None or self.control_stats.empty:
            self.logger.error("No control pen statistics available. Cannot compare with outbreaks.")
            return None

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
        outbreak_cols = {metric: metric for metric in metrics_to_compare}
        control_cols = {metric: metric for metric in metrics_to_compare}
        # Special case for value_at_removal
        if 'value_at_removal' in control_cols:
            control_cols['value_at_removal'] = 'value_at_reference'

        # Create a dictionary to store results
        comparison_dict = {}

        for metric in metrics_to_compare:
            outbreak_col = outbreak_cols[metric]
            control_col = control_cols[metric]

            # Skip if column doesn't exist in either dataset
            if outbreak_col not in self.pre_outbreak_stats.columns or control_col not in self.control_stats.columns:
                self.logger.warning(f"Skipping comparison for {metric}: columns not found.")
                continue

            # Get values for comparison
            outbreak_values = self.pre_outbreak_stats[outbreak_col].dropna()
            control_values = self.control_stats[control_col].dropna()

            if len(outbreak_values) < self.config.get('min_sample_values_comparison', 2) or len(control_values) < self.config.get('min_sample_values_comparison', 2):
                self.logger.warning(f"Insufficient data for {metric} comparison.")
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
                comparison_results['metrics'].append(metric)
                comparison_results['outbreak_stats'].append((outbreak_mean, outbreak_std))
                comparison_results['control_stats'].append((control_mean, control_std))
                comparison_results['p_values'].append(p_value)
                comparison_results['is_significant'].append(is_significant)
                comparison_results['effect_size'].append(effect_size)

                # Add to dictionary
                comparison_dict[metric] = {
                    'outbreak_mean': outbreak_mean,
                    'outbreak_std': outbreak_std,
                    'control_mean': control_mean,
                    'control_std': control_std,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'effect_size': effect_size
                }

                self.logger.info(f"Comparison for {metric}: Outbreak={outbreak_mean:.3f}±{outbreak_std:.3f}, " +
                        f"Control={control_mean:.3f}±{control_std:.3f}, p={p_value:.4f}, " +
                        f"Significant={is_significant}, Effect Size={effect_size:.2f}")

            except Exception as e:
                self.logger.error(f"Error comparing {metric}: {e}")

        # Save comparison results
        filename = self.config.get('comparison_stats_filename', 'outbreak_vs_control_comparison.csv')
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
            self.logger.info(f"Saved outbreak vs control comparison to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save comparison CSV: {e}")

        # Ensure self.comparison_results is set for the report
        self.comparison_results = comparison_dict
        return comparison_dict

    def analyze_individual_outbreak_variation(self):
        """Analyze individual variation in outbreak patterns."""
        self.logger.info("Analyzing individual variation in outbreak patterns...")

        if not hasattr(self, 'pre_outbreak_stats') or self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot analyze individual variation.")
            self.outbreak_patterns = pd.DataFrame() # Ensure empty
            self.pattern_stats = pd.DataFrame() # Ensure empty
            self.pen_consistency = {}
            return None

        outbreaks_df = self.pre_outbreak_stats.copy()

        # Define required columns for categorization AND aggregation
        categorization_cols = ['pen', 'value_at_removal', 'value_7d_before',
                               'abs_change_7d', '7d_window_slope', '3d_window_slope']
        aggregation_cols_tuples = [
            ('value_at_removal', ['mean', 'std', 'count']),
            ('value_7d_before', ['mean', 'std']),
            ('abs_change_7d', ['mean', 'std']),
            ('7d_window_slope', ['mean', 'std']),
            ('3d_window_slope', ['mean', 'std']),
        ]
        # Flatten list of required column names for checking existence
        required_cols_flat = categorization_cols + [col[0] for col in aggregation_cols_tuples]
        required_cols_flat = list(set(required_cols_flat)) # Unique columns

        missing_cols = [col for col in required_cols_flat if col not in outbreaks_df.columns]
        if missing_cols:
            # Log specific missing columns that are strictly required for categorization logic
            essential_missing = [c for c in ['abs_change_7d', '3d_window_slope', '7d_window_slope'] if c in missing_cols]
            if essential_missing:
                 self.logger.error(f"Missing essential columns for pattern categorization logic: {essential_missing}. Aborting pattern analysis.")
                 self.outbreak_patterns = pd.DataFrame() # Ensure empty
                 self.pattern_stats = pd.DataFrame() # Ensure empty
                 self.pen_consistency = {}
                 return None
            else:
                 self.logger.warning(f"Missing optional/aggregation columns for pattern analysis: {missing_cols}. Proceeding with categorization.")


        # Ensure that there is enough data
        min_outbreaks = self.config.get('min_outbreaks_for_variation_analysis', 3)
        if len(outbreaks_df) < min_outbreaks:
            self.logger.warning(f"Insufficient data for pattern analysis (only {len(outbreaks_df)} outbreaks < {min_outbreaks}). Skipping detailed stats.")

        # Calculate additional features for categorization (handle potential NaNs)
        if 'abs_change_7d' in outbreaks_df.columns:
            outbreaks_df['overall_change_rate'] = outbreaks_df['abs_change_7d'] / 7
        else: outbreaks_df['overall_change_rate'] = np.nan

        if '3d_window_slope' in outbreaks_df.columns and '7d_window_slope' in outbreaks_df.columns:
            # Ensure both slopes are numeric before calculating acceleration
            outbreaks_df['change_acceleration'] = np.where(
                pd.to_numeric(outbreaks_df['3d_window_slope'], errors='coerce').notna() & pd.to_numeric(outbreaks_df['7d_window_slope'], errors='coerce').notna(),
                (outbreaks_df['3d_window_slope'] - outbreaks_df['7d_window_slope']) / 4,
                np.nan
            )
        else: outbreaks_df['change_acceleration'] = np.nan


        # Define thresholds based on data distributions (handle potential NaNs)
        # Ensure columns exist before calculating quantiles
        slope_q = self.config.get('variation_slope_quantile', 0.33)
        accel_q = self.config.get('variation_accel_quantile', 0.33)

        # Calculate thresholds robustly, defaulting to -inf if data is missing/all NaN
        slope_3d_series = pd.to_numeric(outbreaks_df.get('3d_window_slope'), errors='coerce').dropna()
        slope_3d_threshold = slope_3d_series.quantile(slope_q) if not slope_3d_series.empty else -np.inf

        accel_series = pd.to_numeric(outbreaks_df.get('change_acceleration'), errors='coerce').dropna()
        acc_threshold = accel_series.quantile(accel_q) if not accel_series.empty else -np.inf


        def categorize_pattern(row):
            """
            Categorizes tail posture patterns into three distinct categories based on 
            statistical analysis of the pre-outbreak dataset.
            
            Categories:
            1. Stabil: Minimal/no decline or even improvement
            2. Gleichmäßige Abnahme: Steady, consistent decline 
            3. Steile Abnahme: Steep, potentially accelerating decline
            """
            # Check required fields are not NaN
            required_fields = ['abs_change_7d', '3d_window_slope', '7d_window_slope', 'value_at_removal']
            if any(pd.isna(row.get(field)) for field in required_fields):
                return "Undefiniert"
            
            # 1. Stable: Minimal change cases or improved posture
            if row['abs_change_7d'] > -0.3:  # Minimal decline threshold based on statistics
                return "Stabil"
            
            # Get 3-day and 7-day slopes for pattern determination
            slope_3d = row['3d_window_slope']
            slope_7d = row['7d_window_slope']
            
            # 3. Steep-decline: Steep decline criteria
            if slope_3d < -0.25:  # Steeper than 25th percentile of observed slopes
                return "Steile Abnahme"
                    
            # 2. Consistent-decline: Steady decline 
            return "Gleichmäßige Abnahme"

        outbreaks_df['pattern_category'] = outbreaks_df.apply(categorize_pattern, axis=1)
        self.outbreak_patterns = outbreaks_df # Store the df with categories

        # Count instances of each pattern (include Undefined if any)
        pattern_counts = outbreaks_df['pattern_category'].value_counts()
        self.logger.info(f"Pattern categorization counts: {pattern_counts.to_dict()}")
        if "Undefined" in pattern_counts:
             self.logger.warning(f"{pattern_counts['Undefined']} outbreaks could not be categorized due to missing data.")

        # Calculate Pen Consistency
        pens_with_multiple = 0
        pens_consistent = 0
        consistency_pct = 0
        try:
            pen_patterns = outbreaks_df[outbreaks_df['pattern_category'] != 'Undefined'].groupby('pen')['pattern_category'].agg(list) # Exclude undefined
            if not pen_patterns.empty:
                pens_with_multiple = sum(len(patterns) > 1 for patterns in pen_patterns)
                pens_consistent = sum(len(set(patterns)) == 1 for patterns in pen_patterns if len(patterns) > 1)
                consistency_pct = (pens_consistent / pens_with_multiple * 100) if pens_with_multiple > 0 else (100.0 if not pen_patterns.empty and pens_with_multiple == 0 else 0.0) # 100% if all pens had only 1 outbreak
                self.logger.info(f"Pen consistency (excluding 'Undefined'): {pens_consistent}/{pens_with_multiple} pens ({consistency_pct:.1f}%) have consistent pattern categories.")
            else:
                 self.logger.info("No pens with defined patterns found for consistency check.")

        except Exception as e:
            self.logger.warning(f"Error calculating pen consistency: {e}")

        self.pen_consistency = {
                'pens_with_multiple': pens_with_multiple,
                'pens_consistent': pens_consistent,
                'consistency_percentage': consistency_pct
            }

        # Calculate Aggregated Pattern Statistics
        pattern_stats = pd.DataFrame() # Initialize as empty
        try:
            # Filter out 'Undefined' category before aggregation if it exists
            valid_outbreaks_for_stats = outbreaks_df[outbreaks_df['pattern_category'] != "Undefined"]

            if not valid_outbreaks_for_stats.empty and len(valid_outbreaks_for_stats) >= min_outbreaks:
                 # Define aggregation dictionary based on available columns AND config
                 agg_dict = {}
                 for col_name, agg_funcs in aggregation_cols_tuples:
                     if col_name in valid_outbreaks_for_stats.columns:
                         # Ensure column is numeric before trying to aggregate
                         if pd.api.types.is_numeric_dtype(valid_outbreaks_for_stats[col_name]):
                            agg_dict[col_name] = agg_funcs
                         else:
                            self.logger.warning(f"Column '{col_name}' is not numeric, skipping aggregation for it.")


                 if agg_dict: # Proceed only if there's something valid to aggregate
                     # Use observed=True for category grouping if pandas version supports it
                     try:
                         pattern_stats = valid_outbreaks_for_stats.groupby('pattern_category', observed=True).agg(agg_dict)
                     except TypeError: # Older pandas might not support 'observed'
                         pattern_stats = valid_outbreaks_for_stats.groupby('pattern_category').agg(agg_dict)

                     # Flatten MultiIndex columns if created
                     pattern_stats.columns = ['_'.join(col).strip() for col in pattern_stats.columns.values]
                     pattern_stats = pattern_stats.reset_index() # Make category a column

                     self.logger.info("Successfully calculated pattern statistics.")
                 else:
                      self.logger.warning("No valid numeric columns found for pattern statistics aggregation.")
            elif not valid_outbreaks_for_stats.empty:
                 self.logger.warning(f"Not enough valid outbreaks ({len(valid_outbreaks_for_stats)}) to calculate detailed pattern statistics (min: {min_outbreaks}).")
            else:
                 self.logger.warning("No valid outbreaks (excluding 'Undefined') found for calculating pattern statistics.")

        except KeyError as e:
             self.logger.error(f"KeyError during pattern statistics aggregation: {e}. Check column names.", exc_info=True)
        except Exception as e:
             self.logger.error(f"Unexpected error calculating pattern statistics: {e}", exc_info=True)

        self.pattern_stats = pattern_stats # Store the aggregated stats DataFrame (might be empty)


        # Saving
        output_dir = self.config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed outbreak patterns (with category)
        filename_patterns = self.config.get('outbreak_patterns_filename', 'outbreak_patterns.csv')
        output_path = os.path.join(output_dir, filename_patterns)
        try:
            self.outbreak_patterns.to_csv(output_path, index=False)
            self.logger.info(f"Saved outbreak pattern analysis to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save outbreak patterns CSV: {e}")

        # Save aggregated pattern statistics (if not empty)
        filename_stats = self.config.get('pattern_stats_filename', 'pattern_stats.csv')
        stats_path = os.path.join(output_dir, filename_stats)
        if not self.pattern_stats.empty:
            try:
                self.pattern_stats.to_csv(stats_path, index=False)
                self.logger.info(f"Saved pattern statistics to {stats_path}")
            except Exception as e:
                 self.logger.error(f"Failed to save pattern statistics CSV: {e}")
        else:
            self.logger.warning("Pattern statistics DataFrame is empty, not saving statistics CSV.")

        # Return results as dictionary (using instance attributes)
        return {
            'outbreak_patterns': self.outbreak_patterns,
            'pattern_counts': pattern_counts.to_dict(),
            'pattern_stats': self.pattern_stats,
            'pen_consistency': self.pen_consistency
        }
        
        
    def analyze_posture_components(self):
        """Analyze the individual components (upright and hanging tails) separately."""
         
        self.logger.info("Analyzing individual posture components (upright vs. hanging tails)...")
        
        if self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot analyze components.")
            return None
        
        # Create a dictionary to store the results
        component_analysis = {}
        
        # Initialize the outbreak component tracking dataframe
        outbreak_components_data = []
        
        # Process control data for later comparison
        control_components_data = []
        
        # Load the JSON data once
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        
        # Process outbreak pens
        for processed_data in self.processed_results:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            pen_type, culprit_removal, _ = get_pen_info(camera, date_span, json_data)
            
            # Skip non-tail biting pens
            if pen_type != "tail biting" or not culprit_removal:
                continue
                
            camera_label = camera.replace("Kamera", "Pen ")
            interpolated_data = processed_data.get('interpolated_data')
            
            if interpolated_data is None or interpolated_data.empty:
                continue
                
            # Check if this pen/datespan is in analyzed outbreaks
            outbreak_match = self.pre_outbreak_stats[(self.pre_outbreak_stats['pen'] == camera_label) & 
                                                (self.pre_outbreak_stats['datespan'] == date_span)]
            
            if not outbreak_match.empty:
                # Get earliest removal date from pre_outbreak_stats
                outbreak_row = outbreak_match.iloc[0]
                removal_date_str = outbreak_row['culprit_removal_date']
                removal_date = pd.to_datetime(removal_date_str)
                
                # Extract component time series with sufficient days before
                if interpolated_data is not None and not interpolated_data.empty:
                    # Get data from window before removal
                    window_days = self.config.get('component_analysis_window_days', 10)
                    window_start = removal_date - timedelta(days=window_days)
                    
                    # Ensure datetime index is properly set
                    if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                        if 'datetime' in interpolated_data.columns:
                            interpolated_data = interpolated_data.set_index('datetime')
                        else:
                            self.logger.warning(f"Cannot set datetime index for {camera_label} / {date_span}")
                            continue
                    
                    # Filter data to window
                    window_data = interpolated_data[(interpolated_data.index >= window_start) & 
                                                (interpolated_data.index <= removal_date)]
                    
                    if not window_data.empty:
                        # Check if required columns exist
                        if all(col in window_data.columns for col in ['num_tails_upright', 'num_tails_hanging', 'posture_diff']):
                            # Extract daily component values for key days before removal
                            for days_before in range(window_days + 1):
                                target_date = removal_date - timedelta(days=days_before)
                                # Find nearest data point before or at the target date
                                nearest_data = window_data.loc[window_data.index <= target_date]
                                
                                if not nearest_data.empty:
                                    nearest_point = nearest_data.iloc[-1]
                                    
                                    upright_val = nearest_point['num_tails_upright']
                                    hanging_val = nearest_point['num_tails_hanging']
                                    diff_val = nearest_point['posture_diff']
                                    
                                    # Store the data point if components are not NaN
                                    if not (pd.isna(upright_val) or pd.isna(hanging_val)):
                                        outbreak_components_data.append({
                                            'pen': camera_label,
                                            'datespan': date_span,
                                            'days_before_removal': days_before,
                                            'date': target_date,
                                            'upright_tails': upright_val,
                                            'hanging_tails': hanging_val,
                                            'posture_diff': diff_val,
                                            'group': 'outbreak'
                                        })
        
        # Process control pens for comparison
        if hasattr(self, 'control_stats') and self.control_stats is not None and not self.control_stats.empty:
            for processed_data in self.processed_results:
                camera = processed_data['camera']
                date_span = processed_data['date_span']
                pen_type, _, _ = get_pen_info(camera, date_span, json_data)
                
                # Only process control pens
                if pen_type != "control":
                    continue
                    
                camera_label = camera.replace("Kamera", "Pen ")
                interpolated_data = processed_data.get('interpolated_data')
                
                if interpolated_data is None or interpolated_data.empty:
                    continue
                    
                # Check if this control pen is in our analyzed controls
                control_match = self.control_stats[(self.control_stats['pen'] == camera_label) & 
                                                (self.control_stats['datespan'] == date_span)]
                
                if not control_match.empty:
                    # For each reference point in this control pen
                    for _, control_row in control_match.iterrows():
                        reference_date_str = control_row['reference_date']
                        reference_date = pd.to_datetime(reference_date_str)
                        
                        # Extract component time series around reference date
                        if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                            if 'datetime' in interpolated_data.columns:
                                interpolated_data = interpolated_data.set_index('datetime')
                            else:
                                continue
                        
                        # Filter for same window as outbreaks
                        window_start = reference_date - timedelta(days=window_days)
                        window_data = interpolated_data[(interpolated_data.index >= window_start) & 
                                                    (interpolated_data.index <= reference_date)]
                        
                        if not window_data.empty:
                            # Same processing as outbreaks, but for control reference points
                            if all(col in window_data.columns for col in ['num_tails_upright', 'num_tails_hanging', 'posture_diff']):
                                for days_before in range(window_days + 1):
                                    target_date = reference_date - timedelta(days=days_before)
                                    nearest_data = window_data.loc[window_data.index <= target_date]
                                    
                                    if not nearest_data.empty:
                                        nearest_point = nearest_data.iloc[-1]
                                        
                                        upright_val = nearest_point['num_tails_upright']
                                        hanging_val = nearest_point['num_tails_hanging']
                                        diff_val = nearest_point['posture_diff']
                                        
                                        if not (pd.isna(upright_val) or pd.isna(hanging_val)):
                                            control_components_data.append({
                                                'pen': camera_label,
                                                'datespan': date_span,
                                                'days_before_removal': days_before,
                                                'date': target_date,
                                                'upright_tails': upright_val,
                                                'hanging_tails': hanging_val,
                                                'posture_diff': diff_val,
                                                'group': 'control',
                                                'sample_id': control_row.get('sample_id', '1')
                                            })
        
        # Create DataFrames from collected data
        if not outbreak_components_data:
            self.logger.warning("No component data could be extracted from the processed outbreaks.")
            return None
            
        outbreak_components_df = pd.DataFrame(outbreak_components_data)
        component_analysis['outbreak_components'] = outbreak_components_df
        
        if control_components_data:
            control_components_df = pd.DataFrame(control_components_data)
            component_analysis['control_components'] = control_components_df
            
            # Combine for easier comparison
            all_components_df = pd.concat([outbreak_components_df, control_components_df], ignore_index=True)
            component_analysis['all_components'] = all_components_df
        
        # Calculate statistics for key timepoints (outbreaks)
        timepoint_stats = {}
        report_days = self.config.get('component_timepoint_days', [0, 1, 3, 7])
        for days_before in report_days:
            day_data = outbreak_components_df[outbreak_components_df['days_before_removal'] == days_before]
            if not day_data.empty:
                timepoint_stats[f'day_minus_{days_before}'] = {
                    'upright_mean': day_data['upright_tails'].mean(),
                    'upright_std': day_data['upright_tails'].std(),
                    'hanging_mean': day_data['hanging_tails'].mean(),
                    'hanging_std': day_data['hanging_tails'].std(),
                    'diff_mean': day_data['posture_diff'].mean(),
                    'diff_std': day_data['posture_diff'].std(),
                    'count': len(day_data)
                }
        
        component_analysis['timepoint_stats'] = timepoint_stats
        
        # Calculate change metrics for each component
        change_stats = {}
        if all(f'day_minus_{days_before}' in timepoint_stats for days_before in [0, 3, 7]):
            # 7-day changes
            change_stats['upright_7d_change'] = timepoint_stats['day_minus_0']['upright_mean'] - timepoint_stats['day_minus_7']['upright_mean']
            change_stats['hanging_7d_change'] = timepoint_stats['day_minus_0']['hanging_mean'] - timepoint_stats['day_minus_7']['hanging_mean']
            change_stats['diff_7d_change'] = timepoint_stats['day_minus_0']['diff_mean'] - timepoint_stats['day_minus_7']['diff_mean']
            
            # 3-day changes
            change_stats['upright_3d_change'] = timepoint_stats['day_minus_0']['upright_mean'] - timepoint_stats['day_minus_3']['upright_mean']
            change_stats['hanging_3d_change'] = timepoint_stats['day_minus_0']['hanging_mean'] - timepoint_stats['day_minus_3']['hanging_mean']
            change_stats['diff_3d_change'] = timepoint_stats['day_minus_0']['diff_mean'] - timepoint_stats['day_minus_3']['diff_mean']
        
        component_analysis['change_stats'] = change_stats
        
        # Calculate percent contributions to the overall change
        contribution_stats = {}
        if 'change_stats' in component_analysis:
            change_stats = component_analysis['change_stats']
            diff_7d_change = change_stats.get('diff_7d_change', 0)
            diff_3d_change = change_stats.get('diff_3d_change', 0)
            
            # 7-day contributions
            if abs(diff_7d_change) > 0.001:  # Avoid division by near-zero
                upright_contrib_7d = change_stats['upright_7d_change'] / abs(diff_7d_change) * 100
                hanging_contrib_7d = -change_stats['hanging_7d_change'] / abs(diff_7d_change) * 100
                
                contribution_stats['upright_contribution_7d'] = upright_contrib_7d
                contribution_stats['hanging_contribution_7d'] = hanging_contrib_7d
                contribution_stats['primary_driver_7d'] = 'upright' if abs(upright_contrib_7d) > abs(hanging_contrib_7d) else 'hanging'
            
            # 3-day contributions
            if abs(diff_3d_change) > 0.001:
                upright_contrib_3d = change_stats['upright_3d_change'] / abs(diff_3d_change) * 100
                hanging_contrib_3d = -change_stats['hanging_3d_change'] / abs(diff_3d_change) * 100
                
                contribution_stats['upright_contribution_3d'] = upright_contrib_3d
                contribution_stats['hanging_contribution_3d'] = hanging_contrib_3d
                contribution_stats['primary_driver_3d'] = 'upright' if abs(upright_contrib_3d) > abs(hanging_contrib_3d) else 'hanging'
        
        component_analysis['contribution_stats'] = contribution_stats
        
        # Save the component data
        filename = self.config.get('posture_components_filename', 'posture_components.csv')
        output_path = os.path.join(self.config['output_dir'], filename)
        outbreak_components_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved posture component data to {output_path}")
        
        # Save control component data if available
        if 'control_components' in component_analysis:
            ctrl_filename = self.config.get('control_components_filename', 'control_posture_components.csv')
            control_output_path = os.path.join(self.config['output_dir'], ctrl_filename)
            component_analysis['control_components'].to_csv(control_output_path, index=False)
            self.logger.info(f"Saved control posture component data to {control_output_path}")
        
        return component_analysis