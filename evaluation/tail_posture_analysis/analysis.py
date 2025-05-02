import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import timedelta

from .processing import DataProcessor
from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info
from evaluation.tail_posture_analysis.threshold_monitoring import ThresholdMonitoringMixin


class TailPostureAnalyzer(ThresholdMonitoringMixin, DataProcessor):
    """Methods for analyzing tail posture data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the tracking structures for exclusions
        self.excluded_elements = {
            'consecutive_missing': [],
            'missing_percentage': [],
            'other_reasons': []
        }
        
    def _log_exclusion(self, exclusion_type, camera, date_span, pen_type, value, threshold, message, analysis_type="unknown"):
        """Log an exclusion with enhanced tracking of which analysis excluded it."""
        if not hasattr(self, 'excluded_elements'):
            self.excluded_elements = {
                'consecutive_missing': [],
                'missing_percentage': [],
                'other_reasons': []
            }
            
        if not hasattr(self, 'exclusion_by_analysis'):
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
        
        self.logger.info(message)
        
        # Track in the original structure
        if exclusion_type in self.excluded_elements:
            if exclusion_type in ['consecutive_missing', 'missing_percentage']:
                self.excluded_elements[exclusion_type].append((camera, date_span, pen_type, value, threshold))
            else:
                self.excluded_elements[exclusion_type].append((camera, date_span, pen_type, value))
        
        # Also track in the new analysis-specific structure
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
    
    def analyze_pre_outbreak_statistics(self):
        """Analyze pre-outbreak statistics using earliest removal date per event."""
        self.logger.info("Analyzing pre-outbreak statistics (using earliest removal date per event)...")
        json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        results = []
        self.excluded_events_count = 0
        self.excluded_events_missing_pct_count = 0
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            self.pre_outbreak_stats = pd.DataFrame()
            return self.pre_outbreak_stats
            
        max_missing_threshold = self.config.get('max_allowed_consecutive_missing_days', 3)
        max_missing_pct_threshold = self.config.get('max_allowed_missing_days_pct', 50.0)

        for processed_data in self.processed_results:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            quality_metrics = processed_data.get('quality_metrics', {})
            consecutive_missing = quality_metrics.get('max_consecutive_missing_resampled', 0)
            
            # Get pen type for tracking
            pen_type, _, _ = get_pen_info(camera, date_span, json_data)
            
            # Check for consecutive missing days threshold
            if consecutive_missing > max_missing_threshold:
                message = (
                    f"Excluding {camera}/{date_span} from analysis due to {consecutive_missing} consecutive missing periods "
                    f"in resampled data (threshold: {max_missing_threshold}). Raw missing days: {quality_metrics.get('missing_days_detected', 'unknown')}"
                )
                self.excluded_events_count += self._log_exclusion(
                    'consecutive_missing', camera, date_span, pen_type, consecutive_missing, 
                    max_missing_threshold, message, analysis_type="tail_biting_analysis"
                )
                continue
                
            # check for percentage of missing days
            missing_days = quality_metrics.get('missing_days_detected', 0)
            total_days = quality_metrics.get('total_expected_days', 0)
            missing_pct = (missing_days / total_days * 100) if total_days > 0 else 0
            
            if missing_pct > max_missing_pct_threshold:
                message = (
                    f"Excluding {camera}/{date_span} from analysis due to excessive missing days "
                    f"({missing_days}/{total_days} = {missing_pct:.1f}% > {max_missing_pct_threshold}%)"
                )
                self.excluded_events_missing_pct_count += self._log_exclusion(
                    'missing_percentage', camera, date_span, pen_type, missing_pct, 
                    max_missing_pct_threshold, message, analysis_type="tail_biting_analysis"
                )
                continue
            
            pen_type, culprit_removal, datespan_gt = get_pen_info(camera, date_span, json_data)
            if pen_type != "tail biting" or culprit_removal is None or culprit_removal == "Unknown" or culprit_removal == []:
                reason = "Not a tail biting event or missing culprit removal info"
                self._log_exclusion('other_reasons', camera, date_span, pen_type, reason, 
                                    None, f"Excluding {camera}/{date_span}: {reason}", 
                                    analysis_type="tail_biting_analysis")
                self.excluded_elements['other_reasons'].append((camera, date_span, pen_type, reason))
                continue
            
            camera_label = camera.replace("Kamera", "Pen ")
            self.logger.debug(f"Processing tail biting event: {camera_label} / {date_span}")
            interpolated_data = processed_data.get('interpolated_data')
            if interpolated_data is None or interpolated_data.empty:
                reason = "Empty interpolated data"
                self.excluded_elements['other_reasons'].append((camera, date_span, pen_type, reason))
                self.logger.warning(f"Skipping {camera_label} / {date_span} due to empty interpolated data.")
                continue
            if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                if 'datetime' in interpolated_data.columns: interpolated_data = interpolated_data.set_index('datetime')
                elif isinstance(interpolated_data.index, pd.RangeIndex) and 'datetime' in interpolated_data.index.name: interpolated_data.index = pd.to_datetime(interpolated_data.index)
                else: self.logger.error(f"Cannot set datetime index for {camera_label} / {date_span}"); continue

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
                self.excluded_elements['other_reasons'].append((camera, date_span, pen_type, reason))
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
        total_excluded = self.excluded_events_count + getattr(self, 'excluded_events_missing_pct_count', 0)
        num_other_reasons = num_total_biting - num_analyzed - total_excluded

        self.logger.info(f"Attempted to analyze {num_total_biting} potential tail biting events.")
        self.logger.info(f"Successfully analyzed {num_analyzed} events.")
        self.logger.info(f"Excluded {self.excluded_events_count} events due to >{max_missing_threshold} consecutive missing days.")
        self.logger.info(f"Excluded {self.excluded_events_missing_pct_count} events due to excessive missing days (>{max_missing_pct_threshold}%).")
        self.logger.info(f"Excluded {num_other_reasons} events for other reasons (e.g., no valid removal date, empty data).")

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
        self.excluded_controls_count = 0
        self.excluded_controls_missing_pct_count = 0
        
        if not self.processed_results:
            self.logger.error("No processed data available. Run preprocessing steps first.")
            self.control_stats = pd.DataFrame()
            return self.control_stats
            
        max_missing_threshold = self.config.get('max_allowed_consecutive_missing_days', 3)
        max_missing_pct_threshold = self.config.get('max_allowed_missing_days_pct', 50.0)

        # Filter for control pens
        control_processed_data = [p for p in self.processed_results if 
                                get_pen_info(p['camera'], p['date_span'], json_data)[0] == "control"]
        
        self.logger.info(f"Found {len(control_processed_data)} control pen datasets for analysis")
        
        for processed_data in control_processed_data:
            camera = processed_data['camera']
            date_span = processed_data['date_span']
            quality_metrics = processed_data.get('quality_metrics', {})
            consecutive_missing = quality_metrics.get('max_consecutive_missing_resampled', 0)
            pen_type = "control"
            
            if consecutive_missing > max_missing_threshold:
                message = (
                    f"Excluding control pen {camera}/{date_span} due to {consecutive_missing} consecutive missing periods "
                    f"in resampled data (threshold: {max_missing_threshold}). Raw missing days: {quality_metrics.get('missing_days_detected', 'unknown')}"
                )
                self.excluded_controls_count += self._log_exclusion(
                    'consecutive_missing', camera, date_span, pen_type, consecutive_missing, 
                    max_missing_threshold, message, analysis_type="control_analysis"
                )
                continue
                
            # check for percentage of missing days
            missing_days = quality_metrics.get('missing_days_detected', 0)
            total_days = quality_metrics.get('total_expected_days', 0)
            missing_pct = (missing_days / total_days * 100) if total_days > 0 else 0
            
            if missing_pct > max_missing_pct_threshold:
                message = (
                    f"Excluding control pen {camera}/{date_span} due to excessive missing days "
                    f"({missing_days}/{total_days} = {missing_pct:.1f}% > {max_missing_pct_threshold}%)"
                )
                self.excluded_controls_missing_pct_count += self._log_exclusion(
                    'missing_percentage', camera, date_span, pen_type, missing_pct, 
                    max_missing_pct_threshold, message, analysis_type="control_analysis"
                )
                continue
                
            camera_label = camera.replace("Kamera", "Pen ")
            interpolated_data = processed_data.get('interpolated_data')
            
            if interpolated_data is None or interpolated_data.empty:
                reason = "Empty interpolated data"
                self._log_exclusion('other_reasons', camera, date_span, pen_type, reason, 
                   None, f"Skipping control pen {camera_label} / {date_span}: {reason}", 
                   analysis_type="control_analysis")
                self.excluded_elements['other_reasons'].append((camera, date_span, pen_type, reason))
                self.logger.warning(f"Skipping control pen {camera_label} / {date_span} due to empty interpolated data.")
                continue
                
            if not isinstance(interpolated_data.index, pd.DatetimeIndex):
                if 'datetime' in interpolated_data.columns:
                    interpolated_data = interpolated_data.set_index('datetime')
                elif isinstance(interpolated_data.index, pd.RangeIndex) and 'datetime' in interpolated_data.index.name:
                    interpolated_data.index = pd.to_datetime(interpolated_data.index)
                else:
                    self.logger.error(f"Cannot set datetime index for control pen {camera_label} / {date_span}")
                    continue
            
            # Sample 3 random timepoints with sufficient data for analysis
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
            
        num_total_control = len(control_processed_data)
        num_analyzed = len(self.control_stats[['pen', 'datespan']].drop_duplicates()) if not self.control_stats.empty else 0
        num_reference_points = len(self.control_stats)
        
        self.logger.info(f"Analyzed {num_total_control} control pens.")
        self.logger.info(f"Successfully analyzed {num_analyzed} control pen datasets (from {len(self.control_stats['pen'].unique())} unique pens) with {num_reference_points} reference points.")
        self.logger.info(f"Excluded {self.excluded_controls_count} control pens due to >{max_missing_threshold} consecutive missing days.")
        self.logger.info(f"Excluded {self.excluded_controls_missing_pct_count} control pens due to excessive missing days (>{max_missing_pct_threshold}%).")
        
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

            if len(outbreak_values) < 2 or len(control_values) < 2:
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


        # Define pattern categories function (robust to NaNs)
        def categorize_pattern(row):
            # Check required fields are not NaN for categorization logic
            if pd.isna(row.get('abs_change_7d')) or pd.isna(row.get('3d_window_slope')) or pd.isna(row.get('7d_window_slope')) or pd.isna(row.get('change_acceleration')):
                return "Undefined" # Assign if key metrics are missing

            # Non-declining pattern
            if row['abs_change_7d'] >= 0: return "Non-declining"
            # Sudden decline: Steep recent slope AND accelerating decline
            if row['3d_window_slope'] < slope_3d_threshold and row['change_acceleration'] < acc_threshold: return "Sudden-decline"
            # Gradual consistent decline: 7d slope is negative AND 3d/7d slopes are similar
            gradual_diff = self.config.get('variation_gradual_slope_diff', 0.05)
            if row['7d_window_slope'] < 0 and abs(row['3d_window_slope'] - row['7d_window_slope']) < gradual_diff: return "Gradual-decline"
            # Erratic decline: 3d/7d slopes differ significantly
            erratic_diff = self.config.get('variation_erratic_slope_diff', 0.1)
            if abs(row['3d_window_slope'] - row['7d_window_slope']) > erratic_diff: return "Erratic-decline"
            # Default - moderate decline (if none of the above match and change is negative)
            return "Moderate-decline"

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
                                                'days_before_removal': days_before,  # Same name for consistency
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
    
    # def analyze_and_identify_monitoring_thresholds(self):
    #     """
    #     Perform complete analysis including monitoring threshold identification.
        
    #     This method runs the complete analysis pipeline:
    #     1. Analyze pre-outbreak statistics
    #     2. Analyze control pen statistics
    #     3. Compare outbreak vs. control
    #     4. Analyze individual outbreak variation
    #     5. Analyze posture components
    #     6. Identify monitoring thresholds and validate
        
    #     Returns:
    #         dict: Complete analysis results including monitoring thresholds
    #     """
    #     results = {}
        
    #     # 1. Analyze pre-outbreak statistics
    #     pre_outbreak_stats = self.analyze_pre_outbreak_statistics()
    #     results['pre_outbreak_stats'] = True if not pre_outbreak_stats.empty else False
        
    #     # 2. Analyze control pen statistics
    #     control_stats = self.analyze_control_pen_statistics()
    #     results['control_stats'] = True if not control_stats.empty else False
        
    #     # 3. Compare outbreak vs. control
    #     comparison = self.compare_outbreak_vs_control_statistics()
    #     results['comparison'] = True if comparison else False
        
    #     # 4. Analyze individual outbreak variation
    #     variation = self.analyze_individual_outbreak_variation()
    #     results['variation'] = True if variation else False
        
    #     # 5. Analyze posture components
    #     components = self.analyze_posture_components()
    #     results['components'] = True if components else False
        
    #     # Proceed with monitoring threshold analysis only if we have comparison data
    #     if comparison:
    #         # 6. Identify monitoring thresholds
    #         thresholds = self.identify_monitoring_thresholds()
    #         results['thresholds'] = True if thresholds else False
            
    #         # 7. Validate thresholds
    #         validation_method = self.config.get('threshold_validation_method', 'cv')
    #         validation = self.validate_monitoring_thresholds(validation_method=validation_method)
    #         results['validation'] = True if validation else False
            
    #         # 8. Identify optimal thresholds
    #         optimal = self.identify_optimal_monitoring_thresholds()
    #         results['optimal_thresholds'] = True if optimal else False
            
    #         # 9. Apply to example data
    #         monitoring = self.apply_monitoring_thresholds()
    #         results['monitoring'] = True if monitoring else False
    #     else:
    #         self.logger.warning("Skipping threshold monitoring due to lack of comparison data")
    #         results['thresholds'] = False
    #         results['validation'] = False
    #         results['optimal_thresholds'] = False
    #         results['monitoring'] = False
        
    #     return results
    
    # def analyze_monitoring_thresholds(self):
    #     """
    #     Translate statistical findings into potential monitoring thresholds
    #     and evaluate their practical effectiveness with robust validation.
        
    #     This method:
    #     1. Analyzes different metrics (posture difference, slope, components) 
    #     2. Determines potential threshold values at different sensitivity levels
    #     3. Calculates sensitivity/specificity for each threshold
    #     4. Estimates advance warning time
    #     5. Validates thresholds using cross-validation
    #     6. Evaluates on a holdout test set
    #     7. Recommends optimal thresholds for practical monitoring
        
    #     Parameters:
    #         cv_method (str): Validation method: 'lopocv' (leave-one-pen-out) or 'kfold'
    #         n_folds (int): Number of folds for k-fold cross-validation
    #         holdout_fraction (float): Fraction of data to reserve as final test set
    #         random_seed (int): Random seed for reproducibility
            
    #     Returns:
    #         dict: Monitoring threshold analysis results with validation metrics
    #     """
    #     cv_method = self.config.get('cv_method', 'lopocv')
    #     self.logger.info(f"Analyzing monitoring thresholds using {cv_method} validation...")
    #     n_folds = self.config.get('n_folds', 5)
    #     holdout_fraction = self.config.get('holdout_fraction', 0.2)
    #     random_seed = self.config.get('random_seed', 42)
        
    #     # Ensure the necessary data
    #     if self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
    #         self.logger.error("No pre-outbreak statistics available. Cannot analyze monitoring thresholds.")
    #         return None
                
    #     # Initialize results structure
    #     threshold_results = {
    #         'metrics': {},
    #         'thresholds': {},
    #         'performance': {},
    #         'warning_time': {},
    #         'recommendations': {},
    #         'validation': {
    #             'method': cv_method,
    #             'holdout_fraction': holdout_fraction,
    #             'cross_validation': {}
    #         },
    #         'threshold_sensitivity': {}
    #     }
        
    #     # Load necessary data
    #     json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        
    #     # Extract outbreak trajectories for threshold testing
    #     outbreak_trajectories = []
    #     for processed_data in self.processed_results:
    #         camera = processed_data['camera']
    #         date_span = processed_data['date_span']
    #         pen_type, culprit_removal, _ = get_pen_info(camera, date_span, json_data)
            
    #         # Skip non-tail biting pens
    #         if pen_type != "tail biting" or not culprit_removal:
    #             continue
                
    #         camera_label = camera.replace("Kamera", "Pen ")
    #         interpolated_data = processed_data.get('interpolated_data')
            
    #         if interpolated_data is None or interpolated_data.empty:
    #             continue
                
    #         # Check if this pen/datespan is in analyzed outbreaks
    #         outbreak_match = self.pre_outbreak_stats[(self.pre_outbreak_stats['pen'] == camera_label) & 
    #                                             (self.pre_outbreak_stats['datespan'] == date_span)]
            
    #         if not outbreak_match.empty:
    #             # Get earliest removal date from pre_outbreak_stats
    #             outbreak_row = outbreak_match.iloc[0]
    #             removal_date_str = outbreak_row['culprit_removal_date']
    #             removal_date = pd.to_datetime(removal_date_str)
                
    #             # Extract time series leading to the outbreak
    #             if interpolated_data is not None and not interpolated_data.empty:
    #                 # Ensure datetime index is properly set
    #                 if not isinstance(interpolated_data.index, pd.DatetimeIndex):
    #                     if 'datetime' in interpolated_data.columns:
    #                         interpolated_data = interpolated_data.set_index('datetime')
    #                     else:
    #                         continue
                    
    #                 # Get data for 14 days before removal (for extended analysis)
    #                 lookback_days = self.config.get('threshold_analysis_lookback_days', 14)
    #                 lookback_start = removal_date - timedelta(days=lookback_days)
    #                 trajectory_data = interpolated_data.loc[(interpolated_data.index >= lookback_start) & 
    #                                                     (interpolated_data.index <= removal_date)].copy()
                    
    #                 # Ensure enough data (at least 7 days)
    #                 min_len = self.config.get('min_trajectory_length_days', 7)
    #                 if not trajectory_data.empty and len(trajectory_data) >= min_len:
    #                     # Add columns for days-before-removal
    #                     trajectory_data['days_before_removal'] = (removal_date - trajectory_data.index).total_seconds() / (24 * 3600)
    #                     trajectory_data['pen'] = camera_label
    #                     trajectory_data['datespan'] = date_span
    #                     trajectory_data['removal_date'] = removal_date
                        
    #                     # Append to collection of trajectories
    #                     outbreak_trajectories.append(trajectory_data)
        
    #     # Extract control trajectories (for false positive testing)
    #     control_trajectories = []
    #     if hasattr(self, 'control_stats') and self.control_stats is not None and not self.control_stats.empty:
    #         for processed_data in self.processed_results:
    #             camera = processed_data['camera']
    #             date_span = processed_data['date_span']
    #             pen_type, _, _ = get_pen_info(camera, date_span, json_data)
                
    #             # Only process control pens
    #             if pen_type != "control":
    #                 continue
                    
    #             camera_label = camera.replace("Kamera", "Pen ")
    #             interpolated_data = processed_data.get('interpolated_data')
                
    #             if interpolated_data is None or interpolated_data.empty:
    #                 continue
                    
    #             # Check if this control pen is in our analyzed controls
    #             control_match = self.control_stats[(self.control_stats['pen'] == camera_label) & 
    #                                             (self.control_stats['datespan'] == date_span)]
                
    #             if not control_match.empty:
    #                 # For each reference point in this control pen
    #                 for _, control_row in control_match.iterrows():
    #                     reference_date_str = control_row['reference_date']
    #                     reference_date = pd.to_datetime(reference_date_str)
                        
    #                     # Ensure datetime index is properly set
    #                     if not isinstance(interpolated_data.index, pd.DatetimeIndex):
    #                         if 'datetime' in interpolated_data.columns:
    #                             interpolated_data = interpolated_data.set_index('datetime')
    #                         else:
    #                             continue
                        
    #                     # Get the same window length as outbreak data
    #                     lookback_days = 14
    #                     lookback_start = reference_date - timedelta(days=lookback_days)
    #                     control_data = interpolated_data.loc[(interpolated_data.index >= lookback_start) & 
    #                                                     (interpolated_data.index <= reference_date)].copy()
                        
    #                     # Ensure we have enough data
    #                     if not control_data.empty and len(control_data) >= 7:
    #                         # Add columns for days-before-reference (matching outbreak naming for consistency)
    #                         control_data['days_before_removal'] = (reference_date - control_data.index).total_seconds() / (24 * 3600)
    #                         control_data['pen'] = camera_label
    #                         control_data['datespan'] = date_span
    #                         control_data['removal_date'] = reference_date
    #                         control_data['sample_id'] = control_row.get('sample_id', '1')
                            
    #                         # Append to collection of control trajectories
    #                         control_trajectories.append(control_data)
        
    #     # Check if we have enough data to proceed
    #     if not outbreak_trajectories:
    #         self.logger.error("No valid outbreak trajectories for threshold analysis.")
    #         threshold_results['status'] = "No valid outbreak trajectories available."
    #         return threshold_results
        
    #     # Combine all trajectory data into DataFrames 
    #     outbreak_df_list = []
    #     for trajectory in outbreak_trajectories:
    #         outbreak_df_list.append(trajectory)
        
    #     if outbreak_df_list:
    #         outbreaks_df = pd.concat(outbreak_df_list)
    #         threshold_results['data_counts'] = {
    #             'outbreak_pens': len(set(outbreaks_df['pen'])),
    #             'outbreak_trajectories': len(outbreak_trajectories)
    #         }
    #     else:
    #         self.logger.warning("No valid outbreak data for threshold analysis.")
    #         outbreaks_df = pd.DataFrame()
    #         threshold_results['data_counts'] = {
    #             'outbreak_pens': 0,
    #             'outbreak_trajectories': 0
    #         }
        
    #     control_df_list = []
    #     for trajectory in control_trajectories:
    #         control_df_list.append(trajectory)
        
    #     if control_df_list:
    #         controls_df = pd.concat(control_df_list)
    #         threshold_results['data_counts']['control_pens'] = len(set(controls_df['pen']))
    #         threshold_results['data_counts']['control_trajectories'] = len(control_trajectories)
    #     else:
    #         self.logger.warning("No control data for false positive testing.")
    #         controls_df = pd.DataFrame()
    #         threshold_results['data_counts']['control_pens'] = 0
    #         threshold_results['data_counts']['control_trajectories'] = 0
        
    #     # Log the data counts
    #     self.logger.info(f"Threshold analysis data: {threshold_results['data_counts']['outbreak_trajectories']} outbreak trajectories from {threshold_results['data_counts']['outbreak_pens']} pens")
    #     self.logger.info(f"Threshold analysis data: {threshold_results['data_counts'].get('control_trajectories', 0)} control trajectories from {threshold_results['data_counts'].get('control_pens', 0)} pens")
        
    #     # ===== CROSS-VALIDATION SETUP =====
    #     # Create holdout test set
    #     np.random.seed(random_seed)
        
    #     # Get unique pens
    #     outbreak_pens = list(set([traj['pen'].iloc[0] for traj in outbreak_trajectories if 'pen' in traj.columns]))
    #     n_pens = len(outbreak_pens)
        
    #     # Determine holdout pens (20% of pens, minimum 1)
    #     n_holdout_pens = max(1, int(holdout_fraction * n_pens))
    #     holdout_pens = np.random.choice(outbreak_pens, size=n_holdout_pens, replace=False)
        
    #     # Split trajectories into holdout and training sets
    #     holdout_trajectories = [traj for traj in outbreak_trajectories 
    #                         if 'pen' in traj.columns and traj['pen'].iloc[0] in holdout_pens]
    #     train_trajectories = [traj for traj in outbreak_trajectories 
    #                     if 'pen' in traj.columns and traj['pen'].iloc[0] not in holdout_pens]
        
    #     # ENHANCEMENT 1: Create control holdout set
    #     control_pens = list(set([traj['pen'].iloc[0] for traj in control_trajectories if 'pen' in traj.columns]))
    #     n_control_pens = len(control_pens)
        
    #     # Determine holdout control pens (use same fraction as for outbreak pens)
    #     n_holdout_control_pens = max(1, int(holdout_fraction * n_control_pens))
    #     holdout_control_pens = np.random.choice(control_pens, size=n_holdout_control_pens, replace=False)
        
    #     # Split control trajectories into holdout and training sets
    #     holdout_control_trajectories = [traj for traj in control_trajectories 
    #                         if 'pen' in traj.columns and traj['pen'].iloc[0] in holdout_control_pens]
    #     training_control_trajectories = [traj for traj in control_trajectories 
    #                         if 'pen' in traj.columns and traj['pen'].iloc[0] not in holdout_control_pens]
        
    #     # Adjust strategy if too few pens for cross-validation
    #     min_cv_pens = self.config.get('min_pens_for_cv', 3)
    #     if len(set([traj['pen'].iloc[0] for traj in train_trajectories if 'pen' in traj.columns])) < min_cv_pens:
    #         self.logger.warning(f"Too few training pens ({len(set([traj['pen'].iloc[0] for traj in train_trajectories if 'pen' in traj.columns]))} < 3) for cross-validation. Switching to simple holdout validation.")
    #         cv_method = 'none'
    #         # Return holdout pens to training set
    #         train_trajectories = outbreak_trajectories
    #         holdout_trajectories = []
    #         training_control_trajectories = control_trajectories
    #         holdout_control_trajectories = []
        
    #     # Update validation info
    #     threshold_results['validation'].update({
    #         'n_holdout_pens': n_holdout_pens,
    #         'holdout_pens': holdout_pens.tolist() if isinstance(holdout_pens, np.ndarray) else holdout_pens,
    #         'n_train_pens': len(set([traj['pen'].iloc[0] for traj in train_trajectories if 'pen' in traj.columns])),
    #         'n_train_trajectories': len(train_trajectories),
    #         'n_holdout_trajectories': len(holdout_trajectories),
    #         'n_holdout_control_pens': n_holdout_control_pens,
    #         'holdout_control_pens': holdout_control_pens.tolist() if isinstance(holdout_control_pens, np.ndarray) else holdout_control_pens,
    #         'n_holdout_control_trajectories': len(holdout_control_trajectories)
    #     })
        
    #     # Save the raw trajectories for analysis
    #     threshold_results['raw_data'] = {
    #         'outbreak_trajectories': outbreak_trajectories,
    #         'control_trajectories': control_trajectories,
    #         'train_trajectories': train_trajectories,
    #         'holdout_trajectories': holdout_trajectories,
    #         'training_control_trajectories': training_control_trajectories,
    #         'holdout_control_trajectories': holdout_control_trajectories
    #     }
        
    #     # Define metrics to evaluate for thresholds
    #     metrics_to_evaluate = []
        
    #     # 1. Absolute posture difference
    #     if 'posture_diff' in outbreaks_df.columns:
    #         metrics_to_evaluate.append({
    #             'name': 'posture_diff', 
    #             'display_name': 'Posture Difference',
    #             'direction': 'below',  # threshold is crossed when metric goes below the value
    #             'units': ''
    #         })
        
    #     # 2. Rolling slope over different windows
    #     analysis_windows = self.config.get('analysis_window_days', [3, 7])
    #     for window in analysis_windows:
    #         if f'posture_diff_{window}d_slope' in outbreaks_df.columns:
    #             metrics_to_evaluate.append({
    #                 'name': f'posture_diff_{window}d_slope', 
    #                 'display_name': f'{window}d Slope',
    #                 'direction': 'below',  # negative slope indicates decreasing trend
    #                 'units': '/day'
    #             })
        
    #     # 3. Component values
    #     if 'num_tails_upright' in outbreaks_df.columns:
    #         metrics_to_evaluate.append({
    #             'name': 'num_tails_upright', 
    #             'display_name': 'Upright Tails',
    #             'direction': 'below',
    #             'units': ''
    #         })
        
    #     if 'num_tails_hanging' in outbreaks_df.columns:
    #         metrics_to_evaluate.append({
    #             'name': 'num_tails_hanging', 
    #             'display_name': 'Hanging Tails',
    #             'direction': 'above',
    #             'units': ''
    #         })
        
    #     # Save metrics to evaluate
    #     threshold_results['metrics']['evaluated'] = [m['name'] for m in metrics_to_evaluate]
    #     threshold_results['metrics_to_evaluate'] = metrics_to_evaluate  # Save full metric info
        
    #     # Create results storage for each metric
    #     for metric in metrics_to_evaluate:
    #         metric_name = metric['name']
    #         threshold_results['thresholds'][metric_name] = {}
    #         threshold_results['performance'][metric_name] = {}
    #         threshold_results['warning_time'][metric_name] = {}
    #         threshold_results['validation']['cross_validation'][metric_name] = {
    #             'folds': [],
    #             'thresholds': [],
    #             'sensitivities': [],
    #             'specificities': [],
    #             'warning_times': []
    #         }
    #         threshold_results['threshold_sensitivity'][metric_name] = {}  # Storage for sensitivity analysis
        
    #     if cv_method == 'lopocv':
    #         train_pens = list(set([traj['pen'].iloc[0] for traj in train_trajectories if 'pen' in traj.columns]))
    #         control_pens = list(set([traj['pen'].iloc[0] for traj in training_control_trajectories if 'pen' in traj.columns]))
    #         self.logger.info(f"Performing leave-one-pen-out cross-validation with {len(train_pens)} pens")
            
    #         for i, test_pen in enumerate(train_pens):
    #             # Split outbreak data for this fold
    #             cv_test_trajectories = [traj for traj in train_trajectories 
    #                                 if 'pen' in traj.columns and traj['pen'].iloc[0] == test_pen]
    #             cv_train_trajectories = [traj for traj in train_trajectories 
    #                                 if 'pen' in traj.columns and traj['pen'].iloc[0] != test_pen]
                
    #             # Split control data for this fold using the same approach
    #             # For control, keep roughly the same amount of control data in test vs train
    #             # Randomly assign control pens to test or train
    #             np.random.seed(random_seed + i)  # Different seed for each fold
    #             test_control_pens = np.random.choice(control_pens, 
    #                                             size=max(1, int(len(control_pens) * holdout_fraction)), 
    #                                             replace=False)
                
    #             # Split control trajectories
    #             cv_test_control_trajectories = [traj for traj in training_control_trajectories 
    #                                         if 'pen' in traj.columns and traj['pen'].iloc[0] in test_control_pens]
    #             cv_train_control_trajectories = [traj for traj in training_control_trajectories 
    #                                         if 'pen' in traj.columns and traj['pen'].iloc[0] not in test_control_pens]
                
    #             # Process this fold with separate control sets
    #             self._process_cv_fold(metrics_to_evaluate, cv_train_trajectories, cv_test_trajectories, 
    #                             cv_train_control_trajectories, cv_test_control_trajectories,
    #                             threshold_results, fold=i, pen=test_pen)
        
    #     elif cv_method == 'kfold':
    #         # K-fold cross-validation (group by pen)
    #         train_pens = list(set([traj['pen'].iloc[0] for traj in train_trajectories if 'pen' in traj.columns]))
    #         control_pens = list(set([traj['pen'].iloc[0] for traj in training_control_trajectories if 'pen' in traj.columns]))
    #         n_folds = min(n_folds, len(train_pens))  # Can't have more folds than pens
    #         self.logger.info(f"Performing {n_folds}-fold cross-validation with {len(train_pens)} pens")
            
    #         # Create folds based on pens (not trajectories)
    #         np.random.seed(random_seed)
    #         np.random.shuffle(train_pens)
    #         pen_folds = np.array_split(train_pens, n_folds)
            
    #         # Also create folds for control pens
    #         np.random.shuffle(control_pens)
    #         control_pen_folds = np.array_split(control_pens, n_folds) if control_pens else []
            
    #         for i, test_pens in enumerate(pen_folds):
    #             # Split outbreak data for this fold
    #             cv_test_trajectories = [traj for traj in train_trajectories 
    #                             if 'pen' in traj.columns and traj['pen'].iloc[0] in test_pens]
    #             cv_train_trajectories = [traj for traj in train_trajectories 
    #                             if 'pen' in traj.columns and traj['pen'].iloc[0] not in test_pens]
                
    #             # Split control data for this fold
    #             if i < len(control_pen_folds) and control_pen_folds:
    #                 test_control_pens = control_pen_folds[i]
    #                 cv_test_control_trajectories = [traj for traj in training_control_trajectories 
    #                                         if 'pen' in traj.columns and traj['pen'].iloc[0] in test_control_pens]
    #                 cv_train_control_trajectories = [traj for traj in training_control_trajectories 
    #                                         if 'pen' in traj.columns and traj['pen'].iloc[0] not in test_control_pens]
    #             else:
    #                 # Handle case where there are no control pens or not enough folds
    #                 cv_test_control_trajectories = []
    #                 cv_train_control_trajectories = training_control_trajectories
                
    #             # Process this fold with separate control sets
    #             self._process_cv_fold(metrics_to_evaluate, cv_train_trajectories, cv_test_trajectories, 
    #                             cv_train_control_trajectories, cv_test_control_trajectories,
    #                             threshold_results, fold=i, pen=f"{len(test_pens)} pens")
                
    #     # Calculate CV summary statistics for each metric
    #     self.logger.info("Summarizing cross-validation results...")
    #     cv_performance_metrics = {} # Store avg CV performance per metric
    #     for metric in metrics_to_evaluate:
    #         metric_name = metric['name']
    #         if metric_name in threshold_results['validation']['cross_validation']:
    #             cv_data = threshold_results['validation']['cross_validation'][metric_name]

    #             if cv_data['sensitivities']: # Check if CV ran successfully for this metric
    #                 # Calculate mean performance across folds (using TEST results)
    #                 mean_cv_sens = np.mean(cv_data['sensitivities'])
    #                 mean_cv_spec = np.mean(cv_data['specificities'])
    #                 mean_cv_warn = np.mean(cv_data['warning_times'])
    #                 mean_cv_thresh = np.mean(cv_data['thresholds'])
    #                 std_cv_thresh = np.std(cv_data['thresholds'])
    #                 n_folds_run = len(cv_data['folds'])

    #                 # Calculate CV Balanced Accuracy (average test performance)
    #                 cv_balanced_accuracy = (mean_cv_sens + mean_cv_spec) / 2
    #                 cv_performance_metrics[metric_name] = cv_balanced_accuracy # Store for final scoring

    #                 cv_summary = {
    #                     'mean_threshold_selected_in_folds': mean_cv_thresh,
    #                     'std_threshold_selected_in_folds': std_cv_thresh,
    #                     'mean_test_sensitivity': mean_cv_sens,
    #                     'mean_test_specificity': mean_cv_spec,
    #                     'mean_test_warning_time': mean_cv_warn,
    #                     'mean_cv_balanced_accuracy': cv_balanced_accuracy,
    #                     'n_folds': n_folds_run
    #                     # Add aggregate counts if needed:
    #                     # 'total_cv_tp': np.sum(cv_data.get('true_positives', [])),
    #                     # ... etc. ...
    #                 }
    #                 threshold_results['validation']['cross_validation'][metric_name]['summary'] = cv_summary

    #                 sens_str = f"{mean_cv_sens:.2f}" if mean_cv_sens is not None else "N/A"
    #                 spec_str = f"{mean_cv_spec:.2f}" if mean_cv_spec is not None else "N/A"
    #                 warn_str = f"{mean_cv_warn:.1f}d" if mean_cv_warn is not None else "N/A"
    #                 bal_acc_str = f"{cv_balanced_accuracy:.2f}" if cv_balanced_accuracy is not None else "N/A"
    #                 thresh_str = f"{mean_cv_thresh:.3f}" if mean_cv_thresh is not None else "N/A"
    #                 std_str = f"{std_cv_thresh:.3f}" if std_cv_thresh is not None else "N/A"

    #                 self.logger.info(f"  CV Summary for {metric_name}: "
    #                             f"Avg Test Sens: {sens_str}, "
    #                             f"Avg Test Spec: {spec_str}, "
    #                             f"Avg Test Warn: {warn_str}, "
    #                             f"Avg Bal Acc: {bal_acc_str}, "
    #                             f"(Avg Fold Thresh: {thresh_str} ± {std_str})")
    #             else:
    #                  self.logger.warning(f"  No CV results to summarize for {metric_name}.")
    #                  cv_performance_metrics[metric_name] = 0.0 # Assign 0 if CV failed

    #     # ===== FINAL THRESHOLD CALCULATION (using ALL training data) =====
    #     self.logger.info("Calculating final thresholds using the entire training set...")
    #     # Ensure train_trajectories is not empty before concatenating
    #     if not train_trajectories:
    #          self.logger.error("No training trajectories available for final threshold calculation.")
    #          return threshold_results # Or handle appropriately

    #     # Combine ALL training trajectories for final threshold selection
    #     train_df_full = pd.concat([traj for traj in train_trajectories if isinstance(traj, pd.DataFrame) and not traj.empty])
    #     if train_df_full.empty:
    #          self.logger.error("Concatenated training trajectories resulted in an empty DataFrame.")
    #          return threshold_results

    #     # Also combine ALL training control trajectories
    #     training_controls_full_df = None
    #     if training_control_trajectories:
    #         training_controls_full_df = pd.concat([traj for traj in training_control_trajectories if isinstance(traj, pd.DataFrame) and not traj.empty])
    #         if training_controls_full_df.empty:
    #             training_controls_full_df = None # Treat as no control data if concatenation is empty
    #             self.logger.warning("Concatenated training control trajectories resulted in an empty DataFrame.")

    #     # Analyze each metric for final threshold selection
    #     for metric in metrics_to_evaluate:
    #         metric_name = metric['name']
    #         display_name = metric['display_name']
    #         direction = metric['direction']
    #         self.logger.info(f"  Analyzing final thresholds for metric: {display_name}")

    #         # Check if metric exists in full training data
    #         if metric_name not in train_df_full.columns:
    #             self.logger.warning(f"  Metric {metric_name} not found in full training data, skipping.")
    #             continue

    #         # 1. Determine candidate threshold values from full training set
    #         # Use values near removal from the *entire* training set
    #         removal_values_full = train_df_full[train_df_full['days_before_removal'] <= 0][metric_name].dropna()

    #         if removal_values_full.empty:
    #             self.logger.warning(f"  No non-NaN values at removal for metric {metric_name} in full training data.")
    #             continue

    #         # Calculate candidate thresholds using configured percentiles
    #         percentiles = self.config.get('threshold_percentiles', [5, 10, 25, 50, 75, 90, 95])
    #         # Adjust percentile based on direction for calculating quantiles
    #         candidate_threshold_values = {}
    #         for p in percentiles:
    #              quantile_to_use = p / 100.0
    #              if direction == 'above':
    #                  quantile_to_use = 1.0 - quantile_to_use
    #              try:
    #                   candidate_threshold_values[p] = removal_values_full.quantile(quantile_to_use)
    #              except Exception as e:
    #                   self.logger.warning(f"    Error calculating {p}th percentile for {metric_name}: {e}")


    #         if not candidate_threshold_values:
    #             self.logger.warning(f"  No candidate thresholds generated for {metric_name}.")
    #             continue

    #         # Store candidate info
    #         threshold_results['thresholds'][metric_name]['candidate_percentiles'] = list(candidate_threshold_values.keys())
    #         threshold_results['thresholds'][metric_name]['candidate_values'] = candidate_threshold_values

    #         # 2. Evaluate EACH candidate threshold on the FULL training set
    #         candidate_performance = {}
    #         candidate_warning_times = {}
    #         candidate_scores = {}

    #         # Get the pre-calculated average CV performance for this metric
    #         # Use 0 if CV failed or metric wasn't evaluated
    #         cv_perf_score = cv_performance_metrics.get(metric_name, 0.0)

    #         for percentile, threshold_value in candidate_threshold_values.items():
    #             # Evaluate this candidate threshold on ALL training outbreaks
    #             train_tp, train_fn = 0, 0
    #             current_warning_times = []
    #             for trajectory in train_trajectories: # Iterate through original list of DFs
    #                 if metric_name not in trajectory.columns: continue
    #                 first_crossing_time = np.nan
    #                 traj_sorted = trajectory.sort_values(by='days_before_removal', ascending=False)
    #                 for _, row in traj_sorted.iterrows():
    #                     value = row[metric_name]
    #                     if pd.isna(value): continue
    #                     threshold_is_crossed = (direction == 'below' and value <= threshold_value) or \
    #                                            (direction == 'above' and value >= threshold_value)
    #                     if threshold_is_crossed and pd.isna(first_crossing_time):
    #                         first_crossing_time = row['days_before_removal']

    #                 if not pd.isna(first_crossing_time):
    #                     train_tp += 1
    #                     current_warning_times.append(first_crossing_time)
    #                 else:
    #                     train_fn += 1

    #             # Evaluate this candidate threshold on ALL training controls
    #             train_fp, train_tn = 0, 0
    #             if training_controls_full_df is not None:
    #                  # Use unique identifier if multiple samples per pen exist
    #                  unique_control_ids = training_controls_full_df[['pen', 'datespan', 'sample_id']].drop_duplicates().values.tolist()
    #                  for pen_id, span_id, samp_id in unique_control_ids:
    #                       trajectory = training_controls_full_df[
    #                            (training_controls_full_df['pen'] == pen_id) &
    #                            (training_controls_full_df['datespan'] == span_id) &
    #                            (training_controls_full_df['sample_id'] == samp_id)
    #                       ]
    #                       if metric_name not in trajectory.columns: continue

    #                       threshold_crossed = False
    #                       for _, row in trajectory.iterrows():
    #                           value = row[metric_name]
    #                           if pd.isna(value): continue
    #                           threshold_is_crossed = (direction == 'below' and value <= threshold_value) or \
    #                                                 (direction == 'above' and value >= threshold_value)
    #                           if threshold_is_crossed:
    #                               threshold_crossed = True
    #                               break
    #                       if threshold_crossed: train_fp += 1
    #                       else: train_tn += 1
    #             else:
    #                 # If no control data, specificity cannot be calculated
    #                 train_fp, train_tn = 0, 0 # Ensure these are zero


    #             # Calculate performance metrics ON THE FULL TRAINING SET
    #             train_sensitivity = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
    #             train_specificity = train_tn / (train_tn + train_fp) if (train_tn + train_fp) > 0 else 0 # Handle no controls
    #             train_mean_warning = np.mean(current_warning_times) if current_warning_times else 0

    #             candidate_performance[percentile] = {
    #                 'sensitivity': train_sensitivity, 'specificity': train_specificity,
    #                 'true_positives': train_tp, 'false_negatives': train_fn,
    #                 'true_negatives': train_tn, 'false_positives': train_fp
    #             }
    #             candidate_warning_times[percentile] = {
    #                  'mean': train_mean_warning, 'median': np.median(current_warning_times) if current_warning_times else 0,
    #                  'std': np.std(current_warning_times) if current_warning_times else 0,
    #                  'count': len(current_warning_times), 'raw_times': current_warning_times
    #             }

    #             # Calculate warning score for this candidate
    #             target_d = self.config.get('warning_score_target_days', 7.0)
    #             max_d = self.config.get('warning_score_max_days', 14.0)
    #             warning_score = 0
    #             if train_mean_warning > 0:
    #                 if train_mean_warning <= target_d:
    #                     warning_score = train_mean_warning / target_d if target_d > 0 else 0
    #                 else:
    #                      # Penalize very early warnings gradually
    #                      warning_score = max(0, 1.0 - (train_mean_warning - target_d) / (max_d - target_d)) if max_d > target_d else 0

    #             warning_score = max(0, min(1, warning_score)) # Clamp to 0-1

    #             # Calculate the combined score using configured weights
    #             # Uses performance on full training set + average CV performance
    #             weights = self.config.get('threshold_score_weights', {})
    #             combined_score = (
    #                 weights.get('train_sensitivity', 0) * train_sensitivity +
    #                 weights.get('train_specificity', 0) * train_specificity +
    #                 weights.get('train_warning_score', 0) * warning_score +
    #                 weights.get('cv_balanced_accuracy', 0) * cv_perf_score # Use the stored avg CV perf
    #             )
    #             candidate_scores[percentile] = combined_score

    #         # 3. Select the best FINAL threshold based on the combined score
    #         if candidate_scores:
    #             best_percentile = max(candidate_scores, key=candidate_scores.get)
    #             best_threshold = candidate_threshold_values[best_percentile]
    #             best_performance_on_train = candidate_performance[best_percentile]
    #             best_warning_on_train = candidate_warning_times[best_percentile]
    #             best_combined_score = candidate_scores[best_percentile]

    #             # --- Perform threshold sensitivity analysis on the CHOSEN best_threshold ---
    #             threshold_sensitivity_results = self._run_threshold_sensitivity_analysis(
    #                                                 metric_name, direction, best_threshold,
    #                                                 train_trajectories, training_control_trajectories)

    #             # Store final recommendation
    #             threshold_results['recommendations'][metric_name] = {
    #                 'best_percentile': best_percentile,
    #                 'threshold_value': best_threshold,
    #                 'performance_on_training': best_performance_on_train,
    #                 'warning_time_on_training': best_warning_on_train,
    #                 'combined_score': best_combined_score,
    #                 'cv_balanced_accuracy_used_in_score': cv_perf_score,
    #                 'candidate_scores': candidate_scores, # Store scores for all candidates
    #                 'threshold_sensitivity': threshold_sensitivity_results
    #             }

    #             # Log the chosen threshold and its performance on training data + CV estimate
    #             self.logger.info(f"  --> Optimal final threshold for {metric_name}: {best_threshold:.3f} ({best_percentile}th percentile)")
    #             self.logger.info(f"      Performance on Training Set: Sens={best_performance_on_train['sensitivity']:.2f}, Spec={best_performance_on_train['specificity']:.2f}, Warn={best_warning_on_train['mean']:.1f}d")
    #             self.logger.info(f"      Avg CV Balanced Accuracy: {cv_perf_score:.2f}")
    #             self.logger.info(f"      Stability (Sens/Spec/Overall): {threshold_sensitivity_results.get('sensitivity_stability', 0):.2f} / {threshold_sensitivity_results.get('specificity_stability', 0):.2f} / {threshold_sensitivity_results.get('overall_stability', 0):.2f}")

    #         else:
    #             self.logger.warning(f"  No candidates scored for {metric_name}.")


    #     # ===== HOLDOUT SET EVALUATION =====
    #     self.logger.info("Evaluating final recommended thresholds on holdout set...")
    #     if holdout_trajectories or holdout_control_trajectories:
    #          holdout_results = {}
    #          for metric in metrics_to_evaluate:
    #               metric_name = metric['name']
    #               if metric_name not in threshold_results['recommendations']:
    #                    continue

    #               rec = threshold_results['recommendations'][metric_name]
    #               threshold_value = rec['threshold_value']
    #               direction = metric['direction']
    #               threshold_sensitivity_info = rec.get('threshold_sensitivity', {})

    #               # Evaluate on holdout outbreak data
    #               holdout_tp, holdout_fn, holdout_warning_times = 0, 0, []
    #               if holdout_trajectories:
    #                    for trajectory in holdout_trajectories:
    #                         # ... (same logic as training evaluation loop) ...
    #                         if metric_name not in trajectory.columns: continue
    #                         first_crossing_time = np.nan
    #                         traj_sorted = trajectory.sort_values(by='days_before_removal', ascending=False)
    #                         for _, row in traj_sorted.iterrows():
    #                              value = row[metric_name]
    #                              if pd.isna(value): continue
    #                              threshold_is_crossed = (direction == 'below' and value <= threshold_value) or \
    #                                                   (direction == 'above' and value >= threshold_value)
    #                              if threshold_is_crossed and pd.isna(first_crossing_time):
    #                                   first_crossing_time = row['days_before_removal']

    #                         if not pd.isna(first_crossing_time):
    #                              holdout_tp += 1
    #                              holdout_warning_times.append(first_crossing_time)
    #                         else:
    #                              holdout_fn += 1

    #               # Evaluate on holdout control data
    #               holdout_fp, holdout_tn = 0, 0
    #               if holdout_control_trajectories:
    #                    for trajectory in holdout_control_trajectories:
    #                        # ... (same logic as training evaluation loop) ...
    #                         if metric_name not in trajectory.columns: continue
    #                         threshold_crossed = False
    #                         for _, row in trajectory.iterrows():
    #                             value = row[metric_name]
    #                             if pd.isna(value): continue
    #                             threshold_is_crossed = (direction == 'below' and value <= threshold_value) or \
    #                                                   (direction == 'above' and value >= threshold_value)
    #                             if threshold_is_crossed:
    #                                 threshold_crossed = True
    #                                 break
    #                         if threshold_crossed: holdout_fp += 1
    #                         else: holdout_tn += 1

    #               # Calculate holdout metrics
    #               holdout_sens = holdout_tp / (holdout_tp + holdout_fn) if (holdout_tp + holdout_fn) > 0 else 0
    #               holdout_spec = holdout_tn / (holdout_tn + holdout_fp) if (holdout_tn + holdout_fp) > 0 else 0
    #               holdout_warn = np.mean(holdout_warning_times) if holdout_warning_times else 0
    #               holdout_bal_acc = (holdout_sens + holdout_spec) / 2

    #               holdout_results[metric_name] = {
    #                   'sensitivity': holdout_sens, 'specificity': holdout_spec, 'balanced_accuracy': holdout_bal_acc,
    #                   'mean_warning_time': holdout_warn,
    #                   'n_outbreak_trajectories': len(holdout_trajectories) if holdout_trajectories else 0,
    #                   'n_control_trajectories': len(holdout_control_trajectories) if holdout_control_trajectories else 0,
    #                   'true_positives': holdout_tp, 'false_negatives': holdout_fn,
    #                   'true_negatives': holdout_tn, 'false_positives': holdout_fp,
    #                   'raw_warning_times': holdout_warning_times,
    #                   'threshold_stability_overall': threshold_sensitivity_info.get('overall_stability', 0) # From training analysis
    #               }

    #               self.logger.info(f"  Holdout evaluation for {metric_name}: "
    #                             f"Sens: {holdout_sens:.2f}, Spec: {holdout_spec:.2f}, "
    #                             f"Warn: {holdout_warn:.1f}d (Bal Acc: {holdout_bal_acc:.2f}) "
    #                             f"(n={holdout_results[metric_name]['n_outbreak_trajectories']} outbreaks, {holdout_results[metric_name]['n_control_trajectories']} controls)")

    #          threshold_results['validation']['holdout_evaluation'] = holdout_results # Changed key slightly
    #     else:
    #         self.logger.warning("No holdout data available for final evaluation.")


    #     # ===== DETERMINE BEST OVERALL METRIC =====
    #     # ... (Logic uses 'combined_score' and 'overall_stability' from recommendations - should be okay) ...
    #     self.logger.info("Determining best overall metric for monitoring...")
    #     metric_scores = {}
    #     for metric, results in threshold_results.get('recommendations', {}).items():
    #          combined_score = results.get('combined_score', 0)
    #          stability_score = results.get('threshold_sensitivity', {}).get('overall_stability', 0)
    #          overall_weights = self.config.get('overall_metric_score_weights', {'combined': 0.8, 'stability': 0.2})
    #          metric_scores[metric] = (overall_weights.get('combined', 0.8) * combined_score +
    #                                 overall_weights.get('stability', 0.2) * stability_score)

    #     if metric_scores:
    #          best_metric = max(metric_scores, key=metric_scores.get)
    #          threshold_results['overall_best_metric'] = best_metric
    #          threshold_results['overall_best_score'] = metric_scores[best_metric]
    #          best_metric_info = next((m for m in metrics_to_evaluate if m['name'] == best_metric), None)
    #          best_metric_display = best_metric_info['display_name'] if best_metric_info else best_metric
    #          threshold_results['overall_best_metric_display'] = best_metric_display
    #          self.logger.info(f"--> Best overall metric identified: {best_metric_display} (Score: {metric_scores[best_metric]:.3f})")
    #     else:
    #          self.logger.warning("Could not determine best overall metric.")
        
    #     # ===== SAVE RESULTS =====
    #     # Save results to file
    #     filename = self.config.get('threshold_analysis_filename', 'monitoring_threshold_analysis.json')
    #     output_path = os.path.join(self.config['output_dir'], filename)
    #     # Need to handle non-serializable objects like DataFrames
    #     try:
    #         serializable_results = self._prepare_for_json(threshold_results)
    #         with open(output_path, 'w') as f:
    #             json.dump(serializable_results, f, indent=2)
    #         self.logger.info(f"Saved monitoring threshold analysis to {output_path}")
    #     except Exception as e:
    #         self.logger.error(f"Failed to save threshold analysis: {e}")
        
    #     # Return the full results
    #     return threshold_results

    # def _process_cv_fold(self, metrics_to_evaluate, train_trajectories, test_trajectories,
    #                 control_trajectories, test_control_trajectories, threshold_results, fold, pen=None):
    #     """
    #     Process one cross-validation fold.

    #     Selects a representative threshold based on the fold's training data using a
    #     predefined percentile, then evaluates its performance on the fold's test data.
    #     """
    #     self.logger.debug(f"Processing CV Fold {fold} (Test Pen/Group: {pen})")

    #     # Combine fold's training trajectories into a DataFrame
    #     if not train_trajectories:
    #         self.logger.warning(f"CV Fold {fold}: No training trajectories provided. Skipping.")
    #         return
    #     train_df = pd.concat([traj for traj in train_trajectories if isinstance(traj, pd.DataFrame) and not traj.empty])
    #     if train_df.empty:
    #         self.logger.warning(f"CV Fold {fold}: Training trajectories resulted in empty DataFrame. Skipping.")
    #         return

    #     # Get the percentile to use for threshold selection within the fold
    #     cv_select_percentile = self.config.get('cv_fold_threshold_percentile', 10) # Default to 10th percentile

    #     # For each metric
    #     for metric in metrics_to_evaluate:
    #         metric_name = metric['name']
    #         direction = metric['direction']
    #         self.logger.debug(f"  Metric: {metric_name}")

    #         # Skip if metric not in fold's training data
    #         if metric_name not in train_df.columns:
    #             self.logger.debug(f"    Metric {metric_name} not in fold's training data. Skipping.")
    #             continue

    #         # --- 1. Select Threshold based on Fold's Training Data ---
    #         # Get values near removal from the fold's training outbreak data
    #         removal_values = train_df[train_df['days_before_removal'] <= 0][metric_name].dropna()
    #         if removal_values.empty:
    #             self.logger.debug(f"    No removal values for {metric_name} in fold's training data. Skipping.")
    #             continue

    #         # Determine the threshold for this fold using the specified percentile
    #         try:
    #             # Adjust percentile based on direction (lower for 'below', higher for 'above')
    #             quantile_to_use = cv_select_percentile / 100.0
    #             if direction == 'above': # e.g., for hanging tails
    #                 quantile_to_use = 1.0 - quantile_to_use

    #             fold_threshold = removal_values.quantile(quantile_to_use)
    #             self.logger.debug(f"    Selected fold threshold ({cv_select_percentile}th percentile): {fold_threshold:.3f}")

    #         except Exception as e:
    #             self.logger.warning(f"    Error calculating quantile for {metric_name} in fold {fold}: {e}")
    #             continue

    #         # --- 2. Evaluate the selected threshold on Fold's Test Data ---
    #         test_tp, test_fn, test_warning_times = 0, 0, []
    #         test_fp, test_tn = 0, 0

    #         # Evaluate on test outbreak trajectories
    #         if test_trajectories:
    #             for trajectory in test_trajectories:
    #                 if metric_name not in trajectory.columns:
    #                     continue

    #                 threshold_crossed = False
    #                 first_crossing_time = np.nan # Use NaN for not crossed

    #                 traj_sorted = trajectory.sort_values(by='days_before_removal', ascending=False)
    #                 for _, row in traj_sorted.iterrows():
    #                     value = row[metric_name]
    #                     days_before = row['days_before_removal']

    #                     if pd.isna(value):
    #                         continue

    #                     threshold_is_crossed = False
    #                     if direction == 'below' and value <= fold_threshold:
    #                         threshold_is_crossed = True
    #                     elif direction == 'above' and value >= fold_threshold:
    #                         threshold_is_crossed = True

    #                     # Record the *first* time it crosses
    #                     if threshold_is_crossed and pd.isna(first_crossing_time):
    #                          first_crossing_time = days_before

    #                 # Determine outcome for this trajectory
    #                 if not pd.isna(first_crossing_time):
    #                     test_tp += 1
    #                     test_warning_times.append(first_crossing_time)
    #                 else:
    #                     test_fn += 1
    #         else:
    #              self.logger.debug(f"    No test outbreak trajectories for fold {fold}.")


    #         # Evaluate on test control trajectories (for specificity)
    #         if test_control_trajectories:
    #             for trajectory in test_control_trajectories:
    #                 if metric_name not in trajectory.columns:
    #                     continue

    #                 threshold_crossed = False
    #                 # Check if the threshold is *ever* crossed in the control trajectory
    #                 for _, row in trajectory.iterrows():
    #                     value = row[metric_name]
    #                     if pd.isna(value):
    #                         continue

    #                     threshold_is_crossed = False
    #                     if direction == 'below' and value <= fold_threshold:
    #                         threshold_is_crossed = True
    #                     elif direction == 'above' and value >= fold_threshold:
    #                         threshold_is_crossed = True

    #                     if threshold_is_crossed:
    #                         threshold_crossed = True
    #                         break # Only need one crossing to count as FP

    #                 if threshold_crossed:
    #                     test_fp += 1
    #                 else:
    #                     test_tn += 1
    #         else:
    #             self.logger.debug(f"    No test control trajectories for fold {fold}.")

    #         # Calculate final performance metrics for this fold's *test* data
    #         test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    #         test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0
    #         test_mean_warning = np.mean(test_warning_times) if test_warning_times else 0 # Use 0 if no TPs

    #         # Store fold results (test performance and the threshold used)
    #         cv_results_storage = threshold_results['validation']['cross_validation'][metric_name]
    #         cv_results_storage['folds'].append(fold)
    #         cv_results_storage['thresholds'].append(fold_threshold) # Store the threshold chosen for this fold
    #         cv_results_storage['sensitivities'].append(test_sensitivity)
    #         cv_results_storage['specificities'].append(test_specificity)
    #         cv_results_storage['warning_times'].append(test_mean_warning)
    #         # Add raw counts for potential later analysis if needed
    #         cv_results_storage.setdefault('true_positives', []).append(test_tp)
    #         cv_results_storage.setdefault('false_negatives', []).append(test_fn)
    #         cv_results_storage.setdefault('true_negatives', []).append(test_tn)
    #         cv_results_storage.setdefault('false_positives', []).append(test_fp)


    #         self.logger.info(f"  CV Fold {fold} ({pen if pen else 'multiple pens'}): "
    #                     f"{metric_name} - Fold Threshold: {fold_threshold:.3f}, "
    #                     f"Test Sens: {test_sensitivity:.2f}, Test Spec: {test_specificity:.2f}, "
    #                     f"Test Warn: {test_mean_warning:.1f}d "
    #                     f"(TP:{test_tp}, FN:{test_fn}, TN:{test_tn}, FP:{test_fp})")
    
    # def _run_threshold_sensitivity_analysis(self, metric_name, direction, best_threshold,
    #                                     train_trajectories, training_control_trajectories):
    #     """ Helper function to perform threshold sensitivity analysis. """
    #     sensitivity_results = {}
    #     base_tp, base_fn, base_fp, base_tn = 0, 0, 0, 0 # Recalculate base counts for clarity

    #     # Evaluate base threshold on training outbreaks
    #     for trajectory in train_trajectories:
    #          if metric_name not in trajectory.columns: continue
    #          first_crossing_time = np.nan
    #          traj_sorted = trajectory.sort_values(by='days_before_removal', ascending=False)
    #          for _, row in traj_sorted.iterrows():
    #               value = row[metric_name]
    #               if pd.isna(value): continue
    #               threshold_is_crossed = (direction == 'below' and value <= best_threshold) or \
    #                                     (direction == 'above' and value >= best_threshold)
    #               if threshold_is_crossed and pd.isna(first_crossing_time):
    #                    first_crossing_time = row['days_before_removal']
    #          if not pd.isna(first_crossing_time): base_tp += 1
    #          else: base_fn += 1

    #     # Evaluate base threshold on training controls
    #     if training_control_trajectories:
    #         for trajectory in training_control_trajectories:
    #             if metric_name not in trajectory.columns: continue
    #             threshold_crossed = False
    #             for _, row in trajectory.iterrows():
    #                 value = row[metric_name]
    #                 if pd.isna(value): continue
    #                 threshold_is_crossed = (direction == 'below' and value <= best_threshold) or \
    #                                        (direction == 'above' and value >= best_threshold)
    #                 if threshold_is_crossed:
    #                     threshold_crossed = True
    #                     break
    #             if threshold_crossed: base_fp += 1
    #             else: base_tn += 1

    #     base_sensitivity = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0
    #     base_specificity = base_tn / (base_tn + base_fp) if (base_tn + base_fp) > 0 else 0

    #     perturbations = self.config.get('threshold_sensitivity_perturbations', [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05])
    #     for perturbation in perturbations:
    #         perturbed_threshold = best_threshold * (1 + perturbation)
    #         perturbation_key = f"{perturbation*100:+.0f}%"

    #         # Evaluate perturbed threshold
    #         tp, fn, warning_times = 0, 0, []
    #         for trajectory in train_trajectories:
    #              if metric_name not in trajectory.columns: continue
    #              first_crossing_time = np.nan
    #              traj_sorted = trajectory.sort_values(by='days_before_removal', ascending=False)
    #              for _, row in traj_sorted.iterrows():
    #                   value = row[metric_name]
    #                   if pd.isna(value): continue
    #                   threshold_is_crossed = (direction == 'below' and value <= perturbed_threshold) or \
    #                                         (direction == 'above' and value >= perturbed_threshold)
    #                   if threshold_is_crossed and pd.isna(first_crossing_time):
    #                        first_crossing_time = row['days_before_removal']
    #              if not pd.isna(first_crossing_time):
    #                   tp += 1
    #                   warning_times.append(first_crossing_time)
    #              else: fn += 1

    #         fp, tn = 0, 0
    #         if training_control_trajectories:
    #             for trajectory in training_control_trajectories:
    #                  if metric_name not in trajectory.columns: continue
    #                  threshold_crossed = False
    #                  for _, row in trajectory.iterrows():
    #                       value = row[metric_name]
    #                       if pd.isna(value): continue
    #                       threshold_is_crossed = (direction == 'below' and value <= perturbed_threshold) or \
    #                                             (direction == 'above' and value >= perturbed_threshold)
    #                       if threshold_is_crossed:
    #                            threshold_crossed = True
    #                            break
    #                  if threshold_crossed: fp += 1
    #                  else: tn += 1

    #         perturbed_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    #         perturbed_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    #         perturbed_mean_warning = np.mean(warning_times) if warning_times else 0

    #         sensitivity_results[perturbation_key] = {
    #             'threshold': perturbed_threshold,
    #             'sensitivity': perturbed_sensitivity,
    #             'specificity': perturbed_specificity,
    #             'mean_warning_time': perturbed_mean_warning,
    #             'true_positives': tp, 'false_negatives': fn,
    #             'true_negatives': tn, 'false_positives': fp
    #         }

    #     # Calculate stability scores
    #     sensitivity_stability, specificity_stability, overall_stability = 0.0, 0.0, 0.0
    #     sensitivity_ci, specificity_ci = [base_sensitivity]*2, [base_specificity]*2 # Default CI is just the value itself

    #     if sensitivity_results:
    #         sensitivity_changes = [abs(result['sensitivity'] - base_sensitivity) for result in sensitivity_results.values()]
    #         specificity_changes = [abs(result['specificity'] - base_specificity) for result in sensitivity_results.values()]

    #         stab_factor = self.config.get('threshold_stability_factor', 5.0)
    #         sensitivity_stability = max(0.0, 1.0 - np.mean(sensitivity_changes) * stab_factor)
    #         specificity_stability = max(0.0, 1.0 - np.mean(specificity_changes) * stab_factor)
    #         overall_stability = (sensitivity_stability + specificity_stability) / 2

    #         # Calculate confidence intervals based on stability
    #         ci_factor = self.config.get('threshold_stability_ci_factor', 0.2)
    #         sens_ci_width = (1.0 - sensitivity_stability) * ci_factor * base_sensitivity # Relative width
    #         spec_ci_width = (1.0 - specificity_stability) * ci_factor * base_specificity # Relative width
    #         sensitivity_ci = [max(0, base_sensitivity - sens_ci_width), min(1.0, base_sensitivity + sens_ci_width)]
    #         specificity_ci = [max(0, base_specificity - spec_ci_width), min(1.0, base_specificity + spec_ci_width)]

    #     return {
    #         'base_performance': {'sensitivity': base_sensitivity, 'specificity': base_specificity},
    #         'perturbation_results': sensitivity_results,
    #         'sensitivity_stability': sensitivity_stability,
    #         'specificity_stability': specificity_stability,
    #         'overall_stability': overall_stability,
    #         'sensitivity_confidence_interval': sensitivity_ci,
    #         'specificity_confidence_interval': specificity_ci
    #     }
    
    # def _prepare_for_json(self, data):
    #     """
    #     Convert non-serializable data (like NumPy and Pandas objects) to regular Python types.
    #     """
    #     try:
    #         if isinstance(data, dict):
    #             return {k: self._prepare_for_json(v) for k, v in data.items() if k != 'raw_data'}
    #         elif isinstance(data, list):
    #             return [self._prepare_for_json(item) for item in data]
    #         elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
    #             return int(data)
    #         elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
    #             return float(data)
    #         elif isinstance(data, (np.ndarray, pd.Series)):
    #             return self._prepare_for_json(data.tolist())
    #         elif isinstance(data, pd.DataFrame):
    #             return "DataFrame(too large for JSON)"
    #         elif isinstance(data, bytes):
    #             return str(data)
    #         elif isinstance(data, (datetime.datetime, pd.Timestamp, np.datetime64)):
    #             return str(data)
    #         else:
    #             return data
    #     except Exception as e:
    #         self.logger.warning(f"Error in _prepare_for_json: {e}, returning string representation")
    #         return str(data)