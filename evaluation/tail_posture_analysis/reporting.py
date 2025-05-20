import os
import numpy as np
import pandas as pd
import time

from evaluation.tail_posture_analysis.visualization import TailPostureVisualizer

from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info


class TailPostureReporter(TailPostureVisualizer):
    """Methods for generating reports from tail posture analysis."""
        
    def generate_summary_report(self):
        """Generate an summary report."""
        self.logger.info("Generating descriptive summary report...")

        def format_value(value, fmt_str='.3f', suffix="", na_val="N/A"):
            """Helper to format potentially None/NaN values."""
            if value is not None and not pd.isna(value):
                try: 
                    return f"{value:{fmt_str}}{suffix}"
                except (TypeError, ValueError): 
                    return na_val  # Handle unexpected types
            return na_val

        summary = {'dataset': {}, 'data_quality': {}, 'outbreak_analysis': {}, 
                'control_comparison': {}, 'individual_variation': {}, 
                'component_analysis': {}, 'monitoring_thresholds': {}}

        # --- Dataset Info ---
        if self.monitoring_results:
            summary['dataset']['num_cameras'] = len(set(r.get('camera', 'Unknown') for r in self.monitoring_results))
            summary['dataset']['num_datespans_loaded'] = len(self.monitoring_results)
            
            # Track both actual and reference datespans
            all_start_dates = []
            all_end_dates = []
            all_gt_start_dates = []
            all_gt_end_dates = []
            
            json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
            
            # Initialize counters and collections for statistics
            total_tb_pens = 0
            total_control_pens = 0
            removal_dates = []
            removal_dates_per_pen = {}
            removal_counts_per_pen = {}
            datespan_coverage = {}  # Track coverage percentages
            
            relative_removal_positions = {
                "first_quintile": 0,
                "second_quintile": 0,
                "third_quintile": 0,
                "fourth_quintile": 0,
                "fifth_quintile": 0
            }
            removal_position_percentages = []
            
            for r in self.monitoring_results:
                camera = r.get('camera')
                ds = r.get('date_span')
                
                if ds:
                    try:
                        start, end = ds.split('_')
                        start_date = pd.to_datetime(start, format='%y%m%d')
                        end_date = pd.to_datetime(end, format='%y%m%d')
                        all_start_dates.append(start_date)
                        all_end_dates.append(end_date)
                    except: 
                        pass
                
                # Get reference datespan (datespan_gt)
                pen_type, _, datespan_gt = get_pen_info(camera, ds, json_data)
                
                # Count pen types
                if pen_type == "tail biting":
                    total_tb_pens += 1
                elif pen_type == "control":
                    total_control_pens += 1
                    
                if datespan_gt and datespan_gt != "Unknown":
                    try:
                        gt_start, gt_end = datespan_gt.split('_')
                        gt_start_date = pd.to_datetime(gt_start, format='%y%m%d')
                        gt_end_date = pd.to_datetime(gt_end, format='%y%m%d')
                        all_gt_start_dates.append(gt_start_date)
                        all_gt_end_dates.append(gt_end_date)
                        
                        # Calculate coverage percentage
                        if ds:
                            try:
                                gt_duration = (gt_end_date - gt_start_date).days + 1
                                
                                # Calculate overlap
                                overlap_start = max(start_date, gt_start_date)
                                overlap_end = min(end_date, gt_end_date)
                                
                                if overlap_end >= overlap_start:  # Valid overlap
                                    overlap_days = (overlap_end - overlap_start).days + 1
                                    coverage_pct = (overlap_days / gt_duration) * 100
                                    datespan_coverage[f"{camera}_{ds}"] = coverage_pct
                            except:
                                pass
                    except:
                        pass
                
                # Extract culprit removal information
                if pen_type == "tail biting":
                    for entry in json_data:
                        if entry.get('camera') == camera and entry.get('datespan') == ds:
                            culprit_data = entry.get('culpritremoval')
                            pen_id = f"{camera}_{ds}"
                            
                            if culprit_data:
                                datespan_start = pd.to_datetime(ds.split('_')[0], format='%y%m%d')
                                datespan_end = pd.to_datetime(ds.split('_')[1], format='%y%m%d')
                                datespan_length = (datespan_end - datespan_start).days
                                
                                if isinstance(culprit_data, list):
                                    # Multiple removal dates
                                    removal_counts_per_pen[pen_id] = len(culprit_data)
                                    for date_str in culprit_data:
                                        try:
                                            removal_date = pd.to_datetime(date_str)
                                            removal_dates.append(removal_date)
                                            if pen_id not in removal_dates_per_pen:
                                                removal_dates_per_pen[pen_id] = []
                                            removal_dates_per_pen[pen_id].append(removal_date)
                                            
                                            if datespan_length > 0:
                                                days_from_start = (removal_date - datespan_start).days
                                                position_pct = (days_from_start / datespan_length) * 100
                                                removal_position_percentages.append(position_pct)
                                                
                                                if position_pct <= 20:
                                                    relative_removal_positions["first_quintile"] += 1
                                                elif position_pct <= 40:
                                                    relative_removal_positions["second_quintile"] += 1
                                                elif position_pct <= 60:
                                                    relative_removal_positions["third_quintile"] += 1
                                                elif position_pct <= 80:
                                                    relative_removal_positions["fourth_quintile"] += 1
                                                else:
                                                    relative_removal_positions["fifth_quintile"] += 1
                                        except:
                                            pass
                                else:
                                    # Single removal date
                                    removal_counts_per_pen[pen_id] = 1
                                    try:
                                        removal_date = pd.to_datetime(culprit_data)
                                        removal_dates.append(removal_date)
                                        removal_dates_per_pen[pen_id] = [removal_date]
                                        
                                        if datespan_length > 0:
                                            days_from_start = (removal_date - datespan_start).days
                                            position_pct = (days_from_start / datespan_length) * 100
                                            removal_position_percentages.append(position_pct)
                                            
                                            if position_pct <= 20:
                                                relative_removal_positions["first_quintile"] += 1
                                            elif position_pct <= 40:
                                                relative_removal_positions["second_quintile"] += 1
                                            elif position_pct <= 60:
                                                relative_removal_positions["third_quintile"] += 1
                                            elif position_pct <= 80:
                                                relative_removal_positions["fourth_quintile"] += 1
                                            else:
                                                relative_removal_positions["fifth_quintile"] += 1
                                    except:
                                        pass
                            break
            
            summary['dataset']['relative_removal_positions'] = relative_removal_positions

            if removal_position_percentages:
                summary['dataset']['removal_position_stats'] = {
                    'avg_position_pct': sum(removal_position_percentages) / len(removal_position_percentages),
                    'median_position_pct': np.median(removal_position_percentages),
                    'min_position_pct': min(removal_position_percentages),
                    'max_position_pct': max(removal_position_percentages)
                }
            
            # Analyze removal dates temporally
            if removal_dates:
                # Sort the dates for analysis
                removal_dates.sort()
                first_removal = min(removal_dates)
                last_removal = max(removal_dates)
                
                # Analyze distribution by quarters
                quarters = pd.Series(removal_dates).dt.to_period('Q').value_counts().sort_index()
                
                summary['dataset']['removal_date_range'] = {
                    'first': first_removal.date(),
                    'last': last_removal.date(),
                    'span_days': (last_removal - first_removal).days
                }
                
                summary['dataset']['removal_dates_quarterly'] = {
                    str(quarter): count for quarter, count in quarters.items()
                }
                
                # Statistics on removals per pen
                summary['dataset']['removals_per_pen'] = {
                    'min': min(removal_counts_per_pen.values()) if removal_counts_per_pen else 0,
                    'max': max(removal_counts_per_pen.values()) if removal_counts_per_pen else 0,
                    'avg': sum(removal_counts_per_pen.values()) / len(removal_counts_per_pen) if removal_counts_per_pen else 0,
                    'total_removals': sum(removal_counts_per_pen.values()) if removal_counts_per_pen else 0,
                    'pens_with_multiple': sum(1 for count in removal_counts_per_pen.values() if count > 1) if removal_counts_per_pen else 0
                }
            
            # Add datespan coverage statistics
            if datespan_coverage:
                coverage_values = list(datespan_coverage.values())
                summary['dataset']['datespan_coverage'] = {
                    'avg_pct': sum(coverage_values) / len(coverage_values) if coverage_values else 0,
                    'min_pct': min(coverage_values) if coverage_values else 0,
                    'max_pct': max(coverage_values) if coverage_values else 0,
                    'full_coverage_count': sum(1 for pct in coverage_values if pct >= 99.0)  # Effectively 100% coverage
                }
            
            if all_start_dates and all_end_dates:
                summary['dataset']['date_range'] = { 
                    'min': min(all_start_dates).date(), 
                    'max': max(all_end_dates).date(),
                    'span_days': (max(all_end_dates) - min(all_start_dates)).days
                }
            
            # Add reference date range if available
            if all_gt_start_dates and all_gt_end_dates:
                summary['dataset']['reference_date_range'] = { 
                    'min': min(all_gt_start_dates).date(), 
                    'max': max(all_gt_end_dates).date(),
                    'span_days': (max(all_gt_end_dates) - min(all_gt_start_dates)).days
                }
            
            # Count pen types
            pen_types = {}
            for result in self.monitoring_results:
                pen_type, _, _ = get_pen_info(result.get('camera'), result.get('date_span'), json_data)
                pen_types[pen_type] = pen_types.get(pen_type, 0) + 1
            summary['dataset']['pen_types_loaded'] = pen_types
            summary['dataset']['total_tb_pens'] = total_tb_pens
            summary['dataset']['total_control_pens'] = total_control_pens
        else: 
            summary['dataset']['status'] = "No monitoring results loaded."

        # --- Data Quality Metrics ---
        if self.processed_results:
            json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
            
            # Calculate metrics for actual datespans
            total_expected_days = sum(p['quality_metrics'].get('total_expected_days', 0) for p in self.processed_results)
            total_missing_days = sum(p['quality_metrics'].get('missing_days_detected', 0) for p in self.processed_results)
            
            # Calculate metrics for reference datespans (GT)
            total_gt_expected_days = 0
            total_gt_missing_days = 0
            total_gt_available_days = 0
            incomplete_camera_datespan = 0
            missing_days_within_datespan = 0  # Initialize this variable here

            # Track individual missing days counts for min/max/avg/median calculations
            missing_days_per_datespan = []

            for p in self.processed_results:
                camera = p.get('camera')
                datespan = p.get('date_span')
                _, _, datespan_gt = get_pen_info(camera, datespan, json_data)
                
                missing_days_files = p['quality_metrics'].get('missing_days_detected', 0)
                missing_days_within_datespan += missing_days_files
                
                missing_days_per_datespan.append(missing_days_files)
                
                if datespan_gt and datespan_gt != "Unknown":
                    try:
                        # Parse GT datespan
                        gt_start, gt_end = datespan_gt.split('_')
                        gt_start_date = pd.to_datetime(gt_start, format='%y%m%d')
                        gt_end_date = pd.to_datetime(gt_end, format='%y%m%d')
                        
                        # Parse actual datespan
                        start, end = datespan.split('_')
                        start_date = pd.to_datetime(start, format='%y%m%d')
                        end_date = pd.to_datetime(end, format='%y%m%d')
                        
                        # Calculate days
                        gt_days = (gt_end_date - gt_start_date).days + 1
                        actual_days = (end_date - start_date).days + 1
                        
                        total_gt_expected_days += gt_days
                        total_gt_available_days += actual_days
                        
                        # Calculate missing days due to datespan difference
                        # Days missing at beginning
                        days_missing_start = max(0, (start_date - gt_start_date).days)
                        # Days missing at end
                        days_missing_end = max(0, (gt_end_date - end_date).days)
                        # Total missing due to datespan difference
                        total_missing_due_to_span = days_missing_start + days_missing_end
                        
                        total_gt_missing_days += total_missing_due_to_span
                        
                        # Check for incomplete datespans
                        if gt_start_date < start_date or gt_end_date > end_date:
                            incomplete_camera_datespan += 1
                    except:
                        pass

            # Calculate avg/min/max/median of missing days per datespan
            avg_missing_days_per_datespan = np.mean(missing_days_per_datespan) if missing_days_per_datespan else 0
            min_missing_days_per_datespan = min(missing_days_per_datespan) if missing_days_per_datespan else 0
            max_missing_days_per_datespan = max(missing_days_per_datespan) if missing_days_per_datespan else 0
            median_missing_days_per_datespan = np.median(missing_days_per_datespan) if missing_days_per_datespan else 0
                        
            # Calculate average quality metrics
            avg_missing_raw = np.nanmean([p['quality_metrics'].get('percent_missing_rows_raw') for p in self.processed_results])
            avg_missing_resampled = np.nanmean([p['quality_metrics'].get('percent_missing_resampled') for p in self.processed_results])
            avg_max_consecutive = np.nanmean([p['quality_metrics'].get('max_consecutive_missing_resampled') for p in self.processed_results])
            
            # Store all metrics in summary
            summary['data_quality']['total_processed_events'] = len(self.processed_results)
            summary['data_quality']['total_expected_days'] = total_expected_days
            summary['data_quality']['total_missing_days_files'] = total_missing_days
            summary['data_quality']['overall_missing_days_pct'] = (total_missing_days / total_expected_days * 100) if total_expected_days > 0 else 0.0

            # Add GT datespan metrics
            summary['data_quality']['total_gt_expected_days'] = total_gt_expected_days
            summary['data_quality']['total_gt_available_days'] = total_gt_available_days
            summary['data_quality']['total_gt_missing_days'] = total_gt_missing_days
            summary['data_quality']['overall_gt_missing_days_pct'] = (total_gt_missing_days / total_gt_expected_days * 100) if total_gt_expected_days > 0 else 0.0
            summary['data_quality']['incomplete_camera_datespans'] = incomplete_camera_datespan
            summary['data_quality']['incomplete_camera_datespans_pct'] = (incomplete_camera_datespan / len(self.processed_results) * 100) if len(self.processed_results) > 0 else 0.0

            # Add metrics about missing days within datespan
            summary['data_quality']['missing_days_within_datespan'] = missing_days_within_datespan
            summary['data_quality']['missing_days_within_datespan_pct'] = (missing_days_within_datespan / total_gt_available_days * 100) if total_gt_available_days > 0 else 0.0

            # Add new metrics for min/max/avg/median missing days per datespan
            summary['data_quality']['avg_missing_days_per_datespan'] = avg_missing_days_per_datespan
            summary['data_quality']['min_missing_days_per_datespan'] = min_missing_days_per_datespan
            summary['data_quality']['max_missing_days_per_datespan'] = max_missing_days_per_datespan
            summary['data_quality']['median_missing_days_per_datespan'] = median_missing_days_per_datespan
            
            # Other quality metrics
            summary['data_quality']['avg_missing_raw_seconds_pct'] = avg_missing_raw if not np.isnan(avg_missing_raw) else None
            summary['data_quality']['avg_missing_resampled_periods_pct'] = avg_missing_resampled if not np.isnan(avg_missing_resampled) else None
            summary['data_quality']['avg_max_consecutive_missing_resampled'] = avg_max_consecutive if not np.isnan(avg_max_consecutive) else None
            summary['data_quality']['events_excluded_for_consecutive_missing'] = self.excluded_events_count
            summary['data_quality']['events_excluded_for_missing_pct'] = getattr(self, 'excluded_events_missing_pct_count', 0)
            summary['data_quality']['total_events_excluded_for_missing_data'] = (
                self.excluded_events_count + getattr(self, 'excluded_events_missing_pct_count', 0)
            )
            summary['data_quality']['missing_data_exclusion_threshold_days'] = self.config.get('max_allowed_consecutive_missing_days')
            summary['data_quality']['missing_data_exclusion_threshold_pct'] = self.config.get('max_allowed_missing_days_pct', 50.0)
            summary['data_quality']['interpolation_method'] = self.config.get('interpolation_method')
            if self.config.get('interpolation_method') in ['spline', 'polynomial']:
                summary['data_quality']['interpolation_order'] = self.config.get('interpolation_order')
        else: 
            summary['data_quality']['status'] = "No data processed."

        # --- Descriptive Outbreak Analysis ---
        if hasattr(self, 'pre_outbreak_stats') and self.pre_outbreak_stats is not None and not self.pre_outbreak_stats.empty:
            stats_df = self.pre_outbreak_stats
            oa = summary['outbreak_analysis']
            n_analyzed = len(stats_df)
            oa['num_outbreaks_analyzed'] = n_analyzed
            oa['num_pens_analyzed'] = len(stats_df['pen'].unique())

            # Value stats
            val_rem = stats_df['value_at_removal']
            oa['value_at_removal'] = {
                'count': val_rem.count(),
                'mean': val_rem.mean(), 'median': val_rem.median(), 'std': val_rem.std(),
                'p25': val_rem.quantile(0.25), 'p10': val_rem.quantile(0.10)}
            for days in [1, 3, 5, 7, 10]:
                col = f'value_{days}d_before'
                if col in stats_df.columns and stats_df[col].notna().any():
                    data = stats_df[col]
                    oa[col] = {'mean': data.mean(), 'median': data.median(), 'std': data.std()}

            # Change stats
            oa['absolute_change'] = {}
            oa['percentage_change'] = {}
            for days in [1, 3, 7]:
                # Absolute Change
                abs_col = f'abs_change_{days}d'
                if abs_col in stats_df.columns and stats_df[abs_col].notna().any():
                    data = stats_df[abs_col]
                    oa['absolute_change'][f'{days}d'] = {'mean': data.mean(), 'median': data.median(), 'std': data.std()}
                # Percentage Change
                pct_col = f'pct_change_{days}d'
                if pct_col in stats_df.columns:
                    # Handle inf/-inf before dropping NaN
                    valid_pct = stats_df[pct_col].replace([np.inf, -np.inf], np.nan).dropna()
                    if not valid_pct.empty:
                         oa['percentage_change'][f'{days}d'] = {'mean': valid_pct.mean(), 'median': valid_pct.median(), 'std': valid_pct.std()}


            # Window stats
            oa['window_stats'] = {}
            alpha = self.config.get('significance_level', 0.05)
            for days in [3, 7]:
                key_base = f'{days}d_window'
                oa['window_stats'][key_base] = {}
                # Average
                avg_col = f'{key_base}_avg'
                if avg_col in stats_df.columns and stats_df[avg_col].notna().any():
                    data = stats_df[avg_col]
                    oa['window_stats'][key_base]['avg'] = {'mean': data.mean(), 'median': data.median(), 'std': data.std()}
                # Slope
                slope_col = f'{key_base}_slope'
                if slope_col in stats_df.columns and stats_df[slope_col].notna().any():
                    data = stats_df[slope_col]
                    oa['window_stats'][key_base]['slope'] = {'mean': data.mean(), 'median': data.median(), 'std': data.std()}
                    # Calculate percentage significant slopes
                    pval_col = f'{key_base}_slope_p_value'
                    if pval_col in stats_df.columns and stats_df[pval_col].notna().any():
                        p_values = stats_df[pval_col].dropna()
                        if not p_values.empty:
                            significant_count = (p_values < alpha).sum()
                            total_valid_p = len(p_values)
                            pct_significant = (significant_count / total_valid_p * 100) if total_valid_p > 0 else 0
                            oa['window_stats'][key_base]['slope']['percent_significant'] = pct_significant
                            oa['window_stats'][key_base]['slope']['alpha_level'] = alpha # Store alpha used

        elif hasattr(self, 'excluded_events_count') and self.excluded_events_count > 0:
            total_excluded = self.excluded_events_count + getattr(self, 'excluded_events_missing_pct_count', 0)
            summary['outbreak_analysis']['status'] = f"No outbreaks analyzed successfully (due to {total_excluded} exclusions for missing data, or other reasons)."
        else: summary['outbreak_analysis']['status'] = "No tail biting events found or analyzed."

        # --- Control Comparison Analysis ---
        if hasattr(self, 'control_stats') and self.control_stats is not None and not self.control_stats.empty and \
           hasattr(self, 'pre_outbreak_stats') and self.pre_outbreak_stats is not None and not self.pre_outbreak_stats.empty:
            cc = summary['control_comparison']
            cc['num_control_pens_analyzed'] = len(self.control_stats['pen'].unique())
            cc['num_control_reference_points'] = len(self.control_stats)

            # Ensure comparison results are calculated and stored
            if not hasattr(self, 'comparison_results') or self.comparison_results is None:
                 self.logger.info("Calculating outbreak vs control comparison for report...")
                 self.compare_outbreak_vs_control_statistics() # This should store results in self.comparison_results

            # Check again if it exists after calculation
            if hasattr(self, 'comparison_results') and self.comparison_results:
                cc['comparison_results'] = self.comparison_results

                # Extract key metrics for summary
                significant_metrics = [metric for metric, data in self.comparison_results.items()
                                    if data.get('is_significant', False)]
                cc['significant_metrics'] = significant_metrics

                # Count strong effect sizes
                strong_effects = [metric for metric, data in self.comparison_results.items()
                                if data.get('effect_size', 0) > 0.8]
                cc['strong_effect_metrics'] = strong_effects
            else:
                 cc['status'] = "Comparison could not be performed or failed."
        else:
            summary['control_comparison']['status'] = "No control pen data or outbreak data available for comparison."
            
        # --- Individual Variation Analysis ---
        if hasattr(self, 'outbreak_patterns') and self.outbreak_patterns is not None and not self.outbreak_patterns.empty:
            iv = summary['individual_variation']
            
            # Add pattern counts
            pattern_counts = self.outbreak_patterns['pattern_category'].value_counts().to_dict()
            iv['pattern_counts'] = pattern_counts
            
            # Calculate percentage of each pattern
            total_patterns = sum(pattern_counts.values())
            pattern_percentages = {pattern: count/total_patterns*100 for pattern, count in pattern_counts.items()}
            iv['pattern_percentages'] = pattern_percentages
            
            # Add pen consistency information
            pens_with_multiple = 0
            pens_consistent = 0
            
            pen_patterns = self.outbreak_patterns.groupby('pen')['pattern_category'].apply(list)
            pens_with_multiple = sum(len(patterns) > 1 for patterns in pen_patterns)
            pens_consistent = sum(len(set(patterns)) == 1 for patterns in pen_patterns if len(patterns) > 1)
            
            consistency_pct = (pens_consistent / pens_with_multiple * 100) if pens_with_multiple > 0 else 0
            
            iv['pen_consistency'] = {
                'pens_with_multiple': pens_with_multiple,
                'pens_consistent': pens_consistent,
                'consistency_percentage': consistency_pct
            }
            
            # Add average metrics by pattern
            for pattern in pattern_counts.keys():
                pattern_data = self.outbreak_patterns[self.outbreak_patterns['pattern_category'] == pattern]
                
                iv[f"{pattern}_metrics"] = {
                    'count': len(pattern_data),
                    'avg_value_at_removal': pattern_data['value_at_removal'].mean(),
                    'avg_abs_change_7d': pattern_data['abs_change_7d'].mean() if 'abs_change_7d' in pattern_data.columns else np.nan,
                    'avg_7d_slope': pattern_data['7d_window_slope'].mean() if '7d_window_slope' in pattern_data.columns else np.nan,
                    'avg_3d_slope': pattern_data['3d_window_slope'].mean() if '3d_window_slope' in pattern_data.columns else np.nan
                }
        else:
            summary['individual_variation']['status'] = "No pattern analysis available."

        # --- Component Analysis Section ---
        if hasattr(self, 'component_analysis') and self.component_analysis is not None:
            ca = summary['component_analysis']
            
            # Add statistics from component analysis
            if 'timepoint_stats' in self.component_analysis:
                timepoint_stats = self.component_analysis['timepoint_stats']
                ca['timepoint_stats'] = timepoint_stats
                
                # Add summary level metrics for key timepoints
                for days in [0, 3, 7]:
                    key = f'day_minus_{days}'
                    if key in timepoint_stats:
                        ca[f'day_minus_{days}_stats'] = timepoint_stats[key]
            
            # Add change statistics
            if 'change_stats' in self.component_analysis:
                ca['change_stats'] = self.component_analysis['change_stats']
                
            # Add contribution statistics
            if 'contribution_stats' in self.component_analysis:
                ca['contribution_stats'] = self.component_analysis['contribution_stats']
                
                # Highlight primary drivers
                if 'primary_driver_7d' in self.component_analysis['contribution_stats']:
                    ca['primary_driver_7d'] = self.component_analysis['contribution_stats']['primary_driver_7d']
                    ca['upright_contribution_7d'] = self.component_analysis['contribution_stats']['upright_contribution_7d']
                    ca['hanging_contribution_7d'] = self.component_analysis['contribution_stats']['hanging_contribution_7d']
                    
                if 'primary_driver_3d' in self.component_analysis['contribution_stats']:
                    ca['primary_driver_3d'] = self.component_analysis['contribution_stats']['primary_driver_3d']
                    ca['upright_contribution_3d'] = self.component_analysis['contribution_stats']['upright_contribution_3d']
                    ca['hanging_contribution_3d'] = self.component_analysis['contribution_stats']['hanging_contribution_3d']
        else:
            summary['component_analysis']['status'] = "No component analysis available."


        # --- Generate Text Report ---
        report_text = ["=" * 80, "DESCRIPTIVE TAIL POSTURE ANALYSIS REPORT", "=" * 80, ""]

        # Dataset Info 
        report_text.append("DATASET INFORMATION")
        report_text.append("-" * 80)
        ds = summary.get('dataset', {})
        report_text.append(f"Number of cameras found: {ds.get('num_cameras', 'N/A')}")
        report_text.append(f"Number of datespans loaded: {ds.get('num_datespans_loaded', 'N/A')}")

        # Date range information
        if 'date_range' in ds:
            actual_range = ds['date_range']
            report_text.append(f"Actual Date range: {actual_range['min']} to {actual_range['max']} ({actual_range.get('span_days', 'N/A')} days)")

        if 'reference_date_range' in ds:
            ref_range = ds['reference_date_range']
            report_text.append(f"Reference Date range: {ref_range['min']} to {ref_range['max']} ({ref_range.get('span_days', 'N/A')} days)")

        # Add datespan coverage information
        if 'datespan_coverage' in ds:
            coverage = ds['datespan_coverage']
            report_text.append(f"Datespan Coverage Analysis:")
            report_text.append(f"  - Average actual vs. reference coverage: {coverage.get('avg_pct', 0):.1f}%")
            report_text.append(f"  - Range: {coverage.get('min_pct', 0):.1f}% - {coverage.get('max_pct', 0):.1f}%")
            report_text.append(f"  - Full coverage (>99%): {coverage.get('full_coverage_count', 0)} datespans")

        # Pen type information
        if 'pen_types_loaded' in ds:
            report_text.append("\nPen Types Analysis:")
            for pen_type, count in ds['pen_types_loaded'].items():
                report_text.append(f"  - {pen_type}: {count}")
            report_text.append(f"  - Total Tail Biting Pens: {ds.get('total_tb_pens', 'N/A')}")
            report_text.append(f"  - Total Control Pens: {ds.get('total_control_pens', 'N/A')}")

        # Culprit removal information
        if 'removal_date_range' in ds:
            removal_range = ds['removal_date_range']
            report_text.append("\nCulprit Removal Analysis:")
            report_text.append(f"  - Removal date span: {removal_range['first']} to {removal_range['last']} ({removal_range['span_days']} days)")
            
            # Relative position analysis
            if 'relative_removal_positions' in ds:
                positions = ds['relative_removal_positions']
                total_removals = sum(positions.values())
                
                report_text.append("  - Relative Position der Täterentfernungen innerhalb der Durchläufe:")
                report_text.append(f"    * Sehr früh (0-20% des Durchlaufs): {positions.get('first_quintile', 0)} Entfernungen ({positions.get('first_quintile', 0)/total_removals*100:.1f}% der Fälle)")
                report_text.append(f"    * Früh (21-40% des Durchlaufs): {positions.get('second_quintile', 0)} Entfernungen ({positions.get('second_quintile', 0)/total_removals*100:.1f}% der Fälle)")
                report_text.append(f"    * Mitte (41-60% des Durchlaufs): {positions.get('third_quintile', 0)} Entfernungen ({positions.get('third_quintile', 0)/total_removals*100:.1f}% der Fälle)")
                report_text.append(f"    * Spät (61-80% des Durchlaufs): {positions.get('fourth_quintile', 0)} Entfernungen ({positions.get('fourth_quintile', 0)/total_removals*100:.1f}% der Fälle)")
                report_text.append(f"    * Sehr spät (81-100% des Durchlaufs): {positions.get('fifth_quintile', 0)} Entfernungen ({positions.get('fifth_quintile', 0)/total_removals*100:.1f}% der Fälle)")

            if 'removal_position_stats' in ds:
                stats = ds['removal_position_stats']
                report_text.append("  - Statistik zur relativen Position der Täterentfernungen:")
                report_text.append(f"    * Durchschnittliche Position: {stats.get('avg_position_pct', 0):.1f}% des Durchlaufs")
                report_text.append(f"    * Median-Position: {stats.get('median_position_pct', 0):.1f}% des Durchlaufs")
                report_text.append(f"    * Früheste Entfernung: {stats.get('min_position_pct', 0):.1f}% des Durchlaufs")
                report_text.append(f"    * Späteste Entfernung: {stats.get('max_position_pct', 0):.1f}% des Durchlaufs")
            
            # Statistics on removals per pen
            if 'removals_per_pen' in ds:
                removals = ds['removals_per_pen']
                report_text.append("  - Schwanzbeißer pro Bucht:")
                report_text.append(f"    * Insgesamt entfernte Schwanzbeißer: {removals.get('total_removals', 0)}")
                report_text.append(f"    * Durchschnitt: {removals.get('avg', 0):.1f} Schwanzbeißer pro Bucht")
                report_text.append(f"    * Spanne: {removals.get('min', 0)}-{removals.get('max', 0)} Schwanzbeißer pro Bucht")
                report_text.append(f"    * Buchten mit mehreren Entfernungen: {removals.get('pens_with_multiple', 0)}")

            # Add quarterly distribution (optional - falls Sie diese Information dennoch behalten möchten)
            if 'removal_dates_quarterly' in ds:
                report_text.append("  - Quartalsweise Verteilung der Entfernungen:")
                for quarter, count in ds['removal_dates_quarterly'].items():
                    report_text.append(f"    * {quarter}: {count} Entfernungen")

        report_text.append("")

        # Data Quality Info 
        report_text.append("DATA QUALITY & PREPROCESSING")
        report_text.append("-" * 80)
        dq = summary.get('data_quality', {})
        if 'status' in dq: report_text.append(dq['status'])
        else:
            if 'overall_gt_missing_days_pct' in dq and 'overall_missing_days_pct' in dq:
                report_text.append("\nReference vs Actual Datespan Coverage:")
                
                gt_missing_pct = dq.get('overall_gt_missing_days_pct', 0.0)
                missing_pct = dq.get('overall_missing_days_pct', 0.0)
                gt_expected_days = dq.get('total_gt_expected_days', 0)
                gt_available_days = dq.get('total_gt_available_days', 0)
                gt_missing_days = dq.get('total_gt_missing_days', 0)
                
                report_text.append(f"- Based on reference datespans (datespan_gt), overall {gt_missing_days} of {gt_expected_days} expected days ({gt_missing_pct:.1f}%) were missing due to datespan differences.")
                
                missing_days_within = dq.get('missing_days_within_datespan', 0)
                missing_days_within_pct = dq.get('missing_days_within_datespan_pct', 0.0)
                report_text.append(f"- {missing_days_within} days ({missing_days_within_pct:.1f}%) were missing within the actual collected datespans.")
                
                # Add new bilingual section for missing days per datespan statistics
                avg_missing = dq.get('avg_missing_days_per_datespan', 0.0)
                min_missing = dq.get('min_missing_days_per_datespan', 0)
                max_missing = dq.get('max_missing_days_per_datespan', 0)
                median_missing = dq.get('median_missing_days_per_datespan', 0.0)
                
                # German version
                report_text.append(f"- Durchschnittliche Anzahl fehlender Tage innerhalb der verfügbaren Zeitspannen: {avg_missing:.1f} Tage pro Zeitspanne")
                report_text.append(f"  * Minimum: {min_missing} Tage, Maximum: {max_missing} Tage, Median: {median_missing:.1f} Tage")
                
                # English version
                report_text.append(f"- Average number of missing days within available datespans: {avg_missing:.1f} days per datespan")
                report_text.append(f"  * Minimum: {min_missing} days, Maximum: {max_missing} days, Median: {median_missing:.1f} days")
                
                incomplete_pct = dq.get('incomplete_camera_datespans_pct', 0.0)
                report_text.append(f"- {dq.get('incomplete_camera_datespans', 0)} camera/datespan combinations ({incomplete_pct:.1f}%) had incomplete coverage compared to their reference datespan.")
            
            report_text.append(f"Total processed camera/datespan events: {dq.get('total_processed_events', 'N/A')}")
            total_expected = dq.get('total_expected_days', 0); missing_files = dq.get('total_missing_days_files', 0); missing_pct = dq.get('overall_missing_days_pct', 0.0)
            report_text.append(f"Missing Daily Files: {missing_files} of {total_expected} expected days ({missing_pct:.2f}%) were missing source CSV files.")
            report_text.append(f"  - Avg. missing raw seconds within loaded days: {dq.get('avg_missing_raw_seconds_pct', 0.0):.2f}% (approx).")
            report_text.append(f"  - Avg. missing resampled periods ({self.config.get('resample_freq', '?')}) before interpolation: {dq.get('avg_missing_resampled_periods_pct', 0.0):.2f}%.")
            report_text.append(f"  - Avg. max consecutive missing resampled periods: {dq.get('avg_max_consecutive_missing_resampled', 0.0):.1f} days.")
            report_text.append(f"Interpolation Method: '{dq.get('interpolation_method', 'N/A')}'" + (f" (Order: {dq['interpolation_order']})" if dq.get('interpolation_method') in ['spline', 'polynomial'] else "") + " was used.")
            
            # Updated exclusion reporting
            consecutive_excluded = dq.get('events_excluded_for_consecutive_missing', 0)
            pct_excluded = dq.get('events_excluded_for_missing_pct', 0)
            total_excluded = dq.get('total_events_excluded_for_missing_data', 0)
            consec_thresh = dq.get('missing_data_exclusion_threshold_days', 'N/A')
            pct_thresh = dq.get('missing_data_exclusion_threshold_pct', 'N/A')

            report_text.append(f"Event Exclusion: {total_excluded} total event(s) excluded from analysis due to missing data:")
            report_text.append(f"  - {consecutive_excluded} event(s) with >{consec_thresh} consecutive missing {self.config.get('resample_freq', '?')} periods")
            report_text.append(f"  - {pct_excluded} event(s) with >{pct_thresh}% of expected days missing")
        report_text.append("")
        
        report_text.append("DETAILED EXCLUSION REPORT")
        report_text.append("-" * 80)

        if hasattr(self, 'exclusion_by_analysis'):
            reported_cameras = set()
            
            # First report tail biting analysis exclusions
            report_text.append("\nExclusions from Tail Biting Event Analysis:")
            
            # Consecutive missing exclusions
            if 'tail_biting_analysis' in self.exclusion_by_analysis and 'consecutive_missing' in self.exclusion_by_analysis['tail_biting_analysis'] and self.exclusion_by_analysis['tail_biting_analysis']['consecutive_missing']:
                report_text.append("\n  Elements excluded due to consecutive missing periods:")
                for camera, date_span, pen_type, consecutive_missing, threshold in self.exclusion_by_analysis['tail_biting_analysis']['consecutive_missing']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {consecutive_missing} consecutive missing periods (threshold: {threshold})")
            
            # Missing percentage exclusions
            if 'tail_biting_analysis' in self.exclusion_by_analysis and 'missing_percentage' in self.exclusion_by_analysis['tail_biting_analysis'] and self.exclusion_by_analysis['tail_biting_analysis']['missing_percentage']:
                report_text.append("\n  Elements excluded due to excessive missing days percentage:")
                for camera, date_span, pen_type, missing_pct, threshold in self.exclusion_by_analysis['tail_biting_analysis']['missing_percentage']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {missing_pct:.1f}% missing days (threshold: {threshold}%)")
                        
            # Other reasons exclusions
            if 'tail_biting_analysis' in self.exclusion_by_analysis and 'other_reasons' in self.exclusion_by_analysis['tail_biting_analysis'] and self.exclusion_by_analysis['tail_biting_analysis']['other_reasons']:
                report_text.append("\n  Elements excluded for other reasons:")
                for camera, date_span, pen_type, reason in self.exclusion_by_analysis['tail_biting_analysis']['other_reasons']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {reason}")
            
            # Reset tracking set for control analysis exclusions
            reported_cameras = set()
            
            # Next report control analysis exclusions
            report_text.append("\nExclusions from Control Pen Analysis:")
            
            # Consecutive missing exclusions
            if 'control_analysis' in self.exclusion_by_analysis and 'consecutive_missing' in self.exclusion_by_analysis['control_analysis'] and self.exclusion_by_analysis['control_analysis']['consecutive_missing']:
                report_text.append("\n  Elements excluded due to consecutive missing periods:")
                for camera, date_span, pen_type, consecutive_missing, threshold in self.exclusion_by_analysis['control_analysis']['consecutive_missing']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {consecutive_missing} consecutive missing periods (threshold: {threshold})")
            
            # Missing percentage exclusions
            if 'control_analysis' in self.exclusion_by_analysis and 'missing_percentage' in self.exclusion_by_analysis['control_analysis'] and self.exclusion_by_analysis['control_analysis']['missing_percentage']:
                report_text.append("\n  Elements excluded due to excessive missing days percentage:")
                for camera, date_span, pen_type, missing_pct, threshold in self.exclusion_by_analysis['control_analysis']['missing_percentage']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {missing_pct:.1f}% missing days (threshold: {threshold}%)")
                        
            # Other reasons exclusions
            if 'control_analysis' in self.exclusion_by_analysis and 'other_reasons' in self.exclusion_by_analysis['control_analysis'] and self.exclusion_by_analysis['control_analysis']['other_reasons']:
                report_text.append("\n  Elements excluded for other reasons:")
                for camera, date_span, pen_type, reason in self.exclusion_by_analysis['control_analysis']['other_reasons']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {reason}")

        # Fall back to the original method if tracking isn't available
        elif hasattr(self, 'excluded_elements'):
            report_text.append("\nNote: Exclusions are not separated by analysis type. To see which analysis excluded each element, re-run with updated tracking.")
            
            reported_cameras = set()
            
            # Consecutive missing exclusions
            if 'consecutive_missing' in self.excluded_elements and self.excluded_elements['consecutive_missing']:
                report_text.append("\nElements excluded due to consecutive missing periods:")
                for camera, date_span, pen_type, consecutive_missing, threshold in self.excluded_elements['consecutive_missing']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {consecutive_missing} consecutive missing periods (threshold: {threshold})")
            
            # Missing percentage exclusions
            if 'missing_percentage' in self.excluded_elements and self.excluded_elements['missing_percentage']:
                report_text.append("\nElements excluded due to excessive missing days percentage:")
                for camera, date_span, pen_type, missing_pct, threshold in self.excluded_elements['missing_percentage']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {missing_pct:.1f}% missing days (threshold: {threshold}%)")
                        
            # Other reasons exclusions
            if 'other_reasons' in self.excluded_elements and self.excluded_elements['other_reasons']:
                report_text.append("\nElements excluded for other reasons:")
                for camera, date_span, pen_type, reason in self.excluded_elements['other_reasons']:
                    element_id = f"{camera}_{date_span}"
                    if element_id not in reported_cameras:
                        reported_cameras.add(element_id)
                        camera_label = camera.replace("Kamera", "Pen ")
                        report_text.append(f"  - {camera_label} / {date_span} ({pen_type}): {reason}")
                            
        else:
            report_text.append("\nNo detailed exclusion information available.")
                                
        report_text.append("")

        # Outbreak Analysis
        report_text.append("DESCRIPTIVE PRE-OUTBREAK ANALYSIS")
        report_text.append("-" * 80)
        oa = summary.get('outbreak_analysis', {})
        if 'status' in oa:
            report_text.append(oa['status'])
        else:
            report_text.append(f"Number of outbreaks analyzed: {oa.get('num_outbreaks_analyzed', 'N/A')} from {oa.get('num_pens_analyzed', 'N/A')} pens.")

            report_text.append("\nPosture Difference Statistics (Value):")
            if 'value_at_removal' in oa:
                stats = oa['value_at_removal']
                report_text.append(f"  - At Removal (N={stats.get('count', 'N/A')}): " +
                                   f"Mean={format_value(stats.get('mean'))}, Median={format_value(stats.get('median'))}, " +
                                   f"Std={format_value(stats.get('std'))}, P25={format_value(stats.get('p25'))}, P10={format_value(stats.get('p10'))}")
            for days in [1, 3, 7]:
                key = f'value_{days}d_before'
                if key in oa:
                    stats = oa[key]
                    report_text.append(f"  - {days}d Before : Mean={format_value(stats.get('mean'))}, Median={format_value(stats.get('median'))}, Std={format_value(stats.get('std'))}")

            report_text.append("\nChange Statistics (Relative to Removal):")
            if 'absolute_change' in oa and oa['absolute_change']:
                 report_text.append("  Absolute Change:")
                 for days_str, stats in oa['absolute_change'].items():
                     report_text.append(f"    - {days_str} Window: Mean={format_value(stats.get('mean'))}, Median={format_value(stats.get('median'))}, Std={format_value(stats.get('std'))}")
            else: report_text.append("  Absolute Change: No data calculated.")

            if 'percentage_change' in oa and oa['percentage_change']:
                 report_text.append("  Percentage Change:")
                 for days_str, stats in oa['percentage_change'].items():
                     report_text.append(f"    - {days_str} Window: Mean={format_value(stats.get('mean'), '.1f', '%')}, Median={format_value(stats.get('median'), '.1f', '%')}, Std={format_value(stats.get('std'), '.1f', '%')}")
            else: report_text.append("  Percentage Change: No data calculated.")


            report_text.append("\nWindow Slope Statistics (Ending at Removal):")
            if 'window_stats' in oa and oa['window_stats']:
                 for window_key, stats_dict in oa['window_stats'].items():
                      days = window_key.replace('d_window','')
                      if 'slope' in stats_dict:
                           slope_stats = stats_dict['slope']
                           pct_sig_str = ""
                           if 'percent_significant' in slope_stats:
                               alpha = slope_stats.get('alpha_level', self.config.get('significance_level', 0.05))
                               pct_sig_str = f" ({format_value(slope_stats['percent_significant'], '.1f')}% significant at p<{alpha})"
                           report_text.append(f"  - {days}-Day Window: Mean Slope={format_value(slope_stats.get('mean'))}{pct_sig_str}, " +
                                               f"Median={format_value(slope_stats.get('median'))}, Std={format_value(slope_stats.get('std'))}")
                      else:
                          report_text.append(f"  - {days}-Day Window: Slope data not calculated.")
                 report_text.append("  Window Average (Value):")
                 for window_key, stats_dict in oa['window_stats'].items():
                      days = window_key.replace('d_window','')
                      if 'avg' in stats_dict:
                            avg_stats = stats_dict['avg']
                            report_text.append(f"    - {days}-Day Window: Mean Avg={format_value(avg_stats.get('mean'))}, Median={format_value(avg_stats.get('median'))}, Std={format_value(avg_stats.get('std'))}")

            else: report_text.append("  Window Statistics: No data calculated.")

        report_text.append("\n*See 'descriptive_pre_outbreak_patterns.png' for visualization.*")
        report_text.append("")

        # Control Comparison Analysis 
        report_text.append("CONTROL COMPARISON ANALYSIS")
        report_text.append("-" * 80)
        cc = summary.get('control_comparison', {})
        if 'status' in cc:
            report_text.append(cc['status'])
        else:
            report_text.append(f"Control data analyzed: {cc.get('num_control_pens_analyzed', 0)} pens with {cc.get('num_control_reference_points', 0)} reference points.")
            
            # Add comparison results summary
            if 'comparison_results' in cc:
                report_text.append("\nSignificant Differences Between Tail Biting and Control Pens:")
                
                # Format significant metrics
                sig_metrics = cc.get('significant_metrics', [])
                if sig_metrics:
                    for metric in sig_metrics:
                        data = cc['comparison_results'][metric]
                        # Format metric name
                        metric_name = {
                            'value_at_removal': 'Value at Removal',
                            '3d_window_avg': '3d Window Average',
                            '7d_window_avg': '7d Window Average',
                            '3d_window_slope': '3d Window Slope',
                            '7d_window_slope': '7d Window Slope',
                            'abs_change_1d': '1d Absolute Change',
                            'abs_change_3d': '3d Absolute Change',
                            'abs_change_7d': '7d Absolute Change'
                        }.get(metric, metric)
                        
                        # Add effect size interpretation
                        effect_size = data['effect_size']
                        effect_interp = "small" if effect_size < 0.5 else "medium" if effect_size < 0.8 else "large"
                        
                        report_text.append(f"  - {metric_name}: TB={data['outbreak_mean']:.3f}±{data['outbreak_std']:.3f}, " +
                                        f"Control={data['control_mean']:.3f}±{data['control_std']:.3f}, " +
                                        f"p={data['p_value']:.4f}, {effect_interp} effect size ({effect_size:.2f})")
                else:
                    report_text.append("  No statistically significant differences were found.")
                    
            report_text.append("\n*See 'outbreak_vs_control_comparison.png' for detailed visualizations.*")
        report_text.append("")
        
        # Individual Variation Analysis
        report_text.append("INDIVIDUAL VARIATION ANALYSIS")
        report_text.append("-" * 80)
        iv = summary.get('individual_variation', {})
        if 'status' in iv:
            report_text.append(iv['status'])
        else:
            # Report pattern distribution
            if 'pattern_counts' in iv:
                report_text.append("Pattern Distribution in Outbreak Events:")
                for pattern, count in iv['pattern_counts'].items():
                    percentage = iv['pattern_percentages'][pattern]
                    report_text.append(f"  - {pattern}: {count} outbreaks ({percentage:.1f}%)")
                
            # Report metrics by pattern
            report_text.append("\nKey Metrics by Pattern Category:")
            pattern_metrics = [key for key in iv.keys() if key.endswith('_metrics')]
            
            for pattern_key in pattern_metrics:
                pattern = pattern_key.replace('_metrics', '')
                metrics = iv[pattern_key]
                report_text.append(f"  - {pattern} (n={metrics['count']}):")
                report_text.append(f"    * Value at removal: {metrics['avg_value_at_removal']:.3f}")
                report_text.append(f"    * 7-day change: {metrics['avg_abs_change_7d']:.3f}")
                report_text.append(f"    * 7-day slope: {metrics['avg_7d_slope']:.3f}")
                report_text.append(f"    * 3-day slope: {metrics['avg_3d_slope']:.3f}")
                
            report_text.append("\n*See 'individual_variation_analysis.png' for detailed visualizations.*")
        report_text.append("")
        
        # Component Analysis Section
        report_text.append("POSTURE COMPONENT ANALYSIS")
        report_text.append("-" * 80)
        ca = summary.get('component_analysis', {})
        
        if 'status' in ca:
            report_text.append(ca['status'])
        else:
            # Report component values at key timepoints
            report_text.append("Component Values at Key Timepoints:")
            for days in [0, 3, 7]:
                key = f'day_minus_{days}_stats'
                if key in ca:
                    data = ca[key]
                    timepoint = "Removal Day" if days == 0 else f"{days} Days Before Removal"
                    report_text.append(f"  - {timepoint}:")
                    report_text.append(f"    * Upright Tails: {data['upright_mean']:.3f} ± {data['upright_std']:.3f}")
                    report_text.append(f"    * Hanging Tails: {data['hanging_mean']:.3f} ± {data['hanging_std']:.3f}")
                    report_text.append(f"    * Posture Diff:  {data['diff_mean']:.3f} ± {data['diff_std']:.3f}")
            
            # Report component changes
            if 'change_stats' in ca:
                changes = ca['change_stats']
                report_text.append("\nComponent Changes (Removal vs. Earlier Timepoints):")
                if 'upright_7d_change' in changes and 'hanging_7d_change' in changes:
                    upright_change = changes['upright_7d_change']
                    hanging_change = changes['hanging_7d_change']
                    upright_dir = "increase" if upright_change > 0 else "decrease"
                    hanging_dir = "increase" if hanging_change > 0 else "decrease"
                    report_text.append(f"  - 7-Day Window: Upright tails {upright_dir} by {abs(upright_change):.3f}, " +
                                    f"Hanging tails {hanging_dir} by {abs(hanging_change):.3f}")
                    
                if 'upright_3d_change' in changes and 'hanging_3d_change' in changes:
                    upright_change = changes['upright_3d_change']
                    hanging_change = changes['hanging_3d_change']
                    upright_dir = "increase" if upright_change > 0 else "decrease"
                    hanging_dir = "increase" if hanging_change > 0 else "decrease"
                    report_text.append(f"  - 3-Day Window: Upright tails {upright_dir} by {abs(upright_change):.3f}, " +
                                    f"Hanging tails {hanging_dir} by {abs(hanging_change):.3f}")
            
            # Report contributions to posture difference change
            if 'primary_driver_7d' in ca:
                primary_driver = ca['primary_driver_7d']
                upright_contrib = ca.get('upright_contribution_7d', 0)
                hanging_contrib = ca.get('hanging_contribution_7d', 0)
                
                report_text.append("\nComponent Contributions to 7-Day Posture Difference Change:")
                report_text.append(f"  - Primary driver: {primary_driver.title()} tail posture changes")
                report_text.append(f"  - Upright tails contribution: {upright_contrib:.1f}%")
                report_text.append(f"  - Hanging tails contribution: {hanging_contrib:.1f}%")
                
            report_text.append("\n*See 'posture_component_analysis.png' for detailed visualizations.*")
        report_text.append("")

        dq = summary.get('data_quality', {})

        # Save the report
        report_path = os.path.join(self.config['output_dir'], 'analysis_summary_report.txt')
        try:
            with open(report_path, 'w') as f: f.write('\n'.join(report_text))
            self.logger.info(f"Saved summary report to {report_path}")
        except Exception as e: self.logger.error(f"Failed to save summary report to {report_path}: {e}")

        return summary

    def run_complete_analysis(self, output_dir=None):
        """Run the descriptive analysis pipeline."""
        self.logger.info("Running DESCRIPTIVE tail posture analysis pipeline...")

        if output_dir:
            self.config['output_dir'] = output_dir
            os.makedirs(self.config['output_dir'], exist_ok=True)
            self._save_config()

        # Step 1: Load data
        if self.load_data() is None:
            self.logger.critical("Data loading failed. Aborting analysis.")
            return None
        self.logger.info(f"Loaded {len(self.monitoring_results)} monitoring results entries.")

        # Step 1.5: Preprocess all data
        self.processed_results = []
        self.excluded_events_count = 0 # Initialize counter
        self.logger.info("Starting preprocessing...")
        start_time_preprocess = time.time()
        for i, result in enumerate(self.monitoring_results):
            self.logger.info(f"Preprocessing result {i+1}/{len(self.monitoring_results)}: {result.get('camera','?')}/{result.get('date_span','?')}")
            processed_data = self.preprocess_data(result)
            if processed_data:
                self.processed_results.append(processed_data)
            else:
                # Assuming preprocess_data logs the specific failure reason
                self.logger.error(f"Preprocessing failed for {result.get('camera','?')}/{result.get('date_span','?')}. Skipping.")
        self.logger.info(f"Preprocessing finished in {time.time() - start_time_preprocess:.2f} seconds.")
        self.logger.info(f"Total processed results after preprocessing: {len(self.processed_results)}")

        # Step 2: Analyze pre-outbreak statistics (descriptive)
        try:
             self.analyze_pre_outbreak_statistics()
             self.logger.info("Completed pre-outbreak statistics calculation.")
             # The excluded_events_count is now set within analyze_pre_outbreak_statistics
             self.logger.info(f"Excluded {getattr(self, 'excluded_events_count', 'N/A')} outbreak events based on missing data/validity checks.")
        except Exception as e:
             self.logger.error(f"Error during analyze_pre_outbreak_statistics: {e}", exc_info=True)
             self.pre_outbreak_stats = pd.DataFrame() # Ensure it's empty on error

        # Step 2b: Analyze control pen statistics for comparison
        try:
             self.analyze_control_pen_statistics()
             self.logger.info("Completed control pen statistics calculation.")
        except Exception as e:
             self.logger.error(f"Error during analyze_control_pen_statistics: {e}", exc_info=True)
             self.control_stats = pd.DataFrame()

        # Step 2c: Analyze individual variation in outbreaks
        pattern_results = None
        try:
            pattern_results = self.analyze_individual_outbreak_variation()
            if pattern_results and 'outbreak_patterns' in pattern_results:
                self.outbreak_patterns = pattern_results['outbreak_patterns'] # This should be the DataFrame
                self.pattern_stats = pattern_results.get('pattern_stats') # This is the aggregated stats DataFrame
                self.pen_consistency = pattern_results.get('pen_consistency', {})
                self.logger.info("Completed individual outbreak pattern analysis.")
            else:
                self.logger.warning("Individual outbreak pattern analysis did not produce expected results or had insufficient data.")
                self.outbreak_patterns = None
                self.pattern_stats = None
                self.pen_consistency = {}
        except Exception as e:
             self.logger.error(f"Error during analyze_individual_outbreak_variation: {e}", exc_info=True)
             self.outbreak_patterns = None
             self.pattern_stats = None
             self.pen_consistency = {}

        # Step 2d: Analyze upright and hanging components separately
        component_results = None
        try:
            component_results = self.analyze_posture_components()
            if component_results:
                self.component_analysis = component_results
                self.logger.info("Completed posture component analysis (upright vs. hanging tails).")
            else:
                self.logger.warning("Posture component analysis failed or had insufficient data.")
                self.component_analysis = None
        except Exception as e:
             self.logger.error(f"Error during analyze_posture_components: {e}", exc_info=True)
             self.component_analysis = None

        # Step 3: Create visualizations (descriptive)

        # Visualize Data Completeness (Robust check for processed_results)
        if hasattr(self, 'processed_results') and self.processed_results:
            completeness_vis_path = os.path.join(self.config['output_dir'], self.config.get('viz_completeness_filename', 'data_completeness_timeline.png'))
            try:
                 self.visualize_data_completeness(save_path=completeness_vis_path)
                 self.logger.info(f"Created data completeness visualization: {completeness_vis_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create data completeness visualization: {e}", exc_info=True)
        else:
             self.logger.warning("Skipping data completeness visualization as no data was successfully processed.")


        if hasattr(self, 'pre_outbreak_stats') and self.pre_outbreak_stats is not None and not self.pre_outbreak_stats.empty:
            vis_path = os.path.join(self.config['output_dir'], self.config.get('viz_pre_outbreak_filename', 'descriptive_pre_outbreak_patterns.png'))
            try:
                 self.visualize_pre_outbreak_patterns(save_path=vis_path)
                 self.logger.info(f"Created descriptive pre-outbreak patterns visualization: {vis_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create pre-outbreak patterns visualization: {e}", exc_info=True)
        else:
            self.logger.warning("Skipping pre-outbreak patterns visualization as no pre-outbreak statistics were generated.")

        # Step 3b: Create control comparison visualization and table
        if hasattr(self, 'control_stats') and self.control_stats is not None and not self.control_stats.empty \
           and hasattr(self, 'pre_outbreak_stats') and self.pre_outbreak_stats is not None and not self.pre_outbreak_stats.empty:
            comparison_vis_path = os.path.join(self.config['output_dir'], self.config.get('viz_comparison_filename', 'outbreak_vs_control_comparison.png'))
            try:
                self.visualize_comparison_with_controls(save_path=comparison_vis_path)
                self.logger.info(f"Created outbreak vs control comparison visualization: {comparison_vis_path}")
            except Exception as e:
                self.logger.error(f"Failed to create outbreak vs control comparison visualization: {e}", exc_info=True)
            self.logger.warning("Skipping control comparison visualization and table generation due to missing outbreak or control statistics.")

        # Step 3c: Create individual variation visualization and table
        # Use self.outbreak_patterns (DataFrame) for check, not self.pattern_stats (aggregated)
        if hasattr(self, 'outbreak_patterns') and self.outbreak_patterns is not None and not self.outbreak_patterns.empty:
            variation_vis_path = os.path.join(self.config['output_dir'], self.config.get('viz_variation_filename', 'individual_variation_analysis.png'))
            try:
                 self.visualize_individual_variation(save_path=variation_vis_path)
                 self.logger.info(f"Created individual variation visualization: {variation_vis_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create individual variation visualization: {e}", exc_info=True)
        else:
            self.logger.warning("Skipping individual variation visualization and table generation due to insufficient pattern data.")

        # Step 3d: Create component visualization and table
        if hasattr(self, 'component_analysis') and self.component_analysis is not None:
            component_vis_path = os.path.join(self.config['output_dir'], self.config.get('viz_components_filename', 'posture_component_analysis.png'))
            try:
                 # Pass the component_analysis dict directly
                 self.visualize_posture_components(self.component_analysis, save_path=component_vis_path)
                 self.logger.info(f"Created posture component visualization: {component_vis_path}")
            except Exception as e:
                 self.logger.error(f"Failed to create posture component visualization: {e}", exc_info=True)
                 
        # Step 4: Generate summary
        summary = None
        try:
             summary = self.generate_summary_report()
             self.logger.info("Generated descriptive summary report.")
        except Exception as e:
             self.logger.error(f"Failed to generate summary report: {e}", exc_info=True)


        self.logger.info("Descriptive analysis pipeline finished.")
        return summary