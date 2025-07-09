import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from evaluation.utils.utils import COLORS, lighten_color, set_plotting_style


class EarlyWarningVisualizer:
    """Class for visualizing early warning thresholds on tail posture data."""

    def __init__(self, early_warning_analyzer, logger=None, config=None):
        self.ewa = early_warning_analyzer
        self.analyzer = early_warning_analyzer.analyzer
        
        if logger is None:
            class PrintLogger:
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARN: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
            logger = PrintLogger()
        self.logger = logger
        
        self.config = config if config is not None else early_warning_analyzer.config

        self.path_manager = self.analyzer.path_manager

        output_dir = self.config.get('output_dir', '.')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

        set_plotting_style(self.config)
        self.logger.info(f"EarlyWarningVisualizer initialized. Visualizations will be saved to {self.viz_dir}")

    def visualize_pen(self, pen, datespan, thresholds=None, save_plot=True, show_plot=False):
        """
        Visualize the tail posture data with early warning thresholds for a specific pen.
        
        Args:
            pen (str): Pen identifier.
            datespan (str): Datespan identifier.
            thresholds (dict, optional): Dictionary of thresholds to use. If None, uses config thresholds.
            save_plot (bool, optional): Whether to save the plot to a file. Default is True.
            show_plot (bool, optional): Whether to display the plot. Default is False.
            
        Returns:
            matplotlib.figure.Figure or None: The figure if created successfully, None otherwise.
        """
        if thresholds is None:
            thresholds = self.config.get('thresholds', {
                'attention': {'posture_diff': 0.5},
                'alert': {'posture_diff': 0.4},
                'critical': {'posture_diff': 0.25}
            })

        is_outbreak = False
        removal_date = None
        pen_type = "control"

        if hasattr(self.analyzer, 'pre_outbreak_stats') and self.analyzer.pre_outbreak_stats is not None and not self.analyzer.pre_outbreak_stats.empty:
            outbreak_data = self.analyzer.pre_outbreak_stats[
                (self.analyzer.pre_outbreak_stats['pen'] == pen) &
                (self.analyzer.pre_outbreak_stats['datespan'] == datespan)
            ]
            if not outbreak_data.empty:
                is_outbreak = True
                pen_type = "tail_biting"
                removal_date_val = outbreak_data['culprit_removal_date'].iloc[0]
                if pd.notna(removal_date_val):
                    removal_date = pd.to_datetime(removal_date_val)

        pen_data_item_found = None
        for processed_data_item in self.analyzer.processed_results:
            camera_label = processed_data_item['camera'].replace("Kamera", "Pen ")
            if camera_label == pen and processed_data_item['date_span'] == datespan:
                pen_data_item_found = processed_data_item
                break

        if pen_data_item_found is None:
            self.logger.warning(f"No data found for pen {pen} / {datespan}. Cannot create visualization.")
            return None

        use_interpolated = self.config.get('use_interpolated_data', True)
        data_field = 'interpolated_data' if use_interpolated else 'resampled_data'

        if data_field not in pen_data_item_found or pen_data_item_found[data_field].empty:
            self.logger.warning(f"No {data_field} found for pen {pen} / {datespan}. Cannot create visualization.")
            return None

        data = pen_data_item_found[data_field].copy()

        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
                if not isinstance(data.index, pd.DatetimeIndex):
                    try: data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert index to DatetimeIndex for {pen} / {datespan}: {e}. Cannot create visualization.")
                        return None
            else:
                self.logger.warning(f"Cannot set datetime index for {pen} / {datespan}. 'datetime' column missing. Cannot create visualization.")
                return None
        
        data = data.sort_index()

        if data.empty:
            self.logger.warning(f"Data is empty after processing for {pen} / {datespan}. Cannot create visualization.")
            return None

        # Force time resolution based on data type
        if use_interpolated:
            # For hourly data, try to determine time resolution
            data_time_resolution = None
            if hasattr(data.index, 'freq') and data.index.freq is not None:
                try:
                    data_time_resolution = pd.tseries.frequencies.to_offset(data.index.freq)
                except ValueError:
                    self.logger.warning(f"Could not convert data.index.freq '{data.index.freq}' to offset for {pen} / {datespan}. Will try to infer.")
            
            if data_time_resolution is None and len(data.index) > 1:
                diffs = data.index.to_series().diff().dropna()
                if not diffs.empty:
                    median_diff = diffs.median()
                    if pd.notna(median_diff) and median_diff > pd.Timedelta(0):
                        data_time_resolution = median_diff
            
            if data_time_resolution is None:
                self.logger.warning(f"Could not determine data time resolution for {pen} / {datespan}. Defaulting to 1 hour.")
                data_time_resolution = pd.Timedelta(hours=1)
        else:
            # For daily data, force a daily time resolution
            data_time_resolution = pd.Timedelta(days=1)
            self.logger.info(f"Using daily time resolution for non-interpolated data for {pen} / {datespan}")

        # Get ignore_first_percent from config
        ignore_first_percent = self.config.get('ignore_first_percent', 20)
        skip_points = int(len(data) * ignore_first_percent / 100)
        min_date_data = data.index.min()

        if skip_points > 0 and len(data) > 0:
            # Index of the last data point within the skip zone
            last_idx_in_skip_zone_int = min(skip_points - 1, len(data) - 1)
            last_idx_in_skip_zone_time = data.index[last_idx_in_skip_zone_int]

            if skip_points < len(data):
                skip_zone_visual_end_time = data.index[skip_points]
            else:
                skip_zone_visual_end_time = data.index[-1] + data_time_resolution
        else:
            last_idx_in_skip_zone_time = min_date_data - pd.Timedelta(microseconds=1)
            skip_zone_visual_end_time = min_date_data

        fig, ax = plt.subplots(figsize=(12, 8))

        # Add shaded area for the ignore zone (first 20%)
        if skip_zone_visual_end_time > min_date_data:
            ax.axvspan(min_date_data, skip_zone_visual_end_time, color='#AAAAAA', alpha=0.2, zorder=0,
                    label=f"Ignorierbereich (Erste {ignore_first_percent}%)")
            ax.axvline(x=skip_zone_visual_end_time, linestyle='--', color='#555555', linewidth=1.5)

        # Calculate the max_percent_before as lookback from the end
        max_percent_before = self.config.get('max_percent_before', 60)
        
        # Get the endpoint (typically removal date for outbreak pens or the end of the data for control pens)
        endpoint = removal_date if (is_outbreak and removal_date is not None) else data.index[-1]
        
        # Calculate the lookback from the endpoint
        lookback_percent = max_percent_before  # This is now the percent to look back from the end
        total_duration = (endpoint - data.index[0]).total_seconds()
        lookback_seconds = total_duration * lookback_percent / 100
        
        # Calculate the start of analysis time (lookback from endpoint)
        if use_interpolated:
            lookback_hours = lookback_seconds / 3600
            lookback_start_time = endpoint - pd.Timedelta(hours=lookback_hours)
        else:
            lookback_days = lookback_seconds / 86400
            lookback_start_time = endpoint - pd.Timedelta(days=lookback_days)
        
        # Ensure lookback_start_time isn't before the ignore zone
        lookback_start_time = max(lookback_start_time, skip_zone_visual_end_time)
        
        # Find the index position closest to the lookback_start_time
        if lookback_start_time in data.index:
            lookback_start_index = data.index.get_loc(lookback_start_time)
        else:
            # Find closest index before lookback_start_time
            mask = data.index <= lookback_start_time
            if mask.any():
                lookback_start_index = mask.sum() - 1
            else:
                lookback_start_index = 0
        
        lookback_start_index = max(lookback_start_index, skip_points)
        
        if lookback_start_index < len(data):
            analysis_start_time = data.index[lookback_start_index]
        else:
            analysis_start_time = skip_zone_visual_end_time
            
        # Draw the analysis zone shading (between the lookback start and the endpoint)
        if endpoint > analysis_start_time:
            ax.axvspan(analysis_start_time, endpoint, 
                    color='#E6F2FF', alpha=0.2, zorder=-1,  # Light blue, behind other elements
                    label=f"Analysebereich ({max_percent_before}% Rückblick vom Ende)")
                    
        # Add a marker for the lookback start
        ax.axvline(x=analysis_start_time, linestyle=':', color='#888888', linewidth=1.0,
                zorder=1, label=f"Analyse-Start ({max_percent_before}% vom Ende)")

        if 'posture_diff' in data.columns:
            data['posture_diff'].plot(ax=ax, linewidth=2.5, color=COLORS['difference'],
                                    label='Schwanzhaltungsindex', zorder=10)
        else:
            self.logger.warning(f"'posture_diff' column not found in data for {pen} / {datespan}.")

        ax.axhline(y=thresholds['attention']['posture_diff'], linestyle='--', color=COLORS['warning'],
                label=f"Warnschwelle ({thresholds['attention']['posture_diff']})")
        ax.axhline(y=thresholds['alert']['posture_diff'], linestyle='--', color=COLORS['critical'],
                alpha=0.7, label=f"Alarmschwelle ({thresholds['alert']['posture_diff']})")
        ax.axhline(y=thresholds['critical']['posture_diff'], linestyle='--',
                color=lighten_color(COLORS['critical'], 0.2),
                label=f"Kritische Schwelle ({thresholds['critical']['posture_diff']})")

        if is_outbreak and removal_date is not None:
            ax.axvline(x=removal_date, linestyle='-', color='black', linewidth=2.5,
                    label=f"Entfernungsdatum: {removal_date.strftime('%d.%m.%Y')}")

        level_translations = {'attention': 'Warn', 'alert': 'Alarm', 'critical': 'Kritische'}
        
        threshold_mask = {
            'attention': np.zeros(len(data), dtype=bool),
            'alert': np.zeros(len(data), dtype=bool),
            'critical': np.zeros(len(data), dtype=bool)
        }
        
        for i_loop, (idx_timestamp, current_point) in enumerate(data.iterrows()):
            # Skip points in the ignore zone
            if idx_timestamp <= last_idx_in_skip_zone_time:
                continue

            # Skip points outside of our analysis window
            if idx_timestamp < analysis_start_time or idx_timestamp > endpoint:
                continue

            current_posture_diff = current_point.get('posture_diff', np.nan)
            if pd.isna(current_posture_diff): 
                continue
            
            try:
                i_idx = data.index.get_loc(idx_timestamp)
            except KeyError:
                self.logger.warning(f"Could not find index for timestamp {idx_timestamp}. Skipping this point.")
                continue

            # Check thresholds based only on posture_diff
            if current_posture_diff < thresholds['critical']['posture_diff']:
                threshold_mask['critical'][i_idx] = True
            elif current_posture_diff < thresholds['alert']['posture_diff']:
                threshold_mask['alert'][i_idx] = True
            elif current_posture_diff < thresholds['attention']['posture_diff']:
                threshold_mask['attention'][i_idx] = True

        plotted_spans_for_level = {'attention': False, 'alert': False, 'critical': False}
        for level, color_key, alpha_val in [
            ('critical', 'critical', 0.25),
            ('alert', lighten_color(COLORS['critical'], 0.3), 0.2),
            ('attention', lighten_color(COLORS['warning'], 0.5), 0.15)
        ]:
            mask = threshold_mask[level]
            color_to_use = COLORS[color_key] if color_key in COLORS else color_key
            in_span = False
            start_idx_span_plotting = -1

            for i_plot in range(len(data)):
                if mask[i_plot] and not in_span:
                    in_span = True
                    start_idx_span_plotting = i_plot
                elif not mask[i_plot] and in_span:
                    span_start_time = data.index[start_idx_span_plotting]
                    span_end_time = data.index[i_plot]

                    # For daily data, ensure clean date boundaries
                    if not use_interpolated:
                        # Convert to date and add one day for span end
                        if isinstance(span_start_time, pd.Timestamp):
                            span_start_time = pd.Timestamp(span_start_time.date())
                        if isinstance(span_end_time, pd.Timestamp):
                            span_end_time = pd.Timestamp(span_end_time.date()) + pd.Timedelta(days=1)

                    if span_start_time < span_end_time:
                        ax.axvspan(span_start_time, span_end_time, alpha=alpha_val, color=color_to_use, zorder=1,
                                label=f"{level_translations[level]}-Zone" if not plotted_spans_for_level[level] else None)
                        if not plotted_spans_for_level[level]: plotted_spans_for_level[level] = True
                    in_span = False
            
            if in_span:
                span_start_time = data.index[start_idx_span_plotting]
                # Use endpoint as the end of the span
                span_end_time_final = endpoint
                
                # For daily data, ensure clean date boundaries
                if not use_interpolated:
                    if isinstance(span_start_time, pd.Timestamp):
                        span_start_time = pd.Timestamp(span_start_time.date())
                    if isinstance(span_end_time_final, pd.Timestamp):
                        span_end_time_final = pd.Timestamp(span_end_time_final.date()) + pd.Timedelta(days=1)

                if span_start_time < span_end_time_final:
                    ax.axvspan(span_start_time, span_end_time_final, alpha=alpha_val, color=color_to_use, zorder=1,
                            label=f"{level_translations[level]}-Zone" if not plotted_spans_for_level[level] else None)
                    if not plotted_spans_for_level[level]: plotted_spans_for_level[level] = True

        critical_mask_for_hatch = threshold_mask['critical']
        in_critical_hatch_span = False
        start_hatch_idx_plotting = -1

        for i_hatch in range(len(data)):
            if critical_mask_for_hatch[i_hatch] and not in_critical_hatch_span:
                in_critical_hatch_span = True
                start_hatch_idx_plotting = i_hatch
            elif not critical_mask_for_hatch[i_hatch] and in_critical_hatch_span:
                hatch_span_start = data.index[start_hatch_idx_plotting]
                hatch_span_end = data.index[i_hatch]
                
                # For daily data, ensure clean date boundaries
                if not use_interpolated:
                    if isinstance(hatch_span_start, pd.Timestamp):
                        hatch_span_start = pd.Timestamp(hatch_span_start.date())
                    if isinstance(hatch_span_end, pd.Timestamp):
                        hatch_span_end = pd.Timestamp(hatch_span_end.date()) + pd.Timedelta(days=1)
                
                if hatch_span_start < hatch_span_end:
                    ax.axvspan(hatch_span_start, hatch_span_end,
                            alpha=0.08, hatch='///', edgecolor=COLORS['critical'], facecolor='none', zorder=2)
                in_critical_hatch_span = False
        
        if in_critical_hatch_span:
            hatch_span_start = data.index[start_hatch_idx_plotting]
            # Use endpoint as end of span
            hatch_span_end = endpoint
            
            # For daily data, ensure clean date boundaries
            if not use_interpolated:
                if isinstance(hatch_span_start, pd.Timestamp):
                    hatch_span_start = pd.Timestamp(hatch_span_start.date())
                if isinstance(hatch_span_end, pd.Timestamp):
                    hatch_span_end = pd.Timestamp(hatch_span_end.date()) + pd.Timedelta(days=1)
            
            if hatch_span_start < hatch_span_end:
                ax.axvspan(hatch_span_start, hatch_span_end,
                        alpha=0.08, hatch='///', edgecolor=COLORS['critical'], facecolor='none', zorder=2)

        formatted_datespan_for_title = datespan
        if isinstance(datespan, str) and '_' in datespan:
            try:
                start_date_str, end_date_str = datespan.split('_', 1)
                start_dt = pd.to_datetime(start_date_str, format='%y%m%d')
                end_dt = pd.to_datetime(end_date_str, format='%y%m%d')
                formatted_datespan_for_title = f"{start_dt.strftime('%d.%m.%Y')} - {end_dt.strftime('%d.%m.%Y')}"
            except Exception as e:
                self.logger.warning(f"Could not parse datespan '{datespan}' for title formatting. Error: {e}. Using original.")
        elif not isinstance(datespan, str):
            self.logger.warning(f"Datespan '{datespan}' is not a string. Using original for title.")

        ax.set_ylabel('Schwanzhaltungsindex')
        ax.set_title(f'Schwanzhaltungsindex und Schwellenwert-Alarme - {pen} ({formatted_datespan_for_title})', fontsize=14)

        formatter = mdates.DateFormatter('%d.%m.%Y')
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ymin, ymax = -0.1, 1.05 # Defaults
        if 'posture_diff' in data and not data['posture_diff'].empty:
            min_posture_val_data = data['posture_diff'].min()
            min_relevant_threshold = thresholds['critical']['posture_diff']

            effective_min_val = min(min_posture_val_data, min_relevant_threshold)
            ymin = effective_min_val - 0.1
            if effective_min_val >= 0:
                ymin = max(ymin, -0.1)
            if min_posture_val_data < -0.1:
                ymin = min_posture_val_data - 0.15

        ax.set_ylim(ymin, ymax)

        ax.legend(loc='lower left')

        pen_type_german_map = {"control": "Kontrollbucht", "tail_biting": "Schwanzbeißbucht"}
        display_pen_type = pen_type_german_map.get(pen_type, pen_type.replace('_', ' ').title())
        text_color = COLORS['control'] if pen_type == 'control' else COLORS['tail_biting']
        ax.text(0.02, 0.98, display_pen_type, transform=ax.transAxes, color=text_color, fontweight='bold',
                ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor=text_color, boxstyle='round,pad=0.3'))

        fig.tight_layout()

        if save_plot:
            filename = f"{pen.replace(' ', '_')}_{datespan}_{pen_type}_early_warning.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"Visualization saved to {filepath}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_all_outbreak_pens(self, thresholds=None, save_plot=True, show_plot=False):
        """
        Create visualizations for all outbreak pens.
        """
        figures = []
        if hasattr(self.analyzer, 'pre_outbreak_stats') and self.analyzer.pre_outbreak_stats is not None and not self.analyzer.pre_outbreak_stats.empty:
            unique_pens = self.analyzer.pre_outbreak_stats[['pen', 'datespan']].drop_duplicates()
            for _, row in unique_pens.iterrows():
                pen = row['pen']
                datespan = row['datespan']
                self.logger.info(f"Creating visualization for outbreak pen {pen} / {datespan}")
                fig = self.visualize_pen(pen, datespan, thresholds, save_plot, show_plot)
                if fig is not None:
                    figures.append(fig)
        else:
            self.logger.info("No pre_outbreak_stats found or it is empty. Cannot visualize outbreak pens.")
        return figures

    def visualize_sample_control_pens(self, num_samples=5, thresholds=None, save_plot=True, show_plot=False):
        """
        Create visualizations for a sample of control pens.
        """
        figures = []
        if hasattr(self.analyzer, 'control_stats') and self.analyzer.control_stats is not None and not self.analyzer.control_stats.empty:
            unique_pens = self.analyzer.control_stats[['pen', 'datespan']].drop_duplicates()
            if len(unique_pens) > num_samples:
                unique_pens = unique_pens.sample(n=num_samples, random_state=self.config.get('random_seed', 42))

            for _, row in unique_pens.iterrows():
                pen = row['pen']
                datespan = row['datespan']
                self.logger.info(f"Creating visualization for control pen {pen} / {datespan}")
                fig = self.visualize_pen(pen, datespan, thresholds, save_plot, show_plot)
                if fig is not None:
                    figures.append(fig)
        else:
            self.logger.info("No control_stats found or it is empty. Cannot visualize sample control pens.")
        return figures

    def visualize_sample_tail_biting_pens(self, num_samples=5, thresholds=None, save_plot=True, show_plot=False):
        """
        Create visualizations for a sample of tail biting outbreak pens.
        """
        figures = []
        if hasattr(self.analyzer, 'pre_outbreak_stats') and self.analyzer.pre_outbreak_stats is not None and not self.analyzer.pre_outbreak_stats.empty:
            unique_pens = self.analyzer.pre_outbreak_stats[['pen', 'datespan']].drop_duplicates()
            if len(unique_pens) > num_samples:
                unique_pens = unique_pens.sample(n=num_samples, random_state=self.config.get('random_seed', 42))

            for _, row in unique_pens.iterrows():
                pen = row['pen']
                datespan = row['datespan']
                self.logger.info(f"Creating visualization for tail biting pen {pen} / {datespan}")
                fig = self.visualize_pen(pen, datespan, thresholds, save_plot, show_plot)
                if fig is not None:
                    figures.append(fig)
        else:
            self.logger.info("No pre_outbreak_stats found or it is empty. Cannot visualize sample tail biting pens.")
        return figures
    
    def visualize_all_control_pens(self, thresholds=None, save_plot=True, show_plot=False):
        """
        Create visualizations for all control pens.
        """
        figures = []
        if hasattr(self.analyzer, 'control_stats') and self.analyzer.control_stats is not None and not self.analyzer.control_stats.empty:
            unique_pens = self.analyzer.control_stats[['pen', 'datespan']].drop_duplicates()
            for _, row in unique_pens.iterrows():
                pen = row['pen']
                datespan = row['datespan']
                self.logger.info(f"Creating visualization for control pen {pen} / {datespan}")
                fig = self.visualize_pen(pen, datespan, thresholds, save_plot, show_plot)
                if fig is not None:
                    figures.append(fig)
        else:
            self.logger.info("No control_stats found or it is empty. Cannot visualize control pens.")
        return figures

    def visualize_all_pens(self, thresholds=None, save_plot=True, show_plot=False):
        """
        Create visualizations for all pens (both outbreak and control).
        """
        figures = []
        all_unique_pens = set()
        
        # Collect all outbreak pens
        if hasattr(self.analyzer, 'pre_outbreak_stats') and self.analyzer.pre_outbreak_stats is not None and not self.analyzer.pre_outbreak_stats.empty:
            outbreak_pens = self.analyzer.pre_outbreak_stats[['pen', 'datespan']].drop_duplicates()
            for _, row in outbreak_pens.iterrows():
                all_unique_pens.add((row['pen'], row['datespan']))
        
        # Collect all control pens
        if hasattr(self.analyzer, 'control_stats') and self.analyzer.control_stats is not None and not self.analyzer.control_stats.empty:
            control_pens = self.analyzer.control_stats[['pen', 'datespan']].drop_duplicates()
            for _, row in control_pens.iterrows():
                all_unique_pens.add((row['pen'], row['datespan']))
        
        # Create visualizations for all pens
        if all_unique_pens:
            for pen, datespan in all_unique_pens:
                self.logger.info(f"Creating visualization for pen {pen} / {datespan}")
                fig = self.visualize_pen(pen, datespan, thresholds, save_plot, show_plot)
                if fig is not None:
                    figures.append(fig)
        else:
            self.logger.info("No pen data found. Cannot visualize any pens.")
        
        return figures
    
    def visualize_confusion_matrix(self, results, visualization_type='heatmap', 
                                include_levels=None, save_plot=True, show_plot=False,
                                figsize=None, title_suffix=""):
        """
        Visualize confusion matrices for early warning threshold evaluation results.
        
        Args:
            results (dict): Results dictionary from evaluate_thresholds method
            visualization_type (str): Type of visualization ('heatmap', 'subplots', 'metrics_comparison')
            include_levels (list, optional): List of levels to include. If None, uses all from results.
            save_plot (bool): Whether to save the plot to a file
            show_plot (bool): Whether to display the plot
            figsize (tuple, optional): Figure size. If None, uses automatic sizing
            title_suffix (str): Additional suffix for plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        import seaborn as sns
        from matplotlib.patches import Rectangle
        
        # Get included levels from results if not specified
        if include_levels is None:
            include_levels = results.get('include_levels', ['attention', 'alert', 'critical'])
        
        # Validate that requested levels exist in results
        available_levels = []
        for level in include_levels:
            level_key = f'{level}_level'
            if level_key in results:
                available_levels.append(level)
            else:
                self.logger.warning(f"Level '{level}' not found in results. Skipping.")
        
        if not available_levels:
            self.logger.error("No valid levels found in results for visualization.")
            return None
        
        # German translations
        level_translations = {
            'attention': 'Warnschwelle',
            'alert': 'Alarmschwelle', 
            'critical': 'Kritisch Schwelle'
        }
        
        if visualization_type == 'heatmap':
            return self._plot_confusion_heatmap(results, available_levels, level_translations, 
                                            save_plot, show_plot, figsize, title_suffix)
        elif visualization_type == 'subplots':
            return self._plot_confusion_subplots(results, available_levels, level_translations,
                                            save_plot, show_plot, figsize, title_suffix)
        elif visualization_type == 'metrics_comparison':
            return self._plot_metrics_comparison(results, available_levels, level_translations,
                                            save_plot, show_plot, figsize, title_suffix)
        else:
            self.logger.error(f"Unknown visualization_type: {visualization_type}")
            return None

    def _plot_confusion_heatmap(self, results, levels, level_translations, save_plot, show_plot, figsize, title_suffix):
        """Plot confusion matrices as heatmaps in subplots."""
        import seaborn as sns
        
        n_levels = len(levels)
        if figsize is None:
            figsize = (10, 8)  # Adjusted for 2-row layout
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Define confusion matrix labels in German
        cm_labels = ['Tatsächlich\nNegativ', 'Tatsächlich\nPositiv']
        pred_labels = ['Vorhergesagt Negativ', 'Vorhergesagt Positiv']
        
        # Plot positions: first two in top row, third in bottom left (centered)
        positions = [(0, 0), (0, 1), (1, 0)]
        
        for i, level in enumerate(levels):
            level_key = f'{level}_level'
            
            # Extract confusion matrix values
            tn = results[level_key]['true_negatives']
            fp = results[level_key]['false_positives']
            fn = results[level_key]['false_negatives']
            tp = results[level_key]['true_positives']
            
            # Create confusion matrix array
            cm = np.array([[tn, fp], [fn, tp]])
            
            # Get the axis for current position
            row, col = positions[i]
            ax = axes[row, col]
            
            # Use a custom colormap with our color scheme
            colors = [COLORS['background'], COLORS['secondary_metric']]
            n_bins = 100
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            # Create heatmap with annotations
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=pred_labels, yticklabels=cm_labels,
                    ax=ax, cbar_kws={'label': 'Anzahl Fälle'},
                    square=True, linewidths=1, linecolor=COLORS['annotation'])
            
            # Customize the plot (remove individual axis labels)
            ax.set_title(f'{level_translations[level]}', fontweight='bold', color=COLORS['annotation'])
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Hide the unused bottom-right subplot
        axes[1, 1].set_visible(False)
        
        # Add shared axis labels
        fig.supxlabel('Vorhersage', fontweight='bold', y=0.02)
        fig.supylabel('Realität', fontweight='bold', x=0.02)
        
        plt.tight_layout()
        
        if save_plot:
            levels_suffix = '_'.join(levels)
            filename = f"confusion_matrix_heatmap_{levels_suffix}{title_suffix.replace(' ', '_')}.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"Confusion matrix heatmap saved to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    def _plot_confusion_subplots(self, results, levels, level_translations,
                                save_plot, show_plot, figsize, title_suffix):
        """Plot detailed confusion matrix information in subplots."""
        
        n_levels = len(levels)
        if figsize is None:
            figsize = (6 * n_levels, 8)
        
        fig, axes = plt.subplots(2, n_levels, figsize=figsize)
        if n_levels == 1:
            axes = axes.reshape(2, 1)
        
        for i, level in enumerate(levels):
            level_key = f'{level}_level'
            
            # Extract values
            tn = results[level_key]['true_negatives']
            fp = results[level_key]['false_positives']
            fn = results[level_key]['false_negatives'] 
            tp = results[level_key]['true_positives']
            
            # Top subplot: Confusion matrix values as bar chart
            ax1 = axes[0, i]
            categories = ['Richtig\nNegativ', 'Falsch\nPositiv', 'Falsch\nNegativ', 'Richtig\nPositiv']
            values = [tn, fp, fn, tp]
            colors = [COLORS['control'], COLORS['warning'], COLORS['critical'], COLORS['upright']]
            
            bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor=COLORS['annotation'])
            ax1.set_title(f'{level_translations[level]} - Konfusion Matrix Werte', 
                        fontweight='bold')
            ax1.set_ylabel('Anzahl Fälle')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{val}', ha='center', va='bottom', fontweight='bold')
            
            # Bottom subplot: Performance metrics
            ax2 = axes[1, i]
            metrics = results[level_key]['metrics']
            metric_names = ['Sensitivität', 'Spezifität', 'Genauigkeit', 'F1-Score']
            metric_values = [metrics['sensitivity'], metrics['specificity'], 
                            metrics['accuracy'], metrics['f1_score']]
            
            bars2 = ax2.bar(metric_names, metric_values, color=COLORS['secondary_metric'], 
                        alpha=0.7, edgecolor=COLORS['annotation'])
            ax2.set_title(f'{level_translations[level]} - Leistungsmetriken')
            ax2.set_ylabel('Wert')
            ax2.set_ylim(0, 1.1)
            
            # Add horizontal line at 0.5 for reference
            ax2.axhline(y=0.5, color=COLORS['grid'], linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for bar, val in zip(bars2, metric_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            ax2.tick_params(axis='x', rotation=45)
        
        # Main title
        main_title = f"Detaillierte Konfusion Matrix Analyse{title_suffix}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', color=COLORS['annotation'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_plot:
            levels_suffix = '_'.join(levels)
            filename = f"confusion_matrix_detailed_{levels_suffix}{title_suffix.replace(' ', '_')}.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"Detailed confusion matrix plot saved to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    def _plot_metrics_comparison(self, results, levels, level_translations,
                                save_plot, show_plot, figsize, title_suffix):
        """Plot comparison of metrics across different threshold levels."""
        
        if figsize is None:
            figsize = (12, 8)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Prepare data
        level_names = [level_translations[level] for level in levels]
        
        # Extract metrics for all levels
        sensitivity_vals = [results[f'{level}_level']['metrics']['sensitivity'] for level in levels]
        specificity_vals = [results[f'{level}_level']['metrics']['specificity'] for level in levels]
        precision_vals = [results[f'{level}_level']['metrics']['precision'] for level in levels]
        f1_vals = [results[f'{level}_level']['metrics']['f1_score'] for level in levels]
        
        # 1. Sensitivity comparison
        bars1 = ax1.bar(level_names, sensitivity_vals, color=COLORS['upright'], alpha=0.7)
        ax1.set_title('Sensitivität (Richtig-Positiv-Rate)', fontweight='bold')
        ax1.set_ylabel('Wert')
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.5, color=COLORS['grid'], linestyle='--', alpha=0.7)
        for bar, val in zip(bars1, sensitivity_vals):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Specificity comparison  
        bars2 = ax2.bar(level_names, specificity_vals, color=COLORS['control'], alpha=0.7)
        ax2.set_title('Spezifität (Richtig-Negativ-Rate)', fontweight='bold')
        ax2.set_ylabel('Wert')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.5, color=COLORS['grid'], linestyle='--', alpha=0.7)
        for bar, val in zip(bars2, specificity_vals):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Precision comparison
        bars3 = ax3.bar(level_names, precision_vals, color=COLORS['secondary_metric'], alpha=0.7)
        ax3.set_title('Präzision (Positiv-Vorhersage-Wert)', fontweight='bold')
        ax3.set_ylabel('Wert')
        ax3.set_ylim(0, 1.1)
        ax3.axhline(y=0.5, color=COLORS['grid'], linestyle='--', alpha=0.7)
        for bar, val in zip(bars3, precision_vals):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. F1-Score comparison
        bars4 = ax4.bar(level_names, f1_vals, color=COLORS['difference'], alpha=0.7)
        ax4.set_title('F1-Score (Harmonisches Mittel)', fontweight='bold')
        ax4.set_ylabel('Wert')
        ax4.set_ylim(0, 1.1)
        ax4.axhline(y=0.5, color=COLORS['grid'], linestyle='--', alpha=0.7)
        for bar, val in zip(bars4, f1_vals):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Main title
        main_title = f"Leistungsmetriken Vergleich - Schwellenwerte{title_suffix}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold', color=COLORS['annotation'])
        
        # Add summary information
        summary_text = (f"Basis: {results['total_pens_analyzed']} Buchten "
                    f"({results['total_outbreak_pens']} Ausbruch, {results['total_control_pens']} Kontrolle)")
        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10, 
                color=COLORS['annotation'], style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08)
        
        if save_plot:
            levels_suffix = '_'.join(levels)
            filename = f"metrics_comparison_{levels_suffix}{title_suffix.replace(' ', '_')}.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"Metrics comparison plot saved to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    def visualize_roc_curve(self, results, save_plot=True, show_plot=False, 
                        figsize=(10, 8), title_suffix=""):
        """
        Visualize ROC curves for different threshold levels.
        
        Args:
            results (dict): Results dictionary from evaluate_thresholds method
            save_plot (bool): Whether to save the plot
            show_plot (bool): Whether to display the plot
            figsize (tuple): Figure size
            title_suffix (str): Additional suffix for plot title
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        
        # Get included levels from results
        include_levels = results.get('include_levels', ['attention', 'alert', 'critical'])
        
        # German translations
        level_translations = {
            'attention': 'Aufmerksamkeit',
            'alert': 'Alarm',
            'critical': 'Kritisch'
        }
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each level
        colors = [COLORS['warning'], COLORS['critical'], COLORS['secondary_metric']]
        
        for i, level in enumerate(include_levels):
            level_key = f'{level}_level'
            if level_key not in results:
                continue
                
            # Calculate TPR and FPR
            metrics = results[level_key]['metrics']
            tpr = metrics['sensitivity']  # True Positive Rate
            fpr = 1 - metrics['specificity']  # False Positive Rate = 1 - Specificity
            
            # Plot point for this threshold level
            color = colors[i % len(colors)]
            ax.scatter(fpr, tpr, color=color, s=100, 
                    label=f'{level_translations[level]} (AUC ≈ {tpr:.3f})', 
                    zorder=5, edgecolor='white', linewidth=2)
            
            # Add annotation with threshold values
            ax.annotate(f'{level_translations[level]}\n(FPR={fpr:.3f}, TPR={tpr:.3f})', 
                    xy=(fpr, tpr), xytext=(10, 10), 
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color=color))
        
        # Plot diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Zufallsklassifikator')
        
        # Customize plot
        ax.set_xlabel('Falsch-Positiv-Rate (1 - Spezifität)', fontweight='bold')
        ax.set_ylabel('Richtig-Positiv-Rate (Sensitivität)', fontweight='bold')
        ax.set_title(f'ROC Kurve - Frühwarnsystem{title_suffix}', 
                    fontweight='bold', fontsize=14)
        
        # Set equal aspect ratio and limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Add summary text
        summary_text = (f"Basis: {results['total_pens_analyzed']} Buchten "
                    f"({results['total_outbreak_pens']} Ausbruch, {results['total_control_pens']} Kontrolle)")
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            levels_suffix = '_'.join(include_levels)
            filename = f"roc_curve_{levels_suffix}{title_suffix.replace(' ', '_')}.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"ROC curve plot saved to {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def visualize_pen_clean(self, pen, datespan, thresholds=None, save_plot=True, show_plot=False):
        """
        Visualize only the tail posture data for a specific pen without thresholds or analysis zones.
        Uses the same axis formatting and limits as the regular visualize_pen method.
        
        Args:
            pen (str): Pen identifier.
            datespan (str): Datespan identifier.
            thresholds (dict, optional): Dictionary of thresholds (used only for y-axis scaling). If None, uses config thresholds.
            save_plot (bool, optional): Whether to save the plot to a file. Default is True.
            show_plot (bool, optional): Whether to display the plot. Default is False.
            
        Returns:
            matplotlib.figure.Figure or None: The figure if created successfully, None otherwise.
        """
        # Use the same threshold logic as the original method for consistent y-axis scaling
        if thresholds is None:
            thresholds = self.config.get('thresholds', {
                'attention': {'posture_diff': 0.5},
                'alert': {'posture_diff': 0.4},
                'critical': {'posture_diff': 0.25}
            })

        is_outbreak = False
        removal_date = None
        pen_type = "control"

        # Check if this is an outbreak pen
        if hasattr(self.analyzer, 'pre_outbreak_stats') and self.analyzer.pre_outbreak_stats is not None and not self.analyzer.pre_outbreak_stats.empty:
            outbreak_data = self.analyzer.pre_outbreak_stats[
                (self.analyzer.pre_outbreak_stats['pen'] == pen) &
                (self.analyzer.pre_outbreak_stats['datespan'] == datespan)
            ]
            if not outbreak_data.empty:
                is_outbreak = True
                pen_type = "tail_biting"
                removal_date_val = outbreak_data['culprit_removal_date'].iloc[0]
                if pd.notna(removal_date_val):
                    removal_date = pd.to_datetime(removal_date_val)

        # Find the pen data
        pen_data_item_found = None
        for processed_data_item in self.analyzer.processed_results:
            camera_label = processed_data_item['camera'].replace("Kamera", "Pen ")
            if camera_label == pen and processed_data_item['date_span'] == datespan:
                pen_data_item_found = processed_data_item
                break

        if pen_data_item_found is None:
            self.logger.warning(f"No data found for pen {pen} / {datespan}. Cannot create visualization.")
            return None

        # Get the data - use the EXACT same logic as the original visualize_pen method
        use_interpolated = self.config.get('use_interpolated_data', True)
        data_field = 'interpolated_data' if use_interpolated else 'resampled_data'

        if data_field not in pen_data_item_found or pen_data_item_found[data_field].empty:
            self.logger.warning(f"No {data_field} found for pen {pen} / {datespan}. Cannot create visualization.")
            return None

        data = pen_data_item_found[data_field].copy()

        # Ensure datetime index - same logic as original
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'datetime' in data.columns:
                data = data.set_index('datetime')
                if not isinstance(data.index, pd.DatetimeIndex):
                    try: 
                        data.index = pd.to_datetime(data.index)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert index to DatetimeIndex for {pen} / {datespan}: {e}. Cannot create visualization.")
                        return None
            else:
                self.logger.warning(f"Cannot set datetime index for {pen} / {datespan}. 'datetime' column missing. Cannot create visualization.")
                return None
        
        data = data.sort_index()

        if data.empty:
            self.logger.warning(f"Data is empty after processing for {pen} / {datespan}. Cannot create visualization.")
            return None

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot only the tail posture index (Schwanzhaltungsindex)
        if 'posture_diff' in data.columns:
            data['posture_diff'].plot(ax=ax, linewidth=2.5, color=COLORS['difference'],
                                    label='Schwanzhaltungsindex', zorder=10)
        else:
            self.logger.warning(f"'posture_diff' column not found in data for {pen} / {datespan}.")
            return None

        # Add removal date line if it's an outbreak pen
        if is_outbreak and removal_date is not None:
            ax.axvline(x=removal_date, linestyle='-', color='black', linewidth=2.5,
                    label=f"Entfernungsdatum: {removal_date.strftime('%d.%m.%Y')}")

        # Format datespan for title - same logic as original
        formatted_datespan_for_title = datespan
        if isinstance(datespan, str) and '_' in datespan:
            try:
                start_date_str, end_date_str = datespan.split('_', 1)
                start_dt = pd.to_datetime(start_date_str, format='%y%m%d')
                end_dt = pd.to_datetime(end_date_str, format='%y%m%d')
                formatted_datespan_for_title = f"{start_dt.strftime('%d.%m.%Y')} - {end_dt.strftime('%d.%m.%Y')}"
            except Exception as e:
                self.logger.warning(f"Could not parse datespan '{datespan}' for title formatting. Error: {e}. Using original.")
        elif not isinstance(datespan, str):
            self.logger.warning(f"Datespan '{datespan}' is not a string. Using original for title.")

        # Set labels and title
        ax.set_ylabel('Schwanzhaltungsindex')
        ax.set_title(f'Schwanzhaltungsindex - {pen} ({formatted_datespan_for_title})', fontsize=14)

        # Format x-axis - EXACT same logic as original method
        formatter = mdates.DateFormatter('%d.%m.%Y')
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Set y-axis limits - EXACT same logic as original method
        ymin, ymax = -0.1, 1.05  # Defaults
        if 'posture_diff' in data and not data['posture_diff'].empty:
            min_posture_val_data = data['posture_diff'].min()
            min_relevant_threshold = thresholds['critical']['posture_diff']

            effective_min_val = min(min_posture_val_data, min_relevant_threshold)
            ymin = effective_min_val - 0.1
            if effective_min_val >= 0:
                ymin = max(ymin, -0.1)
            if min_posture_val_data < -0.1:
                ymin = min_posture_val_data - 0.15

        ax.set_ylim(ymin, ymax)

        # Add legend only if there are multiple elements to show
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(loc='lower left')

        # Add pen type indicator - same as original
        pen_type_german_map = {"control": "Kontrollbucht", "tail_biting": "Schwanzbeißbucht"}
        display_pen_type = pen_type_german_map.get(pen_type, pen_type.replace('_', ' ').title())
        text_color = COLORS['control'] if pen_type == 'control' else COLORS['tail_biting']
        ax.text(0.02, 0.98, display_pen_type, transform=ax.transAxes, color=text_color, fontweight='bold',
                ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor=text_color, boxstyle='round,pad=0.3'))

        fig.tight_layout()

        # Save and/or show the plot
        if save_plot:
            filename = f"{pen.replace(' ', '_')}_{datespan}_{pen_type}_clean.png"
            filepath = os.path.join(self.viz_dir, filename)
            fig.savefig(filepath)
            self.logger.info(f"Clean visualization saved to {filepath}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig