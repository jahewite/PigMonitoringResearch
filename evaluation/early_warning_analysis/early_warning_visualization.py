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
        if thresholds is None:
            thresholds = self.config.get('thresholds', {
                'attention': {'posture_diff': 0.6},
                'alert': {'posture_diff': 0.5},
                'critical': {'posture_diff': 0.2}
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
            self.logger.warning(f"Could not determine data time resolution for {pen} / {datespan}. Defaulting to 1 day.")
            data_time_resolution = pd.Timedelta(days=1)

        skip_points = int(len(data) * 0.2)
        min_date_data = data.index.min()

        if skip_points > 0 and len(data) > 0 :
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

        if skip_zone_visual_end_time > min_date_data :
            ax.axvspan(min_date_data, skip_zone_visual_end_time, color='#AAAAAA', alpha=0.2, zorder=0,
                       label="Ignorierbereich (Erste 20%)")
            ax.axvline(x=skip_zone_visual_end_time, linestyle='--', color='#555555', linewidth=1.5)

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

            current_posture_diff = current_point.get('posture_diff', np.nan)
            if pd.isna(current_posture_diff): 
                continue
            
            i_idx = data.index.get_loc(idx_timestamp)

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

                    if span_start_time < span_end_time:
                        ax.axvspan(span_start_time, span_end_time, alpha=alpha_val, color=color_to_use, zorder=1,
                                   label=f"{level_translations[level]}-Zone" if not plotted_spans_for_level[level] else None)
                        if not plotted_spans_for_level[level]: plotted_spans_for_level[level] = True
                    in_span = False
            
            if in_span:
                span_start_time = data.index[start_idx_span_plotting]
                span_end_time_final = data.index[-1] + data_time_resolution 

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
                
                if hatch_span_start < hatch_span_end:
                    ax.axvspan(hatch_span_start, hatch_span_end,
                               alpha=0.08, hatch='///', edgecolor=COLORS['critical'], facecolor='none', zorder=2)
                in_critical_hatch_span = False
        
        if in_critical_hatch_span:
            hatch_span_start = data.index[start_hatch_idx_plotting]
            hatch_span_end = data.index[-1] + data_time_resolution
            
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