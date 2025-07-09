# visualization.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
import json
import os
from datetime import datetime

from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info
from evaluation.analysis.tail_posture_analysis.tail_posture_analyzer import TailPostureAnalyzer
from evaluation.utils.utils import COLORS, PATTERN_COLORS, lighten_color, set_plotting_style

class TailPostureVisualizer(TailPostureAnalyzer):
    """Methods for visualizing tail posture analysis results."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'logger'):
             self.logger = logging.getLogger(__name__)
             if not self.logger.hasHandlers():
                 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set the style upon instantiation
        set_plotting_style(self.config)
        self.logger.info("Dissertation quality plotting style set.")
        
    def log_visualization_stats(self, stats_dict, function_name, format='csv'):
        """
        Log visualization statistics to a file for reference in plot descriptions.
        """
        
        # Create output directory specifically for viz statistics
        viz_stats_dir = os.path.join(self.config['output_dir'], 'visualization_stats')
        os.makedirs(viz_stats_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{function_name}_statistics_{timestamp}.{format}"
        file_path = os.path.join(viz_stats_dir, filename)
        
        try:
            if format.lower() == 'csv':
                flat_data = self._flatten_dict(stats_dict)
                df = pd.DataFrame(flat_data.items(), columns=['Metric', 'Value'])
                df.to_csv(file_path, index=False)
            else:
                with open(file_path, 'w') as f:
                    json.dump(stats_dict, f, indent=4, default=str)
                    
            self.logger.info(f"Saved visualization statistics to {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"Failed to log visualization statistics: {e}")
            return None

    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Helper to flatten nested dictionaries for CSV output."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
    
    def _style_violin_box(self, ax, data, labels, violin_color, box_color, scatter_color, violin_width=0.6):
        """Helper to style combined violin and box plots (Dissertation Quality)."""
        if not data or all(d.empty for d in data): # Check if all data series are empty
             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color=COLORS['annotation'], fontsize=plt.rcParams['font.size'])
             return # No data to plot

        try:
            # Violin Plot
            parts = ax.violinplot(data, showmeans=False, showmedians=False, widths=violin_width)
            for pc in parts['bodies']:
                pc.set_facecolor(lighten_color(violin_color, 0.7))
                pc.set_edgecolor(violin_color)
                pc.set_alpha(0.9)
                pc.set_linewidth(0.8)

            # Explicitly set colors/alphas based on input parameters for box fill
            bplot = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                               showmeans=True,
                               widths=violin_width * 0.3,
                               positions=np.arange(1, len(data) + 1))

            for patch in bplot['boxes']:
                patch.set_facecolor(lighten_color(box_color, 0.5))
                patch.set_alpha(0.85)
                patch.set_edgecolor(box_color) # Main color for edge
                patch.set_linewidth(plt.rcParams['boxplot.boxprops.linewidth']) # Use rcParams linewidth
            # Use rcParams for whiskers and caps styling
            for whisker in bplot['whiskers']:
                whisker.set(color=plt.rcParams['axes.edgecolor'], # Match axes edge color
                           linewidth=plt.rcParams['boxplot.whiskerprops.linewidth'],
                           linestyle=plt.rcParams['boxplot.whiskerprops.linestyle'])
            for cap in bplot['caps']:
                 cap.set(color=plt.rcParams['axes.edgecolor'],
                         linewidth=plt.rcParams['boxplot.capprops.linewidth'])

            # Scatter Plot (Jitter) - Smaller, more transparent points
            for i, d in enumerate(data):
                if d is not None and not d.empty: # Check d is not None
                    x_jitter = np.random.normal(i + 1, 0.025, size=len(d)) # Smaller jitter spread
                    ax.scatter(x_jitter, d, alpha=0.3, s=10, color=scatter_color, # Smaller size, lower alpha
                               edgecolor='none', zorder=3) # No edges for less clutter
        except Exception as e:
            self.logger.error(f"Error styling violin/box plot: {e}", exc_info=True)
            ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', color='red')

    def _add_stats_annotation(self, ax, text, loc='upper right', fontsize=None, **kwargs):
        """Helper to add standardized statistics box (Dissertation Quality)."""
        if fontsize is None:
             fontsize = plt.rcParams['legend.fontsize'] # Use legend font size from rcParams

        # Slightly cleaner bbox
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, # Higher alpha
                          edgecolor='#CCCCCC', linewidth=0.5) # Lighter edge

        # Default location settings
        loc_params = {
            'upper right': {'x': 0.98, 'y': 0.98, 'ha': 'right', 'va': 'top'},
            'upper left': {'x': 0.02, 'y': 0.98, 'ha': 'left', 'va': 'top'},
            'lower right': {'x': 0.98, 'y': 0.02, 'ha': 'right', 'va': 'bottom'},
            'lower left': {'x': 0.02, 'y': 0.02, 'ha': 'left', 'va': 'bottom'},
            'center': {'x': 0.5, 'y': 0.5, 'ha': 'center', 'va': 'center'},
        }
        params = loc_params.get(loc, loc_params['upper right'])
        params.update(kwargs)

        ax.text(params['x'], params['y'], text, transform=ax.transAxes,
                fontsize=fontsize, va=params['va'], ha=params['ha'], linespacing=1.4,
                color=plt.rcParams['text.color'], # Use default text color
                bbox=bbox_props)
    
    def visualize_pre_outbreak_patterns(self, save_path=None, show=False):
        """Creates visualizations to understand pre-outbreak patterns."""
        self.logger.info("Visualizing descriptive pre-outbreak patterns...")
        set_plotting_style(self.config)

        if not hasattr(self, 'pre_outbreak_stats') or self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot visualize patterns.")
            return None

        # Calculate descriptive stats needed for plot elements (lines, percentiles)
        value_at_removal = self.pre_outbreak_stats['value_at_removal'].dropna()
        n_analyzed = len(value_at_removal)
        if n_analyzed == 0:
            self.logger.warning("No valid 'value_at_removal' data points.")
            p10, p25, mean_val, median_val = np.nan, np.nan, np.nan, np.nan
        else:
            p10 = value_at_removal.quantile(0.10)
            p25 = value_at_removal.quantile(0.25)
            mean_val = value_at_removal.mean()
            median_val = value_at_removal.median()

        fig_size = self.config.get('fig_size_pre_outbreak', (11, 10))
        fig = plt.figure(figsize=fig_size)

        # Neues GridSpec mit verändertem Layout
        # Erstes Element der height_ratios ist jetzt doppelt so hoch für die Trajektorien
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                            hspace=0.45, wspace=0.3)

        violin_width = self.config.get('violin_width', 0.6)
        base_color = COLORS.get('difference', 'blue') # Use .get for safety
        scatter_color = base_color

        # --- Panel 1 (Row 0, Col 0): Value Distributions ---
        ax0 = fig.add_subplot(gs[0, 0])
        days_list = self.config.get('days_before_list', [7, 3, 1]) # Order for plotting
        data_to_plot = []
        labels = []

        # Collect data in desired plot order (e.g., 7d, 3d, 1d, Removal)
        plot_order_days = sorted([d for d in days_list if f'value_{d}d_before' in self.pre_outbreak_stats.columns], reverse=True) # e.g. [7, 3, 1]
        for d in plot_order_days:
            col_name = f'value_{d}d_before'
            data_series = self.pre_outbreak_stats[col_name].dropna()
            if not data_series.empty:
                data_to_plot.append(data_series)
                # labels.append(f'{d}d Before')
                labels.append(f'{d}T vorher')
                
        if not value_at_removal.empty:
                data_to_plot.append(value_at_removal)
                labels.append('Bei Entfernung')

        if data_to_plot:
            # Use internal styling method if available, otherwise basic plot
            if hasattr(self, '_style_violin_box'):
                self._style_violin_box(ax0, data_to_plot, labels,
                                        violin_color=lighten_color(base_color, 0.6),
                                        box_color=lighten_color(base_color, 0.3),
                                        scatter_color=scatter_color,
                                        violin_width=violin_width)
            else: # Fallback basic plot
                ax0.violinplot(data_to_plot, showmeans=False, showmedians=True)
                ax0.boxplot(data_to_plot, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=lighten_color(base_color, 0.3)),
                            medianprops=dict(color=COLORS.get('warning','orange')))
                for i, d in enumerate(data_to_plot):
                        jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d))
                        ax0.scatter(jitter, d, alpha=0.3, s=10, color=scatter_color)
                ax0.set_xticks(np.arange(1, len(labels) + 1))
                ax0.set_xticklabels(labels)


            # Add horizontal lines for key stats (without text labels in legend)
            if pd.notna(mean_val):
                ax0.axhline(y=mean_val, color=COLORS.get('critical', 'red'), linestyle='--', linewidth=1.5, alpha=0.9, zorder=2)
            if pd.notna(median_val):
                ax0.axhline(y=median_val, color=COLORS.get('warning', 'orange'), linestyle='-', linewidth=1.5, alpha=0.9, zorder=2)
            if pd.notna(p25):
                ax0.axhline(y=p25, color=COLORS.get('secondary_metric', 'green'), linestyle='-.', linewidth=1.5, alpha=0.8, zorder=2)

        else:
            ax0.text(0.5, 0.5, 'No Data to Plot', ha='center', va='center', color=COLORS.get('annotation', 'grey'))

        # ax0.set_title('A) Posture Difference Distribution')
        ax0.set_title('A) Verteilung des Schwanzhaltungsindex')
        # ax0.set_ylabel('Posture Difference')
        ax0.set_ylabel('Schwanzhaltungsindex')
        # ax0.set_xlabel('Time Point Relative to Removal')
        ax0.set_xlabel('Zeitpunkt relativ zur Entfernung')
        ax0.tick_params(axis='x', labelsize=9)
        ax0.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
        ax0.grid(axis='x', visible=False)


        # --- Panel 2 (Row 0, Col 1): Percentage Change (ehemals Plot E) ---
        ax1 = fig.add_subplot(gs[0, 1])
        pct_changes_data = []
        pct_labels = []
        days_list = self.config.get('days_before_list', [7, 3, 1])
        plot_order_pct = sorted([d for d in days_list if f'pct_change_{d}d' in self.pre_outbreak_stats.columns], reverse=True)
        for d in plot_order_pct:
            col = f'pct_change_{d}d'
            data = self.pre_outbreak_stats[col].dropna()
            if not data.empty:
                iqr_factor = self.config.get('pct_change_outlier_iqr_factor', 3)
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - iqr_factor * iqr
                    upper_bound = q3 + iqr_factor * iqr
                    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                else:
                    filtered_data = data
                    
                if not filtered_data.empty:
                    pct_changes_data.append(filtered_data)
                    pct_labels.append(f'{d}T Fenster')

        if pct_changes_data:
            if hasattr(self, '_style_violin_box'):
                self._style_violin_box(ax1, pct_changes_data, pct_labels,
                                        violin_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.6),
                                        box_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3),
                                        scatter_color=COLORS.get('secondary_metric', 'green'),
                                        violin_width=violin_width)
            else: # Fallback basic plot
                ax1.violinplot(pct_changes_data, showmeans=False, showmedians=True)
                ax1.boxplot(pct_changes_data, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3)),
                            medianprops=dict(color=COLORS.get('warning','orange')))
                for i, d in enumerate(pct_changes_data):
                        jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d))
                        ax1.scatter(jitter, d, alpha=0.3, s=10, color=COLORS.get('secondary_metric', 'green'))
                ax1.set_xticks(np.arange(1, len(pct_labels) + 1))
                ax1.set_xticklabels(pct_labels)

            ax1.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0)
            # REMOVED: Stats annotation

        else:
            ax1.text(0.5, 0.5, 'No Percentage Change Data', ha='center', va='center', color=COLORS.get('annotation', 'grey'))

        # ax1.set_title('B) Percentage Change in Posture Difference')
        ax1.set_title('B) Symmetrische prozentuale Änderung des Schwanzhaltungsindex')
        # ax1.set_xlabel('Time Window Before Removal')
        # ax1.set_ylabel('Percentage Change (%)')
        ax1.set_xlabel('Zeitfenster vor Entfernung')
        ax1.set_ylabel('Symmetrische prozentuale Änderung (%)')
        ax1.tick_params(axis='x', labelsize=9)
        ax1.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
        ax1.grid(axis='x', visible=False)

        # --- Panel 3 (Row 2, Col 0 & Col 1): Trajectories über beide Spalten ---
        ax2 = fig.add_subplot(gs[2, :])  # Hier wird die gesamte zweite Zeile verwendet
        trajectory_data = pd.DataFrame()
        # Expand potential columns to include more days if available
        potential_traj_cols = {f'value_{d}d_before': -d for d in [10, 7, 5, 3, 1]}
        potential_traj_cols['value_at_removal'] = 0
        traj_cols_ordered_map = {col: day for col, day in potential_traj_cols.items() if col in self.pre_outbreak_stats.columns}

        plot_cols = []
        plot_days = []

        for col, day in sorted(traj_cols_ordered_map.items(), key=lambda item: item[1]): # Sort by day (-10, -7, ..., 0)
                if not self.pre_outbreak_stats[col].isnull().all(): # Check if not all NaN
                    plot_cols.append(col)
                    plot_days.append(day)
                    # Add the data to trajectory_data for calculations
                    trajectory_data[col] = self.pre_outbreak_stats[col]

        if not trajectory_data.empty and len(plot_cols) > 1:
            # Plot individual trajectories
            for i in range(len(trajectory_data)):
                values = trajectory_data.iloc[i][plot_cols].values
                valid_indices = ~np.isnan(values)
                if sum(valid_indices) > 1:
                    ax2.plot(np.array(plot_days)[valid_indices], values[valid_indices],
                                marker='.', markersize=4, alpha=0.5, color=lighten_color(COLORS.get('difference', 'blue'), 0.5),
                                linewidth=0.8, zorder=2)

            # Calculate and plot average trajectory
            avg_values = trajectory_data[plot_cols].mean().values
            std_values = trajectory_data[plot_cols].std().values
            count_values = trajectory_data[plot_cols].count().values

            valid_indices_avg = count_values > 1
            if sum(valid_indices_avg) > 1:
                plot_days_avg = np.array(plot_days)[valid_indices_avg]
                avg_values_avg = avg_values[valid_indices_avg]
                std_values_avg = std_values[valid_indices_avg]
                count_values_avg = count_values[valid_indices_avg]

                # Plot average line
                ax2.plot(plot_days_avg, avg_values_avg, marker='o', markersize=5, linewidth=2.0,
                            color=COLORS.get('critical', 'red'), label='Average Trajectory', zorder=10)

                # Calculate 95% CI
                ci_level = self.config.get('confidence_level', 0.95)
                # Use T distribution for CI if N is small, otherwise Z
                if np.min(count_values_avg) < 30: # Arbitrary threshold for T vs Z
                    t_crit = stats.t.ppf((1 + ci_level) / 2, df=count_values_avg - 1)
                    crit_val = t_crit
                else:
                    z_crit = stats.norm.ppf((1 + ci_level) / 2)
                    crit_val = z_crit
                sem = std_values_avg / np.sqrt(count_values_avg)
                margin_of_error = crit_val * sem
                upper = avg_values_avg + margin_of_error
                lower = avg_values_avg - margin_of_error
                
                ax2.fill_between(plot_days_avg, upper, lower, color=COLORS.get('critical', 'red'), alpha=0.15, zorder=9)

            # Add percentile bands (using axhspan)
            legend_handles_traj = []
            avg_line = ax2.get_lines()[-1] # Get the average line handle
            legend_handles_traj.append(avg_line)
            if len(ax2.collections) > 0: # Check if CI fill exists
                    ci_fill = ax2.collections[-1] # Get the CI fill handle
                    legend_handles_traj.append(ci_fill)


            # if pd.notna(p10):
            #     hspan1 = ax2.axhspan(ymin=ax2.get_ylim()[0], ymax=p10, color=COLORS.get('critical', 'red'), alpha=0.08, zorder=1, label=f'<10th Percentile')
            #     legend_handles_traj.append(hspan1)
            # if pd.notna(p25) and pd.notna(p10):
            #     hspan2 = ax2.axhspan(ymin=p10, ymax=p25, color=COLORS.get('warning', 'orange'), alpha=0.08, zorder=1, label=f'10-25th Percentile')
            #     legend_handles_traj.append(hspan2)

            # Add legend for lines/fills/spans
            ax2.legend(handles=legend_handles_traj, loc='lower left', frameon=True, facecolor='white', edgecolor=COLORS.get('grid', 'lightgrey'), fontsize=9)


            # Vertical grid lines for time points
            for day in plot_days: ax2.axvline(x=day, color=COLORS.get('grid', 'lightgrey'), linestyle=':', linewidth=0.8)
            ax2.set_xticks(plot_days)
            # xticklabels = [f'{abs(d)}d' if d < 0 else 'Rem.' for d in plot_days]
            xticklabels = [f'{abs(d)}T' if d < 0 else 'Entf.' for d in plot_days]
            ax2.set_xticklabels(xticklabels, rotation=0, ha='center', fontsize=9)

        else:
            ax2.text(0.5, 0.5, 'Insufficient Data for Trajectories', ha='center', va='center', color=COLORS.get('annotation', 'grey'))

        # ax2.set_title('C) Posture Difference Trajectory')
        ax2.set_title('E) Verlauf des Schwanzhaltungsindex')
        # ax2.set_xlabel('Days Before Removal')
        # ax2.set_ylabel('Posture Difference')
        ax2.set_xlabel('Tage vor Entfernung')
        ax2.set_ylabel('Schwanzhaltungsindex')
        ax2.tick_params(axis='y', labelsize=9)
        ax2.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0, zorder=1)
        ax2.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
        ax2.grid(axis='x', visible=False)


        # --- Panel 4 (Row 1, Col 0): Absolute Change ---
        ax3 = fig.add_subplot(gs[1, 0])
        abs_changes_data = []
        abs_labels = []
        plot_order_abs = sorted([d for d in days_list if f'abs_change_{d}d' in self.pre_outbreak_stats.columns], reverse=True)
        for d in plot_order_abs:
            col = f'abs_change_{d}d'
            data = self.pre_outbreak_stats[col].dropna()
            if not data.empty:
                abs_changes_data.append(data)
                # abs_labels.append(f'{d}d Window')
                abs_labels.append(f'{d}T Fenster')

        if abs_changes_data:
            if hasattr(self, '_style_violin_box'):
                self._style_violin_box(ax3, abs_changes_data, abs_labels,
                                    violin_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.6),
                                    box_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3),
                                    scatter_color=COLORS.get('secondary_metric', 'green'),
                                    violin_width=violin_width)
            else: # Fallback basic plot (similar to ax0)
                ax3.violinplot(abs_changes_data, showmeans=False, showmedians=True)
                ax3.boxplot(abs_changes_data, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3)),
                            medianprops=dict(color=COLORS.get('warning','orange')))
                for i, d in enumerate(abs_changes_data):
                        jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d))
                        ax3.scatter(jitter, d, alpha=0.3, s=10, color=COLORS.get('secondary_metric', 'green'))
                ax3.set_xticks(np.arange(1, len(abs_labels) + 1))
                ax3.set_xticklabels(abs_labels)

            ax3.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0)

        else:
            ax3.text(0.5, 0.5, 'No Absolute Change Data', ha='center', va='center', color=COLORS.get('annotation', 'grey'))

        # ax3.set_title('D) Absolute Change vs. Removal Value')
        ax3.set_title('C) Absolute Änderung vs. Entfernungswert')
        # ax3.set_xlabel('Time Window Before Removal')
        # ax3.set_ylabel('Posture Diff. Change')
        ax3.set_xlabel('Zeitfenster vor Entfernung')
        ax3.set_ylabel('Änderung des Schwanzhaltungsindex')
        ax3.tick_params(axis='x', labelsize=9)
        ax3.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
        ax3.grid(axis='x', visible=False)


        # --- Panel 5 (Row 1, Col 1): Window Slope ---
        ax4 = fig.add_subplot(gs[1, 1])
        slope_data_list = []
        slope_labels = []
        analysis_windows = self.config.get('analysis_window_days', [7, 3])
        plot_order_slope = sorted([d for d in analysis_windows if f'{d}d_window_slope' in self.pre_outbreak_stats.columns], reverse=True)
        for d in plot_order_slope:
            col = f'{d}d_window_slope'
            data = self.pre_outbreak_stats[col].dropna()
            if not data.empty:
                slope_data_list.append(data)
                # slope_labels.append(f'{d}d Window')
                slope_labels.append(f'{d}T Fenster')

        if slope_data_list:
            if hasattr(self, '_style_violin_box'):
                self._style_violin_box(ax4, slope_data_list, slope_labels,
                                        violin_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.6),
                                        box_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3),
                                        scatter_color=COLORS.get('secondary_metric', 'green'),
                                        violin_width=violin_width)
            else: # Fallback basic plot
                ax4.violinplot(slope_data_list, showmeans=False, showmedians=True)
                ax4.boxplot(slope_data_list, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3)),
                            medianprops=dict(color=COLORS.get('warning','orange')))
                for i, d in enumerate(slope_data_list):
                        jitter = np.random.normal(loc=i + 1, scale=0.04, size=len(d))
                        ax4.scatter(jitter, d, alpha=0.3, s=10, color=COLORS.get('secondary_metric', 'green'))
                ax4.set_xticks(np.arange(1, len(slope_labels) + 1))
                ax4.set_xticklabels(slope_labels)

            ax4.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0)

        else:
            ax4.text(0.5, 0.5, 'No Slope Data Calculated', ha='center', va='center', color=COLORS.get('annotation', 'grey'))

        # ax4.set_title('E) Slope in Pre-Outbreak Windows')
        ax4.set_title('D) Steigung in Vor-Ausbruch-Fenstern')
        # ax4.set_xlabel('Window Ending at Removal')
        # ax4.set_ylabel('Slope (Change per Day)')
        ax4.set_xlabel('Fenster mit Ende bei Entfernung')
        ax4.set_ylabel('Steigung (Änderung pro Tag)')
        ax4.tick_params(axis='x', labelsize=9)
        ax4.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
        ax4.grid(axis='x', visible=False)

        # --- Overall Figure Adjustments & Saving ---
        # Optional: Add overall title
        # fig.suptitle('Descriptive Pre-Outbreak Tail Posture Patterns', fontsize=16, weight='bold')
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92) # Adjust spacing

        if save_path is None:
            filename = self.config.get('viz_pre_outbreak_filename', 'descriptive_pre_outbreak_patterns.png')
            output_dir = self.config.get('output_dir', '.')
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, filename)
        try:
            dpi = self.config.get('figure_dpi', 300)
            fig.savefig(save_path, dpi=dpi)
            self.logger.info(f"Saved pre-outbreak pattern visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save pre-outbreak visualization to {save_path}: {e}")
        
        if show:
            plt.show()
        plt.close(fig)

        # Collect comprehensive statistics for logging
        stats_to_log = {
            'summary_statistics': {
                'mean_value_at_removal': float(mean_val) if pd.notna(mean_val) else None,
                'median_value_at_removal': float(median_val) if pd.notna(median_val) else None,
                'p25_value_at_removal': float(p25) if pd.notna(p25) else None,
                'p10_value_at_removal': float(p10) if pd.notna(p10) else None,
                'n_analyzed': n_analyzed
            },
            'time_point_distributions': {}
        }
        
        # Add distribution data for each time point
        for i, label in enumerate(labels):
            if i < len(data_to_plot) and len(data_to_plot[i]) > 0:
                series = data_to_plot[i]
                stats_to_log['time_point_distributions'][label] = {
                    'count': len(series),
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()) if len(series) > 1 else 0,
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'percentiles': {
                        '10': float(series.quantile(0.1)),
                        '25': float(series.quantile(0.25)),
                        '75': float(series.quantile(0.75)),
                        '90': float(series.quantile(0.9))
                    }
                }
        
        # Add percentage change data
        if pct_changes_data:
            stats_to_log['percentage_changes'] = {}
            for i, label in enumerate(pct_labels):
                if i < len(pct_changes_data) and len(pct_changes_data[i]) > 0:
                    pct_series = pct_changes_data[i]
                    stats_to_log['percentage_changes'][label] = {
                        'count': len(pct_series),
                        'mean': float(pct_series.mean()),
                        'median': float(pct_series.median()),
                        'std': float(pct_series.std()) if len(pct_series) > 1 else 0,
                        'min': float(pct_series.min()),
                        'max': float(pct_series.max()),
                        'percentiles': {
                            '25': float(pct_series.quantile(0.25)),
                            '75': float(pct_series.quantile(0.75))
                        }
                    }
        
        # Add absolute change data
        if abs_changes_data:
            stats_to_log['absolute_changes'] = {}
            for i, label in enumerate(abs_labels):
                if i < len(abs_changes_data) and len(abs_changes_data[i]) > 0:
                    abs_series = abs_changes_data[i]
                    stats_to_log['absolute_changes'][label] = {
                        'count': len(abs_series),
                        'mean': float(abs_series.mean()),
                        'median': float(abs_series.median()),
                        'std': float(abs_series.std()) if len(abs_series) > 1 else 0,
                        'min': float(abs_series.min()),
                        'max': float(abs_series.max()),
                        'percentiles': {
                            '25': float(abs_series.quantile(0.25)),
                            '75': float(abs_series.quantile(0.75))
                        }
                    }
        
        # Add slope data
        if slope_data_list:
            stats_to_log['slope_data'] = {}
            for i, label in enumerate(slope_labels):
                if i < len(slope_data_list) and len(slope_data_list[i]) > 0:
                    slope_series = slope_data_list[i]
                    stats_to_log['slope_data'][label] = {
                        'count': len(slope_series),
                        'mean': float(slope_series.mean()),
                        'median': float(slope_series.median()),
                        'std': float(slope_series.std()) if len(slope_series) > 1 else 0,
                        'min': float(slope_series.min()),
                        'max': float(slope_series.max()),
                        'percentiles': {
                            '25': float(slope_series.quantile(0.25)),
                            '75': float(slope_series.quantile(0.75))
                        }
                    }
        
        # Add trajectory data
        if not trajectory_data.empty and len(plot_cols) > 1:
            valid_indices_avg = count_values > 1
            if sum(valid_indices_avg) > 1:
                stats_to_log['trajectory'] = {
                    'days': [int(d) for d in plot_days],
                    'day_labels': [f"{abs(d)}T vorher" if d < 0 else "Bei Entfernung" for d in plot_days],
                    'average_values': avg_values.tolist() if isinstance(avg_values, np.ndarray) else [float(v) if pd.notna(v) else None for v in avg_values],
                    'std_values': std_values.tolist() if isinstance(std_values, np.ndarray) else [float(v) if pd.notna(v) else None for v in std_values],
                    'sample_counts': count_values.tolist() if isinstance(count_values, np.ndarray) else [int(v) for v in count_values],
                    'confidence_intervals': {
                        'lower': lower.tolist() if 'lower' in locals() and isinstance(lower, np.ndarray) else None,
                        'upper': upper.tolist() if 'upper' in locals() and isinstance(upper, np.ndarray) else None
                    }
                }
        
        # Log the statistics
        self.log_visualization_stats(stats_to_log, 'pre_outbreak_patterns')
        
        # Return calculated stats (used for the plot elements, could be used elsewhere too)
        return {
            'mean_value_at_removal': mean_val,
            'median_value_at_removal': median_val,
            'p25_value_at_removal': p25,
            'p10_value_at_removal': p10,
            'n_analyzed': n_analyzed
        }
        
    def _style_boxplot(self, ax, data, labels, colors, show_scatter=True, scatter_alpha=0.4, widths=0.6):
        """Helper to style box plots consistently. Returns the boxplot dictionary.""" # Added return info to docstring
        if not data:
            return None # Return None if no data

        medianprops = dict(linestyle='-', linewidth=2.0, color=COLORS['warning'])
        meanprops = dict(marker='o', markersize=5, markerfacecolor=COLORS['critical'], markeredgecolor='white')

        # Store the result of boxplot call
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                        medianprops=medianprops, meanprops=meanprops, showmeans=True,
                        widths=widths)

        # ... (rest of the styling for boxes, whiskers, caps) ...
        for i, box in enumerate(bp['boxes']):
            box_color = colors[i % len(colors)]
            box.set_facecolor(lighten_color(box_color, 0.4))
            box.set_edgecolor(box_color)
            box.set_alpha(0.8)
            box.set_linewidth(1.0)

        for whisker in bp['whiskers']: whisker.set(color=COLORS['annotation'], linewidth=1.0, linestyle='--')
        for cap in bp['caps']: cap.set(color=COLORS['annotation'], linewidth=1.0)

        if show_scatter:
            # ... (scatter plot logic) ...
            for i, d in enumerate(data):
                 if len(d) > 0:
                     scatter_color = colors[i % len(colors)]
                     x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
                     ax.scatter(x_jitter, d, alpha=scatter_alpha, s=12, color=scatter_color,
                                edgecolor='white', linewidth=0.5, zorder=3)

        return bp
    
    def visualize_comparison_with_controls(self, save_path=None, show=False):
        """Create a side-by-side visualization comparing tail biting pens to control pens."""
        self.logger.info("Visualizing comparison between outbreak and control pens...")
        set_plotting_style(self.config) # Apply style

        if self.pre_outbreak_stats is None or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot visualize comparison.")
            return None
        if self.control_stats is None or self.control_stats.empty:
            self.logger.error("No control pen statistics available. Cannot visualize comparison.")
            return None

        # --- Prepare Data (Keep existing logic) ---
        outbreak_data = self.pre_outbreak_stats.copy().reset_index(drop=True)
        control_data = self.control_stats.copy().reset_index(drop=True)
        outbreak_data['group'] = 'Tail Biting'
        control_data['group'] = 'Control'
        control_renamed = control_data.rename(columns={'value_at_reference': 'value_at_removal',
                                                'reference_date': 'culprit_removal_date'})
        # Define columns needed - use configuration instead of hardcoded lists
        plot_cols_base = ['value_at_removal', 'group', 'pen', 'datespan']

        # Get days from config
        days_before_list = self.config.get('days_before_list', [7, 3, 1])
        analysis_window_days = self.config.get('analysis_window_days', [7, 3])

        # Generate metric columns based on config
        value_cols = [f'value_{d}d_before' for d in days_before_list]
        abs_change_cols = [f'abs_change_{d}d' for d in days_before_list]
        slope_cols = [f'{d}d_window_slope' for d in analysis_window_days]
        avg_cols = [f'{d}d_window_avg' for d in analysis_window_days]  # Add window avg metrics

        # Optionally add any other metric types from config
        additional_metrics = self.config.get('additional_comparison_metrics', [])

        # Combine all metric columns
        common_cols_needed = value_cols + abs_change_cols + slope_cols + avg_cols + additional_metrics

        # Find common columns between outbreak and control data
        available_outbreak = set(outbreak_data.columns)
        available_control = set(control_renamed.columns)
        common_metrics = list(set(common_cols_needed) & available_outbreak & available_control)
        common_cols = plot_cols_base + common_metrics
        common_cols = list(set(common_cols))  # Ensure unique columns
        
        # Check if essential columns are present after intersection
        required_viz_cols = ['value_at_removal', 'group']
        if not all(col in common_cols for col in required_viz_cols):
            self.logger.error(f"Missing essential columns {required_viz_cols} for comparison viz.")
            return None

        outbreak_subset = outbreak_data[[col for col in common_cols if col in outbreak_data.columns]]
        control_subset = control_renamed[[col for col in common_cols if col in control_renamed.columns]]
        combined_data = pd.concat([outbreak_subset, control_subset], ignore_index=True)
        # --- End Data Prep ---

        fig_size = self.config.get('fig_size_comparison', (11, 10))
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], # Changed to 2x2
                            hspace=0.4, wspace=0.3) # Adjusted spacing

        group_colors = {
            'Tail Biting': COLORS['tail_biting'],
            'Control': COLORS['control']
        }
        # group_labels = {
        #     'Tail Biting': 'TB',
        #     'Control': 'Ctrl'
        # }
        
        group_labels = {
            'Tail Biting': 'SB',
            'Control': 'Ktrl'
        }

        # --- Panel A (Row 0, Col 0): Value Distribution Comparison ---
        ax0 = fig.add_subplot(gs[0, 0])
        days_list_comp = self.config.get('days_before_list', [7, 3, 1]) # Consistent order
        boxplot_data_comp = []
        boxplot_labels_comp = []
        boxplot_colors_comp = []

        # time_points = {'At Removal/Ref': 'value_at_removal'}
        # for d in days_list_comp:
        #      time_points[f'{d}d Before'] = f'value_{d}d_before'
        
        time_points = {'Bei Entfernung/Ref': 'value_at_removal'}
        for d in days_list_comp:
            time_points[f'{d}T vorher'] = f'value_{d}d_before'

        for time_label, col_name in time_points.items():
            if col_name not in combined_data.columns: continue
            for group, short_label in group_labels.items():
                data_series = combined_data.loc[combined_data['group'] == group, col_name]
                data_vals = data_series.dropna().values
                if len(data_vals) > 0:
                    boxplot_data_comp.append(data_vals)
                    boxplot_labels_comp.append(f'{short_label}: {time_label}')
                    boxplot_colors_comp.append(group_colors[group])

        if boxplot_data_comp:
            self._style_boxplot(ax0, boxplot_data_comp, boxplot_labels_comp, boxplot_colors_comp, widths=0.7)
            # Legend using patches
            # tb_patch = mpatches.Patch(color=group_colors['Tail Biting'], label='Tail Biting')
            # ctrl_patch = mpatches.Patch(color=group_colors['Control'], label='Control')
            tb_patch = mpatches.Patch(color=group_colors['Tail Biting'], label='Schwanzbeißbucht')
            ctrl_patch = mpatches.Patch(color=group_colors['Control'], label='Kontrollbucht')
            ax0.legend(handles=[tb_patch, ctrl_patch], loc='lower right', frameon=True, fontsize=9)
        else:
            ax0.text(0.5, 0.5, 'Insufficient data for boxplots', ha='center', va='center', color=COLORS['annotation'])

        # ax0.set_title('A) Posture Difference Comparison')
        # ax0.set_ylabel('Posture Difference')
        
        ax0.set_title('A) Vergleich des Schwanzhaltungsindex')
        ax0.set_ylabel('Schwanzhaltungsindex')
        
        ax0.tick_params(axis='x', rotation=45, labelsize=9)
        ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax0.grid(axis='x', b=False)


        # --- Panel B (Row 0, Col 1): Trajectory Comparison ---
        ax1 = fig.add_subplot(gs[0, 1])
        # Define days map relative to removal (0)
        trajectory_days_map = {'value_at_removal': 0}
        days_list = self.config.get('days_before_list', [7, 3, 1])
        for d in days_list:
            trajectory_days_map[f'value_{d}d_before'] = -d
        # Get columns available in data, sorted by day
        plot_cols_traj_comp = []
        plot_days_comp = []
        for col, day in trajectory_days_map.items():
            if col in combined_data.columns:
                plot_cols_traj_comp.append(col)
                plot_days_comp.append(day)
        sorted_indices_traj = np.argsort(plot_days_comp)
        plot_cols_traj_comp = [plot_cols_traj_comp[i] for i in sorted_indices_traj]
        plot_days_comp = [plot_days_comp[i] for i in sorted_indices_traj]

        plotted_traj = False
        trajectory_data_by_group = {}
        
        if len(plot_cols_traj_comp) > 1:
            for group, color in group_colors.items():
                group_data = combined_data.loc[combined_data['group'] == group]
                if not group_data.empty:
                    avg_values, std_values, x_values, n_points = [], [], [], []
                    for col, day in zip(plot_cols_traj_comp, plot_days_comp):
                        values = group_data[col].dropna()
                        if len(values) > 1: # Need >1 point for mean/std
                            avg_values.append(values.mean())
                            std_values.append(values.std())
                            x_values.append(day)
                            n_points.append(len(values))

                    if len(avg_values) > 1:
                        plotted_traj = True
                        # Store trajectory data for logging
                        trajectory_data_by_group[group] = {
                            'days': x_values,
                            'avg_values': avg_values,
                            'std_values': std_values,
                            'n_points': n_points
                        }
                        
                        # Plot average line
                        # ax1.plot(x_values, avg_values, marker='o', markersize=5, linewidth=2.0,
                        #     color=color, label=f'{group} Avg (N={len(group_data)})', zorder=10)
                        ax1.plot(x_values, avg_values, marker='o', markersize=5, linewidth=2.0,
                                color=color, label=f'{"Schwanzbeißbucht" if group == "Tail Biting" else "Kontrollbucht"} Ø (N={len(group_data)})', zorder=10)
                        # Plot CI
                        try:
                            ci_level = self.config.get('confidence_level', 0.95)
                            # Use T dist for smaller samples typical here
                            t_crit = [stats.t.ppf((1 + ci_level) / 2, n - 1) if n > 1 else 0 for n in n_points]
                            sem = [std / np.sqrt(n) if n > 0 else 0 for std, n in zip(std_values, n_points)]
                            margin_of_error = [t * s for t, s in zip(t_crit, sem)]
                            upper = [avg + moe for avg, moe in zip(avg_values, margin_of_error)]
                            lower = [avg - moe for avg, moe in zip(avg_values, margin_of_error)]
                            ax1.fill_between(x_values, upper, lower, color=color, alpha=0.15, zorder=9)
                        except Exception as e_ci:
                            self.logger.warning(f"Could not calculate CI for {group} trajectory: {e_ci}")

        if plotted_traj:
            ax1.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
            for day in plot_days_comp:
                ax1.axvline(x=day, color=COLORS['grid'], linestyle=':', linewidth=0.8)
            ax1.set_xticks(plot_days_comp)
            # ax1.set_xticklabels([f'{abs(d)}d' if d < 0 else 'Day 0' for d in plot_days_comp], rotation=0, ha='center', fontsize=9)
            ax1.set_xticklabels([f'{abs(d)}T' if d < 0 else 'Tag 0' for d in plot_days_comp], rotation=0, ha='center', fontsize=9)
            ax1.legend(loc='best', frameon=True, fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for trajectories', ha='center', va='center', color=COLORS['annotation'])

        # ax1.set_title('B) Average Trajectory Comparison')
        # ax1.set_xlabel('Days Before Removal/Reference')
        # ax1.set_ylabel('Posture Difference')
        
        ax1.set_title('B) Vergleich der durchschnittlichen Verläufe')
        ax1.set_xlabel('Tage vor Entfernung/Referenz')
        ax1.set_ylabel('Schwanzhaltungsindex')

        ax1.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax1.grid(axis='x', b=False)

        # --- Panel C (Row 1, Col 0): Absolute Change Comparison ---
        ax2 = fig.add_subplot(gs[1, 0])
        change_data_comp, change_labels_comp, change_colors_comp = [], [], []
        days_list = self.config.get('days_before_list', [7, 3, 1])
        
        change_data_by_group = {}
        
        for d in sorted(days_list, reverse=True): # Consistent order
            col = f'abs_change_{d}d'
            if col in combined_data.columns:
                for group, short_label in group_labels.items():
                    data_series = combined_data.loc[combined_data['group'] == group, col]
                    data_vals = data_series.dropna().values
                    if len(data_vals) > 0:
                        # Store data for logging
                        if group not in change_data_by_group:
                            change_data_by_group[group] = {}
                        change_data_by_group[group][f'{d}d'] = data_vals
                        
                        # Add to plot data
                        change_data_comp.append(data_vals)
                        # change_labels_comp.append(f'{short_label}: {d}d Change')
                        change_labels_comp.append(f'{short_label}: {d}T Änderung')
                        change_colors_comp.append(group_colors[group])

        if change_data_comp:
            self._style_boxplot(ax2, change_data_comp, change_labels_comp, change_colors_comp, widths=0.7)
            ax2.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0)
            # Add significance if calculated/available (similar to slope)
            # self._add_comparison_significance(ax2, combined_data, 'abs_change', [1, 3, 7], group_labels)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for change comparison', ha='center', va='center', color=COLORS['annotation'])

        # ax2.set_title('C) Absolute Change Comparison')
        # ax2.set_ylabel('Absolute Change in Posture Diff.')
        
        ax2.set_title('C) Vergleich der absoluten Änderungen')
        ax2.set_ylabel('Absolute Änderung des Schwanzhaltungsindex')

        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax2.grid(axis='x', b=False)


        # --- Panel D (Row 1, Col 1): Slope Comparison ---
        ax3 = fig.add_subplot(gs[1, 1])
        slope_data_comp, slope_labels_comp, slope_colors_comp = [], [], []
        slope_metrics_compared = []
        
        slope_data_by_group = {}
        
        analysis_windows = self.config.get('analysis_window_days', [7, 3])
        for d in sorted(analysis_windows, reverse=True): # Consistent order
            col = f'{d}d_window_slope'
            if col in combined_data.columns:
                has_data_for_metric = False
                for group, short_label in group_labels.items():
                    data_series = combined_data.loc[combined_data['group'] == group, col]
                    data_vals = data_series.dropna().values
                    if len(data_vals) > 0:
                        has_data_for_metric = True
                        
                        # Store data for logging
                        if group not in slope_data_by_group:
                            slope_data_by_group[group] = {}
                        slope_data_by_group[group][f'{d}d'] = data_vals
                        
                        # Add to plot data
                        slope_data_comp.append(data_vals)
                        # slope_labels_comp.append(f'{short_label}: {d}d Slope')
                        slope_labels_comp.append(f'{short_label}: {d}T Steigung')
                        slope_colors_comp.append(group_colors[group])
                if has_data_for_metric:
                    slope_metrics_compared.append(col)

        bp = None # Initialize bp to None
        if slope_data_comp:
            # Capture the returned bp dictionary here
            bp = self._style_boxplot(ax3, slope_data_comp, slope_labels_comp, slope_colors_comp, widths=0.7)
            ax3.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0)

            # Add significance annotations (NOW bp should be defined if plot was successful)
            comparison_results = self.compare_outbreak_vs_control_statistics()
            if comparison_results and bp is not None: # Check bp is not None
                y_lim = ax3.get_ylim()
                y_range = y_lim[1] - y_lim[0]
                sig_offset = y_range * 0.05

                for i, metric in enumerate(slope_metrics_compared):
                    if metric in comparison_results and comparison_results[metric]['is_significant']:
                        x_pos = 2 * i + 1.5
                        try:
                            idx1 = 2*i
                            idx2 = 2*i + 1
                            max_y_pair = -np.inf

                            # Use data directly for robust max finding (handles outliers better than caps)
                            if idx1 < len(slope_data_comp) and len(slope_data_comp[idx1]) > 0:
                                q1, q3 = np.percentile(slope_data_comp[idx1], [25, 75])
                                upper_whisker = q3 + 1.5 * (q3 - q1)
                                max_y_pair = max(max_y_pair, np.nanmax(slope_data_comp[idx1][slope_data_comp[idx1] <= upper_whisker]))
                            if idx2 < len(slope_data_comp) and len(slope_data_comp[idx2]) > 0:
                                q1, q3 = np.percentile(slope_data_comp[idx2], [25, 75])
                                upper_whisker = q3 + 1.5 * (q3 - q1)
                                max_y_pair = max(max_y_pair, np.nanmax(slope_data_comp[idx2][slope_data_comp[idx2] <= upper_whisker]))

                            # Optional: Use caps as a fallback or addition if data method fails often
                            if 'caps' in bp and len(bp['caps']) > idx2 * 2 + 1:
                                cap_y = bp['caps'][idx2 * 2 + 1].get_ydata()[0]
                                if max_y_pair == -np.inf: # If data method failed
                                        max_y_pair = cap_y
                                else: # Otherwise take the higher of the two
                                        max_y_pair = max(max_y_pair, cap_y)

                            if max_y_pair == -np.inf: # Final fallback
                                max_y_pair = ax3.get_ylim()[1] * 0.9

                            y_pos = max_y_pair + sig_offset

                        except (IndexError, ValueError, KeyError) as e_pos: # Catch potential errors more broadly
                            self.logger.warning(f"Could not determine optimal y-position for significance marker on {metric}: {e_pos}")
                            y_pos = ax3.get_ylim()[1] * 0.9 # Fallback position

                        #  p_val = comparison_results[metric]['p_value']
                        #  # ... (determine sig_str: ***, **, *) ...
                        #  if p_val < 0.001: sig_str = '***'
                        #  elif p_val < 0.01: sig_str = '**'
                        #  elif p_val < 0.05: sig_str = '*'
                        #  else: sig_str = 'ns' # Should not happen if is_significant is True

                        #  ax3.text(x_pos, y_pos, sig_str, ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['critical'])

        else:
            ax3.text(0.5, 0.5, 'Insufficient data for slope comparison', ha='center', va='center', color=COLORS['annotation'])

        # ax3.set_title('D) Slope Comparison Between Groups')
        # ax3.set_ylabel('Slope (Change per Day)')
        
        ax3.set_title('D) Vergleich der Steigungen zwischen Gruppen')
        ax3.set_ylabel('Steigung (Änderung pro Tag)')

        ax3.tick_params(axis='x', rotation=45, labelsize=9)
        ax3.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax3.grid(axis='x', b=False)

        # --- Overall Adjustments & Saving ---
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92) # Adjust bottom for rotated labels

        if save_path is None:
            filename = self.config.get('viz_comparison_filename', 'outbreak_vs_control_comparison.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        try:
            fig.savefig(save_path)
            self.logger.info(f"Saved outbreak vs control comparison visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save control comparison visualization to {save_path}: {e}")
        finally:
            if show:
                plt.show()
            plt.close(fig)

        # Collect statistics for logging
        stats_to_log = {
            'dataset_info': {
                'outbreak_data': {
                    'count': len(outbreak_subset),
                    'pens': len(outbreak_subset['pen'].unique()) if 'pen' in outbreak_subset.columns else 0
                },
                'control_data': {
                    'count': len(control_subset),
                    'pens': len(control_subset['pen'].unique()) if 'pen' in control_subset.columns else 0
                }
            },
            'value_comparison': {},
            'trajectory_comparison': {},
            'change_comparison': {},
            'slope_comparison': {}
        }
        
        # Add boxplot comparison data
        if boxplot_data_comp:
            for i, label in enumerate(boxplot_labels_comp):
                if i < len(boxplot_data_comp) and len(boxplot_data_comp[i]) > 0:
                    data = boxplot_data_comp[i]
                    stats_to_log['value_comparison'][label] = {
                        'count': len(data),
                        'mean': float(np.mean(data)),
                        'median': float(np.median(data)),
                        'std': float(np.std(data)) if len(data) > 1 else 0,
                        'min': float(np.min(data)),
                        'max': float(np.max(data)),
                        'percentiles': {
                            '25': float(np.percentile(data, 25)),
                            '75': float(np.percentile(data, 75))
                        }
                    }
        
        # Add trajectory comparison data
        if plotted_traj:
            for group, data in trajectory_data_by_group.items():
                group_key = "outbreak" if group == "Tail Biting" else "control"
                stats_to_log['trajectory_comparison'][group_key] = {
                    'days': data['days'],
                    'mean_values': [float(v) for v in data['avg_values']],
                    'std_values': [float(v) for v in data['std_values']],
                    'sample_counts': [int(n) for n in data['n_points']]
                }
        
        # Add change comparison data
        if change_data_by_group:
            for group, windows in change_data_by_group.items():
                group_key = "outbreak" if group == "Tail Biting" else "control"
                stats_to_log['change_comparison'][group_key] = {}
                for window, values in windows.items():
                    stats_to_log['change_comparison'][group_key][window] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)) if len(values) > 1 else 0,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'percentiles': {
                            '25': float(np.percentile(values, 25)),
                            '75': float(np.percentile(values, 75))
                        }
                    }
        
        # Add slope comparison data
        if slope_data_by_group:
            for group, windows in slope_data_by_group.items():
                group_key = "outbreak" if group == "Tail Biting" else "control"
                stats_to_log['slope_comparison'][group_key] = {}
                for window, values in windows.items():
                    stats_to_log['slope_comparison'][group_key][window] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)) if len(values) > 1 else 0,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'percentiles': {
                            '25': float(np.percentile(values, 25)),
                            '75': float(np.percentile(values, 75))
                        }
                    }
        
        # Add statistical comparison results if available
        comparison_results = self.compare_outbreak_vs_control_statistics()
        if comparison_results:
            stats_to_log['statistical_comparisons'] = comparison_results
        
        # Log the statistics
        self.log_visualization_stats(stats_to_log, 'comparison_with_controls')

        return True    
    
    def visualize_individual_variation(self, save_path=None, show=False):
        """Create visualizations showing individual variation in outbreak patterns."""
        self.logger.info("Visualizing individual variation in outbreak patterns...")
        set_plotting_style(self.config) # Apply style

        # --- Data Preparation (Keep existing logic) ---
        if not hasattr(self, 'outbreak_patterns') or self.outbreak_patterns is None:
            pattern_results = self.analyze_individual_outbreak_variation()
            if pattern_results is None:
                self.logger.error("Pattern analysis failed. Cannot visualize individual variation.")
                return None
            self.outbreak_patterns = pattern_results.get('outbreak_patterns')
            self.pattern_stats = pattern_results.get('pattern_stats', pd.DataFrame())
            self.pen_consistency = pattern_results.get('pen_consistency', {})
        else:
            pattern_results = {
                'outbreak_patterns': self.outbreak_patterns,
                'pattern_counts': self.outbreak_patterns['pattern_category'].value_counts().to_dict(),
                'pattern_stats': getattr(self, 'pattern_stats', pd.DataFrame()),
                'pen_consistency': getattr(self, 'pen_consistency', {})
            }

        outbreaks_df = pattern_results.get('outbreak_patterns')
        if outbreaks_df is None or outbreaks_df.empty:
            self.logger.error("Outbreak patterns DataFrame is missing or empty.")
            return None
        # --- End Data Prep ---

        fig_size = self.config.get('fig_size_variation', (11, 10)) # Adjusted size
        fig = plt.figure(figsize=fig_size)
        
        # Modified GridSpec layout: 2 rows x 2 cols, but bottom row spans both columns
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                            hspace=0.4, wspace=0.3)

        # Use PATTERN_COLORS defined in utils
        pattern_colors = PATTERN_COLORS
        default_color = COLORS['neutral']

        # Define trajectory columns and day mapping
        traj_cols_map = {'value_7d_before': -7, 'value_5d_before': -5, 'value_3d_before': -3,
                        'value_1d_before': -1, 'value_at_removal': 0}
        traj_cols = [col for col in traj_cols_map if col in outbreaks_df.columns]
        plot_days_ind = sorted([traj_cols_map[col] for col in traj_cols])

        # Dictionary for storing data to log
        visualization_data = {
            'panel_a': {'plotted': False, 'patterns': {}},
            'panel_b': {'plotted': False, 'avg_trajectories': {}},
            'panel_c': {'pattern_counts': {}, 'pen_consistency': {}}
        }

        # --- Panel A (Row 0, Col 0): Individual Trajectories by Pattern (MODIFIED) ---
        ax0 = fig.add_subplot(gs[0, 0])
        plotted_a = False
        if traj_cols:
            pattern_categories = outbreaks_df['pattern_category'].unique()
            valid_patterns_plotted = []
            
            # Improved random seed for reproducibility
            np.random.seed(self.config.get('random_seed', 42))

            for pattern in pattern_categories:
                pattern_outbreaks = outbreaks_df[outbreaks_df['pattern_category'] == pattern]
                if pattern_outbreaks.empty: continue

                color = pattern_colors.get(pattern, default_color)
                plotted_a = True
                valid_patterns_plotted.append(pattern)
                
                # Store pattern data for logging
                visualization_data['panel_a']['patterns'][pattern] = {
                    'count': len(pattern_outbreaks),
                    'color': color
                }
                
                # Take 3 examples from each category (or fewer if not available)
                num_examples = min(3, len(pattern_outbreaks))
                
                # Get indices of random sample
                if num_examples == len(pattern_outbreaks):
                    # Take all if 3 or fewer examples exist
                    sample_indices = pattern_outbreaks.index.tolist()
                else:
                    # Take random sample of 3 examples
                    sample_indices = np.random.choice(pattern_outbreaks.index, num_examples, replace=False)
                
                # Plot sampled trajectories with increased visibility
                for idx in sample_indices:
                    row = pattern_outbreaks.loc[idx]
                    values = [row.get(col, np.nan) for col in traj_cols]
                    valid_indices = ~np.isnan(values)
                    plot_days_row = [traj_cols_map[traj_cols[i]] for i, v in enumerate(valid_indices) if v]
                    plot_values_row = [values[i] for i, v in enumerate(valid_indices) if v]

                    if sum(valid_indices) >= 2:
                        sorted_points = sorted(zip(plot_days_row, plot_values_row))
                        plot_days_sorted, plot_values_sorted = zip(*sorted_points)
                        # Increased visibility with thicker lines and higher alpha
                        ax0.plot(plot_days_sorted, plot_values_sorted, marker='o', markersize=4, 
                                linewidth=1.5, alpha=0.7, color=color, zorder=5)

            if plotted_a:
                visualization_data['panel_a']['plotted'] = True
                # Ensure all categories are in the legend, even if not plotted
                # Old line commented out:
                # legend_elements_a = [plt.Line2D([0], [0], color=pattern_colors.get(p, default_color), lw=2, label=p)
                #                for p in valid_patterns_plotted if p in pattern_colors]
                
                # New lines to include all defined pattern categories in the legend
                pattern_categories_for_legend = ['Stabil', 'Gleichmäßige Abnahme', 'Steile Abnahme']
                legend_elements_a = [plt.Line2D([0], [0], color=pattern_colors.get(p, default_color), lw=2, label=p)
                                for p in pattern_categories_for_legend if p in pattern_colors]
                
                if legend_elements_a:
                    ax0.legend(handles=legend_elements_a, loc='lower left', frameon=True, fontsize=9)

                ax0.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
                ax0.grid(axis='x', b=False)
                ax0.set_xticks(plot_days_ind)
                ax0.set_xticklabels([f'{abs(d)}T' if d < 0 else 'Entf.' for d in plot_days_ind], fontsize=9)
            else:
                # ax0.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', color=COLORS['annotation'])
                ax0.text(0.5, 0.5, 'Keine Verlaufsdaten', ha='center', va='center', color=COLORS['annotation'])

        else:
            # ax0.text(0.5, 0.5, 'No trajectory columns', ha='center', va='center', color=COLORS['annotation'])
            ax0.text(0.5, 0.5, 'Keine Verlaufsspalten', ha='center', va='center', color=COLORS['annotation'])

        # ax0.set_title('A) Individual Outbreak Trajectories by Pattern', fontsize=11, fontweight='bold')
        ax0.set_title('A) Individuelle Ausbruch-Verläufe nach Muster', fontsize=11, fontweight='bold')
        # ax0.set_xlabel('Days before removal')
        ax0.set_xlabel('Tage vor Entfernung')
        # ax0.set_ylabel('Tail posture index')
        ax0.set_ylabel('Schwanzhaltungsindex')

        # --- Plot D moved to top right (formerly Plot B position) ---
        ax1 = fig.add_subplot(gs[0, 1])  # This is now the pattern distribution plot
        pattern_counts = pattern_results.get('pattern_counts', {})
        visualization_data['panel_c']['pattern_counts'] = pattern_counts

        if pattern_counts:
            # Order patterns for consistency if possible (e.g., severity)
            ordered_patterns = [p for p in PATTERN_COLORS if p in pattern_counts] + \
                            [p for p in pattern_counts if p not in PATTERN_COLORS] # Add any others at end
            counts = [pattern_counts[p] for p in ordered_patterns]
            bar_colors = [pattern_colors.get(p, default_color) for p in ordered_patterns]

            bars = ax1.bar(ordered_patterns, counts, color=bar_colors, edgecolor=COLORS['annotation'], linewidth=0.7)

            ax1.bar_label(bars, padding=3, fontsize=9) # Labels above bars

            # Optional: Add percentage labels inside or below
            total_counts = sum(counts)
            for bar, count in zip(bars, counts):
                if total_counts > 0:
                    percentage = (count / total_counts) * 100
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, # Center inside bar
                            f'{percentage:.0f}%', ha='center', va='center', fontsize=9, 
                            color='white', weight='bold')

            # Add pen consistency info using helper
            pen_consistency_data = pattern_results.get('pen_consistency', {})
            visualization_data['panel_c']['pen_consistency'] = pen_consistency_data

            ax1.margins(y=0.2) # Increased margin for labels

        else:
            # ax1.text(0.5, 0.5, 'No pattern counts', ha='center', va='center', color=COLORS['annotation'])
            ax1.text(0.5, 0.5, 'Keine Musteranzahlen', ha='center', va='center', color=COLORS['annotation'])

        # ax1.set_title('B) Pattern Category Distribution', fontsize=11, fontweight='bold')
        ax1.set_title('B) Verteilung der Musterkategorien', fontsize=11, fontweight='bold')
        # ax1.set_ylabel('Number of outbreaks')
        ax1.set_ylabel('Anzahl der Ausbrüche')
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax1.grid(axis='x', b=False)

        # --- Plot B moved to bottom row spanning both columns (formerly Plot B) ---
        ax2 = fig.add_subplot(gs[1, :])  # Span both columns in bottom row
        plotted_b = False
        if traj_cols:
            legend_handles_b = []
            pattern_categories = outbreaks_df['pattern_category'].unique()

            for pattern in pattern_categories:
                pattern_outbreaks = outbreaks_df[outbreaks_df['pattern_category'] == pattern]
                if len(pattern_outbreaks) < 2: continue # Need >1 for mean/std

                color = pattern_colors.get(pattern, default_color)
                avg_values, std_values, avg_days, n_points = [], [], [], []

                for col in traj_cols:
                    values = pattern_outbreaks[col].dropna()
                    if len(values) > 1: # Need >1 point
                        avg_values.append(values.mean())
                        std_values.append(values.std())
                        avg_days.append(traj_cols_map[col])
                        n_points.append(len(values))

                if len(avg_values) > 1:
                    plotted_b = True
                    # Sort by day before plotting average
                    sorted_avg_points = sorted(zip(avg_days, avg_values, std_values, n_points))
                    avg_days_sorted, avg_values_sorted, std_values_sorted, n_points_sorted = zip(*sorted_avg_points)

                    # Store average trajectory data for logging
                    visualization_data['panel_b']['avg_trajectories'][pattern] = {
                        'days': list(avg_days_sorted),
                        'avg_values': list(avg_values_sorted),
                        'std_values': list(std_values_sorted),
                        'n_points': list(n_points_sorted),
                        'total_outbreaks': len(pattern_outbreaks)
                    }
                    
                    # Enhanced line thickness and markers for better visibility
                    line, = ax2.plot(avg_days_sorted, avg_values_sorted, marker='o', markersize=7,
                                linewidth=2.5, color=color, label=f'{pattern} (N={len(pattern_outbreaks)})', zorder=10)
                    
                    legend_handles_b.append(line)

                    # Add CI fill
                    try:
                        ci_level = 0.95
                        t_crit = [stats.t.ppf((1 + ci_level) / 2, n - 1) if n > 1 else 0 for n in n_points_sorted]
                        sem = [std / np.sqrt(n) if n > 0 else 0 for std, n in zip(std_values_sorted, n_points_sorted)]
                        margin_of_error = [t * s for t, s in zip(t_crit, sem)]
                        upper = [avg + moe for avg, moe in zip(avg_values_sorted, margin_of_error)]
                        lower = [avg - moe for avg, moe in zip(avg_values_sorted, margin_of_error)]
                        ax2.fill_between(avg_days_sorted, upper, lower, color=color, alpha=0.2, zorder=9)
                    except Exception as e_ci_b:
                        self.logger.warning(f"Could not plot CI for {pattern}: {e_ci_b}")

            if plotted_b:
                visualization_data['panel_b']['plotted'] = True
                ax2.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                for day in plot_days_ind: ax2.axvline(x=day, color=COLORS['grid'], linestyle=':', linewidth=0.8, alpha=0.5)
                ax2.grid(False, axis='x') # Only vertical lines shown above
                ax2.grid(True, axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
                ax2.set_xticks(plot_days_ind)
                ax2.set_xticklabels([f'{abs(d)}T' if d < 0 else 'Entf.' for d in plot_days_ind], fontsize=10)
                if legend_handles_b: 
                    ax2.legend(handles=legend_handles_b, loc='upper right', frameon=True, fontsize=10)
            else:
                # ax2.text(0.5, 0.5, 'Insufficient data for avg trajectories', ha='center', va='center', color=COLORS['annotation'])
                ax2.text(0.5, 0.5, 'Unzureichende Daten für Durchschnittswerte', ha='center', va='center', color=COLORS['annotation'])

        else:
            # ax2.text(0.5, 0.5, 'No trajectory columns', ha='center', va='center', color=COLORS['annotation'])
            ax2.text(0.5, 0.5, 'Keine Verlaufsspalten', ha='center', va='center', color=COLORS['annotation'])

        # ax2.set_title('C) Average Trajectories by Pattern Type', fontsize=11, fontweight='bold')
        ax2.set_title('C) Durchschnittliche Verläufe nach Mustertyp', fontsize=11, fontweight='bold')
        # ax2.set_xlabel('Days before removal', fontsize=10)
        ax2.set_xlabel('Tage vor Entfernung', fontsize=10)
        # ax2.set_ylabel('Tail posture index', fontsize=10)
        ax2.set_ylabel('Schwanzhaltungsindex', fontsize=10)

        # --- Layout & Saving ---
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.92) # Adjusted spacing

        if save_path is None:
            filename = self.config.get('viz_variation_filename', 'individual_variation_analysis.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        try:
            dpi = self.config.get('figure_dpi', 300)
            fig.savefig(save_path, dpi=dpi)
            self.logger.info(f"Saved individual variation visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save individual variation visualization: {e}")
        finally:
            if show:
                plt.show()
            plt.close(fig)

        # Collect statistics for logging
        stats_to_log = {
            'dataset_info': {
                'outbreak_patterns': {
                    'count': len(outbreaks_df),
                    'pens': len(outbreaks_df['pen'].unique()) if 'pen' in outbreaks_df.columns else 0,
                    'pattern_categories': list(pattern_counts.keys()) if pattern_counts else []
                }
            },
            'pattern_distribution': pattern_counts,
            'pen_consistency': pen_consistency_data,
            'visualization_data': visualization_data
        }
        
        # Log the statistics
        self.log_visualization_stats(stats_to_log, 'individual_variation')

        return True
        
    def visualize_posture_components(self, component_analysis=None, save_path=None, show=False):
        """Create visualizations showing the individual components (Revised to include components in Panel B)."""
        self.logger.info("Visualizing posture components (upright vs. hanging tails)...")
        set_plotting_style(self.config) # Apply style

        # --- Data Prep (Keep existing logic) ---
        if component_analysis is None:
            # Assuming self.analyze_posture_components exists and returns the dict
            component_analysis = self.analyze_posture_components()
        if component_analysis is None:
            self.logger.error("Component analysis failed. Cannot visualize.")
            return None

        outbreak_components = component_analysis.get('outbreak_components')
        control_components = component_analysis.get('control_components')
        change_stats = component_analysis.get('change_stats')
        contribution_stats = component_analysis.get('contribution_stats')

        if outbreak_components is None or outbreak_components.empty:
            self.logger.error("No outbreak component data available.")
            return None

        has_controls = control_components is not None and not control_components.empty
        # --- End Data Prep ---

        fig_size = self.config.get('fig_size_components', (11, 10))
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], width_ratios=[1],
                            hspace=0.4, wspace=0.3)

        # --- Panel A (Row 0, Col 0): Outbreak Component Trajectories ---
        ax0 = fig.add_subplot(gs[0, 0])
        avg_by_day_outbreak = outbreak_components.groupby('days_before_removal').agg({
            'upright_tails': ['mean', 'std', 'count'],
            'hanging_tails': ['mean', 'std', 'count'],
            'posture_diff': ['mean', 'std', 'count']
        })
        days_outbreak = sorted(outbreak_components['days_before_removal'].unique(), reverse=True)
        days_x_outbreak = [-d for d in days_outbreak]
        plotted_a = False
        
        metric_translations = {
            'upright_tails': 'Aufrechte Schwänze',
            'hanging_tails': 'Hängende Schwänze',
            'posture_diff': 'Schwanzhaltungsindex'
        }
        
        outbreak_metric_data = {}
        
        if not avg_by_day_outbreak.empty and len(days_outbreak) > 1:
            plotted_a = True
            for metric, color_name in [('upright_tails', 'upright'), ('hanging_tails', 'hanging'), ('posture_diff', 'difference')]:
                means = avg_by_day_outbreak[(metric, 'mean')].reindex(days_outbreak).values
                stds = avg_by_day_outbreak[(metric, 'std')].reindex(days_outbreak).values
                counts = avg_by_day_outbreak[(metric, 'count')].reindex(days_outbreak).values
                valid_idx = (counts > 1) & ~np.isnan(means)
                
                # Store data for logging
                outbreak_metric_data[metric] = {
                    'days': [int(-d) for d in np.array(days_outbreak)[valid_idx]],
                    'means': [float(v) for v in means[valid_idx]],
                    'stds': [float(v) for v in stds[valid_idx]],
                    'counts': [int(v) for v in counts[valid_idx]]
                }
                
                if np.sum(valid_idx) > 1:
                    x_plot = np.array(days_x_outbreak)[valid_idx]
                    means_plot = means[valid_idx]
                    stds_plot = stds[valid_idx]
                    counts_plot = counts[valid_idx]
                    # label = metric.replace('_', ' ').title()
                    label = metric_translations.get(metric, metric.replace('_', ' ').title())
                    color = COLORS[color_name]
                    line, = ax0.plot(x_plot, means_plot, 'o-', color=color, label=label, linewidth=1.8, markersize=4)
                    try:
                        t_crit = stats.t.ppf((1 + 0.95) / 2, counts_plot - 1)
                        sem = stds_plot / np.sqrt(counts_plot)
                        moe = t_crit * sem
                        ax0.fill_between(x_plot, means_plot - moe, means_plot + moe, alpha=0.15, color=color)
                    except Exception as e_ci:
                        self.logger.warning(f"CI calculation failed for {metric} in Panel A: {e_ci}")
            ax0.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0)
            ax0.axvline(x=0, color=COLORS['annotation'], linestyle=':', linewidth=1.0, alpha=0.7)
            # ax0.text(0.1, ax0.get_ylim()[1]*0.95, 'Removal', ha='left', va='top', fontsize=8, color=COLORS['annotation'], rotation=90)
            ax0.text(0.1, ax0.get_ylim()[1]*0.95, 'Entfernung', ha='left', va='top', fontsize=8, color=COLORS['annotation'], rotation=90)

            ax0.legend(loc='best', frameon=True, fontsize=9)
            # n_outbreak_pens = len(outbreak_components['pen'].unique())
            # self._add_stats_annotation(ax0, f"N(Outbreak Pens) = {n_outbreak_pens}", loc='upper left', fontsize=9)
        else:
            ax0.text(0.5, 0.5, 'Insufficient Outbreak Data', ha='center', va='center', color=COLORS['annotation'])
        # ax0.set_title('A) Posture Component Trajectories (Outbreaks)')
        ax0.set_title('A) Verlauf der Schwanzhaltungskomponenten (Ausbrüche)')
        # ax0.set_xlabel('Days Before Removal')
        # ax0.set_ylabel('Normalized Value')
        ax0.set_xlabel('Tage vor Entfernung')
        ax0.set_ylabel('Normalisierter Wert')
        ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax0.grid(axis='x', b=False)


        # --- Panel B (Row 0, Col 1): Control vs. Outbreak Comparison (Diff & Components) --- MODIFIED ---
        ax1 = fig.add_subplot(gs[1, 0])
        plotted_b = False
        comparison_data = {}
        if has_controls:
            # Calculate averages for controls INCLUDING COMPONENTS
            avg_by_day_control = control_components.groupby('days_before_removal').agg({
                'upright_tails': ['mean', 'std', 'count'], # ADDED
                'hanging_tails': ['mean', 'std', 'count'], # ADDED
                'posture_diff': ['mean', 'std', 'count']
            })
            days_control = sorted(control_components['days_before_removal'].unique(), reverse=True)
            # Use common days that exist in BOTH outbreak and control data
            common_days = sorted(list(set(days_outbreak) & set(days_control)), reverse=True)
            common_days_x = [-d for d in common_days]

            if len(common_days) > 1:
                # --- Extract Data for Common Days ---
                # Initialize dicts to store extracted data
                data_comp = {'outbreak': {}, 'control': {}}
                metrics_to_plot = ['posture_diff', 'upright_tails', 'hanging_tails']

                for group, avg_data, source_counts in [('outbreak', avg_by_day_outbreak, outbreak_components),
                                                        ('control', avg_by_day_control, control_components)]:
                    group_counts = avg_data.xs('count', axis=1, level=1) # Get counts for all metrics
                    # Check if counts for diff > 1 on common days
                    valid_counts_diff = group_counts['posture_diff'].reindex(common_days).fillna(0) > 1
                    data_comp[group]['n_pens'] = len(source_counts['pen'].unique())

                    for metric in metrics_to_plot:
                        if (metric, 'mean') in avg_data.columns:
                            means = avg_data[(metric, 'mean')].reindex(common_days).values
                            stds = avg_data[(metric, 'std')].reindex(common_days).values
                            # Ensure counts are also reindexed and valid
                            counts = group_counts[metric].reindex(common_days).fillna(0).values

                            # Apply the validity mask derived from posture_diff counts (or could use metric's own counts)
                            means[~valid_counts_diff] = np.nan
                            stds[~valid_counts_diff] = np.nan
                            counts[~valid_counts_diff] = 0 # Set count to 0 if invalid

                            data_comp[group][metric] = {'mean': means, 'std': stds, 'count': counts}
                        else:
                            # Fill with NaNs if metric doesn't exist in avg_data
                            data_comp[group][metric] = {'mean': np.full(len(common_days), np.nan),
                                                        'std': np.full(len(common_days), np.nan),
                                                        'count': np.zeros(len(common_days))}
                
                # Store data for logging
                comparison_data = {
                    'common_days': common_days,
                    'days_x': common_days_x,
                    'data_comp': data_comp
                }

                valid_indices_plot = ~np.isnan(data_comp['outbreak']['posture_diff']['mean']) & \
                                    ~np.isnan(data_comp['control']['posture_diff']['mean']) & \
                                    (data_comp['outbreak']['posture_diff']['count'] > 1) & \
                                    (data_comp['control']['posture_diff']['count'] > 1)

                if np.any(valid_indices_plot): # Use any() instead of sum() > 1 for boolean array
                    plotted_b = True
                    x_plot_comp = np.array(common_days_x)[valid_indices_plot] # X-axis for valid points

                    # --- Plot Difference Lines (Solid, Main focus) ---
                    # Outbreak Diff
                    means_o_diff = data_comp['outbreak']['posture_diff']['mean'][valid_indices_plot]
                    stds_o_diff = data_comp['outbreak']['posture_diff']['std'][valid_indices_plot]
                    counts_o_diff = data_comp['outbreak']['posture_diff']['count'][valid_indices_plot]
                    ax1.plot(x_plot_comp, means_o_diff, 'o-', color=COLORS['critical'], label='Outbreak Diff', linewidth=2.0, markersize=5, zorder=10)
                    try: # CI for Outbreak Diff
                        t_crit_o = stats.t.ppf((1 + 0.95) / 2, counts_o_diff - 1)
                        moe_o = t_crit_o * (stds_o_diff / np.sqrt(counts_o_diff))
                        ax1.fill_between(x_plot_comp, means_o_diff - moe_o, means_o_diff + moe_o, alpha=0.15, color=COLORS['critical'], zorder=9)
                    except Exception: pass

                    # Control Diff
                    means_c_diff = data_comp['control']['posture_diff']['mean'][valid_indices_plot]
                    stds_c_diff = data_comp['control']['posture_diff']['std'][valid_indices_plot]
                    counts_c_diff = data_comp['control']['posture_diff']['count'][valid_indices_plot]
                    ax1.plot(x_plot_comp, means_c_diff, 's-', color=COLORS['control'], label='Control Diff', linewidth=2.0, markersize=5, zorder=10) # Different marker
                    try: # CI for Control Diff
                        t_crit_c = stats.t.ppf((1 + 0.95) / 2, counts_c_diff - 1)
                        moe_c = t_crit_c * (stds_c_diff / np.sqrt(counts_c_diff))
                        ax1.fill_between(x_plot_comp, means_c_diff - moe_c, means_c_diff + moe_c, alpha=0.15, color=COLORS['control'], zorder=9)
                    except Exception: pass

                    # --- Plot Component Lines (Dashed, Secondary focus) ---
                    # Outbreak Components
                    means_o_up = data_comp['outbreak']['upright_tails']['mean'][valid_indices_plot]
                    means_o_hang = data_comp['outbreak']['hanging_tails']['mean'][valid_indices_plot]
                    ax1.plot(x_plot_comp, means_o_up, 'o--', color=COLORS['upright'], label='_nolegend_', linewidth=1.5, markersize=3, alpha=0.7, zorder=5)
                    ax1.plot(x_plot_comp, means_o_hang, 'o--', color=COLORS['hanging'], label='_nolegend_', linewidth=1.5, markersize=3, alpha=0.7, zorder=5)

                    # Control Components
                    means_c_up = data_comp['control']['upright_tails']['mean'][valid_indices_plot]
                    means_c_hang = data_comp['control']['hanging_tails']['mean'][valid_indices_plot]
                    ax1.plot(x_plot_comp, means_c_up, 's--', color=COLORS['upright'], label='_nolegend_', linewidth=1.5, markersize=3, alpha=0.5, zorder=5) # Lower alpha
                    ax1.plot(x_plot_comp, means_c_hang, 's--', color=COLORS['hanging'], label='_nolegend_', linewidth=1.5, markersize=3, alpha=0.5, zorder=5) # Lower alpha

                    # --- Axes and Legend ---
                    ax1.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0)
                    ax1.axvline(x=0, color=COLORS['annotation'], linestyle=':', linewidth=1.0, alpha=0.7)

                    # Create Custom Legend
                    #  handles = [
                    #      plt.Line2D([0], [0], color=COLORS['critical'], linewidth=2.0, marker='o', markersize=5, label='Outbreak Diff'),
                    #      plt.Line2D([0], [0], color=COLORS['control'], linewidth=2.0, marker='s', markersize=5, label='Control Diff'),
                    #      plt.Line2D([0], [0], color=COLORS['upright'], linestyle='--', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Outbreak Upright'),
                    #      plt.Line2D([0], [0], color=COLORS['hanging'], linestyle='--', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Outbreak Hanging'),
                    #      plt.Line2D([0], [0], color=COLORS['upright'], linestyle='--', linewidth=1.5, marker='s', markersize=3, alpha=0.5, label='Control Upright'),
                    #      plt.Line2D([0], [0], color=COLORS['hanging'], linestyle='--', linewidth=1.5, marker='s', markersize=3, alpha=0.5, label='Control Hanging'),
                    #  ]
                    handles = [
                        plt.Line2D([0], [0], color=COLORS['critical'], linewidth=2.0, marker='o', markersize=5, label='Ausbruch Index'),
                        plt.Line2D([0], [0], color=COLORS['control'], linewidth=2.0, marker='s', markersize=5, label='Kontrolle Index'),
                        plt.Line2D([0], [0], color=COLORS['upright'], linestyle='--', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Ausbruch Aufrecht'),
                        plt.Line2D([0], [0], color=COLORS['hanging'], linestyle='--', linewidth=1.5, marker='o', markersize=3, alpha=0.7, label='Ausbruch Hängend'),
                        plt.Line2D([0], [0], color=COLORS['upright'], linestyle='--', linewidth=1.5, marker='s', markersize=3, alpha=0.5, label='Kontrolle Aufrecht'),
                        plt.Line2D([0], [0], color=COLORS['hanging'], linestyle='--', linewidth=1.5, marker='s', markersize=3, alpha=0.5, label='Kontrolle Hängend'),
                    ]
                    ax1.legend(handles=handles, loc='lower left', frameon=True, fontsize=8) # Smaller font size

                    # Add N counts
                    # n_outbreak = data_comp['outbreak']['n_pens']
                    # n_control = data_comp['control']['n_pens']
                    # self._add_stats_annotation(ax1, f"N(Outbreak)={n_outbreak}\nN(Control)={n_control}", loc='upper left', fontsize=8)

        if not plotted_b:
            ax1.text(0.5, 0.5, 'Insufficient Data for Comparison', ha='center', va='center', color=COLORS['annotation'])

        # ax1.set_title('B) Posture Diff & Components: Outbreak vs. Control')
        # ax1.set_xlabel('Days Before Removal/Reference')
        # ax1.set_ylabel('Normalized Value')
        
        ax1.set_title('B) Schwanzhaltungsindex & Komponenten: Ausbruch vs. Kontrolle')
        ax1.set_xlabel('Tage vor Entfernung/Referenz')
        ax1.set_ylabel('Normalisierter Wert')
        
        ax1.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax1.grid(axis='x', b=False)

        # --- Layout & Saving ---
        fig.subplots_adjust(left=0.1, right=0.88, bottom=0.1, top=0.92) # Adjust right for legend

        if save_path is None:
            filename = self.config.get('viz_components_filename', 'posture_component_analysis.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        try:
            fig.savefig(save_path) # Use default DPI from rcParams
            self.logger.info(f"Saved posture component visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save posture component visualization: {e}")
        finally:
            if show:
                plt.show()
            plt.close(fig)
        
        # Collect statistics for logging
        stats_to_log = {
            'dataset_info': {
                'outbreak_components': {
                    'count': len(outbreak_components) if outbreak_components is not None else 0,
                    'pens': len(outbreak_components['pen'].unique()) if outbreak_components is not None else 0,
                },
                'control_components': {
                    'count': len(control_components) if has_controls else 0,
                    'pens': len(control_components['pen'].unique()) if has_controls else 0,
                }
            },
            'outbreak_trajectories': {},
            'comparison_data': {}
        }
        
        # Add outbreak component trajectories (Panel A data)
        if plotted_a:
            for metric, data in outbreak_metric_data.items():
                stats_to_log['outbreak_trajectories'][metric] = data
        
        # Add comparison data (Panel B data)
        if plotted_b and comparison_data:
            common_days = comparison_data['common_days']
            days_x = comparison_data['days_x']
            data_comp = comparison_data['data_comp']
            
            stats_to_log['comparison_data'] = {
                'common_days': [int(d) for d in common_days],
                'days_x': [int(d) for d in days_x],
                'valid_indices': valid_indices_plot.tolist() if 'valid_indices_plot' in locals() else [],
                'x_plot_comp': x_plot_comp.tolist() if 'x_plot_comp' in locals() else [],
                'metrics': {}
            }
            
            for metric in ['posture_diff', 'upright_tails', 'hanging_tails']:
                stats_to_log['comparison_data']['metrics'][metric] = {
                    'outbreak': {
                        'mean': [float(v) if not np.isnan(v) else None for v in data_comp['outbreak'][metric]['mean']],
                        'std': [float(v) if not np.isnan(v) else None for v in data_comp['outbreak'][metric]['std']],
                        'count': [int(v) for v in data_comp['outbreak'][metric]['count']]
                    },
                    'control': {
                        'mean': [float(v) if not np.isnan(v) else None for v in data_comp['control'][metric]['mean']],
                        'std': [float(v) if not np.isnan(v) else None for v in data_comp['control'][metric]['std']],
                        'count': [int(v) for v in data_comp['control'][metric]['count']]
                    }
                }
            
            # Add means and stats for valid plot points (after filtering)
            if 'valid_indices_plot' in locals() and 'x_plot_comp' in locals():
                for group in ['outbreak', 'control']:
                    for metric in ['posture_diff', 'upright_tails', 'hanging_tails']:
                        key = f"{group}_{metric}_valid"
                        var_name = f"means_{group[0]}_{metric.split('_')[0]}"
                        if var_name in locals():
                            stats_to_log['comparison_data']['metrics'][metric][f"{group}_valid"] = {
                                'x': x_plot_comp.tolist(),
                                'mean': [float(v) for v in locals()[var_name]]
                            }
        
        # Add change and contribution statistics
        if change_stats:
            stats_to_log['change_statistics'] = change_stats
        if contribution_stats:
            stats_to_log['contribution_statistics'] = contribution_stats
        
        # Log the statistics
        self.log_visualization_stats(stats_to_log, 'posture_components')

        self.component_analysis = component_analysis # Store results
        return component_analysis
        
    def visualize_data_completeness(self, save_path=None, show=False):
        """
        Generates a publication-quality timeline plot visualizing data availability
        with enhanced features for dissertation presentation and highlights excluded events.
        """
        
        def lighten_color(color, amount=0.5):
            """
            Lightens the given color by multiplying (1-luminosity) by the given amount.
            Input can be matplotlib color string, hex string, or RGB tuple.
            """
            import matplotlib.colors as mc
            import colorsys
            try:
                c = mc.cnames[color]
            except:
                c = color
            c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            return colorsys.hls_to_rgb(c[0], min(1, c[1] + amount * (1 - c[1])), c[2])

        def darken_color(color, amount=0.5):
            """
            Darkens the given color by multiplying luminosity by the given amount.
            Input can be matplotlib color string, hex string, or RGB tuple.
            """
            import matplotlib.colors as mc
            import colorsys
            try:
                c = mc.cnames[color]
            except:
                c = color
            c = colorsys.rgb_to_hls(*mc.to_rgb(c))
            return colorsys.hls_to_rgb(c[0], max(0, c[1] * (1 - amount)), c[2])
        
        self.logger.info("Visualizing data completeness across all events (enhanced version with exclusions)...")
        set_plotting_style(self.config)  # Apply style

        if not self.monitoring_results:
            self.logger.error("No monitoring results loaded. Cannot visualize completeness.")
            return

        # --- Data Preparation (Keep existing logic, ensure robustness) ---
        plot_data = []
        try:
            json_data = load_json_data(self.path_manager.path_to_piglet_rearing_info)
        except Exception as e:
            self.logger.warning(f"Could not load rearing info: {e}. Pen types might be unknown.")
            json_data = None

        # Sort results by date and then by camera for better organization
        sorted_results = sorted(self.monitoring_results, 
                                key=lambda x: (pd.to_datetime(x.get('date_span', '000000_000000').split('_')[0], 
                                                            format='%y%m%d', errors='coerce'),
                                            x.get('camera', '')))

        # Group data by pen type for better organization
        pen_type_groups = {"tail biting": [], "control": [], "Unknown": []}
        
        # Create sets to identify excluded events
        excluded_consecutive = set()
        excluded_percentage = set()
        
        # Extract exclusion data if available
        if hasattr(self, 'excluded_elements'):
            # Handle the potential different structures in excluded_elements for consecutive_missing
            for item in self.excluded_elements.get('consecutive_missing', []):
                # Different unpacking approaches depending on tuple structure
                if len(item) >= 2:
                    try:
                        if isinstance(item, tuple) and len(item) == 5:
                            camera, date_span, pen_type, value, threshold = item
                        elif isinstance(item, tuple) and len(item) == 4:
                            camera, date_span, pen_type, value = item
                        elif isinstance(item, tuple) and len(item) == 3:
                            camera, date_span, pen_type = item
                        elif isinstance(item, tuple) and len(item) == 2:
                            camera, date_span = item
                        else:
                            # Try to extract values from any other structure
                            camera = item[0]
                            date_span = item[1]
                            
                        # Also store with just the number to handle "Bucht X" vs "Kamera X" matching
                        pen_number = ''.join([char for char in camera if char.isdigit()])
                        if pen_number:
                            excluded_consecutive.add((pen_number, date_span))
                            
                        excluded_consecutive.add((camera, date_span))
                        self.logger.info(f"Marked for consecutive exclusion: {camera}/{date_span} (pen {pen_number})")
                    except (IndexError, TypeError) as e:
                        self.logger.warning(f"Could not extract camera/date_span from {item}: {e}")
                        continue
            
            # Handle the potential different structures in excluded_elements for missing_percentage 
            for item in self.excluded_elements.get('missing_percentage', []):
                # Different unpacking approaches depending on tuple structure
                if len(item) >= 2:
                    try:
                        if isinstance(item, tuple) and len(item) == 5:
                            camera, date_span, pen_type, value, threshold = item
                        elif isinstance(item, tuple) and len(item) == 4:
                            camera, date_span, pen_type, value = item
                        elif isinstance(item, tuple) and len(item) == 3:
                            camera, date_span, pen_type = item
                        elif isinstance(item, tuple) and len(item) == 2:
                            camera, date_span = item
                        else:
                            # Try to extract values from any other structure
                            camera = item[0]
                            date_span = item[1]
                            
                        # Also store with just the number to handle "Bucht X" vs "Kamera X" matching
                        pen_number = ''.join([char for char in camera if char.isdigit()])
                        if pen_number:
                            excluded_percentage.add((pen_number, date_span))
                            
                        excluded_percentage.add((camera, date_span))
                        self.logger.info(f"Marked for percentage exclusion: {camera}/{date_span} (pen {pen_number})")
                    except (IndexError, TypeError) as e:
                        self.logger.warning(f"Could not extract camera/date_span from {item}: {e}")
                        continue
                    
        # Also check for exclusions in analysis-specific tracking if available
        if hasattr(self, 'exclusion_by_analysis'):
            # Process tail_biting_analysis exclusions
            for analysis_type in ['tail_biting_analysis', 'control_analysis']:
                if analysis_type in self.exclusion_by_analysis:
                    # Handle consecutive missing in each analysis type
                    for item in self.exclusion_by_analysis[analysis_type].get('consecutive_missing', []):
                        if len(item) >= 2:
                            try:
                                camera = item[0]
                                date_span = item[1]
                                
                                # Also store with just the number to handle "Bucht X" vs "Kamera X" matching
                                pen_number = ''.join([char for char in camera if char.isdigit()])
                                if pen_number:
                                    excluded_consecutive.add((pen_number, date_span))
                                    
                                excluded_consecutive.add((camera, date_span))
                                self.logger.info(f"Marked for consecutive exclusion from {analysis_type}: {camera}/{date_span} (pen {pen_number})")
                            except (IndexError, TypeError) as e:
                                self.logger.warning(f"Could not extract camera/date_span from {item} in {analysis_type}: {e}")
                                continue
                    
                    # Handle missing percentage in each analysis type
                    for item in self.exclusion_by_analysis[analysis_type].get('missing_percentage', []):
                        if len(item) >= 2:
                            try:
                                camera = item[0]
                                date_span = item[1]
                                
                                # Also store with just the number to handle "Bucht X" vs "Kamera X" matching
                                pen_number = ''.join([char for char in camera if char.isdigit()])
                                if pen_number:
                                    excluded_percentage.add((pen_number, date_span))
                                    
                                excluded_percentage.add((camera, date_span))
                                self.logger.info(f"Marked for percentage exclusion from {analysis_type}: {camera}/{date_span} (pen {pen_number})")
                            except (IndexError, TypeError) as e:
                                self.logger.warning(f"Could not extract camera/date_span from {item} in {analysis_type}: {e}")
                                continue
        
        for i, result in enumerate(sorted_results):
            camera = result.get('camera', 'UnknownCam')
            date_span = result.get('date_span', 'UnknownSpan')
            pen_type = "Unknown"
            if json_data:
                try: 
                    pen_type, _, _ = get_pen_info(camera, date_span, json_data)
                except Exception: 
                    pass  # Keep pen_type as Unknown if lookup fails

            try:
                start_str, end_str = date_span.split('_')
                start_date = pd.to_datetime(start_str, format='%y%m%d')
                end_date = pd.to_datetime(end_str, format='%y%m%d')
                
                formatted_start = start_date.strftime('%d.%m.%Y')
                formatted_end = end_date.strftime('%d.%m.%Y')
                formatted_datespan = f"{formatted_start} - {formatted_end}"
                
                end_date_plot = end_date + pd.Timedelta(days=1)  # For bar width
                duration = end_date_plot - start_date

                # Handle missing dates with more robust parsing
                missing_dates_str = result.get('missing_dates', [])
                missing_dates_dt = []
                for d in missing_dates_str:
                    try:
                        missing_dates_dt.append(pd.to_datetime(d))
                    except Exception:
                        self.logger.warning(f"Could not parse missing date '{d}' for {camera}/{date_span}")

                # Extrahieren der Täterentfernungsdaten
                culprit_removal_dates = []
                if pen_type == "tail biting" and json_data:
                    for entry in json_data:
                        if entry.get('camera') == camera and entry.get('datespan') == date_span:
                            culprit_data = entry.get('culpritremoval')
                            if culprit_data:
                                if isinstance(culprit_data, list):
                                    # Multiple removal dates
                                    for date_str in culprit_data:
                                        try:
                                            culprit_removal_dates.append(pd.to_datetime(date_str))
                                        except Exception:
                                            self.logger.warning(f"Could not parse culprit removal date '{date_str}' for {camera}/{date_span}")
                                else:
                                    # Single removal date
                                    try:
                                        culprit_removal_dates.append(pd.to_datetime(culprit_data))
                                    except Exception:
                                        self.logger.warning(f"Could not parse culprit removal date '{culprit_data}' for {camera}/{date_span}")
                            break

                # Calculate completeness percentage but don't display it
                total_days = result.get('total_expected_days', 0)
                if total_days > 0:
                    completeness_pct = 100 * (total_days - len(missing_dates_dt)) / total_days
                else:
                    completeness_pct = float('nan')
                
                # Check if this event is excluded
                is_excluded_consecutive = (camera, date_span) in excluded_consecutive
                is_excluded_percentage = (camera, date_span) in excluded_percentage

                event_data = {
                    'camera': camera,
                    'pen_id': camera.replace('Kamera', 'Bucht '),
                    'date_span': date_span,
                    'formatted_datespan': formatted_datespan,
                    'label': f"{camera.replace('Kamera', 'Bucht ')} - {formatted_datespan}",
                    'start_date': start_date,
                    'end_date': end_date_plot,
                    'duration': duration,
                    'missing_dates': missing_dates_dt,
                    'culprit_removal_dates': culprit_removal_dates,
                    'pen_type': pen_type,
                    'total_days': total_days,
                    'missing_count': len(missing_dates_dt),
                    'completeness_pct': completeness_pct,
                    'excluded_consecutive': is_excluded_consecutive,
                    'excluded_percentage': is_excluded_percentage
                }
                
                # Store data by pen type for grouped plotting
                if pen_type in pen_type_groups:
                    pen_type_groups[pen_type].append(event_data)
                else:
                    pen_type_groups["Unknown"].append(event_data)
                    
            except Exception as e:
                self.logger.warning(f"Skipping event due to parsing error: {camera}/{date_span} - {e}")
                continue
        
        # Flatten the groups back into a single list, now ordered by pen type
        plot_data = []
        for pen_type in ["tail biting", "control", "Unknown"]:
            # Sort each group by start date
            group_data = sorted(pen_type_groups[pen_type], key=lambda x: x['start_date'])
            plot_data.extend(group_data)
        
        # Add y-position to each item
        for i, event in enumerate(plot_data):
            event['y_pos'] = i
        # --- End Data Prep ---

        if not plot_data:
            self.logger.error("No valid data prepared for completeness visualization.")
            return

        # --- Create Figure ---
        num_events = len(plot_data)
        
        # Create figure with single panel (no summary panel)
        fig, ax = plt.subplots(figsize=(8, 11))

        # --- Color definitions ---
        pen_type_colors = {
            "tail biting": COLORS['tail_biting'],
            "control": COLORS['control'],
            "Unknown": COLORS['neutral']
        }
        
        # Use grey for missing days
        missing_marker_color = '#777777'  # Appropriate grey for missing data
        # Definiere sanftes Orange für Täterentfernungen
        culprit_removal_color = '#FF8C00'  # Orange für Täterentfernungen
        grid_color = COLORS.get('grid', '#DDDDDD')  # Light gray for grid
        text_color = COLORS.get('text', '#333333')  # Dark gray for text
        
        # New colors for exclusion highlights
        consecutive_exclude_color = '#FF3333'  # Bright red for consecutive exclusions
        percentage_exclude_color = '#9933FF'   # Purple for percentage exclusions
        
        # Create stylized legend elements for pen types
        legend_elements = [
            mpatches.Patch(color=pen_type_colors["tail biting"], label='Schwanzbeißbucht'),
            mpatches.Patch(color=pen_type_colors["control"], label='Kontrollbucht'),
            plt.Line2D([0], [0], marker='|', color=missing_marker_color, linestyle='', 
                    markersize=10, markeredgewidth=2, label='Fehlender Tag'),
            # Legende für Täterentfernung
            plt.Line2D([0], [0], marker='|', color=culprit_removal_color, linestyle='', 
                    markersize=10, markeredgewidth=2, label='Täterentfernung'),
            # New legend elements for exclusion highlighting with improved visibility
            mpatches.Patch(facecolor=consecutive_exclude_color, alpha=0.4, 
                         hatch='////', edgecolor=consecutive_exclude_color, linewidth=1.5,
                         label='Ausgeschlossen: >3 aufeinanderfolgende fehlende Tage'),
            mpatches.Patch(facecolor=percentage_exclude_color, alpha=0.4, 
                         hatch='\\\\\\\\', edgecolor=percentage_exclude_color, linewidth=1.5,
                         label='Ausgeschlossen: >50% fehlende Tage')
        ]
        
        if any(p['pen_type'] == 'Unknown' for p in plot_data):
            legend_elements.append(mpatches.Patch(color=pen_type_colors["Unknown"], 
                                                label='Unbekannter Buchtentyp'))

        # --- Plot data ---
        bar_height = 0.65
        prev_pen_type = None
        group_separator_positions = []
        
        for event in plot_data:
            # Add subtle separator between pen type groups
            if prev_pen_type and prev_pen_type != event['pen_type']:
                group_separator_positions.append(event['y_pos'] - 0.5)
            prev_pen_type = event['pen_type']
            
            # Bar color and style based on pen type
            bar_color = pen_type_colors.get(event['pen_type'], pen_type_colors["Unknown"])
            
            # Calculate completeness-based opacity (more missing data = more transparent)
            completeness = (event['total_days'] - event['missing_count']) / max(event['total_days'], 1)
            # Limit transparency to ensure bars are still visible
            opacity = max(0.6, min(0.95, completeness))
            
            # Main timeline bar - set to lowest z-order
            bar = ax.barh(y=event['y_pos'], width=event['duration'], left=event['start_date'],
                    height=bar_height, color=lighten_color(bar_color, 0.2),  # Lighter fill
                    edgecolor=bar_color, linewidth=1.0, alpha=opacity,
                    zorder=1)  # Set lowest z-order

            # Add exclusion highlighting with improved visibility - set to higher z-order
            if event['excluded_consecutive']:
                # Add hatched overlay for consecutive missing exclusions
                ax.barh(y=event['y_pos'], width=event['duration'], left=event['start_date'],
                       height=bar_height, color=consecutive_exclude_color, alpha=0.4,
                       hatch='////', edgecolor=consecutive_exclude_color, linewidth=1.5,
                       zorder=4)  # Higher z-order to ensure visibility above missing markers
                self.logger.info(f"Highlighting consecutive exclusion: {event['camera']}/{event['date_span']}")
                       
            elif event['excluded_percentage']:
                # Add hatched overlay for percentage missing exclusions
                ax.barh(y=event['y_pos'], width=event['duration'], left=event['start_date'],
                       height=bar_height, color=percentage_exclude_color, alpha=0.4,
                       hatch='\\\\\\\\', edgecolor=percentage_exclude_color, linewidth=1.5,
                       zorder=4)  # Higher z-order to ensure visibility above missing markers
                self.logger.info(f"Highlighting percentage exclusion: {event['camera']}/{event['date_span']}")

            # Visualize missing dates with grey markers - set to middle z-order
            if event['missing_dates']:
                marker_dates = [d + pd.Timedelta(hours=12) for d in event['missing_dates']]  # Center marker
                m_size = 9  # Slightly larger marker for better visibility
                ax.plot(marker_dates, [event['y_pos']] * len(marker_dates),
                        marker='|', markersize=m_size, linestyle='',
                        color=missing_marker_color, markeredgewidth=1.8,  # Thicker marker line
                        alpha=0.9, zorder=3)  # Middle zorder
            
            # Visualisiere Täterentfernungsdaten mit orangefarbenen Markern - highest z-order
            if event['culprit_removal_dates']:
                culprit_dates = [d + pd.Timedelta(hours=12) for d in event['culprit_removal_dates']]  # Center marker
                c_size = 10  # Etwas größerer Marker für Täterentfernungen
                ax.plot(culprit_dates, [event['y_pos']] * len(culprit_dates),
                        marker='|', markersize=c_size, linestyle='',
                        color=culprit_removal_color, markeredgewidth=2.0,  # Etwas dickere Marker-Linie
                        alpha=0.9, zorder=5)  # Highest zorder for best visibility

        # Add subtle separators between pen type groups
        for pos in group_separator_positions:
            ax.axhline(y=pos, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.5)

        # Format Y-axis with enhanced styling
        y_labels = [p['label'] for p in plot_data]
        y_positions = [p['y_pos'] for p in plot_data]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        
        # Entferne leere Abstände oben und unten
        if y_positions:
            # Berechne die Höhe eines Elements inkl. Abstand
            if len(y_positions) > 1:
                bar_spacing = abs(y_positions[0] - y_positions[1])
            else:
                bar_spacing = 1.0
                
            # Setze die Grenzen mit einem halben Balkenabstand als Puffer
            min_pos = min(y_positions) - bar_spacing/2
            max_pos = max(y_positions) + bar_spacing/2
            ax.set_ylim(max_pos, min_pos)  # Umgekehrte Reihenfolge wegen der invertierten y-Achse
        else:
            ax.invert_yaxis()  # Invertiere nur, wenn keine Daten vorhanden sind

        # Format X-axis (Dates) with enhanced date formatting
        ax.xaxis_date()
        
        # Dynamically choose locator based on total duration
        total_duration_days = (max(p['end_date'] for p in plot_data) - 
                            min(p['start_date'] for p in plot_data)).days
        
        if total_duration_days > 365 * 2:  # Multi-year
            major_locator = mdates.YearLocator()
            minor_locator = mdates.MonthLocator(interval=3)
            date_format = mdates.DateFormatter('%Y')
        elif total_duration_days > 180:  # Several months to ~2 years
            major_locator = mdates.MonthLocator(interval=2)
            minor_locator = mdates.MonthLocator(interval=1)
            date_format = mdates.DateFormatter('%b %Y')  # Month + Year
        else:  # Shorter duration
            major_locator = mdates.MonthLocator(interval=1)
            minor_locator = mdates.DayLocator(interval=7)
            date_format = mdates.DateFormatter('%d.%m.%Y')  # Full German date format
        
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(date_format)
        
        # Set X limits with padding
        if plot_data:
            min_lim = min(p['start_date'] for p in plot_data) - pd.Timedelta(days=7)
            max_lim = max(p['end_date'] for p in plot_data) + pd.Timedelta(days=7)
            ax.set_xlim(min_lim, max_lim)

        # Add horizontal gridlines instead of vertical
        ax.grid(False, axis='x')  # Remove vertical gridlines
        ax.grid(True, axis='y', linestyle=':', color=grid_color, alpha=0.5)  # Add horizontal gridlines
        
        ax.set_xlabel("Datum", fontsize=10, fontweight='bold', color=text_color)
        ax.set_ylabel("Bucht - Zeitraum", fontsize=10, fontweight='bold', color=text_color)
        
        # Format date ticks for better readability
        fig.autofmt_xdate(rotation=45, ha='right')
        
        # Add legend with better formatting for multiple rows
        # Use 2 columns for more compact display
        legend = ax.legend(handles=legend_elements, loc='upper right', 
                frameon=True, facecolor='white', edgecolor=grid_color, 
                fontsize=8, framealpha=0.95, ncol=2)
        
        # Adjust legend position to avoid overlap with data
        # Place it in the upper right with some padding
        legend.set_bbox_to_anchor((1.02, 1.02))
        
        # Ensure the legend has a solid background
        frame = legend.get_frame()
        frame.set_linewidth(1.0)

        # Adjust layout
        plt.tight_layout()
        
        # Save high-resolution figure
        if save_path is None:
            filename = self.config.get('viz_completeness_filename', 'data_completeness_timeline_with_exclusions.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        
        try:
            dpi = self.config.get('figure_dpi', 300)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.logger.info(f"Saved enhanced data completeness visualization with exclusion highlighting to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save enhanced completeness visualization: {e}")
        finally:
            if show:
                plt.show()
            plt.close(fig)
            
        return save_path
    
    
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
