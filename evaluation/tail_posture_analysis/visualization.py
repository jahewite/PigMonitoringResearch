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

from pipeline.utils.general import load_json_data
from pipeline.utils.data_analysis_utils import get_pen_info, sorting_key
from evaluation.tail_posture_analysis.analysis import TailPostureAnalyzer
from evaluation.tail_posture_analysis.utils import COLORS, PATTERN_COLORS, lighten_color, set_plotting_style

class TailPostureVisualizer(TailPostureAnalyzer):
    """Methods for visualizing tail posture analysis results."""
    
    def __init__(self, *args, **kwargs):
        # Ensure logger is initialized in the base class or here
        super().__init__(*args, **kwargs) # Call parent __init__
        if not hasattr(self, 'logger'): # If logger wasn't set by parent
             self.logger = logging.getLogger(__name__)
             if not self.logger.hasHandlers(): # Basic config if no handlers attached
                 logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set the style upon instantiation
        set_plotting_style(self.config)
        self.logger.info("Dissertation quality plotting style set.")
    
    def _style_violin_box(self, ax, data, labels, violin_color, box_color, scatter_color, violin_width=0.6):
        """Helper to style combined violin and box plots (Dissertation Quality)."""
        if not data or all(d.empty for d in data): # Check if all data series are empty
             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color=COLORS['annotation'], fontsize=plt.rcParams['font.size'])
             return # No data to plot

        try:
            # Violin Plot - Lighter fill, slightly darker edge
            parts = ax.violinplot(data, showmeans=False, showmedians=False, widths=violin_width) # Mean/Median handled by boxplot
            for pc in parts['bodies']:
                pc.set_facecolor(lighten_color(violin_color, 0.7)) # More lightening for fill
                pc.set_edgecolor(violin_color) # Use main color for edge
                pc.set_alpha(0.9) # Slightly less transparent
                pc.set_linewidth(0.8) # Thin edge

            # Explicitly set colors/alphas based on input parameters for box fill
            bplot = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, # Flier handling in rcParams
                               showmeans=True, # Use rcParams setting
                               widths=violin_width * 0.3, # Narrower box inside violin
                               positions=np.arange(1, len(data) + 1)) # Ensure positions match violin

            for patch in bplot['boxes']:
                patch.set_facecolor(lighten_color(box_color, 0.5)) # Light fill for box
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

    def _style_boxplot(self, ax, data, labels, colors, show_scatter=True, scatter_alpha=0.3, widths=0.6):
        """Helper to style box plots consistently (Dissertation Quality). Returns the boxplot dictionary."""
        if not data or all(d is None or len(d) == 0 for d in data): # Check if data is empty or contains only empty lists/None
             ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color=COLORS['annotation'], fontsize=plt.rcParams['font.size'])
             return None # Return None if no data

        # Use rcParams for median/mean properties directly
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False, # Flier handling in rcParams
                        showmeans=True, # Use rcParams setting
                        widths=widths)

        # Style boxes, whiskers, caps
        for i, box in enumerate(bp['boxes']):
            box_color = colors[i % len(colors)]
            box.set_facecolor(lighten_color(box_color, 0.6)) # Lighter fill
            box.set_edgecolor(box_color) # Main color edge
            box.set_alpha(0.85)
            box.set_linewidth(plt.rcParams['boxplot.boxprops.linewidth'])

        for whisker in bp['whiskers']:
             whisker.set(color=plt.rcParams['axes.edgecolor'],
                         linewidth=plt.rcParams['boxplot.whiskerprops.linewidth'],
                         linestyle=plt.rcParams['boxplot.whiskerprops.linestyle'])
        for cap in bp['caps']:
             cap.set(color=plt.rcParams['axes.edgecolor'],
                     linewidth=plt.rcParams['boxplot.capprops.linewidth'])

        if show_scatter:
            for i, d in enumerate(data):
                 if d is not None and len(d) > 0: # Check d is not None
                     scatter_color = colors[i % len(colors)]
                     x_jitter = np.random.normal(i + 1, 0.03, size=len(d)) # Slightly tighter jitter
                     ax.scatter(x_jitter, d, alpha=scatter_alpha, s=10, color=scatter_color, # Smaller dots
                                edgecolor='none', zorder=3) # No edges

        return bp

    def _add_stats_annotation(self, ax, text, loc='upper right', fontsize=None, **kwargs):
        """Helper to add standardized statistics box (Dissertation Quality)."""
        if fontsize is None:
             fontsize = plt.rcParams['legend.fontsize'] # Use legend font size from rcParams

        # Slightly cleaner bbox
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, # Higher alpha
                          edgecolor='#CCCCCC', linewidth=0.5) # Lighter edge

        # Default location settings remain the same
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
    
    def visualize_pre_outbreak_patterns(self, save_path=None):
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
                # Note: Extreme outlier filtering is less critical with symmetric percentage change,
                # but we'll keep it for consistency and to handle any remaining outliers
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
            # REMOVED: Stats annotation

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
        plt.close(fig)

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
        
    def visualize_comparison_with_controls(self, save_path=None):
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
        for d in sorted(days_list, reverse=True): # Consistent order
            col = f'abs_change_{d}d'
            if col in combined_data.columns:
                 for group, short_label in group_labels.items():
                     data_series = combined_data.loc[combined_data['group'] == group, col]
                     data_vals = data_series.dropna().values
                     if len(data_vals) > 0:
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
            plt.close(fig)

        return True
        
    def visualize_individual_variation(self, save_path=None):
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
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], # Adjusted ratio maybe needed
                            hspace=0.45, wspace=0.3)

        # Use PATTERN_COLORS defined in utils
        pattern_colors = PATTERN_COLORS
        default_color = COLORS['neutral']

        # Define trajectory columns and day mapping
        traj_cols_map = {'value_7d_before': -7, 'value_5d_before': -5, 'value_3d_before': -3,
                         'value_1d_before': -1, 'value_at_removal': 0}
        traj_cols = [col for col in traj_cols_map if col in outbreaks_df.columns]
        plot_days_ind = sorted([traj_cols_map[col] for col in traj_cols])


        # --- Panel A (Row 0, Col 0): Individual Trajectories by Pattern ---
        ax0 = fig.add_subplot(gs[0, 0])
        plotted_a = False
        if traj_cols:
            pattern_categories = outbreaks_df['pattern_category'].unique()
            valid_patterns_plotted = []

            for pattern in pattern_categories:
                pattern_outbreaks = outbreaks_df[outbreaks_df['pattern_category'] == pattern]
                if pattern_outbreaks.empty: continue

                color = pattern_colors.get(pattern, default_color)
                plotted_a = True # Mark that we have data
                valid_patterns_plotted.append(pattern)

                for _, row in pattern_outbreaks.iterrows():
                    values = [row.get(col, np.nan) for col in traj_cols]
                    valid_indices = ~np.isnan(values)
                    plot_days_row = [traj_cols_map[traj_cols[i]] for i, v in enumerate(valid_indices) if v]
                    plot_values_row = [values[i] for i, v in enumerate(valid_indices) if v]

                    if sum(valid_indices) >= 2:
                        sorted_points = sorted(zip(plot_days_row, plot_values_row))
                        plot_days_sorted, plot_values_sorted = zip(*sorted_points)
                        ax0.plot(plot_days_sorted, plot_values_sorted, marker='.', markersize=3, # Small marker
                                 linewidth=0.7, alpha=0.3, color=color, zorder=5) # Thin, light lines

            if plotted_a:
                # Add legend using Line2D for patterns plotted
                legend_elements_a = [plt.Line2D([0], [0], color=pattern_colors.get(p, default_color), lw=2, label=p)
                                   for p in valid_patterns_plotted if p in pattern_colors] # Only patterns with colors
                if legend_elements_a:
                    ax0.legend(handles=legend_elements_a, loc='lower left', frameon=True, fontsize=8) # Smaller font

                ax0.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
                ax0.grid(axis='x', b=False)
                ax0.set_xticks(plot_days_ind)
                ax0.set_xticklabels([f'{abs(d)}d' if d < 0 else 'Rem.' for d in plot_days_ind], fontsize=9)
            else:
                 ax0.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', color=COLORS['annotation'])

        else:
             ax0.text(0.5, 0.5, 'No trajectory columns', ha='center', va='center', color=COLORS['annotation'])

        ax0.set_title('A) Individual Outbreak Trajectories by Pattern')
        ax0.set_xlabel('Days Before Removal')
        ax0.set_ylabel('Posture Difference')


        # --- Panel B (Row 0, Col 1): Trajectory Clusters with Average Lines ---
        ax1 = fig.add_subplot(gs[0, 1])
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

                    line, = ax1.plot(avg_days_sorted, avg_values_sorted, marker='o', markersize=5,
                                     linewidth=2.0, color=color, label=f'{pattern} (N={len(pattern_outbreaks)})', zorder=10)
                    legend_handles_b.append(line)

                    # Add CI fill
                    try:
                        ci_level = 0.95
                        t_crit = [stats.t.ppf((1 + ci_level) / 2, n - 1) if n > 1 else 0 for n in n_points_sorted]
                        sem = [std / np.sqrt(n) if n > 0 else 0 for std, n in zip(std_values_sorted, n_points_sorted)]
                        margin_of_error = [t * s for t, s in zip(t_crit, sem)]
                        upper = [avg + moe for avg, moe in zip(avg_values_sorted, margin_of_error)]
                        lower = [avg - moe for avg, moe in zip(avg_values_sorted, margin_of_error)]
                        ax1.fill_between(avg_days_sorted, upper, lower, color=color, alpha=0.15, zorder=9)
                    except Exception as e_ci_b:
                        self.logger.warning(f"Could not plot CI for {pattern}: {e_ci_b}")

            if plotted_b:
                ax1.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                for day in plot_days_ind: ax1.axvline(x=day, color=COLORS['grid'], linestyle=':', linewidth=0.8, alpha=0.5)
                ax1.grid(False, axis='x') # Only vertical lines shown above
                ax1.grid(True, axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
                ax1.set_xticks(plot_days_ind)
                ax1.set_xticklabels([f'{abs(d)}d' if d < 0 else 'Rem.' for d in plot_days_ind], fontsize=9)
                if legend_handles_b: ax1.legend(handles=legend_handles_b, loc='best', frameon=True, fontsize=8)
            else:
                 ax1.text(0.5, 0.5, 'Insufficient data for avg trajectories', ha='center', va='center', color=COLORS['annotation'])

        else:
            ax1.text(0.5, 0.5, 'No trajectory columns', ha='center', va='center', color=COLORS['annotation'])

        ax1.set_title('B) Average Trajectories by Pattern Type')
        ax1.set_xlabel('Days Before Removal')
        ax1.set_ylabel('Posture Difference')


        # --- Panel C (Row 1, Col 0): Individual Pen Variation ---
        ax2 = fig.add_subplot(gs[1, 0])
        plotted_c = False
        if traj_cols:
            pen_multi_counts = outbreaks_df['pen'].value_counts()
            pens_with_multiple = pen_multi_counts[pen_multi_counts > 1].index.tolist()
            pens_single = pen_multi_counts[pen_multi_counts == 1].index.tolist()

            # Prioritize pens with multiple, then single
            pens_ordered = pens_with_multiple + pens_single

            max_pens_plot = self.config.get('variation_max_pens_plot', 8)
            pens_to_show = pens_ordered[:max_pens_plot]

            if pens_to_show:
                legend_handles_c = []
                # Use a perceptually uniform colormap if many pens
                pen_colors = plt.cm.viridis(np.linspace(0, 1, len(pens_to_show)))

                for i, pen in enumerate(pens_to_show):
                    pen_data = outbreaks_df[outbreaks_df['pen'] == pen]
                    pen_color = pen_colors[i]

                    for j, (_, row) in enumerate(pen_data.iterrows()):
                        values = [row.get(col, np.nan) for col in traj_cols]
                        valid_indices = ~np.isnan(values)
                        plot_days_row = [traj_cols_map[traj_cols[i]] for i, v in enumerate(valid_indices) if v]
                        plot_values_row = [values[i] for i, v in enumerate(valid_indices) if v]

                        if sum(valid_indices) >= 2:
                            plotted_c = True
                            sorted_points = sorted(zip(plot_days_row, plot_values_row))
                            plot_days_sorted, plot_values_sorted = zip(*sorted_points)
                            line, = ax2.plot(plot_days_sorted, plot_values_sorted, marker='o', markersize=3,
                                             linewidth=1.2, alpha=0.6, color=pen_color, zorder=5)
                            # Add to legend only once per pen
                            if j == 0: legend_handles_c.append(line)

                if plotted_c:
                    # Position legend below plot
                    ax2.legend(handles=legend_handles_c, labels=[str(p) for p in pens_to_show], # Ensure labels are strings
                               loc='upper center', bbox_to_anchor=(0.5, -0.15),
                               ncol=min(4, len(pens_to_show)), frameon=True, fontsize=8)

                    ax2.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                    ax2.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
                    ax2.grid(axis='x', b=False)
                    ax2.set_xticks(plot_days_ind)
                    ax2.set_xticklabels([f'{abs(d)}d' if d < 0 else 'Rem.' for d in plot_days_ind], fontsize=9)
                else:
                    ax2.text(0.5, 0.5, 'No valid pen trajectories to plot', ha='center', va='center', color=COLORS['annotation'])
            else:
                 ax2.text(0.5, 0.5, 'No pens found to display', ha='center', va='center', color=COLORS['annotation'])
        else:
             ax2.text(0.5, 0.5, 'No trajectory columns', ha='center', va='center', color=COLORS['annotation'])

        ax2.set_title('C) Individual Pen Variation (Sample)')
        ax2.set_xlabel('Days Before Removal')
        ax2.set_ylabel('Posture Difference')


        # --- Panel D (Row 1, Col 1): Pattern Distribution ---
        ax3 = fig.add_subplot(gs[1, 1])
        pattern_counts = pattern_results.get('pattern_counts', {})

        if pattern_counts:
            # Order patterns for consistency if possible (e.g., severity)
            ordered_patterns = [p for p in PATTERN_COLORS if p in pattern_counts] + \
                               [p for p in pattern_counts if p not in PATTERN_COLORS] # Add any others at end
            counts = [pattern_counts[p] for p in ordered_patterns]
            bar_colors = [pattern_colors.get(p, default_color) for p in ordered_patterns]

            bars = ax3.bar(ordered_patterns, counts, color=bar_colors, edgecolor=COLORS['annotation'], linewidth=0.7)

            ax3.bar_label(bars, padding=3, fontsize=8) # Labels above bars

            # Optional: Add percentage labels inside or below
            # total_counts = sum(counts)
            # for bar, count in zip(bars, counts):
            #     if total_counts > 0:
            #         percentage = (count / total_counts) * 100
            #         ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, # Center inside bar
            #                  f'{percentage:.0f}%', ha='center', va='center', fontsize=7, color='white', weight='bold')

            # Add pen consistency info using helper
            pen_consistency_data = pattern_results.get('pen_consistency', {})
            pens_consistent = pen_consistency_data.get('pens_consistent', 0)
            pens_with_multiple = pen_consistency_data.get('pens_with_multiple', 0)
            if pens_with_multiple > 0:
                consistency_pct = pen_consistency_data.get('consistency_percentage', 0)
                # consistency_text = (f"Pen Consistency: {pens_consistent}/{pens_with_multiple} pens "
                #                     f"({consistency_pct:.0f}%) had same pattern")
                # self._add_stats_annotation(ax3, consistency_text, loc='upper right', fontsize=8)

            ax3.margins(y=0.1) # Add margin for labels

        else:
             ax3.text(0.5, 0.5, 'No pattern counts', ha='center', va='center', color=COLORS['annotation'])

        ax3.set_title('D) Distribution of Pattern Categories')
        ax3.set_ylabel('Number of Outbreaks')
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
        ax3.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax3.grid(axis='x', b=False)

        # --- Layout & Saving ---
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.92, hspace=0.55) # Adjust hspace more?

        if save_path is None:
            filename = self.config.get('viz_variation_filename', 'individual_variation_analysis.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        try:
            fig.savefig(save_path)
            self.logger.info(f"Saved individual variation visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save individual variation visualization: {e}")
        finally:
            plt.close(fig)

        return True
        
    def visualize_posture_components(self, component_analysis=None, save_path=None):
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
        # This panel remains the same as the previous refactored version
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
        
        if not avg_by_day_outbreak.empty and len(days_outbreak) > 1:
            plotted_a = True
            for metric, color_name in [('upright_tails', 'upright'), ('hanging_tails', 'hanging'), ('posture_diff', 'difference')]:
                means = avg_by_day_outbreak[(metric, 'mean')].reindex(days_outbreak).values
                stds = avg_by_day_outbreak[(metric, 'std')].reindex(days_outbreak).values
                counts = avg_by_day_outbreak[(metric, 'count')].reindex(days_outbreak).values
                valid_idx = (counts > 1) & ~np.isnan(means)
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

                 # Check if we have *any* valid data points to plot after alignment
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
            plt.close(fig)

        self.component_analysis = component_analysis # Store results
        return component_analysis
        
    
    def visualize_data_completeness(self, save_path=None):
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
                if len(item) >= 2:  # At minimum we need camera and date_span
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
                if len(item) >= 2:  # At minimum we need camera and date_span
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
                # Log that we're actually highlighting this event
                self.logger.info(f"Highlighting consecutive exclusion: {event['camera']}/{event['date_span']}")
                       
            elif event['excluded_percentage']:
                # Add hatched overlay for percentage missing exclusions
                ax.barh(y=event['y_pos'], width=event['duration'], left=event['start_date'],
                       height=bar_height, color=percentage_exclude_color, alpha=0.4,
                       hatch='\\\\\\\\', edgecolor=percentage_exclude_color, linewidth=1.5,
                       zorder=4)  # Higher z-order to ensure visibility above missing markers
                # Log that we're actually highlighting this event
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
            plt.close(fig)
            
        return save_path

    # def visualize_monitoring_thresholds(self, threshold_results=None, save_path=None):
    #     """
    #     Create visualizations of monitoring threshold analysis showing:
    #     1. ROC curves for different metrics
    #     2. Warning time distributions
    #     3. Threshold crossings on example trajectories
    #     4. Performance metrics across different threshold levels
    #     5. Recommended thresholds with practical guidance

    #     Parameters:
    #         threshold_results: Results from analyze_monitoring_thresholds()
    #         save_path: Path to save visualization

    #     Returns:
    #         dict: Updated threshold results with visualization paths
    #     """
    #     self.logger.info("Visualizing monitoring threshold analysis...")

    #     # Run threshold analysis if not already done
    #     if threshold_results is None:
    #         # In a class context, you would likely call self.analyze_monitoring_thresholds()
    #         # For standalone testing, ensure threshold_results is passed or loaded.
    #         # Re-running analysis here might be redundant if already done.
    #         # threshold_results = self.analyze_monitoring_thresholds() # Assuming this runs it
    #         pass # Assume threshold_results is passed in for now

    #     if threshold_results is None or not isinstance(threshold_results, dict) or 'metrics' not in threshold_results:
    #         self.logger.error("Threshold analysis failed or returned invalid/missing results.")
    #         return None

    #     # --- Create Figure ---
    #     fig_size = self.config.get('fig_size_thresholds', (14, 16))
    #     fig = plt.figure(figsize=fig_size, constrained_layout=False) # Use constrained_layout=False with manual adjustments
    #     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1],
    #                         figure=fig, hspace=0.45, wspace=0.25) # Adjust spacing

    #     # Define colors for different metrics
    #     metric_colors = {
    #         'posture_diff': COLORS.get('difference', '#1f77b4'), # Use .get for safety
    #         'posture_diff_3d_slope': '#0077B6',  # Blue
    #         'posture_diff_7d_slope': '#00B4D8',  # Lighter blue
    #         'num_tails_upright': COLORS.get('upright', '#2ca02c'),
    #         'num_tails_hanging': COLORS.get('hanging', '#d62728'),
    #     }
    #     default_color = '#666666'  # Gray for any undefined metrics

    #     # Get metrics list
    #     # Use the more detailed 'metrics_to_evaluate' if available, otherwise fallback
    #     metrics_info_list = threshold_results.get('metrics_to_evaluate', [])
    #     if not metrics_info_list:
    #         evaluated_metrics = threshold_results.get('metrics', {}).get('evaluated', [])
    #         # Create minimal info if only names are available
    #         metrics_info_list = [{'name': name, 'display_name': name.replace('_', ' ').title()} for name in evaluated_metrics]
    #     else:
    #         evaluated_metrics = [m['name'] for m in metrics_info_list]


    #     # Create display name lookup from metrics_info_list
    #     metrics_display = {info['name']: info.get('display_name', info['name']) for info in metrics_info_list}

    #     # Get the best overall metric if available
    #     best_metric = threshold_results.get('overall_best_metric')
    #     best_metric_display = threshold_results.get('overall_best_metric_display') # Already retrieved, use if available

    #     # --- Panel A: ROC-like Curve (Sensitivity vs. Specificity) ---
    #     ax0 = fig.add_subplot(gs[0, 0])
    #     ax0.set_title('A) ROC Analysis (Training Set)', fontsize=11, fontweight='bold', loc='left', pad=7)

    #     # Check if we have control data for specificity
    #     has_control_data = threshold_results.get('data_counts', {}).get('control_trajectories', 0) > 0

    #     plotted_metrics_roc = [] # Keep track of metrics plotted
    #     if has_control_data:
    #         # Plot ROC-like curves for each metric
    #         for metric_info in metrics_info_list:
    #             metric_name = metric_info['name']
    #             # Get performance data if available in recommendations
    #             if metric_name in threshold_results.get('recommendations', {}):
    #                 recommendation = threshold_results['recommendations'][metric_name]
    #                 best_percentile = recommendation.get('best_percentile')
    #                 # Use performance on TRAINING data for this plot
    #                 performance = recommendation.get('performance_on_training', {})

    #                 # Get points for different candidate thresholds if available
    #                 points = []
    #                 candidate_values = threshold_results.get('thresholds', {}).get(metric_name, {}).get('candidate_values', {})
    #                 candidate_perf = recommendation.get('candidate_performance', {}) # Assuming this might exist, or recalculate

    #                 # Fallback: Plot only the best point if candidate data is missing
    #                 if 'sensitivity' in performance and 'specificity' in performance:
    #                     points.append((1 - performance['specificity'],
    #                                 performance['sensitivity'],
    #                                 best_percentile)) # Use the selected best percentile

    #                 # Sort by specificity (x-axis) for plotting line if multiple points existed
    #                 points.sort(key=lambda p: p[0])

    #                 # Check if we have points to plot
    #                 if points:
    #                     # Unzip points
    #                     x, y, labels = zip(*points)

    #                     # Plot points (use markers only if plotting single best point per metric)
    #                     metric_color = metric_colors.get(metric_name, default_color)
    #                     metric_label = metrics_display.get(metric_name, metric_name)
    #                     ax0.plot(x, y, 'o', color=metric_color, label=metric_label, markersize=7) # Plot points
    #                     plotted_metrics_roc.append(metric_name)

    #                     # Mark best overall point if this is the best metric
    #                     if metric_name == best_metric and best_percentile is not None:
    #                         best_point = points[0] # Since we only have one point per metric currently
    #                         ax0.plot(best_point[0], best_point[1], 'o', color=metric_color,
    #                                 markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    #                         ax0.annotate(f"Best: {best_percentile}%",
    #                                     xy=(best_point[0], best_point[1]),
    #                                     xytext=(10, 0), textcoords="offset points",
    #                                     fontsize=9, fontweight='bold')

    #         # Add reference line
    #         ax0.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    #         # Add grid and labels
    #         ax0.grid(True, linestyle=':', alpha=0.7)
    #         ax0.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    #         ax0.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)

    #         # Set axis limits
    #         ax0.set_xlim([-0.02, 1.02])
    #         ax0.set_ylim([-0.02, 1.02])

    #     else:
    #         # --- Alternative Panel A if no control data ---
    #         ax0.set_title('A) Sensitivity by Threshold (Training Set)', fontsize=11, fontweight='bold', loc='left', pad=7)
    #         # Plot sensitivity curve by threshold percentile
    #         for metric_info in metrics_info_list:
    #             metric_name = metric_info['name']
    #             # Get performance data if available in recommendations
    #             if metric_name in threshold_results.get('recommendations', {}):
    #                 recommendation = threshold_results['recommendations'][metric_name]
    #                 best_percentile = recommendation.get('best_percentile')
    #                 # Get candidate performance data (need sensitivity per percentile)
    #                 candidate_perf = recommendation.get('candidate_scores', {}) # Using scores dict for percentiles

    #                 points = []
    #                 # Need candidate_performance dict mapping percentile -> {sensitivity: val, ...}
    #                 # This structure isn't directly in the provided JSON, let's use the single best point
    #                 if 'performance_on_training' in recommendation:
    #                     perf = recommendation['performance_on_training']
    #                     if 'sensitivity' in perf and best_percentile is not None:
    #                         points.append((best_percentile, perf['sensitivity']))

    #                 # Check if we have points to plot
    #                 if points:
    #                     # Unzip points
    #                     x, y = zip(*points)

    #                     # Plot sensitivity point
    #                     metric_color = metric_colors.get(metric_name, default_color)
    #                     metric_label = metrics_display.get(metric_name, metric_name)
    #                     ax0.plot(x, y, 'o', color=metric_color, label=metric_label, markersize=7)
    #                     plotted_metrics_roc.append(metric_name)

    #                     # Mark best point if this is the best metric
    #                     if metric_name == best_metric and best_percentile is not None:
    #                         best_point = points[0]
    #                         ax0.plot(best_point[0], best_point[1], 'o', color=metric_color,
    #                                 markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    #                         ax0.annotate(f"Best: {best_percentile}%",
    #                                     xy=(best_point[0], best_point[1]),
    #                                     xytext=(10, 0), textcoords="offset points",
    #                                     fontsize=9, fontweight='bold')

    #         # Add grid and labels
    #         ax0.grid(True, linestyle=':', alpha=0.7)
    #         ax0.set_xlabel('Threshold Percentile', fontsize=10)
    #         ax0.set_ylabel('Sensitivity', fontsize=10)

    #         # Set axis limits
    #         ax0.set_xlim([0, 100])
    #         ax0.set_ylim([-0.02, 1.02])

    #     # Add legend for Panel A (only if metrics were plotted)
    #     if plotted_metrics_roc:
    #         ax0.legend(loc='lower right', frameon=True, fontsize=9)
    #     else:
    #         ax0.text(0.5, 0.5, 'No suitable metrics found for ROC plot', ha='center', va='center', color='gray')


    #     # --- Panel B: Warning Time Distribution (Training Set) ---
    #     ax1 = fig.add_subplot(gs[0, 1])
    #     ax1.set_title('B) Warning Time Distribution (Training Set)', fontsize=11, fontweight='bold', loc='left', pad=7)

    #     warning_data_raw = {}
    #     warning_means = {}
    #     metrics_with_warning_data = []

    #     for metric_info in metrics_info_list:
    #         metric_name = metric_info['name']
    #         if metric_name in threshold_results.get('recommendations', {}):
    #             recommendation = threshold_results['recommendations'][metric_name]
    #             # Use warning time on TRAINING data
    #             warning_stats = recommendation.get('warning_time_on_training', {})

    #             # Only add if we have raw warning times for boxplot
    #             if 'raw_times' in warning_stats and warning_stats['raw_times']:
    #                 warning_data_raw[metric_name] = warning_stats['raw_times']
    #                 warning_means[metric_name] = warning_stats.get('mean', 0)
    #                 metrics_with_warning_data.append(metric_name)
    #             elif 'mean' in warning_stats: # Fallback if no raw times but mean exists
    #                 warning_means[metric_name] = warning_stats['mean']
    #                 metrics_with_warning_data.append(metric_name)
    #                 # Optionally create synthetic data here if needed for visualization consistency
    #                 # warning_data_raw[metric_name] = [warning_stats['mean']] # Simple placeholder


    #     if metrics_with_warning_data:
    #         # Sort metrics by mean warning time (descending)
    #         sorted_metrics = sorted(metrics_with_warning_data, key=lambda m: warning_means.get(m, 0), reverse=True)

    #         # Prepare data for box plot (only those with raw data)
    #         bp_data = [warning_data_raw[m] for m in sorted_metrics if m in warning_data_raw]
    #         bp_labels_raw = [metrics_display.get(m, m) for m in sorted_metrics if m in warning_data_raw]
    #         bp_labels_mean = [f"{warning_means.get(m, 0):.1f}d" for m in sorted_metrics if m in warning_data_raw]
    #         bp_labels = [f"{raw}\n({mean})" for raw, mean in zip(bp_labels_raw, bp_labels_mean)]


    #         if bp_data: # Ensure we actually have data for boxplot
    #             # Create box plots
    #             box_colors = [metric_colors.get(m, default_color) for m in sorted_metrics if m in warning_data_raw]
    #             bp = ax1.boxplot(bp_data, labels=bp_labels, patch_artist=True, showfliers=False, widths=0.6)

    #             # Color the boxes
    #             for patch, color in zip(bp['boxes'], box_colors):
    #                 patch.set_facecolor(color)
    #                 patch.set_alpha(0.7)

    #             # Color the medians
    #             for median in bp['medians']:
    #                 median.set_color('black')
    #                 median.set_linewidth(1.5)

    #             # Add vertical grid lines
    #             ax1.yaxis.grid(True, linestyle=':', alpha=0.7)
    #             ax1.set_ylabel("Advance Warning Time (Days)", fontsize=10)

    #             # Highlight the best metric's box if it has data
    #             best_metric_idx = None
    #             if best_metric in warning_data_raw and best_metric in sorted_metrics:
    #                 try:
    #                     # Get index within the subset that has raw data
    #                     sorted_metrics_raw = [m for m in sorted_metrics if m in warning_data_raw]
    #                     best_metric_idx = sorted_metrics_raw.index(best_metric)
    #                     best_metric_x = best_metric_idx + 1  # Box plot indices start at 1

    #                     # Highlight the best metric's box
    #                     if 'boxes' in bp and len(bp['boxes']) > best_metric_idx:
    #                         bp['boxes'][best_metric_idx].set_edgecolor('black')
    #                         bp['boxes'][best_metric_idx].set_linewidth(2)
    #                         # Add annotation slightly above the box
    #                         box_top = bp['caps'][best_metric_idx*2 + 1].get_ydata()[0] # Top whisker
    #                         ax1.annotate("Best Metric",
    #                                 xy=(best_metric_x, box_top),
    #                                 xytext=(0, 5), textcoords="offset points",
    #                                 fontsize=9, fontweight='bold', ha='center', va='bottom',
    #                                 # arrowprops=dict(arrowstyle="->", color='black') # Optional arrow
    #                                 )
    #                 except ValueError:
    #                     self.logger.debug(f"Best metric {best_metric} not found in metrics with raw warning times for highlighting.")


    #             # Add reference lines from config
    #             ref_lines = self.config.get('warning_time_ref_lines', [3.0, 7.0])
    #             line_colors = ['#AE2012', '#0A9396', '#EE9B00'] # Red, Teal, Orange/Yellow
    #             line_labels = ['minimum', 'target', 'ideal'] # Match config intent
    #             for i, line_val in enumerate(ref_lines):
    #                 color = line_colors[i % len(line_colors)]
    #                 label = line_labels[i % len(line_labels)]
    #                 ax1.axhline(y=line_val, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    #                 # Adjust text position to avoid overlap with plot elements
    #                 ax1.text(ax1.get_xlim()[1]*1.01, line_val, f'{line_val:.0f}d ({label})',
    #                         va='center', ha='left', color=color, fontsize=8)

    #             # Add sample size indicator
    #             total_outbreaks_train = threshold_results.get('validation', {}).get('n_train_trajectories', '?')
    #             ax1.text(0.02, 0.98, f"Based on N={total_outbreaks_train} training trajectories", transform=ax1.transAxes,
    #                     fontsize=9, va='top', ha='left',
    #                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    #             ax1.tick_params(axis='x', labelsize=9)
    #             ax1.set_ylim(bottom=max(0, ax1.get_ylim()[0])) # Ensure y-axis starts at 0 or above

    #         else:
    #             ax1.text(0.5, 0.5, 'No raw warning time data available for boxplot', ha='center', va='center', color='gray', fontsize=10)


    #     else:
    #         ax1.text(0.5, 0.5, 'No warning time data calculated', ha='center', va='center', color='gray', fontsize=10)
    #     ax1.set_ylabel("Advance Warning Time (Days)", fontsize=10)


    #     # --- Panel C: Example Threshold Crossings ---
    #     ax2 = fig.add_subplot(gs[1, 0])
    #     ax2.set_title('C) Example Threshold Crossings (Best Metric)', fontsize=11, fontweight='bold', loc='left', pad=7)

    #     # Check if best metric and raw data are available
    #     raw_data = threshold_results.get('raw_data', {})
    #     outbreak_trajectories = raw_data.get('train_trajectories', []) # Use training trajectories for consistency
    #     if not outbreak_trajectories:
    #         outbreak_trajectories = raw_data.get('outbreak_trajectories', []) # Fallback to all

    #     if best_metric and best_metric in threshold_results.get('recommendations', {}) and outbreak_trajectories:
    #         recommendation = threshold_results['recommendations'][best_metric]
    #         best_pct = recommendation.get('best_percentile')
    #         threshold_value = recommendation.get('threshold_value')

    #         # Get direction for threshold crossing
    #         metric_direction = 'below' # Default
    #         metric_units = ''
    #         for metric_info in metrics_info_list:
    #             if metric_info['name'] == best_metric:
    #                 metric_direction = metric_info.get('direction', 'below')
    #                 metric_units = metric_info.get('units', '')
    #                 break

    #         # Select up to N random trajectories to display
    #         n_traj = self.config.get('threshold_example_trajectory_count', 3)
    #         num_to_show = min(n_traj, len(outbreak_trajectories))

    #         if num_to_show > 0:
    #             np.random.seed(self.config.get('random_seed', 42)) # Use configured seed
    #             selected_indices = np.random.choice(len(outbreak_trajectories), num_to_show, replace=False)

    #             # Plot each selected trajectory
    #             pens_plotted = set() # Avoid duplicate labels if multiple trajectories from same pen
    #             plot_lines = []
    #             plot_labels = []

    #             for i, idx in enumerate(selected_indices):
    #                 trajectory = outbreak_trajectories[idx]
    #                 if not isinstance(trajectory, pd.DataFrame) or trajectory.empty:
    #                     continue

    #                 # Skip if best metric not in trajectory
    #                 if best_metric not in trajectory.columns:
    #                     self.logger.warning(f"Best metric '{best_metric}' not found in example trajectory index {idx}.")
    #                     continue

    #                 # Get pen ID for label
    #                 pen_label = "Unknown Pen"
    #                 if 'pen' in trajectory.columns:
    #                     pen_id = trajectory['pen'].iloc[0]
    #                     # Optional: Add datespan if needed for uniqueness
    #                     # datespan = trajectory['datespan'].iloc[0] if 'datespan' in trajectory.columns else ''
    #                     # pen_label = f"{pen_id} ({datespan})" if datespan else pen_id
    #                     pen_label = str(pen_id) # Keep it simple for the plot legend

    #                 # Get x and y data (ensure numeric types)
    #                 trajectory['days_before_removal'] = pd.to_numeric(trajectory['days_before_removal'], errors='coerce')
    #                 trajectory[best_metric] = pd.to_numeric(trajectory[best_metric], errors='coerce')
    #                 # Drop rows where essential data is missing after coercion
    #                 traj_filtered = trajectory.dropna(subset=['days_before_removal', best_metric])
    #                 if traj_filtered.empty: continue

    #                 # Sort by days_before_removal (x-axis)
    #                 traj_sorted = traj_filtered.sort_values(by='days_before_removal', ascending=False)

    #                 x = traj_sorted['days_before_removal']
    #                 y = traj_sorted[best_metric]
    #                 if x.empty or y.empty: continue


    #                 # Find the *first* threshold crossing time
    #                 first_crossing_x = np.nan
    #                 first_crossing_y = np.nan
    #                 for days, value in zip(x, y):
    #                     if pd.isna(value): continue
    #                     is_crossed = (metric_direction == 'below' and value <= threshold_value) or \
    #                                 (metric_direction == 'above' and value >= threshold_value)
    #                     if is_crossed:
    #                         first_crossing_x = days
    #                         first_crossing_y = value
    #                         break # Stop at the first crossing

    #                 # Plot the trajectory
    #                 line_color = plt.cm.viridis(i / num_to_show) # Use viridis colormap for better variation
    #                 line, = ax2.plot(x, y, '-', color=line_color, linewidth=1.5, alpha=0.8)

    #                 # Add label only once per pen if desired, or always add if trajectories are unique events
    #                 # For simplicity, add label for each trajectory line shown
    #                 plot_lines.append(line)
    #                 plot_labels.append(pen_label)


    #                 # Mark first threshold crossing if found
    #                 if not pd.isna(first_crossing_x):
    #                     ax2.plot(first_crossing_x, first_crossing_y, 'o', color=line_color, markersize=7,
    #                             markeredgecolor='black', markeredgewidth=1.0)
    #                     # Add days annotation near the marker
    #                     ax2.annotate(f"{first_crossing_x:.1f}d",
    #                                 xy=(first_crossing_x, first_crossing_y),
    #                                 xytext=(5, -5), textcoords="offset points",
    #                                 fontsize=8, color='black', ha='left', va='top') # Adjusted text position

    #             # Add threshold line
    #             threshold_label = f"Threshold ({best_pct}th perc.) = {threshold_value:.2f}"
    #             hline = ax2.axhline(y=threshold_value, color='#AE2012', linestyle='--', linewidth=1.5, alpha=0.9)
    #             # Add threshold text label (position depends on direction)
    #             text_y_pos = threshold_value
    #             va = 'bottom' if metric_direction == 'below' else 'top'
    #             # Ensure text is within plot bounds
    #             x_range = ax2.get_xlim()
    #             text_x_pos = x_range[0] + 0.02 * (x_range[1] - x_range[0]) # Position near left edge
    #             ax2.text(text_x_pos, text_y_pos, threshold_label, va=va, ha='left',
    #                     color='#AE2012', fontsize=8, backgroundcolor='white', alpha=0.7)


    #             # Add removal day reference line
    #             ax2.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.7)
    #             # Add label for removal line, ensuring it's visible
    #             y_range = ax2.get_ylim()
    #             ax2.text(0.0, y_range[1], ' Removal', ha='left', va='top', fontsize=9, rotation=90, alpha=0.8)


    #             # Add grid and labels
    #             ax2.grid(True, linestyle=':', alpha=0.7)
    #             ax2.set_xlabel('Days Before Removal', fontsize=10)
    #             y_label = best_metric_display or best_metric # Use display name if available
    #             if metric_units: y_label += f" ({metric_units})"
    #             ax2.set_ylabel(y_label, fontsize=10)

    #             # Add legend if multiple lines were plotted
    #             if len(plot_lines) > 1:
    #                 ax2.legend(plot_lines, plot_labels, loc='best', frameon=True, fontsize=8) # Use 'best' location

    #             # Invert X axis if showing days *before* removal
    #             if all(x <= 0 for x in ax2.get_lines()[0].get_xdata(orig=False)): # Check if data is mostly <= 0
    #                 ax2.invert_xaxis()
    #                 ax2.set_xlabel('Days Before Removal', fontsize=10) # Keep label intuitive


    #         else:
    #             ax2.text(0.5, 0.5, 'No valid example trajectories found', ha='center', va='center', color='gray', fontsize=10)

    #     else:
    #         message = 'Best metric or raw trajectory data not available'
    #         if not best_metric: message = 'Best metric not determined'
    #         elif not outbreak_trajectories: message = 'Raw trajectory data unavailable'
    #         elif best_metric not in threshold_results['recommendations']: message = f"Recommendations missing for metric '{best_metric}'"

    #         ax2.text(0.5, 0.5, message, ha='center', va='center', color='gray', fontsize=10)
    #         ax2.set_xlabel('Days Before Removal', fontsize=10)
    #         ax2.set_ylabel('Metric Value', fontsize=10)


    #     # --- Panel D: Performance Metrics (Sensitivity/Specificity) at Best Threshold ---
    #     ax3 = fig.add_subplot(gs[1, 1])
    #     title_metric_name = best_metric_display or (best_metric if best_metric else "Metric")
    #     ax3.set_title(f'D) Performance at Optimal Threshold ({title_metric_name})',
    #                 fontsize=11, fontweight='bold', loc='left', pad=7)

    #     if best_metric and best_metric in threshold_results.get('recommendations', {}):
    #         recommendation = threshold_results['recommendations'][best_metric]
    #         best_pct = recommendation.get('best_percentile')
    #         threshold_value = recommendation.get('threshold_value')

    #         # Get performance on TRAINING set
    #         performance_train = recommendation.get('performance_on_training', {})
    #         sens_train = performance_train.get('sensitivity')
    #         spec_train = performance_train.get('specificity') if has_control_data else None
    #         acc_train = (sens_train + spec_train) / 2 if sens_train is not None and spec_train is not None else sens_train # Balanced accuracy or just sensitivity

    #         # Get performance on HOLDOUT set if available
    #         holdout_eval = threshold_results.get('validation', {}).get('holdout_evaluation', {})
    #         performance_holdout = holdout_eval.get(best_metric, {})
    #         sens_holdout = performance_holdout.get('sensitivity')
    #         spec_holdout = performance_holdout.get('specificity') if has_control_data else None
    #         acc_holdout = performance_holdout.get('balanced_accuracy') # Use pre-calculated BA

    #         metrics_plot = ['Sensitivity', 'Specificity', 'Accuracy']
    #         values_train = [sens_train, spec_train, acc_train]
    #         values_holdout = [sens_holdout, spec_holdout, acc_holdout]

    #         # Filter out None values if specificity is not available
    #         if not has_control_data:
    #             metrics_plot = ['Sensitivity']
    #             values_train = [sens_train]
    #             values_holdout = [sens_holdout]


    #         x_pos = np.arange(len(metrics_plot))
    #         width = 0.35

    #         colors_train = ['#0A9396', '#AE2012', '#EE9B00'] # Teal, Red, Orange
    #         colors_holdout = ['#94D2BD', '#BB3E03', '#E9C46A'] # Lighter versions

    #         bars_train, bars_holdout = None, None # Initialize

    #         # Plot training bars
    #         if any(v is not None for v in values_train): # Check if there's data to plot
    #             bars_train = ax3.bar(x_pos - width/2, [v if v is not None else 0 for v in values_train], width, label='Training Set',
    #                                 color=[colors_train[i] for i,v in enumerate(values_train)])

    #             # Add value labels for training bars
    #             for i, bar in enumerate(bars_train):
    #                 height = bar.get_height()
    #                 if values_train[i] is not None: # Only label if value exists
    #                     ax3.annotate(f'{height:.2f}',
    #                                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                                 xytext=(0, 3), textcoords="offset points",
    #                                 ha='center', va='bottom', fontsize=9)

    #         # Plot holdout bars if data exists
    #         if any(v is not None for v in values_holdout):
    #             bars_holdout = ax3.bar(x_pos + width/2, [v if v is not None else 0 for v in values_holdout], width, label='Holdout Set',
    #                                 color=[colors_holdout[i] for i,v in enumerate(values_holdout)])
    #             # Add value labels for holdout bars
    #             for i, bar in enumerate(bars_holdout):
    #                 height = bar.get_height()
    #                 if values_holdout[i] is not None:
    #                     ax3.annotate(f'{height:.2f}',
    #                                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                                 xytext=(0, 3), textcoords="offset points",
    #                                 ha='center', va='bottom', fontsize=9, alpha=0.9)


    #         # Add grid and labels
    #         ax3.grid(axis='y', linestyle=':', alpha=0.7)
    #         ax3.set_ylabel('Performance Metric Value', fontsize=10)
    #         ax3.set_xticks(x_pos)
    #         ax3.set_xticklabels(metrics_plot, fontsize=10)
    #         ax3.set_ylim([0, 1.1]) # Set fixed ylim for performance

    #         # Add legend
    #         ax3.legend(loc='best', frameon=True, fontsize=9)

    #         # Add threshold info text box
    #         threshold_text = f"Optimal Threshold:\n{best_pct}th percentile\nValue: {threshold_value:.3f}"
    #         ax3.text(0.98, 0.02, threshold_text, transform=ax3.transAxes,
    #                 fontsize=9, va='bottom', ha='right',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    #         # Optional: Add confidence intervals from stability analysis if available
    #         stability_info = recommendation.get('threshold_sensitivity', {})
    #         sens_ci = stability_info.get('sensitivity_confidence_interval')
    #         spec_ci = stability_info.get('specificity_confidence_interval')

    #         # Add error bars (representing CI) if data available
    #         if bars_train and sens_ci and len(sens_ci) == 2:
    #             sens_err = (sens_ci[1] - sens_ci[0]) / 2
    #             ax3.errorbar(x_pos[0] - width/2, values_train[0], yerr=sens_err, fmt='none', ecolor='black', capsize=3)
    #         if bars_train and spec_ci and len(spec_ci) == 2 and has_control_data and len(values_train)>1 and values_train[1] is not None:
    #             spec_err = (spec_ci[1] - spec_ci[0]) / 2
    #             ax3.errorbar(x_pos[1] - width/2, values_train[1], yerr=spec_err, fmt='none', ecolor='black', capsize=3)


    #     else:
    #         ax3.text(0.5, 0.5, 'Performance data not available\nfor the best metric.',
    #                 ha='center', va='center', color='gray', fontsize=10)
    #         ax3.set_ylim([0, 1.1])
    #         ax3.set_ylabel('Performance Metric Value', fontsize=10)


    #     # --- Panel E: Cross-Validation Performance --- (Renamed from F to E)
    #     ax4 = fig.add_subplot(gs[2, 0])
    #     ax4.set_title('E) Cross-Validation Performance (Mean over Folds)', fontsize=11, fontweight='bold', loc='left', pad=7)

    #     cv_results = threshold_results.get('validation', {}).get('cross_validation', {})

    #     if cv_results and any(m in cv_results for m in evaluated_metrics):
    #         metrics_to_plot_cv = []
    #         mean_sensitivities_cv = []
    #         mean_specificities_cv = []
    #         std_sens_cv = [] # For error bars
    #         std_spec_cv = [] # For error bars
    #         labels_cv = []

    #         # Gather CV summary data for evaluated metrics that have results
    #         for metric_info in metrics_info_list:
    #             metric_name = metric_info['name']
    #             if metric_name in cv_results:
    #                 metric_cv_data = cv_results[metric_name]
    #                 if 'summary' in metric_cv_data and metric_cv_data['summary']: # Check summary exists and is not empty
    #                     summary = metric_cv_data['summary']
    #                     metrics_to_plot_cv.append(metric_name)
    #                     mean_sensitivities_cv.append(summary.get('mean_test_sensitivity', 0))
    #                     mean_specificities_cv.append(summary.get('mean_test_specificity', 0) if has_control_data else 0) # Only plot spec if controls exist
    #                     # Get std deviations if available in summary (need to ensure they were calculated)
    #                     std_sens_cv.append(metric_cv_data.get('std_test_sensitivity', 0)) # Look for std in main metric dict
    #                     std_spec_cv.append(metric_cv_data.get('std_test_specificity', 0)) # Look for std in main metric dict

    #                     labels_cv.append(metrics_display.get(metric_name, metric_name))


    #         if metrics_to_plot_cv: # Proceed if we found CV data for at least one metric
    #             x_cv = np.arange(len(metrics_to_plot_cv))
    #             width_cv = 0.35

    #             # Colors consistent with Panel D
    #             sens_color_cv = '#0A9396'
    #             spec_color_cv = '#AE2012'

    #             # Plot Sensitivity bars
    #             sens_bars_cv = ax4.bar(x_cv - width_cv/2, mean_sensitivities_cv, width_cv, label='Mean Sensitivity',
    #                                 color=sens_color_cv, yerr=std_sens_cv, capsize=3, ecolor='darkgrey')

    #             # Plot Specificity bars only if control data exists
    #             spec_bars_cv = None
    #             if has_control_data:
    #                 spec_bars_cv = ax4.bar(x_cv + width_cv/2, mean_specificities_cv, width_cv, label='Mean Specificity',
    #                                     color=spec_color_cv, yerr=std_spec_cv, capsize=3, ecolor='darkgrey')

    #             # Add value labels centered above bars (adjusted for error bars)
    #             for i, bar in enumerate(sens_bars_cv):
    #                 height = bar.get_height()
    #                 y_pos = height + std_sens_cv[i] + 0.01 # Position above error bar
    #                 ax4.annotate(f'{height:.2f}',
    #                             xy=(bar.get_x() + bar.get_width() / 2, y_pos),
    #                             xytext=(0, 3), textcoords="offset points",
    #                             ha='center', va='bottom', fontsize=8)

    #             if spec_bars_cv:
    #                 for i, bar in enumerate(spec_bars_cv):
    #                     height = bar.get_height()
    #                     y_pos = height + std_spec_cv[i] + 0.01
    #                     ax4.annotate(f'{height:.2f}',
    #                                 xy=(bar.get_x() + bar.get_width() / 2, y_pos),
    #                                 xytext=(0, 3), textcoords="offset points",
    #                                 ha='center', va='bottom', fontsize=8)

    #             # Customize plot
    #             ax4.set_ylabel('Mean Test Performance', fontsize=10)
    #             ax4.set_xticks(x_cv)
    #             ax4.set_xticklabels(labels_cv, rotation=45, ha='right', fontsize=9)
    #             ax4.legend(loc='best', frameon=True, fontsize=9)
    #             ax4.grid(axis='y', linestyle=':', alpha=0.7)
    #             ax4.set_ylim([0, 1.15]) # Extend ylim slightly for labels

    #             # Add performance target line
    #             target_line = self.config.get('cv_performance_target_line', 0.8)
    #             ax4.axhline(y=target_line, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    #             ax4.text(ax4.get_xlim()[1]*0.98, target_line + 0.01, f'Target ({target_line:.1f})',
    #                     ha='right', va='bottom', fontsize=8, color='green')

    #             # Add cross-validation details text box
    #             cv_method = threshold_results.get('validation', {}).get('method', 'N/A')
    #             n_folds = 'N/A'
    #             # Try to get n_folds from the first metric's summary
    #             first_metric = metrics_to_plot_cv[0]
    #             n_folds = cv_results.get(first_metric, {}).get('summary', {}).get('n_folds', 'N/A')

    #             cv_details_text = f"Method: {cv_method}\nFolds/Pens: {n_folds}"
    #             ax4.text(0.02, 0.98, cv_details_text, transform=ax4.transAxes,
    #                     ha='left', va='top', fontsize=8, color='black',
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    #         else:
    #             ax4.text(0.5, 0.5, 'No cross-validation summary data found', ha='center', va='center', color='gray')


    #     else:
    #         ax4.text(0.5, 0.5, 'Cross-validation data not available', ha='center', va='center', color='gray')
    #         ax4.set_ylim([0, 1.1])

    #     # --- Overall Figure Adjustments ---
    #     fig.suptitle('Monitoring Threshold Analysis Results', fontsize=14, fontweight='bold', y=0.98)

    #     # Adjust layout manually *after* plotting everything
    #     # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap
    #     plt.subplots_adjust(top=0.92, bottom=0.05, left=0.08, right=0.95, hspace=0.45, wspace=0.25)

    #     # Save figure
    #     if save_path is None:
    #         filename = self.config.get('viz_thresholds_filename', 'monitoring_threshold_analysis.png')
    #         save_path = os.path.join(self.config['output_dir'], filename)

    #     try:
    #         # Ensure output directory exists
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         dpi = self.config.get('figure_dpi', 600)
    #         plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    #         self.logger.info(f"Saved monitoring threshold visualization to {save_path}")
    #     except Exception as e:
    #         self.logger.error(f"Failed to save monitoring threshold visualization: {e}", exc_info=True) # Add traceback info

    #     plt.close(fig)

    #     # Add visualization path to results
    #     if threshold_results and isinstance(threshold_results, dict):
    #         threshold_results['visualization_path'] = save_path

    #     return threshold_results
