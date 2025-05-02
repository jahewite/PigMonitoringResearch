import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import logging

from evaluation.tail_posture_analysis.utils import set_plotting_style, lighten_color, COLORS
from evaluation.activity_analysis.pig_activity_analyzer import PigBehaviorAnalyzer

class PigBehaviorVisualizer(PigBehaviorAnalyzer):
    """Methods for visualizing pig behavior analysis results."""
    
    def __init__(self, *args, **kwargs):
        # Ensure logger is initialized in the base class or here
        super().__init__(*args, **kwargs)  # Call parent __init__
        if not hasattr(self, 'logger'):  # If logger wasn't set by parent
            self.logger = logging.getLogger(__name__)
            if not self.logger.hasHandlers():  # Basic config if no handlers attached
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Set the style upon instantiation
        set_plotting_style(self.config)
        self.logger.info("Dissertation quality plotting style set for behavior visualization.")
        
        if not hasattr(self, 'behavior_metrics'):
            self.behavior_metrics = ['num_pigs_lying', 'num_pigs_notLying', 'activity']

        # Updated metric_colors (only for activity)
        self.metric_colors = {
            'num_pigs_lying': COLORS.get('lying', '#1f77b4'),
            'num_pigs_notLying': COLORS.get('primary_metric', '#ff7f0e'),
            'activity': COLORS.get('activity', '#2ca02c')
        }

        # Updated metric_display_names
        self.metric_display_names = {
            'num_pigs_lying': 'Anzahl liegender Schweine',
            'num_pigs_notLying': 'Anzahl nicht liegender Schweine',
            'activity': 'Aktivitätslevel'
        }
    
    def _style_violin_box(self, ax, data, labels, violin_color, box_color, scatter_color, violin_width=0.6):
        """Helper to style combined violin and box plots (Dissertation Quality)."""
        if not data or all(d.empty for d in data):  # Check if all data series are empty
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color=COLORS['annotation'], fontsize=plt.rcParams['font.size'])
            return  # No data to plot

        try:
            # Violin Plot - Lighter fill, slightly darker edge
            parts = ax.violinplot(data, showmeans=False, showmedians=False, widths=violin_width)  # Mean/Median handled by boxplot
            for pc in parts['bodies']:
                pc.set_facecolor(lighten_color(violin_color, 0.7))  # More lightening for fill
                pc.set_edgecolor(violin_color)  # Use main color for edge
                pc.set_alpha(0.9)  # Slightly less transparent
                pc.set_linewidth(0.8)  # Thin edge

            # Explicitly set colors/alphas based on input parameters for box fill
            bplot = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,  # Flier handling in rcParams
                               showmeans=True,  # Use rcParams setting
                               widths=violin_width * 0.3,  # Narrower box inside violin
                               positions=np.arange(1, len(data) + 1))  # Ensure positions match violin

            for patch in bplot['boxes']:
                patch.set_facecolor(lighten_color(box_color, 0.5))  # Light fill for box
                patch.set_alpha(0.85)
                patch.set_edgecolor(box_color)  # Main color for edge
                patch.set_linewidth(plt.rcParams['boxplot.boxprops.linewidth'])  # Use rcParams linewidth
            # Use rcParams for whiskers and caps styling
            for whisker in bplot['whiskers']:
                whisker.set(color=plt.rcParams['axes.edgecolor'],  # Match axes edge color
                           linewidth=plt.rcParams['boxplot.whiskerprops.linewidth'],
                           linestyle=plt.rcParams['boxplot.whiskerprops.linestyle'])
            for cap in bplot['caps']:
                cap.set(color=plt.rcParams['axes.edgecolor'],
                        linewidth=plt.rcParams['boxplot.capprops.linewidth'])

            # Scatter Plot (Jitter) - Smaller, more transparent points
            for i, d in enumerate(data):
                if d is not None and not d.empty:  # Check d is not None
                    x_jitter = np.random.normal(i + 1, 0.025, size=len(d))  # Smaller jitter spread
                    ax.scatter(x_jitter, d, alpha=0.3, s=10, color=scatter_color,  # Smaller size, lower alpha
                               edgecolor='none', zorder=3)  # No edges for less clutter
        except Exception as e:
            self.logger.error(f"Error styling violin/box plot: {e}", exc_info=True)
            ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', color='red')

    def _style_boxplot(self, ax, data, labels, colors, show_scatter=True, scatter_alpha=0.3, widths=0.6):
        """Helper to style box plots consistently (Dissertation Quality). Returns the boxplot dictionary."""
        if not data or all(d is None or len(d) == 0 for d in data):  # Check if data is empty or contains only empty lists/None
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color=COLORS['annotation'], fontsize=plt.rcParams['font.size'])
            return None  # Return None if no data

        # Use rcParams for median/mean properties directly
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,  # Flier handling in rcParams
                        showmeans=True,  # Use rcParams setting
                        widths=widths)

        # Style boxes, whiskers, caps
        for i, box in enumerate(bp['boxes']):
            box_color = colors[i % len(colors)]
            box.set_facecolor(lighten_color(box_color, 0.6))  # Lighter fill
            box.set_edgecolor(box_color)  # Main color edge
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
                if d is not None and len(d) > 0:  # Check d is not None
                    scatter_color = colors[i % len(colors)]
                    x_jitter = np.random.normal(i + 1, 0.03, size=len(d))  # Slightly tighter jitter
                    ax.scatter(x_jitter, d, alpha=scatter_alpha, s=10, color=scatter_color,  # Smaller dots
                               edgecolor='none', zorder=3)  # No edges

        return bp

    def _add_stats_annotation(self, ax, text, loc='upper right', fontsize=None, **kwargs):
        """Helper to add standardized statistics box (Dissertation Quality)."""
        if fontsize is None:
            fontsize = plt.rcParams['legend.fontsize']  # Use legend font size from rcParams

        # Slightly cleaner bbox
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9,  # Higher alpha
                          edgecolor='#CCCCCC', linewidth=0.5)  # Lighter edge

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
                color=plt.rcParams['text.color'],  # Use default text color
                bbox=bbox_props)
    
    def visualize_behavior_metrics(self, metric=None, save_path=None):
        """
        Create visualizations for behavior metrics analysis.
        
        Parameters:
            metric (str, optional): Specific metric to visualize. If None, visualize all metrics.
            save_path (str, optional): Path to save the visualization. If None, use default path.
            
        Returns:
            dict: Dictionary with metadata about the visualizations created.
        """
        self.logger.info(f"Visualizing behavior metrics analysis{f' for {metric}' if metric else ''}...")
        set_plotting_style(self.config)
        
        metrics_to_visualize = [metric] if metric else self.behavior_metrics
        saved_files = {}
        
        for current_metric in metrics_to_visualize:
            if current_metric not in self.behavior_metrics:
                self.logger.warning(f"Unknown metric: {current_metric}. Skipping visualization.")
                continue
                
            # Check if we have data for this metric
            if current_metric not in self.pre_outbreak_stats or self.pre_outbreak_stats[current_metric].empty:
                self.logger.warning(f"No pre-outbreak statistics available for {current_metric}. Skipping visualization.")
                continue
                
            # Generate visualization for this metric
            self.logger.info(f"Creating visualization for {current_metric}")
            
            # Setup the figure
            fig_size = self.config.get('fig_size_behavior', (11, 10))
            fig = plt.figure(figsize=fig_size)
            
            # Create grid layout: 2x2 grid with equal sizing
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                                  hspace=0.4, wspace=0.3)
            
            # Get the color for this metric
            metric_color = self.metric_colors.get(current_metric, COLORS['difference'])
            scatter_color = metric_color
            
            # Get display name for this metric
            display_name = self.metric_display_names.get(current_metric, current_metric.replace('_', ' ').title())
            
            # --- Panel A (Row 0, Col 0): Value Distributions ---
            ax0 = fig.add_subplot(gs[0, 0])
            days_list = self.config.get('days_before_list', [7, 3, 1])  # Order for plotting
            data_to_plot = []
            labels = []
            
            # Collect data in desired plot order (e.g., 7d, 3d, 1d, Removal)
            plot_order_days = sorted([d for d in days_list if f'value_{d}d_before' in self.pre_outbreak_stats[current_metric].columns], reverse=True)  # e.g. [7, 3, 1]
            for d in plot_order_days:
                col_name = f'value_{d}d_before'
                data_series = self.pre_outbreak_stats[current_metric][col_name].dropna()
                if not data_series.empty:
                    data_to_plot.append(data_series)
                    # labels.append(f'{d}d Before')
                    labels.append(f'{d}T vorher')
            
            # Add removal value
            value_at_removal = self.pre_outbreak_stats[current_metric]['value_at_removal'].dropna()
            if not value_at_removal.empty:
                data_to_plot.append(value_at_removal)
                # labels.append('At Removal')
                labels.append('Bei Entfernung')
                
            # Calculate statistics for reference lines
            if not value_at_removal.empty:
                mean_val = value_at_removal.mean()
                median_val = value_at_removal.median()
                p25 = value_at_removal.quantile(0.25)
                
            if data_to_plot:
                # Style the violin box plot
                self._style_violin_box(ax0, data_to_plot, labels,
                                      violin_color=lighten_color(metric_color, 0.6),
                                      box_color=lighten_color(metric_color, 0.3),
                                      scatter_color=scatter_color,
                                      violin_width=self.config.get('violin_width', 0.6))
                                      
                # Add horizontal lines for key stats
                if pd.notna(mean_val):
                    ax0.axhline(y=mean_val, color=COLORS.get('critical', 'red'), linestyle='--', linewidth=1.5, alpha=0.9, zorder=2)
                if pd.notna(median_val):
                    ax0.axhline(y=median_val, color=COLORS.get('warning', 'orange'), linestyle='-', linewidth=1.5, alpha=0.9, zorder=2)
                if pd.notna(p25):
                    ax0.axhline(y=p25, color=COLORS.get('secondary_metric', 'green'), linestyle='-.', linewidth=1.5, alpha=0.8, zorder=2)
            else:
                ax0.text(0.5, 0.5, 'No Data to Plot', ha='center', va='center', color=COLORS.get('annotation', 'grey'))
                
            # ax0.set_title(f'A) {display_name} Distribution')
            ax0.set_title(f'A) Verteilung des {display_name}')
            # ax0.set_ylabel(display_name)
            ax0.set_ylabel(display_name)
            # ax0.set_xlabel('Time Point Relative to Removal')
            ax0.set_xlabel('Zeitpunkt relativ zur Entfernung')
            ax0.tick_params(axis='x', labelsize=9)
            ax0.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
            ax0.grid(axis='x', visible=False)
            
            # --- Panel B (Row 0, Col 1): Absolute Change ---
            ax1 = fig.add_subplot(gs[0, 1])
            abs_changes_data = []
            abs_labels = []
            plot_order_abs = sorted([d for d in days_list if f'abs_change_{d}d' in self.pre_outbreak_stats[current_metric].columns], reverse=True)
            for d in plot_order_abs:
                col = f'abs_change_{d}d'
                data = self.pre_outbreak_stats[current_metric][col].dropna()
                if not data.empty:
                    # Apply outlier filtering if needed
                    iqr_factor = self.config.get('abs_change_outlier_iqr_factor', 3)
                    q1, q3 = data.quantile(0.25), data.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - iqr_factor * iqr
                        upper_bound = q3 + iqr_factor * iqr
                        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                    else:
                        filtered_data = data
                        
                    if not filtered_data.empty:
                        abs_changes_data.append(filtered_data)
                        # abs_labels.append(f'{d}d Window')
                        abs_labels.append(f'{d}T Fenster')
                        
            if abs_changes_data:
                self._style_violin_box(ax1, abs_changes_data, abs_labels,
                                      violin_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.6),
                                      box_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3),
                                      scatter_color=COLORS.get('secondary_metric', 'green'),
                                      violin_width=self.config.get('violin_width', 0.6))
                ax1.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0)
            else:
                ax1.text(0.5, 0.5, 'No Absolute Change Data', ha='center', va='center', color=COLORS.get('annotation', 'grey'))
                
            # ax1.set_title(f'B) Absolute Change in {display_name}')
            ax1.set_title(f'B) Absolute Änderung des {display_name}')
            # ax1.set_xlabel('Time Window Before Removal')
            ax1.set_xlabel('Zeitfenster vor Entfernung')
            # ax1.set_ylabel(f'Change in {display_name}')
            ax1.set_ylabel(f'Änderung des {display_name}')
            ax1.tick_params(axis='x', labelsize=9)
            ax1.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
            ax1.grid(axis='x', visible=False)
            
            # --- Panel C (Row 1, Col 0): Window Slope ---
            ax2 = fig.add_subplot(gs[1, 0])
            slope_data_list = []
            slope_labels = []
            analysis_windows = self.config.get('analysis_window_days', [7, 3])
            plot_order_slope = sorted([d for d in analysis_windows if f'{d}d_window_slope' in self.pre_outbreak_stats[current_metric].columns], reverse=True)
            for d in plot_order_slope:
                col = f'{d}d_window_slope'
                data = self.pre_outbreak_stats[current_metric][col].dropna()
                if not data.empty:
                    slope_data_list.append(data)
                    # slope_labels.append(f'{d}d Window')
                    slope_labels.append(f'{d}T Fenster')
                    
            if slope_data_list:
                self._style_violin_box(ax2, slope_data_list, slope_labels,
                                      violin_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.6),
                                      box_color=lighten_color(COLORS.get('secondary_metric', 'green'), 0.3),
                                      scatter_color=COLORS.get('secondary_metric', 'green'),
                                      violin_width=self.config.get('violin_width', 0.6))
                ax2.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0)
            else:
                ax2.text(0.5, 0.5, 'No Slope Data Calculated', ha='center', va='center', color=COLORS.get('annotation', 'grey'))
                
            # ax2.set_title(f'C) Slope in Pre-Outbreak Windows ({display_name})')
            ax2.set_title(f'C) Steigung in Vor-Ausbruch-Fenstern ({display_name})')
            # ax2.set_xlabel('Window Ending at Removal')
            ax2.set_xlabel('Fenster mit Ende bei Entfernung')
            # ax2.set_ylabel('Slope (Change per Day)')
            ax2.set_ylabel('Steigung (Änderung pro Tag)')
            ax2.tick_params(axis='x', labelsize=9)
            ax2.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
            ax2.grid(axis='x', visible=False)
            
            # --- Panel D (Row 1, Col 1): Trajectory ---
            ax3 = fig.add_subplot(gs[1, 1])
            trajectory_data = pd.DataFrame()
            # Expand potential columns to include more days if available
            potential_traj_cols = {f'value_{d}d_before': -d for d in [10, 7, 5, 3, 1]}
            potential_traj_cols['value_at_removal'] = 0
            traj_cols_ordered_map = {col: day for col, day in potential_traj_cols.items() 
                                    if col in self.pre_outbreak_stats[current_metric].columns}
            
            plot_cols = []
            plot_days = []
            
            for col, day in sorted(traj_cols_ordered_map.items(), key=lambda item: item[1]):  # Sort by day (-10, -7, ..., 0)
                if not self.pre_outbreak_stats[current_metric][col].isnull().all():  # Check if not all NaN
                    plot_cols.append(col)
                    plot_days.append(day)
                    # Add the data to trajectory_data for calculations
                    trajectory_data[col] = self.pre_outbreak_stats[current_metric][col]
                    
            if not trajectory_data.empty and len(plot_cols) > 1:
                # Plot individual trajectories
                for i in range(len(trajectory_data)):
                    values = trajectory_data.iloc[i][plot_cols].values
                    valid_indices = ~np.isnan(values)
                    if sum(valid_indices) > 1:
                        ax3.plot(np.array(plot_days)[valid_indices], values[valid_indices],
                                marker='.', markersize=4, alpha=0.5, color=lighten_color(metric_color, 0.5),
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
                    # ax3.plot(plot_days_avg, avg_values_avg, marker='o', markersize=5, linewidth=2.0,
                    #         color=COLORS.get('critical', 'red'), label='Average Trajectory', zorder=10)
                    ax3.plot(plot_days_avg, avg_values_avg, marker='o', markersize=5, linewidth=2.0,
                            color=COLORS.get('critical', 'red'), label='Durchschnittlicher Verlauf', zorder=10)
                    
                    # Calculate 95% CI
                    ci_level = self.config.get('confidence_level', 0.95)
                    # Use T distribution for CI if N is small, otherwise Z
                    if np.min(count_values_avg) < 30:  # Arbitrary threshold for T vs Z
                        t_crit = stats.t.ppf((1 + ci_level) / 2, df=count_values_avg - 1)
                        crit_val = t_crit
                    else:
                        z_crit = stats.norm.ppf((1 + ci_level) / 2)
                        crit_val = z_crit
                    sem = std_values_avg / np.sqrt(count_values_avg)
                    margin_of_error = crit_val * sem
                    upper = avg_values_avg + margin_of_error
                    lower = avg_values_avg - margin_of_error
                    
                    ax3.fill_between(plot_days_avg, upper, lower, color=COLORS.get('critical', 'red'), alpha=0.15, zorder=9)
                
                # Add legend for average line
                legend_handles_traj = []
                avg_line = ax3.get_lines()[-1]  # Get the average line handle
                legend_handles_traj.append(avg_line)
                if len(ax3.collections) > 0:  # Check if CI fill exists
                    ci_fill = ax3.collections[-1]  # Get the CI fill handle
                    legend_handles_traj.append(ci_fill)
                    
                ax3.legend(handles=legend_handles_traj, loc='best', frameon=True, facecolor='white', 
                          edgecolor=COLORS.get('grid', 'lightgrey'), fontsize=9)
                
                # Vertical grid lines for time points
                for day in plot_days: 
                    ax3.axvline(x=day, color=COLORS.get('grid', 'lightgrey'), linestyle=':', linewidth=0.8)
                ax3.set_xticks(plot_days)
                # xticklabels = [f'{abs(d)}d' if d < 0 else 'Rem.' for d in plot_days]
                xticklabels = [f'{abs(d)}T' if d < 0 else 'Entf.' for d in plot_days]
                ax3.set_xticklabels(xticklabels, rotation=0, ha='center', fontsize=9)
        
            else:
                ax3.text(0.5, 0.5, 'Insufficient Data for Trajectories', ha='center', va='center', color=COLORS.get('annotation', 'grey'))
                
            # ax3.set_title(f'D) {display_name} Trajectory')
            ax3.set_title(f'D) Verlauf des {display_name}')
            # ax3.set_xlabel('Days Before Removal')
            ax3.set_xlabel('Tage vor Entfernung')
            # ax3.set_ylabel(display_name)
            ax3.set_ylabel(display_name)
            ax3.tick_params(axis='y', labelsize=9)
            ax3.axhline(y=0, color=COLORS.get('grid', 'lightgrey'), linestyle='--', linewidth=1.0, zorder=1)
            ax3.grid(axis='y', linestyle=':', color=COLORS.get('grid', 'lightgrey'), alpha=0.7)
            ax3.grid(axis='x', visible=False)
            
            # --- Overall Figure Adjustments & Saving ---
            # Add overall title
            # overall_title = f"{display_name} Analysis - Pre-Outbreak Patterns"
            # overall_title = f"{display_name} Analyse - Vor-Ausbruch-Muster"
            # fig.suptitle(overall_title, fontsize=16, weight='bold')
            fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.92)  # Adjust spacing
            
            # Generate save path
            if save_path is None:
                filename = self.config.get(f'viz_behavior_{current_metric}_filename', 
                                          f'behavior_analysis_{current_metric}.png')
                output_dir = self.config.get('output_dir', '.')
                os.makedirs(output_dir, exist_ok=True)
                metric_save_path = os.path.join(output_dir, filename)
            else:
                # If specific metric and specific save path, adjust filename
                if metric:
                    metric_save_path = save_path
                else:
                    # Create path with metric name if visualizing multiple metrics
                    base_dir = os.path.dirname(save_path)
                    base_name = os.path.basename(save_path)
                    name, ext = os.path.splitext(base_name)
                    metric_save_path = os.path.join(base_dir, f"{name}_{current_metric}{ext}")
            
            try:
                dpi = self.config.get('figure_dpi', 300)
                fig.savefig(metric_save_path, dpi=dpi)
                self.logger.info(f"Saved {current_metric} visualization to {metric_save_path}")
                saved_files[current_metric] = metric_save_path
            except Exception as e:
                self.logger.error(f"Failed to save {current_metric} visualization to {metric_save_path}: {e}")
            finally:
                plt.close(fig)            
        return saved_files
    
    def visualize_behavior_comparison(self, metric=None, save_path=None):
        """
        Create visualizations comparing outbreak and control statistics for behavior metrics.
        
        Parameters:
            metric (str, optional): Specific metric to visualize. If None, visualize all metrics.
            save_path (str, optional): Path to save the visualization. If None, use default path.
            
        Returns:
            dict: Dictionary with metadata about the visualizations created.
        """
        self.logger.info(f"Visualizing behavior metrics comparison{f' for {metric}' if metric else ''}...")
        set_plotting_style(self.config)
        
        metrics_to_visualize = [metric] if metric else self.behavior_metrics
        saved_files = {}
        
        for current_metric in metrics_to_visualize:
            if current_metric not in self.behavior_metrics:
                self.logger.warning(f"Unknown metric: {current_metric}. Skipping comparison visualization.")
                continue
                
            # Check if we have data for this metric in both outbreak and control datasets
            if (current_metric not in self.pre_outbreak_stats or self.pre_outbreak_stats[current_metric].empty or
                current_metric not in self.control_stats or self.control_stats[current_metric].empty):
                self.logger.warning(f"Missing data for {current_metric} comparison. Skipping visualization.")
                continue
                
            # Generate visualization for this metric
            self.logger.info(f"Creating comparison visualization for {current_metric}")
            
            # Setup the figure
            fig_size = self.config.get('fig_size_comparison', (11, 10))
            fig = plt.figure(figsize=fig_size)
            
            # Create grid layout: 2x2 grid
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                                 hspace=0.4, wspace=0.3)
            
            # Get display name for this metric
            display_name = self.metric_display_names.get(current_metric, current_metric.replace('_', ' ').title())
            
            # --- Prepare Data ---
            outbreak_data = self.pre_outbreak_stats[current_metric].copy().reset_index(drop=True)
            control_data = self.control_stats[current_metric].copy().reset_index(drop=True)
            # outbreak_data['group'] = 'Tail Biting'
            # control_data['group'] = 'Control'
            outbreak_data['group'] = 'Schwanzbeißbucht'
            control_data['group'] = 'Kontrolle'
            control_renamed = control_data.rename(columns={'value_at_reference': 'value_at_removal',
                                                       'reference_date': 'culprit_removal_date'})
                                                       
            # Define columns needed
            plot_cols_base = ['value_at_removal', 'group', 'pen', 'datespan']
            
            # Get days from config
            days_before_list = self.config.get('days_before_list', [7, 3, 1])
            analysis_window_days = self.config.get('analysis_window_days', [7, 3])
            
            # Generate metric columns based on config
            value_cols = [f'value_{d}d_before' for d in days_before_list]
            abs_change_cols = [f'abs_change_{d}d' for d in days_before_list]
            slope_cols = [f'{d}d_window_slope' for d in analysis_window_days]
            avg_cols = [f'{d}d_window_avg' for d in analysis_window_days]
            
            # Combine all metric columns
            common_cols_needed = value_cols + abs_change_cols + slope_cols + avg_cols
            
            # Find common columns between outbreak and control data
            available_outbreak = set(outbreak_data.columns)
            available_control = set(control_renamed.columns)
            common_metrics = list(set(common_cols_needed) & available_outbreak & available_control)
            common_cols = plot_cols_base + common_metrics
            common_cols = list(set(common_cols))  # Ensure unique columns
            
            # Check if essential columns are present after intersection
            required_viz_cols = ['value_at_removal', 'group']
            if not all(col in common_cols for col in required_viz_cols):
                self.logger.error(f"Missing essential columns {required_viz_cols} for comparison viz of {current_metric}.")
                continue
                
            outbreak_subset = outbreak_data[[col for col in common_cols if col in outbreak_data.columns]]
            control_subset = control_renamed[[col for col in common_cols if col in control_renamed.columns]]
            combined_data = pd.concat([outbreak_subset, control_subset], ignore_index=True)
            
            # Colors for the groups
            group_colors = {
                # 'Tail Biting': COLORS['tail_biting'],
                # 'Control': COLORS['control']
                'Schwanzbeißbucht': COLORS['tail_biting'],
                'Kontrolle': COLORS['control']
            }
            group_labels = {
                # 'Tail Biting': 'TB',
                # 'Control': 'Ctrl'
                'Schwanzbeißbucht': 'SB',
                'Kontrolle': 'Ktrl'
            }
            
            # --- Panel A (Row 0, Col 0): Value Distribution Comparison ---
            ax0 = fig.add_subplot(gs[0, 0])
            boxplot_data_comp = []
            boxplot_labels_comp = []
            boxplot_colors_comp = []
            
            # time_points = {'At Removal/Ref': 'value_at_removal'}
            # for d in days_before_list:
            #     time_points[f'{d}d Before'] = f'value_{d}d_before'
            time_points = {'Bei Entfernung/Ref': 'value_at_removal'}
            for d in days_before_list:
                time_points[f'{d}T vorher'] = f'value_{d}d_before'
                
            for time_label, col_name in time_points.items():
                if col_name not in combined_data.columns: 
                    continue
                for group, short_label in group_labels.items():
                    data_series = combined_data.loc[combined_data['group'] == group, col_name]
                    data_vals = data_series.dropna().values
                    if len(data_vals) > 0:
                        boxplot_data_comp.append(data_vals)
                        boxplot_labels_comp.append(f'{short_label}: {time_label}')
                        boxplot_colors_comp.append(group_colors[group])
                        
            if boxplot_data_comp:
                self._style_boxplot(ax0, boxplot_data_comp, boxplot_labels_comp, boxplot_colors_comp, widths=0.7)
                # Add legend
                # tb_patch = mpatches.Patch(color=group_colors['Tail Biting'], label='Tail Biting')
                # ctrl_patch = mpatches.Patch(color=group_colors['Control'], label='Control')
                tb_patch = mpatches.Patch(color=group_colors['Schwanzbeißbucht'], label='Schwanzbeißbucht')
                ctrl_patch = mpatches.Patch(color=group_colors['Kontrolle'], label='Kontrollbucht')
                ax0.legend(handles=[tb_patch, ctrl_patch], loc='best', frameon=True, fontsize=9)
                
            else:
                ax0.text(0.5, 0.5, 'Insufficient data for boxplots', ha='center', va='center', color=COLORS['annotation'])
                
            # ax0.set_title(f'A) {display_name} Comparison')
            ax0.set_title(f'A) Vergleich des {display_name}')
            # ax0.set_ylabel(display_name)
            ax0.set_ylabel(display_name)
            ax0.tick_params(axis='x', rotation=45, labelsize=9)
            ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
            ax0.grid(axis='x', b=False)
            
            # --- Panel B (Row 0, Col 1): Trajectory Comparison ---
            ax1 = fig.add_subplot(gs[0, 1])
            # Define days map relative to removal (0)
            trajectory_days_map = {'value_at_removal': 0}
            for d in days_before_list:
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
                            if len(values) > 1:  # Need >1 point for mean/std
                                avg_values.append(values.mean())
                                std_values.append(values.std())
                                x_values.append(day)
                                n_points.append(len(values))
                                
                        if len(avg_values) > 1:
                            plotted_traj = True
                            # Plot average line
                            # ax1.plot(x_values, avg_values, marker='o', markersize=5, linewidth=2.0,
                            #         color=color, label=f'{group} Avg (N={len(group_data)})', zorder=10)
                            ax1.plot(x_values, avg_values, marker='o', markersize=5, linewidth=2.0,
                                   color=color, label=f'{group} Ø (N={len(group_data)})', zorder=10)
                                   
                            # Plot CI
                            try:
                                ci_level = self.config.get('confidence_level', 0.95)
                                # Use T dist for smaller samples
                                t_crit = [stats.t.ppf((1 + ci_level) / 2, n - 1) if n > 1 else 0 for n in n_points]
                                sem = [std / np.sqrt(n) if n > 0 else 0 for std, n in zip(std_values, n_points)]
                                margin_of_error = [t * s for t, s in zip(t_crit, sem)]
                                upper = [avg + moe for avg, moe in zip(avg_values, margin_of_error)]
                                lower = [avg - moe for avg, moe in zip(avg_values, margin_of_error)]
                                ax1.fill_between(x_values, upper, lower, color=color, alpha=0.15, zorder=9)
                            except Exception as e_ci:
                                self.logger.warning(f"Could not calculate CI for {group} trajectory: {e_ci}")
                                
            if plotted_traj:
                # Add reference lines
                ax1.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0, zorder=1)
                for day in plot_days_comp:
                    ax1.axvline(x=day, color=COLORS['grid'], linestyle=':', linewidth=0.8)
                ax1.set_xticks(plot_days_comp)
                # ax1.set_xticklabels([f'{abs(d)}d' if d < 0 else 'Day 0' for d in plot_days_comp], 
                #                    rotation=0, ha='center', fontsize=9)
                ax1.set_xticklabels([f'{abs(d)}T' if d < 0 else 'Tag 0' for d in plot_days_comp], 
                                   rotation=0, ha='center', fontsize=9)
                ax1.legend(loc='best', frameon=True, fontsize=9)
            else:
                ax1.text(0.5, 0.5, 'Insufficient data for trajectories', ha='center', va='center', color=COLORS['annotation'])
                
            # ax1.set_title(f'B) Average {display_name} Trajectory')
            ax1.set_title(f'B) Durchschnittlicher Verlauf des {display_name}')
            # ax1.set_xlabel('Days Before Removal/Reference')
            ax1.set_xlabel('Tage vor Entfernung/Referenz')
            # ax1.set_ylabel(display_name)
            ax1.set_ylabel(display_name)
            ax1.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
            ax1.grid(axis='x', b=False)
            
            # --- Panel C (Row 1, Col 0): Absolute Change Comparison ---
            ax2 = fig.add_subplot(gs[1, 0])
            change_data_comp, change_labels_comp, change_colors_comp = [], [], []
            for d in sorted(days_before_list, reverse=True):
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
            else:
                ax2.text(0.5, 0.5, 'Insufficient data for change comparison', ha='center', va='center', color=COLORS['annotation'])
                
            # ax2.set_title(f'C) Absolute Change Comparison ({display_name})')
            ax2.set_title(f'C) Vergleich der absoluten Änderungen ({display_name})')
            # ax2.set_ylabel(f'Absolute Change in {display_name}')
            ax2.set_ylabel(f'Absolute Änderung des {display_name}')
            ax2.tick_params(axis='x', rotation=45, labelsize=9)
            ax2.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
            ax2.grid(axis='x', b=False)
            
            # --- Panel D (Row 1, Col 1): Slope Comparison ---
            ax3 = fig.add_subplot(gs[1, 1])
            slope_data_comp, slope_labels_comp, slope_colors_comp = [], [], []
            slope_metrics_compared = []
            
            for d in sorted(analysis_window_days, reverse=True):
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
                        
            bp = None  # Initialize bp to None
            if slope_data_comp:
                # Capture the returned bp dictionary
                bp = self._style_boxplot(ax3, slope_data_comp, slope_labels_comp, slope_colors_comp, widths=0.7)
                ax3.axhline(y=0, color=COLORS['grid'], linestyle='--', linewidth=1.0)
                
                # Add significance annotations if comparison results available
                if current_metric in self.comparison_results:
                    for i, metric in enumerate(slope_metrics_compared):
                        metric_key = f"{current_metric}_{metric}"
                        if metric_key in self.comparison_results[current_metric] and self.comparison_results[current_metric][metric_key]['is_significant']:
                            # Here you would add the significance annotation code if needed
                            pass
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for slope comparison', ha='center', va='center', color=COLORS['annotation'])
                
            # ax3.set_title(f'D) Slope Comparison Between Groups ({display_name})')
            ax3.set_title(f'D) Steigungsvergleich zwischen Gruppen ({display_name})')
            # ax3.set_ylabel('Slope (Change per Day)')
            ax3.set_ylabel('Steigung (Änderung pro Tag)')
            ax3.tick_params(axis='x', rotation=45, labelsize=9)
            ax3.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
            ax3.grid(axis='x', b=False)
            
            # --- Overall Figure Adjustments & Saving ---
            # Add overall title
            # fig.suptitle(f"{display_name} - Outbreak vs Control Comparison", fontsize=16, weight='bold')
            # fig.suptitle(f"{display_name} - Ausbruch vs Kontrolle Vergleich", fontsize=16, weight='bold')
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)  # Adjust for rotated labels
            
            # Generate save path
            if save_path is None:
                filename = self.config.get(f'viz_comparison_{current_metric}_filename', 
                                         f'behavior_comparison_{current_metric}.png')
                output_dir = self.config.get('output_dir', '.')
                os.makedirs(output_dir, exist_ok=True)
                metric_save_path = os.path.join(output_dir, filename)
            else:
                # If specific metric and specific save path, adjust filename
                if metric:
                    metric_save_path = save_path
                else:
                    # Create path with metric name if visualizing multiple metrics
                    base_dir = os.path.dirname(save_path)
                    base_name = os.path.basename(save_path)
                    name, ext = os.path.splitext(base_name)
                    metric_save_path = os.path.join(base_dir, f"{name}_{current_metric}{ext}")
                    
            try:
                dpi = self.config.get('figure_dpi', 300)
                fig.savefig(metric_save_path, dpi=dpi)
                self.logger.info(f"Saved {current_metric} comparison visualization to {metric_save_path}")
                saved_files[current_metric] = metric_save_path
            except Exception as e:
                self.logger.error(f"Failed to save {current_metric} comparison visualization: {e}")
            finally:
                plt.close(fig)
        return saved_files
    
    def visualize_activity_components(self, save_path=None):
        """Create visualizations showing the individual components (lying vs not lying pigs)."""
        self.logger.info("Visualizing lying components (lying vs not lying pigs)...")
        set_plotting_style(self.config)
        
        # We need the raw lying and not lying data
        if ('num_pigs_lying' not in self.pre_outbreak_stats or 
            'num_pigs_notLying' not in self.pre_outbreak_stats or
            self.pre_outbreak_stats['num_pigs_lying'].empty or 
            self.pre_outbreak_stats['num_pigs_notLying'].empty):
            self.logger.error("No component data (lying/not lying) available for visualization.")
            return None
        
        outbreak_lying = self.pre_outbreak_stats['num_pigs_lying'].copy()
        outbreak_not_lying = self.pre_outbreak_stats['num_pigs_notLying'].copy()
        
        # Initialize control variables
        control_lying = pd.DataFrame()
        control_not_lying = pd.DataFrame()
        
        # Check for control data
        has_controls = True
        if ('num_pigs_lying' in self.control_stats and 
            'num_pigs_notLying' in self.control_stats and
            not self.control_stats['num_pigs_lying'].empty and
            not self.control_stats['num_pigs_notLying'].empty):
            control_lying = self.control_stats['num_pigs_lying'].copy()
            control_not_lying = self.control_stats['num_pigs_notLying'].copy()
        else:
            has_controls = False
            self.logger.warning("No control data available for component comparison")
        
        # Set up the figure
        fig_size = self.config.get('fig_size_components', (11, 10))
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)
        
        # Define colors
        colors = {
            'lying': COLORS.get('lying', '#1f77b4'),
            'not_lying': COLORS.get('primary_metric', '#ff7f0e'),
            'outbreak': COLORS.get('critical', '#d62728'),
            'control': COLORS.get('control', '#7f7f7f')
        }
        
        # --- Panel A: Outbreak Component Trajectories ---
        ax0 = fig.add_subplot(gs[0, 0])
        plotted_a = False
        
        # Prepare data by combining lying and not lying
        days_before_list = self.config.get('days_before_list', [7, 3, 1])
        days_list_with_removal = days_before_list + [0]  # Add day 0 for removal day
        
        # Create a consolidated dataframe
        trajectory_data = []
        
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
                    trajectory_data.append({
                        'pen': pen,
                        'datespan': datespan,
                        'days_before_removal': d,
                        'lying_pigs': lying_val,
                        'not_lying_pigs': not_lying_val
                    })
        
        if trajectory_data:
            trajectory_df = pd.DataFrame(trajectory_data)
            
            # Calculate averages by day
            avg_by_day = trajectory_df.groupby('days_before_removal').agg({
                'lying_pigs': ['mean', 'std', 'count'],
                'not_lying_pigs': ['mean', 'std', 'count']
            })
            
            days_available = sorted(trajectory_df['days_before_removal'].unique(), reverse=True)
            days_x = [-d for d in days_available]
            
            if len(days_available) > 1:
                plotted_a = True
                
                # Plot lying pigs
                means_lying = avg_by_day[('lying_pigs', 'mean')].reindex(days_available).values
                stds_lying = avg_by_day[('lying_pigs', 'std')].reindex(days_available).values
                counts_lying = avg_by_day[('lying_pigs', 'count')].reindex(days_available).values
                
                # Plot not lying pigs
                means_not_lying = avg_by_day[('not_lying_pigs', 'mean')].reindex(days_available).values
                stds_not_lying = avg_by_day[('not_lying_pigs', 'std')].reindex(days_available).values
                counts_not_lying = avg_by_day[('not_lying_pigs', 'count')].reindex(days_available).values
                
                # Plot lying
                ax0.plot(days_x, means_lying, 'o-', color=colors['lying'], 
                        label='Liegende Schweine', linewidth=1.8, markersize=4)
                
                # Plot not lying
                ax0.plot(days_x, means_not_lying, 's-', color=colors['not_lying'], 
                        label='Nicht liegende Schweine', linewidth=1.8, markersize=4)
                
                # Add confidence intervals
                for means, stds, counts, color in [(means_lying, stds_lying, counts_lying, colors['lying']),
                                                (means_not_lying, stds_not_lying, counts_not_lying, colors['not_lying'])]:
                    valid_idx = counts > 1
                    if np.sum(valid_idx) > 1:
                        x_valid = np.array(days_x)[valid_idx]
                        try:
                            t_crit = stats.t.ppf((1 + 0.95) / 2, counts[valid_idx] - 1)
                            sem = stds[valid_idx] / np.sqrt(counts[valid_idx])
                            moe = t_crit * sem
                            ax0.fill_between(x_valid, means[valid_idx] - moe, means[valid_idx] + moe, 
                                        alpha=0.15, color=color)
                        except Exception as e_ci:
                            self.logger.warning(f"CI calculation failed: {e_ci}")
                
                ax0.axvline(x=0, color=COLORS['annotation'], linestyle=':', linewidth=1.0, alpha=0.7)
                ax0.text(0.1, ax0.get_ylim()[1]*0.95, 'Entfernung', ha='left', va='top', 
                        fontsize=8, color=COLORS['annotation'], rotation=90)
                ax0.legend(loc='best', frameon=True, fontsize=9)
        
        if not plotted_a:
            ax0.text(0.5, 0.5, 'Insufficient Outbreak Data', ha='center', va='center', 
                    color=COLORS['annotation'])
        
        ax0.set_title('A) Verlauf von liegenden und nicht liegenden Schweinen (Ausbrüche)')
        ax0.set_xlabel('Tage vor Entfernung')
        ax0.set_ylabel('Anzahl der Schweine')
        ax0.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax0.grid(axis='x', visible=False)
        
        # --- Panel B: Control vs. Outbreak Comparison ---
        ax1 = fig.add_subplot(gs[1, 0])
        plotted_b = False
        
        if has_controls:
            # Prepare control data (including day 0)
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
                            'days_before_removal': d,
                            'lying_pigs': lying_val,
                            'not_lying_pigs': not_lying_val
                        })
            
            if control_trajectory_data and trajectory_data:
                control_df = pd.DataFrame(control_trajectory_data)
                
                # Find common days
                outbreak_days = set(trajectory_df['days_before_removal'].unique())
                control_days = set(control_df['days_before_removal'].unique())
                common_days = sorted(list(outbreak_days & control_days), reverse=True)
                
                if len(common_days) > 1:
                    common_days_x = [-d for d in common_days]
                    
                    # Calculate averages for outbreak
                    avg_outbreak = trajectory_df[trajectory_df['days_before_removal'].isin(common_days)].groupby('days_before_removal').agg({
                        'lying_pigs': ['mean', 'std', 'count'],
                        'not_lying_pigs': ['mean', 'std', 'count']
                    })
                    
                    # Calculate averages for control
                    avg_control = control_df[control_df['days_before_removal'].isin(common_days)].groupby('days_before_removal').agg({
                        'lying_pigs': ['mean', 'std', 'count'],
                        'not_lying_pigs': ['mean', 'std', 'count']
                    })
                    
                    # Plot lying/not lying data
                    for component, color, label_de in [
                        ('lying_pigs', colors['lying'], 'Liegend'),
                        ('not_lying_pigs', colors['not_lying'], 'Nicht liegend')
                    ]:
                        # Outbreak data
                        means_o = avg_outbreak[(component, 'mean')].reindex(common_days).values
                        
                        # Control data
                        means_c = avg_control[(component, 'mean')].reindex(common_days).values
                        
                        # Plot outbreak line
                        ax1.plot(common_days_x, means_o, 'o-', color=color, 
                                label=f'Ausbruch {label_de}', linewidth=2.0, markersize=5, alpha=1.0)
                        
                        # Plot control line (lighter, dotted)
                        ax1.plot(common_days_x, means_c, 's--', color=color, 
                                label=f'Kontrolle {label_de}', linewidth=1.5, markersize=4, alpha=0.6)
                    
                    # Style the axis
                    ax1.axvline(x=0, color=COLORS['annotation'], linestyle=':', linewidth=1.0, alpha=0.7)
                    
                    # Set y-axis label
                    ax1.set_ylabel('Anzahl der Schweine')
                    
                    # Add legend
                    ax1.legend(loc='best', frameon=True, fontsize=8)
                    
                    plotted_b = True
        
        if not plotted_b:
            ax1.text(0.5, 0.5, 'Insufficient Data for Comparison', ha='center', va='center', 
                    color=COLORS['annotation'])
        
        ax1.set_title('B) Liegende Komponenten: Ausbruch vs. Kontrolle')
        ax1.set_xlabel('Tage vor Entfernung/Referenz')
        ax1.grid(axis='y', linestyle=':', color=COLORS['grid'], alpha=0.7)
        ax1.grid(axis='x', visible=False)
        
        # Layout adjustments
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        
        # Save the figure
        if save_path is None:
            filename = self.config.get('viz_activity_components_filename', 'lying_component_analysis.png')
            save_path = os.path.join(self.config['output_dir'], filename)
        
        try:
            dpi = self.config.get('figure_dpi', 300)
            fig.savefig(save_path, dpi=dpi)
            self.logger.info(f"Saved lying component visualization to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save lying component visualization: {e}")
        finally:
            plt.close(fig)
        
        return save_path