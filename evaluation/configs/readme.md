# Configuration Directory

This directory contains JSON configuration files for different analysis modules in the pig behavior monitoring system. Each configuration file controls various aspects of data processing, analysis parameters, and output settings.

## Configuration Files

### 1. `config_activity.json` - Activity Analysis Configuration

Used by `pig_activity_analyzer.py` and `pig_activity_visualization.py` for analyzing pig lying behavior and activity patterns.

#### Core Settings
- **`output_dir`**: Directory for saving analysis results and visualizations
- **`random_seed`**: Random seed for reproducible results (default: 42)
- **`resample_freq`**: Data resampling frequency (default: "D" for daily)
- **`normalize`**: Whether to normalize tail detection counts (default: true)

#### Data Processing
- **`smoothing_method`**: Method for data smoothing ("rolling", "ewm", "savgol", "loess")
- **`smoothing_strength`**: Strength/window size for smoothing (default: 3)
- **`interpolation_method`**: Method for interpolating missing data ("linear", "spline", "polynomial")
- **`interpolation_order`**: Order for spline/polynomial interpolation (default: 3)
- **`max_allowed_consecutive_missing_days`**: Maximum consecutive missing days before exclusion (default: 3)
- **`max_allowed_missing_days_pct`**: Maximum percentage of missing days before exclusion (default: 50.0)

#### Analysis Parameters
- **`days_before_list`**: List of days before outbreak to analyze (default: [1, 3, 5, 7])
- **`analysis_window_days`**: Window sizes for rolling statistics (default: [1, 3, 5, 7])
- **`confidence_level`**: Confidence level for statistical intervals (default: 0.95)
- **`significance_level`**: P-value threshold for statistical significance (default: 0.05)

#### Control Pen Analysis
- **`min_control_dates_threshold`**: Minimum dates required for control pen analysis (default: 8)
- **`control_date_margin`**: Days to exclude from edges of control pen data (default: 5)
- **`min_control_analysis_dates`**: Minimum dates after margin removal (default: 2)
- **`control_samples_per_pen`**: Number of random reference points per control pen (default: 5)
- **`min_sample_values_comparison`**: Minimum samples required for statistical comparison (default: 10)

#### Comparison Settings
- **`comparison_metrics`**: List of metrics to compare between outbreak and control pens

#### Visualization Settings
- **`figure_dpi`**: DPI for saved figures (default: 600)
- **`violin_width`**: Width of violin plots (default: 0.4)
- **`fig_size_behavior`**: Figure size for behavior analysis plots (default: [11, 10])
- **`fig_size_comparison`**: Figure size for comparison plots (default: [11, 10])
- **`fig_size_components`**: Figure size for component analysis plots (default: [11, 10])
- **`abs_change_outlier_iqr_factor`**: IQR factor for outlier filtering in change plots (default: 3)

#### Output Filenames
Specific filenames for different metrics and analysis types:
- **`pre_outbreak_stats_*_filename`**: Files for pre-outbreak statistics
- **`control_stats_*_filename`**: Files for control pen statistics
- **`comparison_stats_*_filename`**: Files for outbreak vs control comparisons
- **`viz_*_filename`**: Files for visualization outputs

---

### 2. `config_early_warning.json` - Early Warning System Configuration

Used by `early_warning_analysis.py` and `early_warning_visualization.py` for threshold-based early warning system evaluation.

#### Core Settings
- **`random_seed`**: Random seed for reproducible results (default: 42)
- **`use_interpolated_data`**: Whether to use interpolated hourly data (false = daily data)
- **`output_dir`**: Directory for saving analysis results

#### Analysis Parameters
- **`max_percent_before`**: Maximum percentage of run duration to look back from end (default: 60)
- **`ignore_first_percent`**: Percentage of run beginning to ignore (default: 20)

#### Threshold Configuration
- **`thresholds`**: Multi-level threshold system:
  - **`attention`**: Early warning level (posture_diff: 0.5)
  - **`alert`**: Medium warning level (posture_diff: 0.4)
  - **`critical`**: Critical warning level (posture_diff: 0.25)

#### Output Filenames
- **`summary_filename`**: Summary evaluation results file
- **`details_filename`**: Detailed per-pen results file
---

### 3. `config_tail_posture.json` - Tail Posture Analysis Configuration

Used by `tail_posture_analyzer.py` and `tail_posture_visualization.py` for comprehensive tail posture pattern analysis.

#### Core Settings
Same as activity analysis for basic processing parameters.

#### Additional Analysis Parameters
- **`min_outbreaks_for_variation_analysis`**: Minimum outbreaks required for pattern variation analysis (default: 3)
- **`variation_slope_quantile`**: Quantile threshold for slope-based pattern categorization (default: 0.33)
- **`variation_accel_quantile`**: Quantile threshold for acceleration-based categorization (default: 0.33)

#### Component Analysis
- **`component_analysis_window_days`**: Window size for component analysis (default: 10)
- **`component_timepoint_days`**: Specific timepoints to analyze (default: [0, 1, 3, 5, 7])

#### Data Saving Options
- **`save_preprocessed_data`**: Whether to save filtered DataFrames to CSV (default: false)
- **`interpolate_resampled_data`**: Whether to interpolate NaN values in resampled data (default: true)

#### Output Filenames
Comprehensive set of filenames for different analysis outputs:
- **`pre_outbreak_stats_filename`**: Pre-outbreak statistics file
- **`control_stats_filename`**: Control pen statistics file
- **`comparison_stats_filename`**: Outbreak vs control comparison file
- **`outbreak_patterns_filename`**: Individual outbreak patterns file
- **`pattern_stats_filename`**: Pattern statistics file
- **`posture_components_filename`**: Posture component analysis file
- **`control_components_filename`**: Control pen component analysis file
- **`viz_*_filename`**: Various visualization output files

---

## Parameter Usage Notes

### Common Parameters Across Configs
- **Processing parameters** (`resample_freq`, `normalize`, `smoothing_*`, `interpolation_*`) are used by the `DataProcessor` class
- **Quality filtering parameters** (`max_allowed_*`) are used by the `DataFilter` class
- **Analysis parameters** (`days_before_list`, `analysis_window_days`, `confidence_level`) are used throughout the analysis classes
- **Visualization parameters** (`figure_dpi`, `violin_width`, `fig_size_*`) are used by the visualization classes

### File Organization
- All output files are saved to the specified `output_dir`
- Visualization files are typically saved as PNG format
- Statistics files are saved as CSV format
- Custom filenames allow for organized output management

### Reproducibility
- Set consistent `random_seed` values across configs for reproducible results
- Control pen sampling and statistical analyses use this seed

### Performance Considerations
- Higher `figure_dpi` values increase file sizes but improve print quality
- Larger `analysis_window_days` values increase computation time
- More `control_samples_per_pen` improves statistical power but increases runtime

## Configuration Best Practices

1. **Consistent Naming**: Use consistent `output_dir` structure across related analyses
2. **Balanced Thresholds**: Adjust quality thresholds based on your data quality requirements
3. **Appropriate Window Sizes**: Choose `days_before_list` and `analysis_window_days` that match your biological expectations
4. **Statistical Power**: Ensure sufficient `min_sample_values_comparison` for reliable statistical tests
5. **Visualization Quality**: Balance `figure_dpi` with file size requirements

## Modifying Configurations

To customize analysis parameters:
1. Edit the relevant JSON file
2. Ensure parameter values are appropriate for your data characteristics
3. Test with a subset of data before running full analysis
4. Update `output_dir` to avoid overwriting previous results

## Dependencies

These configurations work with the following Python classes:
- `DataProcessor` (data loading and preprocessing)
- `DataFilter` (quality-based filtering)
- `TailPostureAnalyzer` / `PigBehaviorAnalyzer` (statistical analysis)
- `TailPostureVisualizer` / `PigBehaviorVisualizer` (visualization)
- `EarlyWarningAnalyzer` / `EarlyWarningVisualizer` (threshold analysis)