# Config

This guide explains the configuration settings for the monitoring pipeline evaluation.

## Core Settings

- **`output_dir`**: Directory to save all outputs (default: `'results'`)
- **`random_seed`**: Seed for reproducibility where randomness is involved (default: `42`)

## Data Processing Settings

- **`resample_freq`**: Resampling frequency (e.g., `'D'` for daily)
- **`normalize`**: Whether to normalize posture difference (0-1)
- **`smoothing_method`**: Method for smoothing (`'rolling'`, `'ewm'`, or `None`)
- **`smoothing_strength`**: Window size or alpha for smoothing (default: `3`)

## Analysis Parameters

- **`max_allowed_consecutive_missing_days`**: Maximum consecutive missing days allowed (default: `3`)
- **`max_allowed_missing_days_pct`**: Maximum percentage of missing days allowed (default: `50.0`)
- **`days_before_list`**: List of days before removal/reference to analyze point statistics (default: `[1, 3, 5, 7]`)
- **`analysis_window_days`**: List of window sizes (days) for rolling stats calculations (default: `[1, 3, 5, 7]`)
- **`confidence_level`**: Confidence level for CI calculations (default: `0.95`)
- **`significance_level`**: Alpha level for statistical significance tests (default: `0.05`)

## Control Pen Analysis Settings

- **`min_control_dates_threshold`**: Minimum total days required for a control pen dataset (default: `8`)
- **`control_date_margin`**: Days to trim from start/end of control data before sampling (default: `5`)
- **`min_control_analysis_dates`**: Minimum days required *after* trimming margin for sampling (default: `2`)
- **`control_samples_per_pen`**: Number of random reference dates to sample per control pen (default: `5`)

## Outbreak vs Control Comparison Settings

- **`comparison_metrics`**: Metrics to compare between outbreak and control groups (includes various windows, slopes, and absolute changes)

## Individual Variation Analysis Settings

- **`min_outbreaks_for_variation_analysis`**: Minimum outbreaks needed for detailed pattern stats (default: `3`)
- **`variation_slope_quantile`**: Quantile for defining 'Sudden-decline' slope threshold (default: `0.33`)
- **`variation_accel_quantile`**: Quantile for defining 'Sudden-decline' acceleration threshold (default: `0.33`)
- **`variation_gradual_slope_diff`**: Max difference between 3d/7d slope for 'Gradual-decline' (default: `0.05`)
- **`variation_erratic_slope_diff`**: Min difference between 3d/7d slope for 'Erratic-decline' (default: `0.1`)

## Component Analysis Settings

- **`component_analysis_window_days`**: Lookback window (days) for component time series analysis (default: `10`)
- **`component_timepoint_days`**: Days before removal to report specific component stats (default: `[0, 1, 3, 5, 7]`)

## Visualization Settings

- **`figure_dpi`**: Resolution for saved figures (default: `600`)
- **`violin_width`**: Width of violins in plots (default: `0.4`)
- **`variation_max_pens_plot`**: Max number of individual pens shown in variation plot (Panel C) (default: `6`)

## Output Filenames

The configuration includes customizable filenames for all output files:

| Setting | Purpose |
|---------|---------|
| `pre_outbreak_stats_filename` | Pre-outbreak statistics filtered data |
| `control_stats_filename` | Control group statistics |
| `comparison_stats_filename` | Outbreak vs control comparison results |
| `outbreak_patterns_filename` | Outbreak pattern classifications |
| `pattern_stats_filename` | Pattern statistics summary |
| `posture_components_filename` | Posture component analysis |
| `control_components_filename` | Control group posture components |
| `threshold_analysis_filename` | Monitoring threshold analysis (JSON) |
| `viz_pre_outbreak_filename` | Pre-outbreak patterns visualization |
| `viz_comparison_filename` | Outbreak vs control comparison plots |
| `viz_variation_filename` | Individual variation analysis plots |
| `viz_components_filename` | Posture component analysis plots |
| `viz_completeness_filename` | Data completeness timeline visualization |
