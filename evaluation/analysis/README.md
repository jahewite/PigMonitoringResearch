# Analysis Directory

This directory contains the comprehensive analysis framework for pig behavior monitoring data, specifically designed for studying tail posture patterns and activity behaviors in relation to tail biting outbreaks.

## Analysis Modules

### 1. Tail Posture Analysis (`tail_posture_analysis/`)

**Primary Module**: Comprehensive analysis of tail posture patterns before tail biting outbreaks.

**Key Features**:
- **Descriptive Pre-outbreak Analysis**: Statistical analysis of tail posture changes leading up to culprit removal
- **Control Comparison**: Statistical comparison between outbreak and control pens
- **Individual Variation Analysis**: Pattern categorization of different outbreak trajectories
- **Component Analysis**: Decomposition of posture differences into upright vs hanging tail components
- **Timeline Visualization**: Data completeness and quality assessment

**Main Classes**:
- `TailPostureAnalyzer`: Statistical analysis engine
- `TailPostureVisualizer`: Visualization generation with dissertation-quality plots

**Output Examples**:
- `descriptive_pre_outbreak_patterns.png`: Multi-panel overview of pre-outbreak patterns
- `outbreak_vs_control_comparison.png`: Statistical comparison visualizations
- `individual_variation_analysis.png`: Pattern categorization results
- `posture_component_analysis.png`: Component decomposition analysis

### 2. Activity Analysis (`activity_analysis/`)

**Purpose**: Analysis of pig lying behavior and activity patterns using the same analytical framework as tail posture analysis.

**Key Features**:
- Analysis of three behavioral metrics: `num_pigs_lying`, `num_pigs_notLying`, `activity`
- Separate analysis and visualization for each metric
- Component analysis showing relationship between lying and not-lying behaviors
- Statistical comparison between outbreak and control conditions

**Main Classes**:
- `PigBehaviorAnalyzer`: Extends the core analysis framework for activity data
- `PigBehaviorVisualizer`: Specialized visualizations for activity metrics

**Output Examples**:
- `behavior_analysis_[metric].png`: Individual metric analysis
- `behavior_comparison_[metric].png`: Outbreak vs control comparisons
- `lying_component_analysis.png`: Component relationship analysis

### 3. Early Warning Analysis (`early_warning_analysis/`)

**Purpose**: Evaluation of threshold-based early warning systems for tail biting prediction.

**Key Features**:
- Multi-level threshold system (attention, alert, critical)
- Configurable lookback periods and ignore zones
- Statistical evaluation (sensitivity, specificity, precision, F1-score)
- Time-to-warning analysis
- Individual pen visualization with threshold zones

**Main Classes**:
- `EarlyWarningAnalyzer`: Threshold evaluation engine
- `EarlyWarningVisualizer`: Individual pen timeline visualizations with threshold overlays

**Output Examples**:
- `early_warning_evaluation.csv`: Statistical performance metrics
- `[Pen]_[datespan]_early_warning.png`: Individual pen visualizations
- ROC curves and confusion matrices

## Core Framework Components

### Data Processing Pipeline (`utils/processing.py`)

**`DataProcessor` Class**:
- **Data Loading**: Loads monitoring pipeline results with missing data tracking
- **Quality Assessment**: Calculates comprehensive quality metrics
- **Data Preprocessing**: Resampling, normalization, smoothing, and interpolation
- **Configurable Processing**: Supports multiple smoothing methods and interpolation techniques

**Processing Steps**:
1. **Load Raw Data**: From monitoring pipeline outputs
2. **Quality Metrics**: Calculate missing data percentages and consecutive missing periods
3. **Resampling**: Convert to specified frequency (hourly/daily)
4. **Normalization**: Normalize tail detection counts
5. **Smoothing**: Apply rolling, EWM, Savitzky-Golay, or LOESS smoothing
6. **Interpolation**: Fill gaps using linear, spline, or polynomial methods

### Quality Filtering (`utils/data_filter.py`)

**`DataFilter` Class**:
- **Quality Thresholds**: Configurable thresholds for consecutive missing days and missing percentage
- **Exclusion Tracking**: Detailed logging of why datasets were excluded
- **Analysis-Specific Filtering**: Separate filtering for different analysis types
- **Pen Type Filtering**: Separate filtering for tail biting events vs control pens

**Filtering Criteria**:
- Maximum consecutive missing days (default: 3)
- Maximum missing percentage (default: 50%)
- Valid tail biting events with culprit removal dates
- Sufficient data for statistical analysis

### Visualization Framework (`utils/utils.py`)

**Key Features**:
- **Dissertation-Quality Styling**: Professional plot styling with consistent fonts and colors
- **Color Scheme**: Carefully chosen color palette for accessibility and print quality
- **Pattern Recognition**: Specific colors for different outbreak patterns
- **Flexible Configuration**: Customizable DPI, fonts, and figure sizes

**Color Palette**:
- **Tail Posture**: Upright (teal), Hanging (orange), Difference (purple)
- **Pen Types**: Tail Biting (red), Control (teal)
- **Patterns**: Stable (teal), Consistent Decline (blue), Steep Decline (gray)

## Configuration System

### Configuration Files (`configs/`)

Each analysis module has its own JSON configuration file:

**`config_tail_posture.json`**:
- Core processing parameters (resampling, normalization, smoothing)
- Analysis windows and statistical parameters
- Pattern analysis thresholds
- Visualization settings
- Output file naming

**`config_activity.json`**:
- Similar structure to tail posture but optimized for activity metrics
- Separate configuration for each behavioral metric
- Component analysis parameters

**`config_early_warning.json`**:
- Threshold definitions for different warning levels
- Lookback periods and ignore zones
- Evaluation parameters

### Key Configuration Parameters

- **`resample_freq`**: Data resampling frequency ("D" for daily, "H" for hourly)
- **`days_before_list`**: Days before outbreak to analyze (e.g., [1, 3, 5, 7])
- **`analysis_window_days`**: Rolling window sizes for statistics
- **`max_allowed_consecutive_missing_days`**: Quality threshold for exclusion
- **`max_allowed_missing_days_pct`**: Percentage threshold for exclusion
- **`confidence_level`**: Statistical confidence level (default: 0.95)
- **`figure_dpi`**: Resolution for saved plots (default: 600)

## Usage Examples

### Basic Tail Posture Analysis

```python
from evaluation.analysis.tail_posture_analysis.tail_posture_analyzer import TailPostureAnalyzer
from evaluation.analysis.tail_posture_analysis.tail_posture_visualization import TailPostureVisualizer

# Load configuration
config = load_json_data('evaluation/configs/config_tail_posture.json')

# Initialize analyzer
analyzer = TailPostureAnalyzer(config)

# Load and preprocess data
analyzer.load_data()
analyzer.preprocess_monitoring_results()

# Run complete analysis
analyzer.analyze_pre_outbreak_statistics()
analyzer.analyze_control_pen_statistics()
analyzer.compare_outbreak_vs_control_statistics()
analyzer.analyze_individual_outbreak_variation()
analyzer.analyze_posture_components()

# Generate visualizations
visualizer = TailPostureVisualizer(config)
visualizer.visualize_pre_outbreak_patterns()
visualizer.visualize_comparison_with_controls()
visualizer.visualize_individual_variation()
visualizer.visualize_posture_components()
visualizer.visualize_data_completeness()

# Generate report
analyzer.generate_summary_report()
```

### Activity Analysis

```python
from evaluation.analysis.activity_analysis.pig_activity_analyzer import PigBehaviorAnalyzer
from evaluation.analysis.activity_analysis.pig_activity_visualization import PigBehaviorVisualizer

# Load configuration
config = load_json_data('evaluation/configs/config_activity.json')

# Initialize analyzer
analyzer = PigBehaviorAnalyzer(config)

# Run complete behavior analysis
results = analyzer.run_complete_behavior_analysis()

# Generate visualizations for each metric
visualizer = PigBehaviorVisualizer(config)
for metric in ['num_pigs_lying', 'num_pigs_notLying', 'activity']:
    visualizer.visualize_behavior_metrics(metric=metric)
    visualizer.visualize_behavior_comparison(metric=metric)

# Activity component analysis
visualizer.visualize_activity_components()
```

### Early Warning System Evaluation

```python
from evaluation.analysis.early_warning_analysis.early_warning_analysis import EarlyWarningAnalyzer
from evaluation.analysis.early_warning_analysis.early_warning_visualization import EarlyWarningVisualizer

# Initialize with tail posture analyzer
tail_analyzer = TailPostureAnalyzer(config)
tail_analyzer.load_data()
tail_analyzer.preprocess_monitoring_results()
tail_analyzer.analyze_pre_outbreak_statistics()
tail_analyzer.analyze_control_pen_statistics()

# Initialize early warning analyzer
ew_config = load_json_data('evaluation/configs/config_early_warning.json')
ew_analyzer = EarlyWarningAnalyzer(tail_analyzer, config=ew_config)

# Evaluate thresholds
results = ew_analyzer.evaluate_thresholds()

# Generate visualizations
visualizer = EarlyWarningVisualizer(ew_analyzer, config=ew_config)
visualizer.visualize_all_outbreak_pens()
visualizer.visualize_sample_control_pens()
visualizer.visualize_confusion_matrix(results)
visualizer.visualize_roc_curve(results)
```

## Output Structure

### Analysis Results
- **CSV Files**: Statistical results, comparison metrics, pattern classifications
- **PNG Files**: High-resolution visualizations (600 DPI default)
- **Text Reports**: Comprehensive summary reports with key findings
- **Log Files**: Detailed processing logs with exclusion tracking

### Visualization Types
- **Multi-panel Overviews**: Comprehensive analysis summaries
- **Statistical Comparisons**: Outbreak vs control visualizations
- **Individual Trajectories**: Pattern analysis and categorization
- **Component Analysis**: Decomposition of complex behaviors
- **Quality Assessment**: Data completeness and processing quality
- **Performance Evaluation**: Early warning system metrics

## Advanced Features

### Statistical Analysis
- **Non-parametric Tests**: Mann-Whitney U tests for group comparisons
- **Effect Size Calculation**: Cohen's d for practical significance
- **Confidence Intervals**: Bootstrap and t-distribution based intervals
- **Linear Regression**: Trend analysis with slope calculations
- **Pattern Classification**: Automated categorization of outbreak patterns

### Quality Assurance
- **Missing Data Handling**: Comprehensive tracking and interpolation
- **Exclusion Logging**: Detailed records of why data was excluded
- **Processing Validation**: Quality metrics at each processing step
- **Reproducibility**: Configurable random seeds for consistent results

### Extensibility
- **Modular Design**: Easy to add new analysis types
- **Configuration System**: Flexible parameter adjustment
- **Visualization Framework**: Consistent styling across all plots
- **Data Pipeline**: Standardized preprocessing for all analysis types

## Dependencies

### Core Scientific Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Statistical analysis and signal processing
- `matplotlib`: Visualization framework
- `seaborn`: Statistical visualization enhancements

### Specialized Libraries
- `statsmodels`: Advanced statistical modeling (LOESS smoothing)
- `scikit-learn`: Machine learning utilities (if needed)

### Custom Dependencies
- `pipeline.utils`: Data loading and path management utilities
- `evaluation.utils`: Shared analysis utilities

## Performance Considerations

### Memory Management
- Efficient data structures for large datasets
- Incremental processing to avoid memory issues
- Configurable batch sizes for large-scale analysis

### Processing Speed
- Vectorized operations using NumPy
- Efficient pandas operations
- Parallel processing capabilities where applicable

### Quality vs Speed Trade-offs
- Configurable interpolation methods (linear vs spline)
- Adjustable smoothing parameters
- Flexible quality thresholds

## Contributing

When adding new analysis modules:

1. **Follow the established pattern**: Analyzer + Visualizer classes
2. **Use the shared utilities**: `DataProcessor`, `DataFilter`, styling functions
3. **Add configuration files**: JSON-based configuration in `configs/`
4. **Document thoroughly**: Include docstrings and usage examples
5. **Test with sample data**: Ensure robust error handling
6. **Update this README**: Add new modules to the documentation

## Troubleshooting

### Common Issues

**Missing Data Handling**:
- Check quality thresholds in configuration
- Review exclusion logs for filtering reasons
- Adjust interpolation methods if needed

**Memory Issues**:
- Reduce batch sizes in configuration
- Use more efficient data types
- Consider processing subsets of data

**Visualization Problems**:
- Check DPI settings for file size issues
- Verify font availability on system
- Adjust figure sizes for different outputs

**Statistical Analysis Issues**:
- Ensure minimum sample sizes are met
- Check for outliers affecting results
- Verify assumptions of statistical tests

For detailed troubleshooting, check the log files generated during analysis runs.