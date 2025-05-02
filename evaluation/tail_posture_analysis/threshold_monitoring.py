import numpy as np
import pandas as pd
from scipy import stats
import os
import json
import datetime
from datetime import timedelta
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, StratifiedKFold
import itertools
import copy


class ThresholdMonitoringMixin:
    """
    Mixin class implementing monitoring threshold identification and validation methods.
    
    This class is designed to be inherited by the TailPostureAnalyzer class to add
    threshold monitoring capabilities.
    """
    
    def identify_monitoring_thresholds(self):
        """
        Identify potential monitoring thresholds for detecting upcoming tail biting events.
        
        This function analyzes the statistical differences between outbreak and control pens
        to define thresholds that could serve as early warning signals. Multiple threshold
        types are calculated, and initial performance metrics are estimated.
        
        Returns:
            dict: Dictionary of identified thresholds and their initial performance metrics
        """
        self.logger.info("Identifying potential monitoring thresholds for tail biting detection...")
        
        # Check if we have the necessary data
        if not hasattr(self, 'pre_outbreak_stats') or self.pre_outbreak_stats.empty:
            self.logger.error("No pre-outbreak statistics available. Cannot identify thresholds.")
            return None
            
        if not hasattr(self, 'control_stats') or self.control_stats.empty:
            self.logger.error("No control pen statistics available. Cannot identify thresholds.")
            return None
            
        # Initialize thresholds dictionary
        self.monitoring_thresholds = {
            'absolute_value': {},
            'change_based': {},
            'slope_based': {},
            'component_based': {},
            'combined': {},
            'metadata': {
                'generated_datetime': datetime.datetime.now().isoformat(),
                'n_outbreaks': len(self.pre_outbreak_stats),
                'n_controls': len(self.control_stats[['pen', 'datespan']].drop_duplicates()) if not self.control_stats.empty else 0,
                'n_control_reference_points': len(self.control_stats)
            }
        }
        
        # 1. Generate absolute value thresholds
        self._identify_absolute_value_thresholds()
        
        # 2. Generate change-based thresholds
        self._identify_change_based_thresholds()
        
        # 3. Generate slope-based thresholds
        self._identify_slope_based_thresholds()
        
        # 4. Generate component-based thresholds (if component data available)
        if hasattr(self, 'analyze_posture_components'):
            component_analysis = self.analyze_posture_components()
            if component_analysis:
                self._identify_component_based_thresholds(component_analysis)
        
        # 5. Generate combined thresholds
        self._identify_combined_thresholds()
        
        # Save initial thresholds
        self._save_monitoring_thresholds("initial_thresholds")
        
        return self.monitoring_thresholds
        
    def _identify_absolute_value_thresholds(self):
        """Identify absolute value-based thresholds."""
        self.logger.info("Identifying absolute value thresholds...")
        
        metrics = ['value_at_removal', '3d_window_avg', '7d_window_avg']
        control_column_mapping = {
            'value_at_removal': 'value_at_reference', 
            '3d_window_avg': '3d_window_avg',
            '7d_window_avg': '7d_window_avg'
        }
        
        for metric in metrics:
            control_metric = control_column_mapping.get(metric, metric)
            
            # Skip if either column doesn't exist
            if metric not in self.pre_outbreak_stats.columns or control_metric not in self.control_stats.columns:
                self.logger.warning(f"Skipping {metric} threshold calculation - column not found")
                continue
                
            # Get outbreak and control distributions
            outbreak_values = self.pre_outbreak_stats[metric].dropna()
            control_values = self.control_stats[control_metric].dropna()
            
            if len(outbreak_values) < 5 or len(control_values) < 5:
                self.logger.warning(f"Insufficient data for {metric} threshold calculation")
                continue
                
            # Calculate thresholds using various methods
            methods = self.config.get('threshold_determination_methods', 
                                      ['percentile', 'roc_optimization', 'distribution_overlap'])
                
            thresholds = {}
            performance = {}
            
            # 1. Percentile-based thresholds
            if 'percentile' in methods:
                percentiles = self.config.get('threshold_percentiles', [5, 10, 25])
                for p in percentiles:
                    threshold_value = np.percentile(outbreak_values, p)
                    threshold_key = f'percentile_{p}'
                    thresholds[threshold_key] = threshold_value
                    
                    # Calculate initial performance metrics
                    performance[threshold_key] = self._calculate_threshold_performance(
                        outbreak_values, control_values, threshold_value, direction='lower')
            
            # 2. ROC curve optimization
            if 'roc_optimization' in methods:
                optimal_threshold, optimal_metrics = self._optimize_threshold_roc(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['roc_optimal'] = optimal_threshold
                performance['roc_optimal'] = optimal_metrics
            
            # 3. Distribution overlap method
            if 'distribution_overlap' in methods:
                # Find the point of minimum overlap between distributions
                min_overlap_threshold = self._find_min_overlap_threshold(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['min_overlap'] = min_overlap_threshold
                performance['min_overlap'] = self._calculate_threshold_performance(
                    outbreak_values, control_values, min_overlap_threshold, direction='lower')
            
            # 4. Standard deviation based (control mean - N*std)
            if 'std_deviation' in methods:
                std_multiples = self.config.get('threshold_std_multiples', [1.0, 1.5, 2.0, 2.5, 3.0])
                for std_mult in std_multiples:
                    threshold_value = control_values.mean() - std_mult * control_values.std()
                    threshold_key = f'std_{std_mult}'
                    thresholds[threshold_key] = threshold_value
                    
                    performance[threshold_key] = self._calculate_threshold_performance(
                        outbreak_values, control_values, threshold_value, direction='lower')
            
            # Store all calculated thresholds and their performance
            self.monitoring_thresholds['absolute_value'][metric] = {
                'thresholds': thresholds,
                'performance': performance,
                'outbreak_stats': {
                    'mean': outbreak_values.mean(),
                    'std': outbreak_values.std(),
                    'min': outbreak_values.min(),
                    'max': outbreak_values.max(),
                    'distribution': {
                        str(p): outbreak_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                },
                'control_stats': {
                    'mean': control_values.mean(),
                    'std': control_values.std(),
                    'min': control_values.min(),
                    'max': control_values.max(),
                    'distribution': {
                        str(p): control_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                }
            }
    
    def _identify_change_based_thresholds(self):
        """Identify change-based thresholds."""
        self.logger.info("Identifying change-based thresholds...")
        
        metrics = ['abs_change_1d', 'abs_change_3d', 'abs_change_7d']
        
        for metric in metrics:
            # Skip if column doesn't exist
            if metric not in self.pre_outbreak_stats.columns or metric not in self.control_stats.columns:
                self.logger.warning(f"Skipping {metric} threshold calculation - column not found")
                continue
                
            # Get outbreak and control distributions
            outbreak_values = self.pre_outbreak_stats[metric].dropna()
            control_values = self.control_stats[metric].dropna()
            
            if len(outbreak_values) < 5 or len(control_values) < 5:
                self.logger.warning(f"Insufficient data for {metric} threshold calculation")
                continue
                
            # For change metrics, we're looking for negative changes
            # that are larger in magnitude than what's seen in controls
            
            # Calculate thresholds using various methods
            methods = self.config.get('threshold_determination_methods', 
                                     ['percentile', 'roc_optimization', 'distribution_overlap'])
                
            thresholds = {}
            performance = {}
            
            # 1. Percentile-based thresholds
            if 'percentile' in methods:
                percentiles = self.config.get('threshold_percentiles', [5, 10, 25])
                for p in percentiles:
                    threshold_value = np.percentile(outbreak_values, p)
                    threshold_key = f'percentile_{p}'
                    thresholds[threshold_key] = threshold_value
                    
                    # Calculate initial performance metrics
                    performance[threshold_key] = self._calculate_threshold_performance(
                        outbreak_values, control_values, threshold_value, direction='lower')
            
            # 2. ROC curve optimization
            if 'roc_optimization' in methods:
                optimal_threshold, optimal_metrics = self._optimize_threshold_roc(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['roc_optimal'] = optimal_threshold
                performance['roc_optimal'] = optimal_metrics
            
            # 3. Distribution overlap method
            if 'distribution_overlap' in methods:
                # Find the point of minimum overlap between distributions
                min_overlap_threshold = self._find_min_overlap_threshold(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['min_overlap'] = min_overlap_threshold
                performance['min_overlap'] = self._calculate_threshold_performance(
                    outbreak_values, control_values, min_overlap_threshold, direction='lower')
            
            # Store all calculated thresholds and their performance
            self.monitoring_thresholds['change_based'][metric] = {
                'thresholds': thresholds,
                'performance': performance,
                'outbreak_stats': {
                    'mean': outbreak_values.mean(),
                    'std': outbreak_values.std(),
                    'min': outbreak_values.min(),
                    'max': outbreak_values.max(),
                    'distribution': {
                        str(p): outbreak_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                },
                'control_stats': {
                    'mean': control_values.mean(),
                    'std': control_values.std(),
                    'min': control_values.min(),
                    'max': control_values.max(),
                    'distribution': {
                        str(p): control_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                }
            }
    
    def _identify_slope_based_thresholds(self):
        """Identify slope-based thresholds."""
        self.logger.info("Identifying slope-based thresholds...")
        
        metrics = ['3d_window_slope', '7d_window_slope']
        
        for metric in metrics:
            # Skip if column doesn't exist
            if metric not in self.pre_outbreak_stats.columns or metric not in self.control_stats.columns:
                self.logger.warning(f"Skipping {metric} threshold calculation - column not found")
                continue
                
            # Get outbreak and control distributions
            outbreak_values = self.pre_outbreak_stats[metric].dropna()
            control_values = self.control_stats[metric].dropna()
            
            if len(outbreak_values) < 5 or len(control_values) < 5:
                self.logger.warning(f"Insufficient data for {metric} threshold calculation")
                continue
                
            # For slope metrics, we're looking for negative slopes
            # that are steeper than what's seen in controls
            
            # Calculate thresholds using various methods
            methods = self.config.get('threshold_determination_methods', 
                                     ['percentile', 'roc_optimization', 'distribution_overlap'])
                
            thresholds = {}
            performance = {}
            
            # 1. Percentile-based thresholds
            if 'percentile' in methods:
                percentiles = self.config.get('threshold_percentiles', [5, 10, 25])
                for p in percentiles:
                    threshold_value = np.percentile(outbreak_values, p)
                    threshold_key = f'percentile_{p}'
                    thresholds[threshold_key] = threshold_value
                    
                    # Calculate initial performance metrics
                    performance[threshold_key] = self._calculate_threshold_performance(
                        outbreak_values, control_values, threshold_value, direction='lower')
            
            # 2. ROC curve optimization
            if 'roc_optimization' in methods:
                optimal_threshold, optimal_metrics = self._optimize_threshold_roc(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['roc_optimal'] = optimal_threshold
                performance['roc_optimal'] = optimal_metrics
            
            # 3. Distribution overlap method
            if 'distribution_overlap' in methods:
                # Find the point of minimum overlap between distributions
                min_overlap_threshold = self._find_min_overlap_threshold(
                    outbreak_values, control_values, direction='lower')
                
                thresholds['min_overlap'] = min_overlap_threshold
                performance['min_overlap'] = self._calculate_threshold_performance(
                    outbreak_values, control_values, min_overlap_threshold, direction='lower')
            
            # Store all calculated thresholds and their performance
            self.monitoring_thresholds['slope_based'][metric] = {
                'thresholds': thresholds,
                'performance': performance,
                'outbreak_stats': {
                    'mean': outbreak_values.mean(),
                    'std': outbreak_values.std(),
                    'min': outbreak_values.min(),
                    'max': outbreak_values.max(),
                    'distribution': {
                        str(p): outbreak_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                },
                'control_stats': {
                    'mean': control_values.mean(),
                    'std': control_values.std(),
                    'min': control_values.min(),
                    'max': control_values.max(),
                    'distribution': {
                        str(p): control_values.quantile(p/100) for p in range(5, 100, 5)
                    }
                }
            }
    
    def _identify_component_based_thresholds(self, component_analysis):
        """Identify component-based thresholds."""
        self.logger.info("Identifying component-based thresholds...")
        
        # Check if we have both outbreak and control component data
        if ('outbreak_components' not in component_analysis or 
            component_analysis['outbreak_components'].empty):
            self.logger.warning("No outbreak component data available for threshold calculation")
            return
            
        if ('control_components' not in component_analysis or 
            component_analysis['control_components'].empty):
            self.logger.warning("No control component data available for threshold calculation")
            return
        
        # Get component data
        outbreak_comp = component_analysis['outbreak_components']
        control_comp = component_analysis['control_components']
        
        # Define components to analyze
        components = ['upright_tails', 'hanging_tails']
        
        # Get days before removal to analyze
        days_before_list = self.config.get('component_threshold_days', [0, 1, 3, 7])
        
        for component in components:
            for days_before in days_before_list:
                # Filter data for specific days before
                outbreak_values = outbreak_comp[outbreak_comp['days_before_removal'] == days_before][component].dropna()
                control_values = control_comp[control_comp['days_before_removal'] == days_before][component].dropna()
                
                if len(outbreak_values) < 5 or len(control_values) < 5:
                    self.logger.warning(f"Insufficient data for {component} at {days_before} days before threshold calculation")
                    continue
                
                # Determine appropriate direction for threshold
                # For upright tails, we expect lower values before outbreaks (direction='lower')
                # For hanging tails, we expect higher values before outbreaks (direction='higher')
                direction = 'lower' if component == 'upright_tails' else 'higher'
                
                # Calculate thresholds using various methods
                methods = self.config.get('threshold_determination_methods', 
                                         ['percentile', 'roc_optimization', 'distribution_overlap'])
                    
                thresholds = {}
                performance = {}
                
                # 1. Percentile-based thresholds
                if 'percentile' in methods:
                    percentiles = self.config.get('threshold_percentiles', [5, 10, 25])
                    for p in percentiles:
                        # For upright tails (lower values), use low percentiles
                        # For hanging tails (higher values), use high percentiles
                        actual_p = p if direction == 'lower' else (100 - p)
                        threshold_value = np.percentile(outbreak_values, actual_p)
                        threshold_key = f'percentile_{actual_p}'
                        thresholds[threshold_key] = threshold_value
                        
                        # Calculate initial performance metrics
                        performance[threshold_key] = self._calculate_threshold_performance(
                            outbreak_values, control_values, threshold_value, direction=direction)
                
                # 2. ROC curve optimization
                if 'roc_optimization' in methods:
                    optimal_threshold, optimal_metrics = self._optimize_threshold_roc(
                        outbreak_values, control_values, direction=direction)
                    
                    thresholds['roc_optimal'] = optimal_threshold
                    performance['roc_optimal'] = optimal_metrics
                
                # 3. Distribution overlap method
                if 'distribution_overlap' in methods:
                    # Find the point of minimum overlap between distributions
                    min_overlap_threshold = self._find_min_overlap_threshold(
                        outbreak_values, control_values, direction=direction)
                    
                    thresholds['min_overlap'] = min_overlap_threshold
                    performance['min_overlap'] = self._calculate_threshold_performance(
                        outbreak_values, control_values, min_overlap_threshold, direction=direction)
                
                # Create a unique key for this component and timepoint
                component_key = f"{component}_{days_before}d_before"
                
                # Store all calculated thresholds and their performance
                self.monitoring_thresholds['component_based'][component_key] = {
                    'thresholds': thresholds,
                    'performance': performance,
                    'direction': direction,
                    'outbreak_stats': {
                        'mean': outbreak_values.mean(),
                        'std': outbreak_values.std(),
                        'min': outbreak_values.min(),
                        'max': outbreak_values.max(),
                        'distribution': {
                            str(p): outbreak_values.quantile(p/100) for p in range(5, 100, 5)
                        }
                    },
                    'control_stats': {
                        'mean': control_values.mean(),
                        'std': control_values.std(),
                        'min': control_values.min(),
                        'max': control_values.max(),
                        'distribution': {
                            str(p): control_values.quantile(p/100) for p in range(5, 100, 5)
                        }
                    }
                }
    
    def _identify_combined_thresholds(self):
        """Identify combined multi-factor thresholds."""
        self.logger.info("Identifying combined multi-factor thresholds...")
        
        # Define combinations to evaluate
        combinations = self.config.get('threshold_combinations', [
            # Tuple format: (name, [(metric, threshold_type), ...])
            ('value_slope_combo', [
                ('value_at_removal', 'roc_optimal'),
                ('3d_window_slope', 'roc_optimal')
            ]),
            ('multi_slope_combo', [
                ('3d_window_slope', 'roc_optimal'),
                ('7d_window_slope', 'roc_optimal')
            ]),
            ('change_value_combo', [
                ('abs_change_3d', 'roc_optimal'),
                ('value_at_removal', 'roc_optimal')
            ])
        ])
        
        # Initialize combined thresholds container
        self.monitoring_thresholds['combined'] = {}
        
        # Create and evaluate each combination
        for combo_name, combo_metrics in combinations:
            # Check if all required metrics exist
            valid_combo = True
            for metric, threshold_type in combo_metrics:
                metric_category = self._get_metric_category(metric)
                if (metric_category is None or 
                    metric not in self.monitoring_thresholds[metric_category] or
                    threshold_type not in self.monitoring_thresholds[metric_category][metric]['thresholds']):
                    valid_combo = False
                    self.logger.warning(f"Missing metric {metric} with threshold {threshold_type} for combination {combo_name}")
                    break
            
            if not valid_combo:
                continue
            
            # Get the threshold values for each metric in the combination
            combo_threshold_values = {}
            for metric, threshold_type in combo_metrics:
                metric_category = self._get_metric_category(metric)
                threshold_value = self.monitoring_thresholds[metric_category][metric]['thresholds'][threshold_type]
                combo_threshold_values[metric] = threshold_value
            
            # Evaluate performance of the combination
            # This requires applying all thresholds simultaneously to the data
            combo_performance = self._evaluate_threshold_combination(combo_metrics, combo_threshold_values)
            
            # Store combination information
            self.monitoring_thresholds['combined'][combo_name] = {
                'component_metrics': combo_metrics,
                'threshold_values': combo_threshold_values,
                'performance': combo_performance
            }
    
    def _get_metric_category(self, metric):
        """Helper to determine which category a metric belongs to."""
        category_mapping = {
            'value_at_removal': 'absolute_value',
            '3d_window_avg': 'absolute_value',
            '7d_window_avg': 'absolute_value',
            'abs_change_1d': 'change_based',
            'abs_change_3d': 'change_based',
            'abs_change_7d': 'change_based',
            '3d_window_slope': 'slope_based',
            '7d_window_slope': 'slope_based'
        }
        
        # If it's in the mapping, return the category
        if metric in category_mapping:
            return category_mapping[metric]
        
        # Check if it's a component metric
        if metric.startswith(('upright_tails', 'hanging_tails')):
            return 'component_based'
            
        return None
    
    def _calculate_threshold_performance(self, outbreak_values, control_values, threshold, direction='lower'):
        """
        Calculate basic performance metrics for a threshold.
        
        Args:
            outbreak_values (Series): Values from outbreak events
            control_values (Series): Values from control events
            threshold (float): Threshold value to evaluate
            direction (str): Whether values below ('lower') or above ('higher') threshold indicate outbreak
            
        Returns:
            dict: Performance metrics
        """
        # For lower thresholds, values below threshold are positive (outbreak)
        # For higher thresholds, values above threshold are positive (outbreak)
        if direction == 'lower':
            true_positives = sum(outbreak_values <= threshold)
            false_negatives = sum(outbreak_values > threshold)
            true_negatives = sum(control_values > threshold)
            false_positives = sum(control_values <= threshold)
        else:  # direction == 'higher'
            true_positives = sum(outbreak_values >= threshold)
            false_negatives = sum(outbreak_values < threshold)
            true_negatives = sum(control_values < threshold)
            false_positives = sum(control_values >= threshold)
        
        # Calculate performance metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # Also calculate balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # Calculate the Youden's J statistic (sensitivity + specificity - 1)
        youdens_j = sensitivity + specificity - 1
        
        # Calculate likelihood ratios, handling special cases
        
        # Positive likelihood ratio (sensitivity / (1-specificity))
        # When specificity = 1, PLR is undefined (division by zero)
        if specificity < 1:
            plr = sensitivity / (1 - specificity)
            # Cap extremely high values for reporting purposes
            if plr > 1000:
                plr_display = ">1000"
            else:
                plr_display = f"{plr:.2f}"
        else:
            # When specificity = 1, use special notation
            plr = None
            plr_display = "∞"  # Unicode infinity symbol
        
        # Negative likelihood ratio ((1-sensitivity) / specificity)
        # When sensitivity = 1, NLR = 0
        # When specificity = 0, NLR is undefined (division by zero)
        if specificity > 0:
            if sensitivity < 1:
                nlr = (1 - sensitivity) / specificity
                # Cap extremely low values for reporting purposes
                if nlr < 0.001:
                    nlr_display = "<0.001"
                else:
                    nlr_display = f"{nlr:.3f}"
            else:
                # When sensitivity = 1, NLR = 0
                nlr = 0
                nlr_display = "0.000"
        else:
            # When specificity = 0, use special notation
            nlr = None
            nlr_display = "∞"  # Unicode infinity symbol
        
        # Return all metrics
        return {
            'counts': {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives)
            },
            'metrics': {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'npv': float(npv),
                'accuracy': float(accuracy),
                'f1_score': float(f1_score),
                'balanced_accuracy': float(balanced_accuracy),
                'youdens_j': float(youdens_j),
                'positive_likelihood_ratio': plr,
                'positive_likelihood_ratio_display': plr_display,
                'negative_likelihood_ratio': nlr,
                'negative_likelihood_ratio_display': nlr_display
            }
        }
        
    def _optimize_threshold_roc(self, outbreak_values, control_values, direction='lower'):
        """
        Optimize threshold using ROC curve analysis.
        
        Args:
            outbreak_values (Series): Values from outbreak events
            control_values (Series): Values from control events
            direction (str): Whether values below ('lower') or above ('higher') threshold indicate outbreak
            
        Returns:
            tuple: (optimal_threshold, performance_metrics)
        """
        # Combine data and create labels
        all_values = pd.concat([outbreak_values, control_values])
        labels = np.concatenate([np.ones(len(outbreak_values)), np.zeros(len(control_values))])
        
        # For 'higher' direction, we need to negate the values to use with ROC curve
        # so that higher values correspond to higher probability of positive class
        if direction == 'lower':
            all_values = -all_values  # Negate values so that lower values -> higher scores
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(labels, all_values)
        
        # Calculate Youden's J statistic (sensitivity + specificity - 1)
        j_scores = tpr - fpr
        
        # Find the optimal threshold
        optimal_idx = np.argmax(j_scores)
        optimal_j = j_scores[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        # Get the corresponding threshold (need to negate back)
        if direction == 'lower':
            optimal_threshold = -thresholds[optimal_idx]
        else:
            optimal_threshold = thresholds[optimal_idx]
        
        # Calculate performance at this threshold
        performance = self._calculate_threshold_performance(
            outbreak_values, control_values, optimal_threshold, direction=direction)
        
        # Add ROC curve metrics
        roc_auc = auc(fpr, tpr)
        performance['metrics']['roc_auc'] = float(roc_auc)
        performance['metrics']['optimal_j'] = float(optimal_j)
        performance['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist() if direction == 'higher' else (-thresholds).tolist(),
            'optimal_point': {
                'fpr': float(optimal_fpr),
                'tpr': float(optimal_tpr),
                'threshold': float(optimal_threshold),
                'j_score': float(optimal_j)
            }
        }
        
        return optimal_threshold, performance
    
    def _find_min_overlap_threshold(self, outbreak_values, control_values, direction='lower'):
        """
        Find the threshold with minimum distribution overlap.
        
        Args:
            outbreak_values (Series): Values from outbreak events
            control_values (Series): Values from control events
            direction (str): Whether values below ('lower') or above ('higher') threshold indicate outbreak
            
        Returns:
            float: Threshold value with minimum overlap
        """
        # Combine all values to test potential thresholds
        all_values = np.concatenate([outbreak_values, control_values])
        unique_values = np.unique(all_values)
        
        # Test each possible threshold
        min_overlap = float('inf')
        best_threshold = None
        
        for threshold in unique_values:
            # Calculate overlap (false positives + false negatives)
            if direction == 'lower':
                false_positives = sum(control_values <= threshold)
                false_negatives = sum(outbreak_values > threshold)
            else:  # direction == 'higher'
                false_positives = sum(control_values >= threshold)
                false_negatives = sum(outbreak_values < threshold)
                
            overlap = false_positives + false_negatives
            
            if overlap < min_overlap:
                min_overlap = overlap
                best_threshold = threshold
        
        return best_threshold
    
    def _evaluate_threshold_combination(self, combo_metrics, threshold_values):
        """
        Evaluate the performance of a combination of thresholds.
        
        Args:
            combo_metrics (list): List of (metric, threshold_type) tuples
            threshold_values (dict): Dictionary of metric to threshold value
            
        Returns:
            dict: Performance metrics for the combination
        """
        # Get the outbreak and control data for all metrics in the combination
        outbreak_data = []
        control_data = []
        
        for metric, _ in combo_metrics:
            metric_category = self._get_metric_category(metric)
            
            # Skip if metric not found
            if metric_category is None:
                continue
                
            # Get control column name (handling the special case of value_at_removal)
            control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
            
            # Get the values
            outbreak_values = self.pre_outbreak_stats[metric].dropna()
            control_values = self.control_stats[control_metric].dropna()
            
            # Store data
            outbreak_data.append((metric, outbreak_values))
            control_data.append((metric, control_values))
        
        # Check if we have all required data
        if len(outbreak_data) != len(combo_metrics) or len(control_data) != len(combo_metrics):
            self.logger.warning("Missing data for some metrics in the combination")
            return None
        
        # Now, for each outbreak and control event, determine if it would trigger the combined threshold
        outbreak_results = []
        control_results = []
        
        # Process outbreak data
        # Create a dataframe with outbreak values for all metrics
        outbreak_df = pd.DataFrame()
        for metric, values in outbreak_data:
            outbreak_df[metric] = values
        
        # Check for each row if it meets all threshold criteria
        for _, row in outbreak_df.iterrows():
            # Check each threshold
            all_thresholds_met = True
            for metric, _ in combo_metrics:
                threshold = threshold_values[metric]
                
                # Determine direction based on metric
                if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                    # For these metrics, lower values (more negative) indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
                elif metric.startswith('upright_tails'):
                    # Lower upright tail values indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
                elif metric.startswith('hanging_tails'):
                    # Higher hanging tail values indicate outbreaks
                    if row[metric] < threshold:
                        all_thresholds_met = False
                        break
                else:
                    # For value metrics, lower values indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
            
            outbreak_results.append(all_thresholds_met)
        
        # Process control data
        # Create a dataframe with control values for all metrics
        control_df = pd.DataFrame()
        for metric, values in control_data:
            control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
            control_df[metric] = values
        
        # Check for each row if it meets all threshold criteria
        for _, row in control_df.iterrows():
            # Check each threshold
            all_thresholds_met = True
            for metric, _ in combo_metrics:
                threshold = threshold_values[metric]
                
                # Determine direction based on metric
                if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                    # For these metrics, lower values (more negative) indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
                elif metric.startswith('upright_tails'):
                    # Lower upright tail values indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
                elif metric.startswith('hanging_tails'):
                    # Higher hanging tail values indicate outbreaks
                    if row[metric] < threshold:
                        all_thresholds_met = False
                        break
                else:
                    # For value metrics, lower values indicate outbreaks
                    if row[metric] > threshold:
                        all_thresholds_met = False
                        break
            
            control_results.append(all_thresholds_met)
        
        # Calculate performance metrics
        true_positives = sum(outbreak_results)
        false_negatives = len(outbreak_results) - true_positives
        true_negatives = len(control_results) - sum(control_results)
        false_positives = sum(control_results)
        
        # Calculate derived metrics
        sensitivity = true_positives / len(outbreak_results) if len(outbreak_results) > 0 else 0
        specificity = true_negatives / len(control_results) if len(control_results) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        youdens_j = sensitivity + specificity - 1
        
        # Calculate positive likelihood ratio
        plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
        
        # Calculate negative likelihood ratio
        nlr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # Return performance metrics
        return {
            'counts': {
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'true_negatives': int(true_negatives),
                'false_negatives': int(false_negatives)
            },
            'metrics': {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'npv': float(npv),
                'accuracy': float(accuracy),
                'f1_score': float(f1_score),
                'balanced_accuracy': float(balanced_accuracy),
                'youdens_j': float(youdens_j),
                'positive_likelihood_ratio': float(plr),
                'negative_likelihood_ratio': float(nlr)
            }
        }
    
    def validate_monitoring_thresholds(self, validation_method='group_cv'):
        """
        Validate the identified monitoring thresholds using group-based cross-validation.
        
        This method performs validation while properly accounting for non-independent data
        by ensuring observations from the same group (pen) are never split between 
        training and testing sets, addressing the methodological flaw in standard CV.
        
        Args:
            validation_method (str): Validation method to use: 'group_cv', 'logo_cv'
                - 'group_cv': Group-based K-fold cross-validation
                - 'logo_cv': Leave-one-group-out cross-validation
                
        Returns:
            dict: Validation results
        """
        self.logger.info(f"Validating monitoring thresholds using {validation_method}...")
        
        # Check if we have identified thresholds
        if not hasattr(self, 'monitoring_thresholds'):
            self.logger.error("No monitoring thresholds identified. Run identify_monitoring_thresholds() first.")
            return None
        
        # Check if we have sufficient data for group-based validation
        outbreak_pens = self.pre_outbreak_stats['pen'].unique() if hasattr(self, 'pre_outbreak_stats') else []
        control_pens = self.control_stats['pen'].unique() if hasattr(self, 'control_stats') else []
        
        if len(outbreak_pens) < 3 or len(control_pens) < 3:
            self.logger.error(f"Insufficient number of unique groups for {validation_method}. "
                            f"Found {len(outbreak_pens)} outbreak pens and {len(control_pens)} control pens. "
                            f"At least 3 of each are required.")
            return None
        
        # Initialize validation results structure
        self.validation_results = {
            'absolute_value': {},
            'change_based': {},
            'slope_based': {},
            'component_based': {},
            'combined': {},
            'metadata': {
                'validation_method': validation_method,
                'generated_datetime': datetime.datetime.now().isoformat(),
                'n_outbreak_groups': len(outbreak_pens),
                'n_control_groups': len(control_pens)
            }
        }
        
        # Configure n_splits based on validation method
        if validation_method == 'group_cv':
            n_folds = min(self.config.get('validation_cv_folds', 5), 
                        len(outbreak_pens), len(control_pens))
            self.validation_results['metadata']['n_folds'] = n_folds
            self.logger.info(f"Using {n_folds}-fold group-based cross-validation")
        elif validation_method == 'logo_cv':
            n_folds = 'logo'
            self.validation_results['metadata']['validation_type'] = "Leave-one-group-out"
            self.logger.info("Using leave-one-group-out cross-validation")
        else:
            self.logger.error(f"Unknown validation method: {validation_method}")
            return None
        
        # Perform the cross-validation
        self._cross_validate_thresholds(n_folds=n_folds)
        
        # Save validation results
        self._save_validation_results(f"{validation_method}_validation")
        
        return self.validation_results
    
    def _cross_validate_thresholds(self, n_folds=5):
        """
        Validate thresholds using k-fold cross-validation or leave-one-out cross-validation.
        Implements group-based cross-validation to handle non-independent data points.
        
        Args:
            n_folds (int or str): Number of folds or 'loocv' for leave-one-out
        """
        # Prepare data for cross-validation
        
        # For each threshold category and metric
        for category in ['absolute_value', 'change_based', 'slope_based', 'component_based']:
            if category not in self.monitoring_thresholds:
                continue
                
            for metric, metric_data in self.monitoring_thresholds[category].items():
                # Prepare outbreak and control data
                if category == 'absolute_value' and metric == 'value_at_removal':
                    control_metric = 'value_at_reference'
                else:
                    control_metric = metric
                
                # Skip if metric doesn't exist in pre_outbreak_stats or control_stats
                if metric not in self.pre_outbreak_stats.columns:
                    self.logger.warning(f"Metric {metric} not found in pre_outbreak_stats. Skipping validation.")
                    continue
                    
                if control_metric not in self.control_stats.columns:
                    self.logger.warning(f"Metric {control_metric} not found in control_stats. Skipping validation.")
                    continue
                
                # Get outbreak and control values with group IDs
                outbreak_data = self.pre_outbreak_stats[['pen', metric]].dropna()
                control_data = self.control_stats[['pen', control_metric]].dropna()
                
                # Skip if not enough data
                if len(outbreak_data) < 3 or len(control_data) < 3:
                    self.logger.warning(f"Insufficient data for {metric} validation. Skipping.")
                    continue
                
                # Get unique pen IDs for group-based cross-validation
                outbreak_pens = outbreak_data['pen'].unique()
                control_pens = control_data['pen'].unique()
                
                if len(outbreak_pens) < 3 or len(control_pens) < 3:
                    self.logger.warning(f"Insufficient unique pens for {metric} group-based validation. Skipping.")
                    continue
                
                # Determine direction based on metric category
                if category in ['change_based', 'slope_based'] or (category == 'component_based' and 'upright_tails' in metric):
                    direction = 'lower'
                elif category == 'component_based' and 'hanging_tails' in metric:
                    direction = 'higher'
                else:  # absolute_value
                    direction = 'lower'
                
                # Initialize group-based cross-validation
                if n_folds == 'loocv':
                    # Use Leave-One-Group-Out cross-validation
                    from sklearn.model_selection import LeaveOneGroupOut
                    outbreak_cv = LeaveOneGroupOut()
                    control_cv = LeaveOneGroupOut()
                else:
                    # Use Group K-Fold cross-validation
                    from sklearn.model_selection import GroupKFold
                    outbreak_cv = GroupKFold(n_splits=min(n_folds, len(outbreak_pens)))
                    control_cv = GroupKFold(n_splits=min(n_folds, len(control_pens)))
                
                # Initialize validation metrics
                validation_metrics = {
                    'thresholds': {},
                    'performance': {}
                }
                
                # Get threshold types to validate
                threshold_types = list(metric_data['thresholds'].keys())
                
                # Validate each threshold type
                for threshold_type in threshold_types:
                    threshold_values = []
                    fold_performances = []
                    
                    # Prepare data for group-based cross-validation
                    outbreak_groups = outbreak_data['pen'].values
                    control_groups = control_data['pen'].values
                    
                    # Perform cross-validation
                    for train_index, test_index in outbreak_cv.split(outbreak_data, groups=outbreak_groups):
                        # Get training and test data for outbreaks
                        train_outbreak_data = outbreak_data.iloc[train_index]
                        test_outbreak_data = outbreak_data.iloc[test_index]
                        
                        train_outbreak_values = train_outbreak_data[metric]
                        test_outbreak_values = test_outbreak_data[metric]
                        
                        # Get training and test data for controls
                        # For each outbreak fold, use all control data
                        # This is simple but effective for validating outbreak detection against controls
                        train_control_values = control_data[control_metric]
                        
                        # Optimize threshold on training data
                        if threshold_type == 'roc_optimal':
                            fold_threshold, _ = self._optimize_threshold_roc(
                                train_outbreak_values, train_control_values, direction=direction)
                        elif threshold_type == 'min_overlap':
                            fold_threshold = self._find_min_overlap_threshold(
                                train_outbreak_values, train_control_values, direction=direction)
                        elif threshold_type.startswith('percentile_'):
                            p = int(threshold_type.split('_')[1])
                            fold_threshold = np.percentile(train_outbreak_values, p)
                        elif threshold_type.startswith('std_'):
                            std_mult = float(threshold_type.split('_')[1])
                            fold_threshold = train_control_values.mean() - std_mult * train_control_values.std()
                        else:
                            # Use the threshold value from the full dataset
                            fold_threshold = metric_data['thresholds'][threshold_type]
                        
                        # Record threshold value
                        threshold_values.append(fold_threshold)
                        
                        # Now split control data for testing as well
                        for c_train_index, c_test_index in control_cv.split(control_data, groups=control_groups):
                            test_control_data = control_data.iloc[c_test_index]
                            test_control_values = test_control_data[control_metric]
                            
                            # If either test set is empty, skip this fold
                            if len(test_outbreak_values) == 0 or len(test_control_values) == 0:
                                continue
                            
                            # Calculate performance
                            fold_performance = self._calculate_threshold_performance(
                                test_outbreak_values, test_control_values, fold_threshold, direction=direction)
                                
                            fold_performances.append(fold_performance)
                    
                    # Calculate average performance across folds
                    avg_performance = self._average_fold_performances(fold_performances)
                    
                    # Calculate threshold statistics
                    threshold_stats = {
                        'mean': np.mean(threshold_values),
                        'std': np.std(threshold_values),
                        'min': np.min(threshold_values),
                        'max': np.max(threshold_values),
                        'values': threshold_values
                    }
                    
                    # Store validation results
                    validation_metrics['thresholds'][threshold_type] = threshold_stats
                    validation_metrics['performance'][threshold_type] = avg_performance
                
                # Store validation metrics for this metric
                self.validation_results[category][metric] = validation_metrics
        
        # Validate combined thresholds
        if 'combined' in self.monitoring_thresholds:
            for combo_name, combo_data in self.monitoring_thresholds['combined'].items():
                # Get component metrics and threshold values
                combo_metrics = combo_data['component_metrics']
                threshold_values = combo_data['threshold_values']
                
                # Initialize validation metrics
                validation_metrics = {
                    'component_metrics': combo_metrics,
                    'threshold_values': {},
                    'performance': {}
                }
                
                # Extract pen IDs from outbreak and control data
                outbreak_pens = self.pre_outbreak_stats['pen'].unique()
                control_pens = self.control_stats['pen'].unique()
                
                # Initialize group-based cross-validation for both outbreak and control data
                if n_folds == 'loocv':
                    from sklearn.model_selection import LeaveOneGroupOut
                    outbreak_cv = LeaveOneGroupOut()
                    control_cv = LeaveOneGroupOut()
                else:
                    from sklearn.model_selection import GroupKFold
                    outbreak_cv = GroupKFold(n_splits=min(n_folds, len(outbreak_pens)))
                    control_cv = GroupKFold(n_splits=min(n_folds, len(control_pens)))
                
                # Group data by pen for group-based cross-validation
                outbreak_groups = self.pre_outbreak_stats['pen'].values
                control_groups = self.control_stats['pen'].values
                
                # Perform group-based cross-validation
                fold_performances = []
                fold_thresholds = {metric: [] for metric, _ in combo_metrics}
                
                # For each fold of outbreak pens
                for outbreak_train_idx, outbreak_test_idx in outbreak_cv.split(
                        self.pre_outbreak_stats, groups=outbreak_groups):
                    
                    # Get training and test outbreak data
                    train_outbreak_data = self.pre_outbreak_stats.iloc[outbreak_train_idx]
                    test_outbreak_data = self.pre_outbreak_stats.iloc[outbreak_test_idx]
                    
                    # For each fold of control pens
                    for control_train_idx, control_test_idx in control_cv.split(
                            self.control_stats, groups=control_groups):
                        
                        # Get training and test control data
                        train_control_data = self.control_stats.iloc[control_train_idx]
                        test_control_data = self.control_stats.iloc[control_test_idx]
                        
                        # Calculate threshold for each metric in the combination
                        fold_threshold_values = {}
                        
                        for metric, threshold_type in combo_metrics:
                            # Skip if metric is missing in either dataset
                            if metric not in train_outbreak_data.columns:
                                self.logger.warning(f"Metric {metric} not found in training outbreak data. Skipping combo {combo_name}.")
                                continue
                            
                            control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                            if control_metric not in train_control_data.columns:
                                self.logger.warning(f"Metric {control_metric} not found in training control data. Skipping combo {combo_name}.")
                                continue
                            
                            # Get training values
                            train_outbreak_values = train_outbreak_data[metric].dropna()
                            train_control_values = train_control_data[control_metric].dropna()
                            
                            # Skip if not enough data
                            if len(train_outbreak_values) < 3 or len(train_control_values) < 3:
                                self.logger.warning(f"Insufficient training data for {metric} in combo {combo_name}. Skipping fold.")
                                continue
                            
                            # Determine direction based on metric
                            if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                                direction = 'lower'
                            elif metric.startswith('upright_tails'):
                                direction = 'lower'
                            elif metric.startswith('hanging_tails'):
                                direction = 'higher'
                            else:
                                direction = 'lower'
                            
                            # Calculate threshold
                            if threshold_type == 'roc_optimal':
                                fold_threshold, _ = self._optimize_threshold_roc(
                                    train_outbreak_values, train_control_values, direction=direction)
                            elif threshold_type == 'min_overlap':
                                fold_threshold = self._find_min_overlap_threshold(
                                    train_outbreak_values, train_control_values, direction=direction)
                            elif threshold_type.startswith('percentile_'):
                                p = int(threshold_type.split('_')[1])
                                fold_threshold = np.percentile(train_outbreak_values, p)
                            elif threshold_type.startswith('std_'):
                                std_mult = float(threshold_type.split('_')[1])
                                fold_threshold = train_control_values.mean() - std_mult * train_control_values.std()
                            else:
                                # Use the threshold value from the full dataset
                                fold_threshold = threshold_values[metric]
                            
                            # Store threshold value
                            fold_threshold_values[metric] = fold_threshold
                            fold_thresholds[metric].append(fold_threshold)
                        
                        # Skip if any component metrics are missing thresholds
                        if len(fold_threshold_values) != len(combo_metrics):
                            continue
                        
                        # Apply combined threshold to test data
                        
                        # Get test data for each metric
                        test_outbreak_results = []
                        
                        # Check if all metrics are available in test data
                        all_metrics_available = True
                        for metric, _ in combo_metrics:
                            if metric not in test_outbreak_data.columns:
                                all_metrics_available = False
                                break
                                
                            control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                            if control_metric not in test_control_data.columns:
                                all_metrics_available = False
                                break
                        
                        if not all_metrics_available:
                            continue
                        
                        # Process test outbreak data
                        for _, row in test_outbreak_data.iterrows():
                            all_thresholds_met = True
                            for metric, _ in combo_metrics:
                                # Skip if value is NaN
                                if pd.isna(row[metric]) or metric not in fold_threshold_values:
                                    all_thresholds_met = False
                                    break
                                    
                                threshold = fold_threshold_values[metric]
                                
                                # Determine direction based on metric
                                if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                                    if row[metric] > threshold:
                                        all_thresholds_met = False
                                        break
                                elif metric.startswith('upright_tails'):
                                    if row[metric] > threshold:
                                        all_thresholds_met = False
                                        break
                                elif metric.startswith('hanging_tails'):
                                    if row[metric] < threshold:
                                        all_thresholds_met = False
                                        break
                                else:
                                    if row[metric] > threshold:
                                        all_thresholds_met = False
                                        break
                            
                            test_outbreak_results.append(all_thresholds_met)
                        
                        # Process test control data
                        test_control_results = []
                        for _, row in test_control_data.iterrows():
                            all_thresholds_met = True
                            for metric, _ in combo_metrics:
                                control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                                
                                # Skip if value is NaN
                                if pd.isna(row[control_metric]) or metric not in fold_threshold_values:
                                    all_thresholds_met = False
                                    break
                                    
                                threshold = fold_threshold_values[metric]
                                
                                # Determine direction based on metric
                                if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                                    if row[control_metric] > threshold:
                                        all_thresholds_met = False
                                        break
                                elif metric.startswith('upright_tails'):
                                    if row[control_metric] > threshold:
                                        all_thresholds_met = False
                                        break
                                elif metric.startswith('hanging_tails'):
                                    if row[control_metric] < threshold:
                                        all_thresholds_met = False
                                        break
                                else:
                                    if row[control_metric] > threshold:
                                        all_thresholds_met = False
                                        break
                            
                            test_control_results.append(all_thresholds_met)
                        
                        # Skip if either test set is empty
                        if len(test_outbreak_results) == 0 or len(test_control_results) == 0:
                            continue
                        
                        # Calculate performance
                        true_positives = sum(test_outbreak_results)
                        false_negatives = len(test_outbreak_results) - true_positives
                        true_negatives = len(test_control_results) - sum(test_control_results)
                        false_positives = sum(test_control_results)
                        
                        # Skip if any category is empty
                        if (true_positives + false_negatives) == 0 or (true_negatives + false_positives) == 0:
                            continue
                        
                        # Calculate metrics
                        sensitivity = true_positives / (true_positives + false_negatives)
                        specificity = true_negatives / (true_negatives + false_positives)
                        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                        npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
                        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                        balanced_accuracy = (sensitivity + specificity) / 2
                        youdens_j = sensitivity + specificity - 1
                        
                        # Calculate likelihood ratios (safely)
                        if (1 - specificity) > 0:
                            plr = sensitivity / (1 - specificity)
                        else:
                            plr = None  # Will handle special cases later
                            
                        if specificity > 0:
                            nlr = (1 - sensitivity) / specificity
                        else:
                            nlr = None  # Will handle special cases later
                        
                        # Store performance
                        fold_performance = {
                            'counts': {
                                'true_positives': int(true_positives),
                                'false_positives': int(false_positives),
                                'true_negatives': int(true_negatives),
                                'false_negatives': int(false_negatives)
                            },
                            'metrics': {
                                'sensitivity': float(sensitivity),
                                'specificity': float(specificity),
                                'precision': float(precision),
                                'npv': float(npv),
                                'accuracy': float(accuracy),
                                'f1_score': float(f1_score),
                                'balanced_accuracy': float(balanced_accuracy),
                                'youdens_j': float(youdens_j),
                                'positive_likelihood_ratio': plr,
                                'negative_likelihood_ratio': nlr
                            }
                        }
                        
                        fold_performances.append(fold_performance)
                
                # Calculate average threshold values
                threshold_stats = {}
                for metric in fold_thresholds:
                    values = fold_thresholds[metric]
                    if values:  # Check if we have values
                        threshold_stats[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'values': values
                        }
                
                # Calculate average performance across folds
                avg_performance = self._average_fold_performances(fold_performances)
                
                # Store validation metrics
                validation_metrics['threshold_values'] = threshold_stats
                validation_metrics['performance'] = avg_performance
                
                # Store in validation results
                self.validation_results['combined'][combo_name] = validation_metrics
    
    def _bootstrap_validate_thresholds(self, n_iterations=100):
        """
        Validate thresholds using bootstrap resampling with out-of-bag evaluation.
        This provides a less biased estimate of performance compared to standard bootstrap.
        
        Args:
            n_iterations (int): Number of bootstrap iterations
        """
        self.logger.info(f"Validating thresholds using bootstrap resampling with out-of-bag evaluation ({n_iterations} iterations)...")
        
        # For each threshold category and metric
        for category in ['absolute_value', 'change_based', 'slope_based', 'component_based']:
            if category not in self.monitoring_thresholds:
                continue
                
            for metric, metric_data in self.monitoring_thresholds[category].items():
                # Prepare outbreak and control data
                if category == 'absolute_value' and metric == 'value_at_removal':
                    control_metric = 'value_at_reference'
                else:
                    control_metric = metric
                
                # Skip if metric doesn't exist in pre_outbreak_stats or control_stats
                if metric not in self.pre_outbreak_stats.columns:
                    self.logger.warning(f"Metric {metric} not found in pre_outbreak_stats. Skipping validation.")
                    continue
                    
                if control_metric not in self.control_stats.columns:
                    self.logger.warning(f"Metric {control_metric} not found in control_stats. Skipping validation.")
                    continue
                
                # Get outbreak and control data with IDs for tracking in-bag vs out-of-bag
                outbreak_data = self.pre_outbreak_stats[['pen', metric]].dropna()
                control_data = self.control_stats[['pen', control_metric]].dropna()
                
                # Skip if not enough data
                if len(outbreak_data) < 3 or len(control_data) < 3:
                    self.logger.warning(f"Insufficient data for {metric} bootstrap validation. Skipping.")
                    continue
                
                # Determine direction based on metric category
                if category in ['change_based', 'slope_based'] or (category == 'component_based' and 'upright_tails' in metric):
                    direction = 'lower'
                elif category == 'component_based' and 'hanging_tails' in metric:
                    direction = 'higher'
                else:  # absolute_value
                    direction = 'lower'
                
                # Initialize validation metrics
                validation_metrics = {
                    'thresholds': {},
                    'performance': {}
                }
                
                # Get threshold types to validate
                threshold_types = list(metric_data['thresholds'].keys())
                
                # Validate each threshold type
                for threshold_type in threshold_types:
                    threshold_values = []
                    bootstrap_performances = []
                    
                    # Perform bootstrap iterations
                    for _ in range(n_iterations):
                        # Get unique indices for outbreak data
                        outbreak_indices = np.arange(len(outbreak_data))
                        
                        # Sample with replacement for in-bag indices
                        in_bag_outbreak_indices = np.random.choice(
                            outbreak_indices, size=len(outbreak_indices), replace=True)
                        
                        # Determine out-of-bag indices (those not in the in-bag sample)
                        oob_outbreak_indices = np.array(list(set(outbreak_indices) - set(in_bag_outbreak_indices)))
                        
                        # Similarly for control data
                        control_indices = np.arange(len(control_data))
                        in_bag_control_indices = np.random.choice(
                            control_indices, size=len(control_indices), replace=True)
                        oob_control_indices = np.array(list(set(control_indices) - set(in_bag_control_indices)))
                        
                        # Skip if either OOB set is empty
                        if len(oob_outbreak_indices) == 0 or len(oob_control_indices) == 0:
                            continue
                        
                        # Get in-bag and out-of-bag samples
                        in_bag_outbreak = outbreak_data.iloc[in_bag_outbreak_indices][metric]
                        in_bag_control = control_data.iloc[in_bag_control_indices][control_metric]
                        
                        oob_outbreak = outbreak_data.iloc[oob_outbreak_indices][metric]
                        oob_control = control_data.iloc[oob_control_indices][control_metric]
                        
                        # Calculate threshold on in-bag sample
                        if threshold_type == 'roc_optimal':
                            bootstrap_threshold, _ = self._optimize_threshold_roc(
                                in_bag_outbreak, in_bag_control, direction=direction)
                        elif threshold_type == 'min_overlap':
                            bootstrap_threshold = self._find_min_overlap_threshold(
                                in_bag_outbreak, in_bag_control, direction=direction)
                        elif threshold_type.startswith('percentile_'):
                            p = int(threshold_type.split('_')[1])
                            bootstrap_threshold = np.percentile(in_bag_outbreak, p)
                        elif threshold_type.startswith('std_'):
                            std_mult = float(threshold_type.split('_')[1])
                            bootstrap_threshold = in_bag_control.mean() - std_mult * in_bag_control.std()
                        else:
                            # Use the threshold value from the full dataset
                            bootstrap_threshold = metric_data['thresholds'][threshold_type]
                        
                        # Record threshold value
                        threshold_values.append(bootstrap_threshold)
                        
                        # Calculate performance on the out-of-bag sample
                        # This provides an unbiased estimate of performance
                        bootstrap_performance = self._calculate_threshold_performance(
                            oob_outbreak, oob_control, bootstrap_threshold, direction=direction)
                            
                        bootstrap_performances.append(bootstrap_performance)
                    
                    # Skip if no valid iterations
                    if not bootstrap_performances:
                        self.logger.warning(f"No valid bootstrap iterations for {metric} with {threshold_type}. Skipping.")
                        continue
                    
                    # Calculate average performance across bootstrap iterations
                    avg_performance = self._average_fold_performances(bootstrap_performances)
                    
                    # Calculate confidence intervals for metrics
                    confidence_level = self.config.get('validation_confidence_level', 0.95)
                    low_percentile = 100 * (1 - confidence_level) / 2
                    high_percentile = 100 - low_percentile
                    
                    # Add confidence intervals to performance metrics
                    for metric_name in avg_performance['metrics']:
                        # Skip likelihood ratios that might be None
                        metric_values = [perf['metrics'][metric_name] for perf in bootstrap_performances 
                                        if metric_name in perf['metrics'] and perf['metrics'][metric_name] is not None]
                        
                        if metric_values:
                            avg_performance['metrics'][f'{metric_name}_ci_lower'] = np.percentile(metric_values, low_percentile)
                            avg_performance['metrics'][f'{metric_name}_ci_upper'] = np.percentile(metric_values, high_percentile)
                    
                    # Calculate threshold statistics
                    threshold_stats = {
                        'mean': np.mean(threshold_values),
                        'std': np.std(threshold_values),
                        'min': np.min(threshold_values),
                        'max': np.max(threshold_values),
                        'ci_lower': np.percentile(threshold_values, low_percentile),
                        'ci_upper': np.percentile(threshold_values, high_percentile)
                    }
                    
                    # Store validation results
                    validation_metrics['thresholds'][threshold_type] = threshold_stats
                    validation_metrics['performance'][threshold_type] = avg_performance
                
                # Store validation metrics for this metric
                self.validation_results[category][metric] = validation_metrics
                
        # Bootstrap validation for combined thresholds with out-of-bag evaluation
        if 'combined' in self.monitoring_thresholds:
            for combo_name, combo_data in self.monitoring_thresholds['combined'].items():
                # Get component metrics and threshold values
                combo_metrics = combo_data['component_metrics']
                
                # Initialize validation metrics
                validation_metrics = {
                    'component_metrics': combo_metrics,
                    'threshold_values': {},
                    'performance': {}
                }
                
                # Prepare for bootstrap
                threshold_values = {metric: [] for metric, _ in combo_metrics}
                bootstrap_performances = []
                
                # Perform bootstrap iterations
                for _ in range(n_iterations):
                    # Group by pen to maintain data independence
                    outbreak_pens = self.pre_outbreak_stats['pen'].unique()
                    control_pens = self.control_stats['pen'].unique()
                    
                    # Sample pens with replacement
                    in_bag_outbreak_pens = np.random.choice(outbreak_pens, size=len(outbreak_pens), replace=True)
                    in_bag_control_pens = np.random.choice(control_pens, size=len(control_pens), replace=True)
                    
                    # Determine out-of-bag pens
                    oob_outbreak_pens = np.array(list(set(outbreak_pens) - set(in_bag_outbreak_pens)))
                    oob_control_pens = np.array(list(set(control_pens) - set(in_bag_control_pens)))
                    
                    # Skip if either OOB set is empty
                    if len(oob_outbreak_pens) == 0 or len(oob_control_pens) == 0:
                        continue
                    
                    # Filter data by in-bag and out-of-bag pens
                    in_bag_outbreak_data = self.pre_outbreak_stats[self.pre_outbreak_stats['pen'].isin(in_bag_outbreak_pens)]
                    in_bag_control_data = self.control_stats[self.control_stats['pen'].isin(in_bag_control_pens)]
                    
                    oob_outbreak_data = self.pre_outbreak_stats[self.pre_outbreak_stats['pen'].isin(oob_outbreak_pens)]
                    oob_control_data = self.control_stats[self.control_stats['pen'].isin(oob_control_pens)]
                    
                    # Calculate thresholds for each component metric
                    fold_threshold_values = {}
                    
                    for metric, threshold_type in combo_metrics:
                        # Skip if metric is missing in either dataset
                        if metric not in in_bag_outbreak_data.columns:
                            self.logger.warning(f"Metric {metric} not found in bootstrap outbreak data. Skipping combo {combo_name}.")
                            continue
                        
                        control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                        if control_metric not in in_bag_control_data.columns:
                            self.logger.warning(f"Metric {control_metric} not found in bootstrap control data. Skipping combo {combo_name}.")
                            continue
                        
                        # Get in-bag values
                        in_bag_outbreak_values = in_bag_outbreak_data[metric].dropna()
                        in_bag_control_values = in_bag_control_data[control_metric].dropna()
                        
                        # Skip if not enough data
                        if len(in_bag_outbreak_values) < 3 or len(in_bag_control_values) < 3:
                            self.logger.warning(f"Insufficient bootstrap data for {metric} in combo {combo_name}. Skipping.")
                            continue
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            direction = 'lower'
                        elif metric.startswith('upright_tails'):
                            direction = 'lower'
                        elif metric.startswith('hanging_tails'):
                            direction = 'higher'
                        else:
                            direction = 'lower'
                        
                        # Calculate threshold on in-bag sample
                        if threshold_type == 'roc_optimal':
                            bootstrap_threshold, _ = self._optimize_threshold_roc(
                                in_bag_outbreak_values, in_bag_control_values, direction=direction)
                        elif threshold_type == 'min_overlap':
                            bootstrap_threshold = self._find_min_overlap_threshold(
                                in_bag_outbreak_values, in_bag_control_values, direction=direction)
                        elif threshold_type.startswith('percentile_'):
                            p = int(threshold_type.split('_')[1])
                            bootstrap_threshold = np.percentile(in_bag_outbreak_values, p)
                        elif threshold_type.startswith('std_'):
                            std_mult = float(threshold_type.split('_')[1])
                            bootstrap_threshold = in_bag_control_values.mean() - std_mult * in_bag_control_values.std()
                        else:
                            # Use the threshold value from the full dataset
                            bootstrap_threshold = combo_data['threshold_values'][metric]
                        
                        # Store threshold value
                        fold_threshold_values[metric] = bootstrap_threshold
                        threshold_values[metric].append(bootstrap_threshold)
                    
                    # Skip if we don't have thresholds for all metrics
                    if len(fold_threshold_values) != len(combo_metrics):
                        continue
                    
                    # Apply combined threshold to out-of-bag data
                    # Process out-of-bag outbreak data
                    oob_outbreak_results = []
                    for _, row in oob_outbreak_data.iterrows():
                        all_thresholds_met = True
                        for metric, _ in combo_metrics:
                            # Skip if value is NaN
                            if pd.isna(row[metric]) or metric not in fold_threshold_values:
                                all_thresholds_met = False
                                break
                                
                            threshold = fold_threshold_values[metric]
                            
                            # Determine direction based on metric
                            if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                                if row[metric] > threshold:
                                    all_thresholds_met = False
                                    break
                            elif metric.startswith('upright_tails'):
                                if row[metric] > threshold:
                                    all_thresholds_met = False
                                    break
                            elif metric.startswith('hanging_tails'):
                                if row[metric] < threshold:
                                    all_thresholds_met = False
                                    break
                            else:
                                if row[metric] > threshold:
                                    all_thresholds_met = False
                                    break
                        
                        oob_outbreak_results.append(all_thresholds_met)
                    
                    # Process out-of-bag control data
                    oob_control_results = []
                    for _, row in oob_control_data.iterrows():
                        all_thresholds_met = True
                        for metric, _ in combo_metrics:
                            control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                            
                            # Skip if value is NaN
                            if pd.isna(row[control_metric]) or metric not in fold_threshold_values:
                                all_thresholds_met = False
                                break
                                
                            threshold = fold_threshold_values[metric]
                            
                            # Determine direction based on metric
                            if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                                if row[control_metric] > threshold:
                                    all_thresholds_met = False
                                    break
                            elif metric.startswith('upright_tails'):
                                if row[control_metric] > threshold:
                                    all_thresholds_met = False
                                    break
                            elif metric.startswith('hanging_tails'):
                                if row[control_metric] < threshold:
                                    all_thresholds_met = False
                                    break
                            else:
                                if row[control_metric] > threshold:
                                    all_thresholds_met = False
                                    break
                        
                        oob_control_results.append(all_thresholds_met)
                    
                    # Skip if either OOB set is empty
                    if len(oob_outbreak_results) == 0 or len(oob_control_results) == 0:
                        continue
                    
                    # Calculate performance on out-of-bag data
                    true_positives = sum(oob_outbreak_results)
                    false_negatives = len(oob_outbreak_results) - true_positives
                    true_negatives = len(oob_control_results) - sum(oob_control_results)
                    false_positives = sum(oob_control_results)
                    
                    # Skip if any category has zero samples
                    if (true_positives + false_negatives) == 0 or (true_negatives + false_positives) == 0:
                        continue
                    
                    # Calculate metrics safely
                    sensitivity = true_positives / (true_positives + false_negatives)
                    specificity = true_negatives / (true_negatives + false_positives)
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    npv = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
                    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                    balanced_accuracy = (sensitivity + specificity) / 2
                    youdens_j = sensitivity + specificity - 1
                    
                    # Calculate likelihood ratios (safely)
                    if specificity < 1.0:
                        plr = sensitivity / (1 - specificity)
                    else:
                        plr = None  # Will be handled specially in reporting
                        
                    if specificity > 0:
                        nlr = (1 - sensitivity) / specificity
                    else:
                        nlr = None  # Will be handled specially in reporting
                    
                    # Store performance
                    fold_performance = {
                        'counts': {
                            'true_positives': int(true_positives),
                            'false_positives': int(false_positives),
                            'true_negatives': int(true_negatives),
                            'false_negatives': int(false_negatives)
                        },
                        'metrics': {
                            'sensitivity': float(sensitivity),
                            'specificity': float(specificity),
                            'precision': float(precision),
                            'npv': float(npv),
                            'accuracy': float(accuracy),
                            'f1_score': float(f1_score),
                            'balanced_accuracy': float(balanced_accuracy),
                            'youdens_j': float(youdens_j),
                            'positive_likelihood_ratio': plr,
                            'negative_likelihood_ratio': nlr
                        }
                    }
                    
                    bootstrap_performances.append(fold_performance)
                
                # Skip if no valid bootstrap iterations
                if not bootstrap_performances:
                    self.logger.warning(f"No valid bootstrap iterations for combo {combo_name}. Skipping.")
                    continue
                
                # Calculate average performance across bootstrap iterations
                avg_performance = self._average_fold_performances(bootstrap_performances)
                
                # Calculate confidence intervals for metrics
                confidence_level = self.config.get('validation_confidence_level', 0.95)
                low_percentile = 100 * (1 - confidence_level) / 2
                high_percentile = 100 - low_percentile
                
                # Add confidence intervals to performance metrics
                for metric_name in avg_performance['metrics']:
                    # Skip likelihood ratios that might be None
                    metric_values = [perf['metrics'][metric_name] for perf in bootstrap_performances 
                                    if metric_name in perf['metrics'] and perf['metrics'][metric_name] is not None]
                    
                    if metric_values:
                        avg_performance['metrics'][f'{metric_name}_ci_lower'] = np.percentile(metric_values, low_percentile)
                        avg_performance['metrics'][f'{metric_name}_ci_upper'] = np.percentile(metric_values, high_percentile)
                
                # Calculate threshold statistics for each component metric
                threshold_stats = {}
                for metric, values in threshold_values.items():
                    if values:  # Check if we have values
                        threshold_stats[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'ci_lower': np.percentile(values, low_percentile),
                            'ci_upper': np.percentile(values, high_percentile)
                        }
                
                # Store validation metrics
                validation_metrics['threshold_values'] = threshold_stats
                validation_metrics['performance'] = avg_performance
                
                # Store in validation results
                self.validation_results['combined'][combo_name] = validation_metrics
    
    def _time_based_validate_thresholds(self):
        """
        Validate thresholds using time-based splitting.
        
        This assumes that the dataset has chronological information and can be
        split into earlier and later periods for validation.
        """
        self.logger.info("Validating thresholds using time-based splitting...")
        
        # Check if we have temporal information
        if 'culprit_removal_date' not in self.pre_outbreak_stats.columns:
            self.logger.error("No removal date information available for time-based validation.")
            return
            
        # Add a temporal validation attribute
        self.validation_results['time_based'] = {}
        
        # Sort outbreaks by date
        self.pre_outbreak_stats['removal_date'] = pd.to_datetime(self.pre_outbreak_stats['culprit_removal_date'])
        self.pre_outbreak_stats = self.pre_outbreak_stats.sort_values('removal_date')
        
        # Get split ratio from config
        train_ratio = self.config.get('time_validation_train_ratio', 0.6)
        
        # Split data into training and testing sets
        train_size = int(len(self.pre_outbreak_stats) * train_ratio)
        train_outbreaks = self.pre_outbreak_stats.iloc[:train_size]
        test_outbreaks = self.pre_outbreak_stats.iloc[train_size:]
        
        # Get split date for reporting
        split_date = train_outbreaks['removal_date'].max()
        self.validation_results['time_based']['metadata'] = {
            'split_date': split_date.strftime('%Y-%m-%d'),
            'train_size': len(train_outbreaks),
            'test_size': len(test_outbreaks)
        }
        
        # Similar split for control data
        # This is approximate since control references might not have clear dates
        # If reference_date is available, use it
        if 'reference_date' in self.control_stats.columns:
            self.control_stats['ref_date'] = pd.to_datetime(self.control_stats['reference_date'])
            self.control_stats = self.control_stats.sort_values('ref_date')
            train_size_control = int(len(self.control_stats) * train_ratio)
            train_controls = self.control_stats.iloc[:train_size_control]
            test_controls = self.control_stats.iloc[train_size_control:]
        else:
            # If no date, just split randomly
            train_controls, test_controls = train_test_split(
                self.control_stats, test_size=(1-train_ratio), random_state=self.config.get('random_seed', 42))
        
        # Now validate each threshold type
        
        # For each threshold category and metric
        for category in ['absolute_value', 'change_based', 'slope_based', 'component_based']:
            if category not in self.monitoring_thresholds:
                continue
                
            self.validation_results['time_based'][category] = {}
                
            for metric, metric_data in self.monitoring_thresholds[category].items():
                # Prepare outbreak and control data
                if category == 'absolute_value' and metric == 'value_at_removal':
                    control_metric = 'value_at_reference'
                else:
                    control_metric = metric
                
                # Skip if metric doesn't exist in training or testing data
                if metric not in train_outbreaks.columns or metric not in test_outbreaks.columns:
                    self.logger.warning(f"Metric {metric} not found in split outbreak data. Skipping validation.")
                    continue
                    
                if control_metric not in train_controls.columns or control_metric not in test_controls.columns:
                    self.logger.warning(f"Metric {control_metric} not found in split control data. Skipping validation.")
                    continue
                
                # Get training values
                train_outbreak_values = train_outbreaks[metric].dropna()
                train_control_values = train_controls[control_metric].dropna()
                
                # Get testing values
                test_outbreak_values = test_outbreaks[metric].dropna()
                test_control_values = test_controls[control_metric].dropna()
                
                # Skip if not enough data
                if (len(train_outbreak_values) < 3 or len(train_control_values) < 3 or
                    len(test_outbreak_values) < 3 or len(test_control_values) < 3):
                    self.logger.warning(f"Insufficient data for {metric} time-based validation. Skipping.")
                    continue
                
                # Determine direction based on metric category
                if category in ['change_based', 'slope_based'] or (category == 'component_based' and 'upright_tails' in metric):
                    direction = 'lower'
                elif category == 'component_based' and 'hanging_tails' in metric:
                    direction = 'higher'
                else:  # absolute_value
                    direction = 'lower'
                
                # Initialize validation metrics
                validation_metrics = {
                    'thresholds': {},
                    'performance': {}
                }
                
                # Get threshold types to validate
                threshold_types = list(metric_data['thresholds'].keys())
                
                # Validate each threshold type
                for threshold_type in threshold_types:
                    # Calculate threshold on training data
                    if threshold_type == 'roc_optimal':
                        train_threshold, _ = self._optimize_threshold_roc(
                            train_outbreak_values, train_control_values, direction=direction)
                    elif threshold_type == 'min_overlap':
                        train_threshold = self._find_min_overlap_threshold(
                            train_outbreak_values, train_control_values, direction=direction)
                    elif threshold_type.startswith('percentile_'):
                        p = int(threshold_type.split('_')[1])
                        train_threshold = np.percentile(train_outbreak_values, p)
                    elif threshold_type.startswith('std_'):
                        std_mult = float(threshold_type.split('_')[1])
                        train_threshold = train_control_values.mean() - std_mult * train_control_values.std()
                    else:
                        # Use the threshold value from the full dataset
                        train_threshold = metric_data['thresholds'][threshold_type]
                    
                    # Calculate performance on training data
                    train_performance = self._calculate_threshold_performance(
                        train_outbreak_values, train_control_values, train_threshold, direction=direction)
                    
                    # Calculate performance on testing data
                    test_performance = self._calculate_threshold_performance(
                        test_outbreak_values, test_control_values, train_threshold, direction=direction)
                    
                    # Store validation results
                    validation_metrics['thresholds'][threshold_type] = {
                        'value': train_threshold
                    }
                    validation_metrics['performance'][threshold_type] = {
                        'train': train_performance,
                        'test': test_performance,
                        'delta': {
                            'sensitivity': test_performance['metrics']['sensitivity'] - train_performance['metrics']['sensitivity'],
                            'specificity': test_performance['metrics']['specificity'] - train_performance['metrics']['specificity'],
                            'f1_score': test_performance['metrics']['f1_score'] - train_performance['metrics']['f1_score'],
                            'balanced_accuracy': test_performance['metrics']['balanced_accuracy'] - train_performance['metrics']['balanced_accuracy']
                        }
                    }
                
                # Store validation metrics for this metric
                self.validation_results['time_based'][category][metric] = validation_metrics
                
        # Validate combined thresholds
        if 'combined' in self.monitoring_thresholds:
            self.validation_results['time_based']['combined'] = {}
            
            for combo_name, combo_data in self.monitoring_thresholds['combined'].items():
                # Get component metrics and threshold values
                combo_metrics = combo_data['component_metrics']
                threshold_values = combo_data['threshold_values']
                
                # Calculate thresholds on training data
                train_threshold_values = {}
                
                for metric, threshold_type in combo_metrics:
                    if metric == 'value_at_removal':
                        control_metric = 'value_at_reference'
                    else:
                        control_metric = metric
                    
                    # Get training values
                    train_outbreak_values = train_outbreaks[metric].dropna()
                    train_control_values = train_controls[control_metric].dropna()
                    
                    # Determine direction based on metric
                    if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                        direction = 'lower'
                    elif metric.startswith('upright_tails'):
                        direction = 'lower'
                    elif metric.startswith('hanging_tails'):
                        direction = 'higher'
                    else:
                        direction = 'lower'
                    
                    # Calculate threshold
                    if threshold_type == 'roc_optimal':
                        train_threshold, _ = self._optimize_threshold_roc(
                            train_outbreak_values, train_control_values, direction=direction)
                    elif threshold_type == 'min_overlap':
                        train_threshold = self._find_min_overlap_threshold(
                            train_outbreak_values, train_control_values, direction=direction)
                    elif threshold_type.startswith('percentile_'):
                        p = int(threshold_type.split('_')[1])
                        train_threshold = np.percentile(train_outbreak_values, p)
                    elif threshold_type.startswith('std_'):
                        std_mult = float(threshold_type.split('_')[1])
                        train_threshold = train_control_values.mean() - std_mult * train_control_values.std()
                    else:
                        # Use the threshold value from the full dataset
                        train_threshold = threshold_values[metric]
                    
                    train_threshold_values[metric] = train_threshold
                
                # Evaluate on training data
                train_outbreak_df = train_outbreaks.copy()
                train_control_df = train_controls.copy()
                
                # Check each outbreak for threshold violations
                train_outbreak_results = []
                for _, row in train_outbreak_df.iterrows():
                    all_thresholds_met = True
                    for metric, _ in combo_metrics:
                        threshold = train_threshold_values[metric]
                        
                        # Skip if NaN
                        if pd.isna(row[metric]) or pd.isna(threshold):
                            continue
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('upright_tails'):
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('hanging_tails'):
                            if row[metric] < threshold:
                                all_thresholds_met = False
                                break
                        else:
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                    
                    train_outbreak_results.append(all_thresholds_met)
                
                # Check each control for threshold violations
                train_control_results = []
                for _, row in train_control_df.iterrows():
                    all_thresholds_met = True
                    for metric, _ in combo_metrics:
                        threshold = train_threshold_values[metric]
                        control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                        
                        # Skip if NaN
                        if pd.isna(row[control_metric]) or pd.isna(threshold):
                            continue
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('upright_tails'):
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('hanging_tails'):
                            if row[control_metric] < threshold:
                                all_thresholds_met = False
                                break
                        else:
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                    
                    train_control_results.append(all_thresholds_met)
                
                # Calculate training performance
                train_tp = sum(train_outbreak_results)
                train_fn = len(train_outbreak_results) - train_tp
                train_tn = len(train_control_results) - sum(train_control_results)
                train_fp = sum(train_control_results)
                
                train_sensitivity = train_tp / len(train_outbreak_results) if len(train_outbreak_results) > 0 else 0
                train_specificity = train_tn / len(train_control_results) if len(train_control_results) > 0 else 0
                train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
                train_npv = train_tn / (train_tn + train_fn) if (train_tn + train_fn) > 0 else 0
                train_accuracy = (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
                train_f1 = 2 * (train_precision * train_sensitivity) / (train_precision + train_sensitivity) if (train_precision + train_sensitivity) > 0 else 0
                train_balanced_acc = (train_sensitivity + train_specificity) / 2
                
                # Evaluate on testing data
                test_outbreak_df = test_outbreaks.copy()
                test_control_df = test_controls.copy()
                
                # Check each outbreak for threshold violations
                test_outbreak_results = []
                for _, row in test_outbreak_df.iterrows():
                    all_thresholds_met = True
                    for metric, _ in combo_metrics:
                        threshold = train_threshold_values[metric]
                        
                        # Skip if NaN
                        if pd.isna(row[metric]) or pd.isna(threshold):
                            continue
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('upright_tails'):
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('hanging_tails'):
                            if row[metric] < threshold:
                                all_thresholds_met = False
                                break
                        else:
                            if row[metric] > threshold:
                                all_thresholds_met = False
                                break
                    
                    test_outbreak_results.append(all_thresholds_met)
                
                # Check each control for threshold violations
                test_control_results = []
                for _, row in test_control_df.iterrows():
                    all_thresholds_met = True
                    for metric, _ in combo_metrics:
                        threshold = train_threshold_values[metric]
                        control_metric = 'value_at_reference' if metric == 'value_at_removal' else metric
                        
                        # Skip if NaN
                        if pd.isna(row[control_metric]) or pd.isna(threshold):
                            continue
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('upright_tails'):
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                        elif metric.startswith('hanging_tails'):
                            if row[control_metric] < threshold:
                                all_thresholds_met = False
                                break
                        else:
                            if row[control_metric] > threshold:
                                all_thresholds_met = False
                                break
                    
                    test_control_results.append(all_thresholds_met)
                
                # Calculate testing performance
                test_tp = sum(test_outbreak_results)
                test_fn = len(test_outbreak_results) - test_tp
                test_tn = len(test_control_results) - sum(test_control_results)
                test_fp = sum(test_control_results)
                
                test_sensitivity = test_tp / len(test_outbreak_results) if len(test_outbreak_results) > 0 else 0
                test_specificity = test_tn / len(test_control_results) if len(test_control_results) > 0 else 0
                test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
                test_npv = test_tn / (test_tn + test_fn) if (test_tn + test_fn) > 0 else 0
                test_accuracy = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)
                test_f1 = 2 * (test_precision * test_sensitivity) / (test_precision + test_sensitivity) if (test_precision + test_sensitivity) > 0 else 0
                test_balanced_acc = (test_sensitivity + test_specificity) / 2
                
                # Store validation results
                self.validation_results['time_based']['combined'][combo_name] = {
                    'component_metrics': combo_metrics,
                    'threshold_values': train_threshold_values,
                    'performance': {
                        'train': {
                            'counts': {
                                'true_positives': int(train_tp),
                                'false_positives': int(train_fp),
                                'true_negatives': int(train_tn),
                                'false_negatives': int(train_fn)
                            },
                            'metrics': {
                                'sensitivity': float(train_sensitivity),
                                'specificity': float(train_specificity),
                                'precision': float(train_precision),
                                'npv': float(train_npv),
                                'accuracy': float(train_accuracy),
                                'f1_score': float(train_f1),
                                'balanced_accuracy': float(train_balanced_acc)
                            }
                        },
                        'test': {
                            'counts': {
                                'true_positives': int(test_tp),
                                'false_positives': int(test_fp),
                                'true_negatives': int(test_tn),
                                'false_negatives': int(test_fn)
                            },
                            'metrics': {
                                'sensitivity': float(test_sensitivity),
                                'specificity': float(test_specificity),
                                'precision': float(test_precision),
                                'npv': float(test_npv),
                                'accuracy': float(test_accuracy),
                                'f1_score': float(test_f1),
                                'balanced_accuracy': float(test_balanced_acc)
                            }
                        },
                        'delta': {
                            'sensitivity': float(test_sensitivity - train_sensitivity),
                            'specificity': float(test_specificity - train_specificity),
                            'f1_score': float(test_f1 - train_f1),
                            'balanced_accuracy': float(test_balanced_acc - train_balanced_acc)
                        }
                    }
                }
    
    def _average_fold_performances(self, fold_performances):
        """
        Calculate average performance metrics across folds.
        Handles special cases like infinite likelihood ratios properly.
        
        Args:
            fold_performances (list): List of performance dictionaries
            
        Returns:
            dict: Averaged performance metrics
        """
        if not fold_performances:
            return {'counts': {}, 'metrics': {}}
        
        # Initialize averaged metrics
        avg_performance = {
            'counts': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            },
            'metrics': {}
        }
        
        # Sum counts
        for perf in fold_performances:
            for count_key in avg_performance['counts']:
                avg_performance['counts'][count_key] += perf['counts'][count_key]
        
        # Calculate metrics from aggregated counts
        tp = avg_performance['counts']['true_positives']
        fp = avg_performance['counts']['false_positives']
        tn = avg_performance['counts']['true_negatives']
        fn = avg_performance['counts']['false_negatives']
        
        # Calculate derived metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        balanced_accuracy = (sensitivity + specificity) / 2
        youdens_j = sensitivity + specificity - 1
        
        # Calculate likelihood ratios, handling special cases
        
        # Positive likelihood ratio
        if specificity < 1:
            plr = sensitivity / (1 - specificity)
            if plr > 1000:
                plr_display = ">1000"
            else:
                plr_display = f"{plr:.2f}"
        else:
            plr = None
            plr_display = "∞"
        
        # Negative likelihood ratio
        if specificity > 0:
            if sensitivity < 1:
                nlr = (1 - sensitivity) / specificity
                if nlr < 0.001:
                    nlr_display = "<0.001"
                else:
                    nlr_display = f"{nlr:.3f}"
            else:
                nlr = 0
                nlr_display = "0.000"
        else:
            nlr = None
            nlr_display = "∞"
        
        # Store metrics
        avg_performance['metrics'] = {
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'npv': float(npv),
            'accuracy': float(accuracy),
            'f1_score': float(f1_score),
            'balanced_accuracy': float(balanced_accuracy),
            'youdens_j': float(youdens_j),
            'positive_likelihood_ratio': plr,
            'positive_likelihood_ratio_display': plr_display,
            'negative_likelihood_ratio': nlr,
            'negative_likelihood_ratio_display': nlr_display
        }
        
        # Additionally calculate average metrics directly from fold metrics
        # This is useful for metrics that don't aggregate well, especially for bootstrap estimates
        direct_averaged_metrics = {}
        count_metrics = {}
        
        for metric_name in ['sensitivity', 'specificity', 'precision', 'npv', 'accuracy', 'f1_score', 
                            'balanced_accuracy', 'youdens_j']:
            values = [perf['metrics'][metric_name] for perf in fold_performances if metric_name in perf['metrics']]
            if values:
                direct_averaged_metrics[f'{metric_name}_direct_avg'] = float(np.mean(values))
                count_metrics[f'{metric_name}_n'] = len(values)
        
        # For likelihood ratios, only include non-None values
        for metric_name in ['positive_likelihood_ratio', 'negative_likelihood_ratio']:
            values = [perf['metrics'][metric_name] for perf in fold_performances 
                    if metric_name in perf['metrics'] and perf['metrics'][metric_name] is not None]
            if values:
                direct_averaged_metrics[f'{metric_name}_direct_avg'] = float(np.mean(values))
                count_metrics[f'{metric_name}_n'] = len(values)
        
        # Add direct averages to output
        avg_performance['metrics'].update(direct_averaged_metrics)
        avg_performance['metrics'].update(count_metrics)
        
        return avg_performance
    
    def _save_monitoring_thresholds(self, suffix=""):
        """
        Save monitoring thresholds to a JSON file.
        
        Args:
            suffix (str): Optional suffix to add to the filename
        """
        if not hasattr(self, 'monitoring_thresholds'):
            self.logger.error("No monitoring thresholds to save.")
            return
            
        # Create a deep copy to avoid modifying the original
        thresholds_to_save = copy.deepcopy(self.monitoring_thresholds)
        
        # Convert any numpy or pandas objects to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.ndarray, pd.Series)):
                return convert_to_native(obj.tolist())
            elif isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            else:
                return obj
                
        thresholds_to_save = convert_to_native(thresholds_to_save)
        
        # Generate filename
        suffix = f"_{suffix}" if suffix else ""
        filename = self.config.get('thresholds_filename', f'monitoring_thresholds{suffix}.json')
        output_path = os.path.join(self.config['output_dir'], filename)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(thresholds_to_save, f, indent=4)
            
        self.logger.info(f"Saved monitoring thresholds to {output_path}")
    
    def _save_validation_results(self, suffix=""):
        """
        Save validation results to a JSON file.
        
        Args:
            suffix (str): Optional suffix to add to the filename
        """
        if not hasattr(self, 'validation_results'):
            self.logger.error("No validation results to save.")
            return
            
        # Create a deep copy to avoid modifying the original
        results_to_save = copy.deepcopy(self.validation_results)
        
        # Convert any numpy or pandas objects to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.ndarray, pd.Series)):
                return convert_to_native(obj.tolist())
            elif isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            else:
                return obj
                
        results_to_save = convert_to_native(results_to_save)
        
        # Generate filename
        suffix = f"_{suffix}" if suffix else ""
        filename = self.config.get('validation_filename', f'threshold_validation{suffix}.json')
        output_path = os.path.join(self.config['output_dir'], filename)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
            
        self.logger.info(f"Saved validation results to {output_path}")
    
    def identify_optimal_monitoring_thresholds(self):
        """
        Identify and select optimal monitoring thresholds based on validation results.
        
        This method analyzes validation results to select the best performing
        thresholds for each metric and creates a simplified monitoring configuration.
        
        Returns:
            dict: Optimal thresholds configuration
        """
        self.logger.info("Identifying optimal monitoring thresholds based on validation...")
        
        # Check if we have validation results
        if not hasattr(self, 'validation_results'):
            self.logger.error("No validation results available. Run validate_monitoring_thresholds() first.")
            return None
            
        # Initialize optimal thresholds structure
        self.optimal_thresholds = {
            'single_metric': {},
            'combined': {},
            'metadata': {
                'generated_datetime': datetime.datetime.now().isoformat(),
                'optimization_criteria': self.config.get('optimization_criteria', 'balanced_accuracy')
            }
        }
        
        # Get optimization criteria from config
        optimization_criteria = self.config.get('optimization_criteria', 'balanced_accuracy')
        secondary_criteria = self.config.get('secondary_optimization_criteria', 'youdens_j')
        
        # For each category and metric, find the best threshold
        categories = ['absolute_value', 'change_based', 'slope_based', 'component_based']
        
        for category in categories:
            if category not in self.validation_results:
                continue
                
            for metric, metric_data in self.validation_results[category].items():
                if 'thresholds' not in metric_data or 'performance' not in metric_data:
                    continue
                    
                # Get the threshold types and their performance
                threshold_types = list(metric_data['thresholds'].keys())
                
                # Find the best threshold based on the optimization criteria
                best_score = -float('inf')
                best_threshold_type = None
                
                for threshold_type in threshold_types:
                    # Get performance for this threshold
                    perf = metric_data['performance'].get(threshold_type, {})
                    
                    # Skip if no performance data
                    if 'metrics' not in perf:
                        continue
                        
                    # Get optimization score
                    # For cross-validation or bootstrap, use the main metrics
                    # For time-based validation, use the test metrics
                    if 'test' in perf:
                        score = perf['test']['metrics'].get(optimization_criteria, 0)
                        secondary_score = perf['test']['metrics'].get(secondary_criteria, 0)
                    else:
                        score = perf['metrics'].get(optimization_criteria, 0)
                        secondary_score = perf['metrics'].get(secondary_criteria, 0)
                    
                    # Check if this is the best score
                    if score > best_score or (score == best_score and secondary_score > best_secondary_score):
                        best_score = score
                        best_secondary_score = secondary_score
                        best_threshold_type = threshold_type
                
                # If we found a best threshold, store it
                if best_threshold_type is not None:
                    # Get threshold value
                    # For cross-validation or bootstrap, use the mean threshold
                    # For time-based validation, use the training threshold
                    if 'test' in metric_data['performance'].get(best_threshold_type, {}):
                        best_threshold_value = metric_data['thresholds'][best_threshold_type]['value']
                    else:
                        best_threshold_value = metric_data['thresholds'][best_threshold_type]['mean']
                    
                    # Get performance
                    if 'test' in metric_data['performance'].get(best_threshold_type, {}):
                        best_performance = metric_data['performance'][best_threshold_type]['test']
                    else:
                        best_performance = metric_data['performance'][best_threshold_type]
                    
                    # Store the optimal threshold
                    self.optimal_thresholds['single_metric'][metric] = {
                        'threshold_value': best_threshold_value,
                        'threshold_type': best_threshold_type,
                        'performance': best_performance,
                        'category': category
                    }
                    
                    # Determine direction
                    if category in ['change_based', 'slope_based'] or (category == 'component_based' and 'upright_tails' in metric):
                        direction = 'lower'
                    elif category == 'component_based' and 'hanging_tails' in metric:
                        direction = 'higher'
                    else:  # absolute_value
                        direction = 'lower'
                        
                    self.optimal_thresholds['single_metric'][metric]['direction'] = direction
        
        # For combined thresholds, find the best combination
        if 'combined' in self.validation_results:
            for combo_name, combo_data in self.validation_results['combined'].items():
                # Get performance 
                # For time-based validation, use the test performance
                if 'performance' in combo_data and 'test' in combo_data['performance']:
                    combo_performance = combo_data['performance']['test']
                else:
                    combo_performance = combo_data['performance']
                
                # Get threshold values
                # For time-based validation, use the training thresholds
                if 'threshold_values' in combo_data and isinstance(combo_data['threshold_values'], dict):
                    combo_thresholds = combo_data['threshold_values']
                else:
                    combo_thresholds = {}
                    for metric in combo_data.get('component_metrics', []):
                        if metric in combo_data.get('threshold_values', {}):
                            if isinstance(combo_data['threshold_values'][metric], dict) and 'mean' in combo_data['threshold_values'][metric]:
                                combo_thresholds[metric] = combo_data['threshold_values'][metric]['mean']
                            else:
                                combo_thresholds[metric] = combo_data['threshold_values'][metric]
                
                # Store the optimal combined threshold
                self.optimal_thresholds['combined'][combo_name] = {
                    'component_metrics': combo_data.get('component_metrics', []),
                    'threshold_values': combo_thresholds,
                    'performance': combo_performance
                }
        
        # Identify the overall best monitoring approach
        single_metrics = list(self.optimal_thresholds['single_metric'].keys())
        combined_metrics = list(self.optimal_thresholds['combined'].keys())
        
        best_single_score = -float('inf')
        best_single_metric = None
        
        for metric in single_metrics:
            perf = self.optimal_thresholds['single_metric'][metric]['performance']
            if isinstance(perf, dict) and 'metrics' in perf:
                score = perf['metrics'].get(optimization_criteria, 0)
                if score > best_single_score:
                    best_single_score = score
                    best_single_metric = metric
        
        best_combined_score = -float('inf')
        best_combined_metric = None
        
        for combo in combined_metrics:
            perf = self.optimal_thresholds['combined'][combo]['performance']
            if isinstance(perf, dict) and 'metrics' in perf:
                score = perf['metrics'].get(optimization_criteria, 0)
                if score > best_combined_score:
                    best_combined_score = score
                    best_combined_metric = combo
        
        # Determine the overall best approach
        if best_single_score >= best_combined_score:
            best_approach = {
                'type': 'single_metric',
                'metric': best_single_metric,
                'score': best_single_score
            }
        else:
            best_approach = {
                'type': 'combined',
                'metric': best_combined_metric,
                'score': best_combined_score
            }
        
        self.optimal_thresholds['best_approach'] = best_approach
        
        # Save optimal thresholds
        self._save_optimal_thresholds()
        
        return self.optimal_thresholds
    
    def _save_optimal_thresholds(self):
        """Save optimal thresholds to a JSON file."""
        if not hasattr(self, 'optimal_thresholds'):
            self.logger.error("No optimal thresholds to save.")
            return
            
        # Create a deep copy to avoid modifying the original
        thresholds_to_save = copy.deepcopy(self.optimal_thresholds)
        
        # Convert any numpy or pandas objects to native Python types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(i) for i in obj]
            elif isinstance(obj, (np.ndarray, pd.Series)):
                return convert_to_native(obj.tolist())
            elif isinstance(obj, (np.integer, np.floating)):
                return int(obj) if isinstance(obj, np.integer) else float(obj)
            else:
                return obj
                
        thresholds_to_save = convert_to_native(thresholds_to_save)
        
        # Generate filename
        filename = self.config.get('optimal_thresholds_filename', 'optimal_monitoring_thresholds.json')
        output_path = os.path.join(self.config['output_dir'], filename)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(thresholds_to_save, f, indent=4)
            
        self.logger.info(f"Saved optimal monitoring thresholds to {output_path}")
        
    def apply_monitoring_thresholds(self, data=None):
        """
        Apply monitoring thresholds to new data or existing data.
        
        Args:
            data (DataFrame, optional): New data to evaluate. If None, uses existing data.
            
        Returns:
            dict: Results of applying thresholds
        """
        self.logger.info("Applying monitoring thresholds to data...")
        
        # Check if we have optimal thresholds
        if not hasattr(self, 'optimal_thresholds'):
            self.logger.error("No optimal thresholds available. Run identify_optimal_monitoring_thresholds() first.")
            return None
        
        # If no data provided, use existing data
        if data is None:
            # Use both outbreak and control data for demonstration
            if hasattr(self, 'pre_outbreak_stats') and not self.pre_outbreak_stats.empty:
                outbreak_data = self.pre_outbreak_stats.copy()
                outbreak_data['is_outbreak'] = True
            else:
                outbreak_data = pd.DataFrame()
                
            if hasattr(self, 'control_stats') and not self.control_stats.empty:
                control_data = self.control_stats.copy()
                control_data['is_outbreak'] = False
                
                # Rename value_at_reference to value_at_removal for consistency
                if 'value_at_reference' in control_data.columns and 'value_at_removal' not in control_data.columns:
                    control_data['value_at_removal'] = control_data['value_at_reference']
            else:
                control_data = pd.DataFrame()
                
            # Combine data
            if not outbreak_data.empty and not control_data.empty:
                # Get common columns
                common_cols = list(set(outbreak_data.columns) & set(control_data.columns))
                data = pd.concat([outbreak_data[common_cols], control_data[common_cols]], ignore_index=True)
            elif not outbreak_data.empty:
                data = outbreak_data
            elif not control_data.empty:
                data = control_data
            else:
                self.logger.error("No data available to apply thresholds.")
                return None
        
        # Initialize results
        monitoring_results = {
            'single_metric': {},
            'combined': {},
            'best_approach': {},
            'metadata': {
                'applied_datetime': datetime.datetime.now().isoformat(),
                'n_samples': len(data)
            }
        }
        
        # Apply single metric thresholds
        for metric, metric_info in self.optimal_thresholds['single_metric'].items():
            # Skip if metric not in data
            if metric not in data.columns:
                self.logger.warning(f"Metric {metric} not found in data. Skipping.")
                continue
                
            # Extract threshold value - handle both float and dict cases
            threshold_value = metric_info['threshold_value']
            if isinstance(threshold_value, dict):
                # If threshold_value is a dict, extract actual value
                if 'mean' in threshold_value:
                    threshold_value = threshold_value['mean']
                elif 'value' in threshold_value:
                    threshold_value = threshold_value['value']
                else:
                    # Use first value found if standard keys not present
                    for key, val in threshold_value.items():
                        if isinstance(val, (int, float)):
                            threshold_value = val
                            break
                    else:
                        self.logger.warning(f"Could not extract a numeric threshold value for {metric}. Skipping.")
                        continue
            
            # Ensure it's now a number
            if not isinstance(threshold_value, (int, float)):
                self.logger.warning(f"Threshold value for {metric} is not numeric ({type(threshold_value)}). Skipping.")
                continue
                
            direction = metric_info['direction']
            
            # Apply threshold
            if direction == 'lower':
                data[f'{metric}_alert'] = data[metric] <= threshold_value
            else:  # direction == 'higher'
                data[f'{metric}_alert'] = data[metric] >= threshold_value
                
            # Calculate alert statistics
            alerts = data[f'{metric}_alert'].sum()
            alert_rate = alerts / len(data)
            
            # Calculate accuracy if we know the true status
            if 'is_outbreak' in data.columns:
                true_positives = ((data[f'{metric}_alert'] == True) & (data['is_outbreak'] == True)).sum()
                false_positives = ((data[f'{metric}_alert'] == True) & (data['is_outbreak'] == False)).sum()
                true_negatives = ((data[f'{metric}_alert'] == False) & (data['is_outbreak'] == False)).sum()
                false_negatives = ((data[f'{metric}_alert'] == False) & (data['is_outbreak'] == True)).sum()
                
                sensitivity = true_positives / data[data['is_outbreak'] == True].shape[0] if data[data['is_outbreak'] == True].shape[0] > 0 else 0
                specificity = true_negatives / data[data['is_outbreak'] == False].shape[0] if data[data['is_outbreak'] == False].shape[0] > 0 else 0
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                accuracy = (true_positives + true_negatives) / len(data)
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                
                performance = {
                    'counts': {
                        'true_positives': int(true_positives),
                        'false_positives': int(false_positives),
                        'true_negatives': int(true_negatives),
                        'false_negatives': int(false_negatives)
                    },
                    'metrics': {
                        'sensitivity': float(sensitivity),
                        'specificity': float(specificity),
                        'precision': float(precision),
                        'accuracy': float(accuracy),
                        'f1_score': float(f1_score)
                    }
                }
            else:
                performance = None
            
            # Store results
            monitoring_results['single_metric'][metric] = {
                'threshold_value': float(threshold_value),  # Ensure it's a float for JSON serialization
                'direction': direction,
                'alerts': int(alerts),
                'alert_rate': float(alert_rate),
                'performance': performance
            }
        
        # Apply combined thresholds
        for combo_name, combo_info in self.optimal_thresholds['combined'].items():
            # Get component metrics and threshold values
            component_metrics = combo_info['component_metrics']
            threshold_values = combo_info['threshold_values']
            
            # Check if all required metrics are in the data
            all_metrics_present = True
            for metric, _ in component_metrics:
                if metric not in data.columns:
                    self.logger.warning(f"Metric {metric} not found in data. Skipping combo {combo_name}.")
                    all_metrics_present = False
                    break
            
            if not all_metrics_present:
                continue
            
            # Apply combined threshold
            alert_flags = []
            for metric, _ in component_metrics:
                threshold = threshold_values[metric]
                
                # Handle case where threshold is a dict
                if isinstance(threshold, dict):
                    if 'mean' in threshold:
                        threshold = threshold['mean']
                    elif 'value' in threshold:
                        threshold = threshold['value']
                    else:
                        # Use first value found if standard keys not present
                        for key, val in threshold.items():
                            if isinstance(val, (int, float)):
                                threshold = val
                                break
                        else:
                            self.logger.warning(f"Could not extract a numeric threshold value for {metric} in combo {combo_name}. Skipping.")
                            continue
                
                # Ensure it's now a number
                if not isinstance(threshold, (int, float)):
                    self.logger.warning(f"Threshold value for {metric} in combo {combo_name} is not numeric ({type(threshold)}). Skipping.")
                    continue
                
                # Determine direction based on metric
                if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                    direction = 'lower'
                    alert = data[metric] <= threshold
                elif metric.startswith('upright_tails'):
                    direction = 'lower'
                    alert = data[metric] <= threshold
                elif metric.startswith('hanging_tails'):
                    direction = 'higher'
                    alert = data[metric] >= threshold
                else:
                    direction = 'lower'
                    alert = data[metric] <= threshold
                
                alert_flags.append(alert)
            
            # Skip if no alert flags were created
            if not alert_flags:
                continue
                
            # Combined alert requires all individual alerts to be True
            data[f'{combo_name}_alert'] = alert_flags[0]
            for flag in alert_flags[1:]:
                data[f'{combo_name}_alert'] &= flag
                
            # Calculate alert statistics
            alerts = data[f'{combo_name}_alert'].sum()
            alert_rate = alerts / len(data)
            
            # Calculate accuracy if we know the true status
            if 'is_outbreak' in data.columns:
                true_positives = ((data[f'{combo_name}_alert'] == True) & (data['is_outbreak'] == True)).sum()
                false_positives = ((data[f'{combo_name}_alert'] == True) & (data['is_outbreak'] == False)).sum()
                true_negatives = ((data[f'{combo_name}_alert'] == False) & (data['is_outbreak'] == False)).sum()
                false_negatives = ((data[f'{combo_name}_alert'] == False) & (data['is_outbreak'] == True)).sum()
                
                sensitivity = true_positives / data[data['is_outbreak'] == True].shape[0] if data[data['is_outbreak'] == True].shape[0] > 0 else 0
                specificity = true_negatives / data[data['is_outbreak'] == False].shape[0] if data[data['is_outbreak'] == False].shape[0] > 0 else 0
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                accuracy = (true_positives + true_negatives) / len(data)
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                
                performance = {
                    'counts': {
                        'true_positives': int(true_positives),
                        'false_positives': int(false_positives),
                        'true_negatives': int(true_negatives),
                        'false_negatives': int(false_negatives)
                    },
                    'metrics': {
                        'sensitivity': float(sensitivity),
                        'specificity': float(specificity),
                        'precision': float(precision),
                        'accuracy': float(accuracy),
                        'f1_score': float(f1_score)
                    }
                }
            else:
                performance = None
            
            # Store results with cleaned up threshold values
            clean_threshold_values = {}
            for metric, threshold in threshold_values.items():
                if isinstance(threshold, dict):
                    if 'mean' in threshold:
                        clean_threshold_values[metric] = float(threshold['mean'])
                    elif 'value' in threshold:
                        clean_threshold_values[metric] = float(threshold['value'])
                    else:
                        # Try to get any numeric value
                        for key, val in threshold.items():
                            if isinstance(val, (int, float)):
                                clean_threshold_values[metric] = float(val)
                                break
                        else:
                            clean_threshold_values[metric] = "unknown"
                else:
                    clean_threshold_values[metric] = float(threshold)
            
            monitoring_results['combined'][combo_name] = {
                'component_metrics': component_metrics,
                'threshold_values': clean_threshold_values,
                'alerts': int(alerts),
                'alert_rate': float(alert_rate),
                'performance': performance
            }
        
        # Apply best approach
        best_approach = self.optimal_thresholds['best_approach']
        if best_approach['type'] == 'single_metric' and best_approach['metric'] in monitoring_results['single_metric']:
            monitoring_results['best_approach'] = {
                'type': 'single_metric',
                'metric': best_approach['metric'],
                'results': monitoring_results['single_metric'][best_approach['metric']]
            }
        elif best_approach['type'] == 'combined' and best_approach['metric'] in monitoring_results['combined']:
            monitoring_results['best_approach'] = {
                'type': 'combined',
                'metric': best_approach['metric'],
                'results': monitoring_results['combined'][best_approach['metric']]
            }
        
        # Store the alert flags
        data_with_alerts = data.copy()
        monitoring_results['data_with_alerts'] = data_with_alerts
        
        return monitoring_results
    
    def generate_threshold_summary_report(self, output_file=None):
        """
        Generate a comprehensive report of threshold performance with statistical summaries.
        
        Args:
            output_file (str, optional): Path to save the report. If None, report is returned as a string.
            
        Returns:
            str: Report text if output_file is None, otherwise None
        """
        if not hasattr(self, 'validation_results'):
            self.logger.error("No validation results available. Run validate_monitoring_thresholds() first.")
            return None
        
        # Initialize report
        report = []
        report.append("# Tail Biting Monitoring Threshold Summary Report")
        report.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Add validation method information
        validation_method = self.validation_results.get('metadata', {}).get('validation_method', 'unknown')
        report.append(f"## Validation Method: {validation_method.upper()}")
        
        if validation_method == 'cv':
            n_folds = self.config.get('validation_cv_folds', 5)
            report.append(f"Cross-validation with {n_folds} folds using group-based splitting")
        elif validation_method == 'loocv':
            report.append(f"Leave-one-group-out cross-validation")
        elif validation_method == 'bootstrap':
            n_iterations = self.config.get('validation_bootstrap_iterations', 100)
            report.append(f"Bootstrap validation with {n_iterations} iterations and out-of-bag evaluation")
        elif validation_method == 'time_based':
            split_date = self.validation_results.get('time_based', {}).get('metadata', {}).get('split_date', 'unknown')
            report.append(f"Time-based validation with split date: {split_date}")
        
        report.append("\n## Dataset Summary")
        
        # Add dataset information
        n_outbreaks = len(self.pre_outbreak_stats['pen'].unique()) if hasattr(self, 'pre_outbreak_stats') else 0
        n_controls = len(self.control_stats['pen'].unique()) if hasattr(self, 'control_stats') else 0
        
        report.append(f"- Outbreak pens: {n_outbreaks}")
        report.append(f"- Control pens: {n_controls}")
        
        # Add information about validation performance
        report.append("\n## Single Metric Thresholds\n")
        
        # Report for each category
        for category in ['absolute_value', 'change_based', 'slope_based', 'component_based']:
            if category not in self.validation_results:
                continue
                
            category_name = {
                'absolute_value': 'Absolute Value Thresholds',
                'change_based': 'Change-Based Thresholds',
                'slope_based': 'Slope-Based Thresholds',
                'component_based': 'Component-Based Thresholds'
            }.get(category, category)
            
            report.append(f"### {category_name}\n")
            
            # Report on each metric
            for metric, metric_data in self.validation_results[category].items():
                report.append(f"#### {metric}")
                
                # Display threshold statistics
                if 'thresholds' in metric_data:
                    report.append("\n**Threshold Values:**\n")
                    report.append("| Method | Mean | Std Dev | Min | Max | 95% CI |")
                    report.append("|--------|------|---------|-----|-----|--------|")
                    
                    for threshold_type, threshold_stats in metric_data['thresholds'].items():
                        if isinstance(threshold_stats, dict) and 'mean' in threshold_stats:
                            mean = threshold_stats.get('mean', 'N/A')
                            std = threshold_stats.get('std', 'N/A')
                            min_val = threshold_stats.get('min', 'N/A')
                            max_val = threshold_stats.get('max', 'N/A')
                            
                            # CI might not be available for all validation methods
                            if 'ci_lower' in threshold_stats and 'ci_upper' in threshold_stats:
                                ci = f"({threshold_stats['ci_lower']:.3f}, {threshold_stats['ci_upper']:.3f})"
                            else:
                                ci = "N/A"
                                
                            report.append(f"| {threshold_type} | {mean:.3f} | {std:.3f} | {min_val:.3f} | {max_val:.3f} | {ci} |")
                
                # Display performance metrics
                if 'performance' in metric_data:
                    report.append("\n**Performance Metrics:**\n")
                    
                    # Handle different validation methods differently
                    if validation_method == 'time_based' and isinstance(metric_data['performance'], dict):
                        # For time-based validation, we have train/test performance
                        report.append("| Metric | Training | Testing | Δ (Test-Train) |")
                        report.append("|--------|----------|---------|---------------|")
                        
                        for threshold_type, perf_data in metric_data['performance'].items():
                            if 'train' in perf_data and 'test' in perf_data:
                                train_sens = perf_data['train']['metrics'].get('sensitivity', 'N/A')
                                train_spec = perf_data['train']['metrics'].get('specificity', 'N/A')
                                train_bacc = perf_data['train']['metrics'].get('balanced_accuracy', 'N/A')
                                train_f1 = perf_data['train']['metrics'].get('f1_score', 'N/A')
                                
                                test_sens = perf_data['test']['metrics'].get('sensitivity', 'N/A')
                                test_spec = perf_data['test']['metrics'].get('specificity', 'N/A')
                                test_bacc = perf_data['test']['metrics'].get('balanced_accuracy', 'N/A')
                                test_f1 = perf_data['test']['metrics'].get('f1_score', 'N/A')
                                
                                delta_sens = perf_data['delta'].get('sensitivity', 'N/A')
                                delta_spec = perf_data['delta'].get('specificity', 'N/A')
                                delta_bacc = perf_data['delta'].get('balanced_accuracy', 'N/A')
                                delta_f1 = perf_data['delta'].get('f1_score', 'N/A')
                                
                                report.append(f"**{threshold_type}**")
                                report.append(f"| Sensitivity | {train_sens:.3f} | {test_sens:.3f} | {delta_sens:+.3f} |")
                                report.append(f"| Specificity | {train_spec:.3f} | {test_spec:.3f} | {delta_spec:+.3f} |")
                                report.append(f"| Balanced Accuracy | {train_bacc:.3f} | {test_bacc:.3f} | {delta_bacc:+.3f} |")
                                report.append(f"| F1 Score | {train_f1:.3f} | {test_f1:.3f} | {delta_f1:+.3f} |")
                                report.append("")
                    else:
                        # For other validation methods, we have performance by threshold type
                        report.append("| Method | Sensitivity | Specificity | Precision | Balanced Acc | F1 Score | PLR | NLR |")
                        report.append("|--------|-------------|------------|-----------|--------------|----------|-----|-----|")
                        
                        for threshold_type, perf_data in metric_data['performance'].items():
                            metrics = perf_data.get('metrics', {})
                            sens = metrics.get('sensitivity', 'N/A')
                            spec = metrics.get('specificity', 'N/A')
                            prec = metrics.get('precision', 'N/A')
                            bacc = metrics.get('balanced_accuracy', 'N/A')
                            f1 = metrics.get('f1_score', 'N/A')
                            
                            # Handle likelihood ratios specially
                            plr_display = metrics.get('positive_likelihood_ratio_display', 'N/A')
                            nlr_display = metrics.get('negative_likelihood_ratio_display', 'N/A')
                            
                            # If display versions aren't available, format the raw values
                            if plr_display == 'N/A' and 'positive_likelihood_ratio' in metrics:
                                plr = metrics['positive_likelihood_ratio']
                                if plr is None:
                                    plr_display = "∞"
                                elif plr > 1000:
                                    plr_display = ">1000"
                                else:
                                    plr_display = f"{plr:.2f}"
                                    
                            if nlr_display == 'N/A' and 'negative_likelihood_ratio' in metrics:
                                nlr = metrics['negative_likelihood_ratio']
                                if nlr is None:
                                    nlr_display = "∞"
                                elif nlr < 0.001:
                                    nlr_display = "<0.001"
                                else:
                                    nlr_display = f"{nlr:.3f}"
                            
                            # Add confidence intervals if available
                            ci_info = ""
                            if validation_method == 'bootstrap':
                                sens_ci_lower = metrics.get('sensitivity_ci_lower', None)
                                sens_ci_upper = metrics.get('sensitivity_ci_upper', None)
                                
                                if sens_ci_lower is not None and sens_ci_upper is not None:
                                    ci_info = f"\n**95% CIs:** Sens ({sens_ci_lower:.3f}, {sens_ci_upper:.3f})"
                                    
                                spec_ci_lower = metrics.get('specificity_ci_lower', None)
                                spec_ci_upper = metrics.get('specificity_ci_upper', None)
                                
                                if spec_ci_lower is not None and spec_ci_upper is not None:
                                    ci_info += f", Spec ({spec_ci_lower:.3f}, {spec_ci_upper:.3f})"
                            
                            report.append(f"| {threshold_type} | {sens:.3f} | {spec:.3f} | {prec:.3f} | {bacc:.3f} | {f1:.3f} | {plr_display} | {nlr_display} |")
                            
                            if ci_info:
                                report.append(ci_info)
                
                report.append("\n")
        
        # Report on combined thresholds
        if 'combined' in self.validation_results:
            report.append("## Combined Multi-Factor Thresholds\n")
            
            for combo_name, combo_data in self.validation_results['combined'].items():
                report.append(f"### {combo_name}")
                
                # Show component metrics
                if 'component_metrics' in combo_data:
                    report.append("\n**Component Metrics:**")
                    for metric, threshold_type in combo_data['component_metrics']:
                        report.append(f"- {metric} ({threshold_type})")
                
                # Display threshold statistics
                if 'threshold_values' in combo_data:
                    report.append("\n**Threshold Values:**\n")
                    report.append("| Metric | Mean | Std Dev | Min | Max | 95% CI |")
                    report.append("|--------|------|---------|-----|-----|--------|")
                    
                    for metric, threshold_stats in combo_data['threshold_values'].items():
                        if isinstance(threshold_stats, dict) and 'mean' in threshold_stats:
                            mean = threshold_stats.get('mean', 'N/A')
                            std = threshold_stats.get('std', 'N/A')
                            min_val = threshold_stats.get('min', 'N/A')
                            max_val = threshold_stats.get('max', 'N/A')
                            
                            # CI might not be available for all validation methods
                            if 'ci_lower' in threshold_stats and 'ci_upper' in threshold_stats:
                                ci = f"({threshold_stats['ci_lower']:.3f}, {threshold_stats['ci_upper']:.3f})"
                            else:
                                ci = "N/A"
                                
                            report.append(f"| {metric} | {mean:.3f} | {std:.3f} | {min_val:.3f} | {max_val:.3f} | {ci} |")
                
                # Display performance metrics
                if 'performance' in combo_data:
                    report.append("\n**Performance Metrics:**\n")
                    
                    # Handle different validation methods differently
                    if validation_method == 'time_based' and 'train' in combo_data['performance'] and 'test' in combo_data['performance']:
                        # For time-based validation, we have train/test performance
                        report.append("| Metric | Training | Testing | Δ (Test-Train) |")
                        report.append("|--------|----------|---------|---------------|")
                        
                        train_perf = combo_data['performance']['train']['metrics']
                        test_perf = combo_data['performance']['test']['metrics']
                        delta = combo_data['performance']['delta']
                        
                        metrics_to_display = [
                            ('Sensitivity', 'sensitivity'),
                            ('Specificity', 'specificity'),
                            ('Precision', 'precision'),
                            ('NPV', 'npv'),
                            ('Accuracy', 'accuracy'),
                            ('F1 Score', 'f1_score'),
                            ('Balanced Accuracy', 'balanced_accuracy')
                        ]
                        
                        for display_name, metric_key in metrics_to_display:
                            train_val = train_perf.get(metric_key, 'N/A')
                            test_val = test_perf.get(metric_key, 'N/A')
                            delta_val = delta.get(metric_key, 'N/A')
                            
                            if isinstance(train_val, (int, float)) and isinstance(test_val, (int, float)) and isinstance(delta_val, (int, float)):
                                report.append(f"| {display_name} | {train_val:.3f} | {test_val:.3f} | {delta_val:+.3f} |")
                            else:
                                report.append(f"| {display_name} | {train_val} | {test_val} | {delta_val} |")
                    else:
                        # For other validation methods
                        metrics = combo_data['performance'].get('metrics', {})
                        
                        report.append("| Metric | Value | 95% CI |")
                        report.append("|--------|-------|--------|")
                        
                        metrics_to_display = [
                            ('Sensitivity', 'sensitivity'),
                            ('Specificity', 'specificity'),
                            ('Precision', 'precision'),
                            ('NPV', 'npv'),
                            ('Accuracy', 'accuracy'),
                            ('F1 Score', 'f1_score'),
                            ('Balanced Accuracy', 'balanced_accuracy'),
                            ('Positive LR', 'positive_likelihood_ratio'),
                            ('Negative LR', 'negative_likelihood_ratio')
                        ]
                        
                        for display_name, metric_key in metrics_to_display:
                            value = metrics.get(metric_key, 'N/A')
                            
                            # Special handling for likelihood ratios
                            if metric_key == 'positive_likelihood_ratio' or metric_key == 'negative_likelihood_ratio':
                                display_key = f"{metric_key}_display"
                                if display_key in metrics:
                                    value = metrics[display_key]
                                elif value is None:
                                    value = "∞"
                                elif metric_key == 'positive_likelihood_ratio' and value > 1000:
                                    value = ">1000"
                                elif metric_key == 'negative_likelihood_ratio' and value < 0.001:
                                    value = "<0.001"
                            
                            # Format value
                            if isinstance(value, (int, float)) and value is not None:
                                formatted_value = f"{value:.3f}"
                            else:
                                formatted_value = str(value)
                            
                            # Add confidence interval if available
                            ci_lower_key = f"{metric_key}_ci_lower"
                            ci_upper_key = f"{metric_key}_ci_upper"
                            
                            if ci_lower_key in metrics and ci_upper_key in metrics:
                                ci_lower = metrics[ci_lower_key]
                                ci_upper = metrics[ci_upper_key]
                                ci = f"({ci_lower:.3f}, {ci_upper:.3f})"
                            else:
                                ci = "N/A"
                            
                            report.append(f"| {display_name} | {formatted_value} | {ci} |")
                
                report.append("\n")
        
        # Add summary of best thresholds if available
        if hasattr(self, 'optimal_thresholds'):
            report.append("## Optimal Monitoring Thresholds\n")
            
            if 'best_approach' in self.optimal_thresholds:
                best = self.optimal_thresholds['best_approach']
                best_type = best.get('type', 'unknown')
                best_metric = best.get('metric', 'unknown')
                best_score = best.get('score', 'N/A')
                
                report.append(f"**Best overall approach:** {best_type}, {best_metric} (score: {best_score:.3f})\n")
                
                if best_type == 'single_metric' and best_metric in self.optimal_thresholds.get('single_metric', {}):
                    metric_info = self.optimal_thresholds['single_metric'][best_metric]
                    threshold = metric_info.get('threshold_value', 'N/A')
                    direction = metric_info.get('direction', 'lower')
                    direction_text = "below" if direction == "lower" else "above"
                    
                    report.append(f"Monitor {best_metric} for values {direction_text} {threshold:.3f}\n")
                    
                    # Add performance metrics
                    if 'performance' in metric_info:
                        perf = metric_info['performance'].get('metrics', {})
                        sens = perf.get('sensitivity', 'N/A')
                        spec = perf.get('specificity', 'N/A')
                        bacc = perf.get('balanced_accuracy', 'N/A')
                        
                        report.append(f"**Performance:** Sensitivity = {sens:.3f}, Specificity = {spec:.3f}, Balanced Accuracy = {bacc:.3f}")
                
                elif best_type == 'combined' and best_metric in self.optimal_thresholds.get('combined', {}):
                    combo_info = self.optimal_thresholds['combined'][best_metric]
                    component_metrics = combo_info.get('component_metrics', [])
                    threshold_values = combo_info.get('threshold_values', {})
                    
                    report.append("**Combined threshold components:**\n")
                    
                    for metric, threshold_type in component_metrics:
                        threshold = threshold_values.get(metric, 'N/A')
                        
                        # Determine direction based on metric
                        if metric in ['3d_window_slope', '7d_window_slope', 'abs_change_1d', 'abs_change_3d', 'abs_change_7d']:
                            direction = "below"
                        elif metric.startswith('upright_tails'):
                            direction = "below"
                        elif metric.startswith('hanging_tails'):
                            direction = "above"
                        else:
                            direction = "below"
                        
                        if isinstance(threshold, (int, float)):
                            report.append(f"- {metric}: values {direction} {threshold:.3f}")
                        else:
                            report.append(f"- {metric}: values {direction} {threshold}")
                    
                    # Add performance metrics
                    if 'performance' in combo_info:
                        perf = combo_info['performance'].get('metrics', {})
                        sens = perf.get('sensitivity', 'N/A')
                        spec = perf.get('specificity', 'N/A')
                        bacc = perf.get('balanced_accuracy', 'N/A')
                        
                        report.append(f"\n**Performance:** Sensitivity = {sens:.3f}, Specificity = {spec:.3f}, Balanced Accuracy = {bacc:.3f}")
        
        # Join all parts of the report
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Threshold summary report saved to {output_file}")
            return None
        
        return report_text