# processing.py
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import timedelta

from evaluation.tail_posture_analysis.core import TailPostureAnalysisBase
from pipeline.utils.data_analysis_utils import (sorting_key, load_monitoring_pipeline_results, get_pen_info)

class DataProcessor(TailPostureAnalysisBase):
    """Handles data loading and preprocessing for tail posture analysis."""
    
    def load_data(self):
        """Load monitoring pipeline results including missing data info."""
        self.logger.info("Loading monitoring pipeline data...")
        start_time = time.time()
        self.monitoring_results = load_monitoring_pipeline_results(
            self.path_manager.path_to_monitoring_pipeline_outputs,
            self.path_manager.path_to_config_files,
            compartment="piglet_rearing",
            as_df_list=True
        )
        if not self.monitoring_results:
             self.logger.error("Failed to load any monitoring results. Check paths and data.")
             return None
        try:
             self.monitoring_results.sort(key=sorting_key)
        except KeyError as e:
             self.logger.error(f"Failed to sort monitoring results, missing key: {e}")
        self.logger.info(f"Loaded {len(self.monitoring_results)} monitoring results in {time.time() - start_time:.2f} seconds")
        cameras = set(result.get('camera', 'Unknown') for result in self.monitoring_results)
        datespans = set(result.get('date_span', 'Unknown') for result in self.monitoring_results)
        total_missing_days = sum(len(result.get('missing_dates', [])) for result in self.monitoring_results)
        total_expected_days = sum(result.get('total_expected_days', 0) for result in self.monitoring_results)
        pct_missing_overall = (total_missing_days / total_expected_days * 100) if total_expected_days > 0 else 0
        self.logger.info(f"Dataset contains {len(cameras)} cameras and {len(datespans)} datespans.")
        self.logger.info(f"Total expected days across all datespans: {total_expected_days}")
        self.logger.info(f"Total missing daily files detected: {total_missing_days} ({pct_missing_overall:.2f}%)")
        return self.monitoring_results

        
    def preprocess_data(self, result):
        """Preprocess a single result dataset."""
        camera = result.get('camera', 'unknown')
        date_span = result.get('date_span', 'unknown')
        self.logger.debug(f"Preprocessing data for {camera} - {date_span}")
        quality_metrics = {
            'initial_row_count': 0, 'rows_after_time_filter': 0, 'rows_after_concat': 0,
            'missing_days_detected': len(result.get('missing_dates', [])),
            'total_expected_days': result.get('total_expected_days', 0),
            'percent_missing_rows_raw': None, 'percent_missing_resampled': None,
            'max_consecutive_missing_resampled': 0 }
        try:
            if not result.get('dataframes') or len(result['dataframes']) == 0:
                self.logger.warning(f"No dataframes found for {camera} - {date_span}")
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': pd.DataFrame(), 'resampled_data': pd.DataFrame(),
                        'smoothed_data': pd.DataFrame(), 'interpolated_data': pd.DataFrame() }
            valid_data_frames = [df for df in result['dataframes'] if df is not None and not df.empty]
            if not valid_data_frames:
                self.logger.warning(f"All dataframes are empty or invalid for {camera} - {date_span}")
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': pd.DataFrame(), 'resampled_data': pd.DataFrame(),
                        'smoothed_data': pd.DataFrame(), 'interpolated_data': pd.DataFrame() }
            quality_metrics['initial_row_count'] = sum(len(df) for df in valid_data_frames)
            data_all = pd.concat(valid_data_frames, ignore_index=True)
            quality_metrics['rows_after_concat'] = len(data_all)
            if data_all.empty:
                self.logger.warning(f"Concatenated data is empty for {camera} - {date_span}")
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': pd.DataFrame(), 'resampled_data': pd.DataFrame(),
                        'smoothed_data': pd.DataFrame(), 'interpolated_data': pd.DataFrame() }
            data_processed = data_all.copy()
            if "datetime" not in data_processed.columns:
                self.logger.error(f"No datetime column found after concat for {camera} - {date_span}")
                return None
            data_processed["datetime"] = pd.to_datetime(data_processed["datetime"])
            data_processed.set_index("datetime", inplace=True)
            data_processed.sort_index(inplace=True)
            required_cols = ["num_tails_hanging", "num_tails_upright", "num_tail_detections"]
            missing_cols = [col for col in required_cols if col not in data_processed.columns]
            if missing_cols:
                self.logger.warning(f"Adding missing columns {missing_cols} with NaN for {camera} - {date_span}")
                for col in missing_cols: data_processed[col] = np.nan
            seconds_per_day_expected = (16.5 - 8) * 3600
            num_loaded_days = len(result.get('found_dates', []))
            total_expected_seconds = num_loaded_days * seconds_per_day_expected
            actual_seconds_present = len(data_processed.index.drop_duplicates())
            missing_seconds = total_expected_seconds - actual_seconds_present
            quality_metrics['percent_missing_rows_raw'] = (missing_seconds / total_expected_seconds * 100) if total_expected_seconds > 0 else 0.0
            self.logger.debug(f"{camera} - {date_span}: Raw data missing approx {quality_metrics['percent_missing_rows_raw']:.2f}% of seconds within loaded days.")
            if self.config['normalize']:
                if "num_tail_detections" in data_processed.columns:
                    denominator = data_processed["num_tail_detections"].replace(0, np.nan)
                    data_processed["num_tails_hanging"] = data_processed["num_tails_hanging"] / denominator
                    data_processed["num_tails_upright"] = data_processed["num_tails_upright"] / denominator
                else:
                    self.logger.error(f"Cannot normalize: 'num_tail_detections' missing for {camera} - {date_span}.")
            if len(data_processed) < 2:
                self.logger.warning(f"Dataset too small for resampling ({len(data_processed)}) for {camera} - {date_span}")
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': data_processed, 'resampled_data': data_processed.assign(posture_diff=np.nan),
                        'smoothed_data': data_processed.assign(posture_diff=np.nan), 'interpolated_data': data_processed.assign(posture_diff=np.nan) }
            try:
                resampled_data = data_processed.resample(self.config['resample_freq']).mean()
                resampled_data['posture_diff'] = resampled_data['num_tails_upright'] - resampled_data['num_tails_hanging']
                expected_resampled_periods = 0
                if not resampled_data.empty:
                    min_date = pd.to_datetime(result['date_span'].split('_')[0], format='%y%m%d')
                    max_date = pd.to_datetime(result['date_span'].split('_')[1], format='%y%m%d')
                    full_range = pd.date_range(start=min_date, end=max_date, freq=self.config['resample_freq'])
                    expected_resampled_periods = len(full_range)
                    nan_periods = resampled_data['posture_diff'].isnull().sum()
                    quality_metrics['percent_missing_resampled'] = (nan_periods / expected_resampled_periods * 100) if expected_resampled_periods > 0 else 0.0
                    quality_metrics['max_consecutive_missing_resampled'] = self._calculate_max_consecutive_nans(resampled_data['posture_diff'])
                    
                    # log message about missing data
                    self.logger.debug(
                        f"{camera} - {date_span}: "
                        f"Raw data: {len(result.get('missing_dates', []))} calendar days missing. "
                        f"After resampling: {quality_metrics['percent_missing_resampled']:.2f}% of periods missing with "
                        f"{quality_metrics['max_consecutive_missing_resampled']} max consecutive missing periods."
                    )
                smoothed_data = self._apply_smoothing(resampled_data)
                smoothed_data['posture_diff_change'] = smoothed_data['posture_diff'].diff()
                smoothed_data['posture_diff_pct_change'] = smoothed_data['posture_diff'].pct_change().replace([np.inf, -np.inf], np.nan) * 100
                for window in [3, 5, 7]:
                    adjusted_window = min(window, smoothed_data['posture_diff'].count())
                    if adjusted_window < 2: continue
                    center_roll = smoothed_data['posture_diff'].rolling(window=adjusted_window, center=True, min_periods=1)
                    smoothed_data[f'posture_diff_{window}d_mean'] = center_roll.mean()
                    if adjusted_window >= 2:
                        smoothed_data[f'posture_diff_{window}d_std'] = center_roll.std()
                        smoothed_data[f'posture_diff_{window}d_slope'] = self._calculate_rolling_slope(smoothed_data['posture_diff'], adjusted_window)
                interpolated_data = smoothed_data.copy()
                if len(smoothed_data.dropna(subset=['posture_diff'])) >= 2:
                    interp_method = self.config.get('interpolation_method', 'linear')
                    interp_order = self.config.get('interpolation_order', 3)
                    self.logger.debug(f"Applying interpolation method: {interp_method} for {camera} - {date_span}")
                    try:
                        hourly_index = pd.date_range(start=smoothed_data.index.min(), end=smoothed_data.index.max(), freq='H')
                        interpolated_data = smoothed_data.reindex(smoothed_data.index.union(hourly_index)).interpolate(method=interp_method, order=interp_order)
                        interpolated_data = interpolated_data.reindex(hourly_index)
                    except ValueError as e:
                        self.logger.warning(f"Interpolation method '{interp_method}' failed for {camera} - {date_span}: {e}. Falling back to linear.")
                        try:
                            hourly_index = pd.date_range(start=smoothed_data.index.min(), end=smoothed_data.index.max(), freq='H')
                            interpolated_data = smoothed_data.reindex(smoothed_data.index.union(hourly_index)).interpolate(method='linear')
                            interpolated_data = interpolated_data.reindex(hourly_index)
                        except Exception as e_lin:
                            self.logger.error(f"Linear interpolation fallback also failed for {camera} - {date_span}: {e_lin}. Using non-interpolated data.")
                            interpolated_data = smoothed_data
                    except Exception as e:
                        self.logger.error(f"Unexpected interpolation error for {camera} - {date_span}: {e}. Using non-interpolated data.", exc_info=True)
                        interpolated_data = smoothed_data
                else:
                    self.logger.warning(f"Not enough non-NaN data points ({len(smoothed_data.dropna(subset=['posture_diff']))}) for interpolation on {camera} - {date_span}. Skipping.")
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': data_processed, 'resampled_data': resampled_data,
                        'smoothed_data': smoothed_data, 'interpolated_data': interpolated_data }
            except Exception as e:
                self.logger.error(f"Error during resampling/smoothing/interpolation for {camera} - {date_span}: {e}", exc_info=True)
                return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                        'raw_data': data_processed, 'resampled_data': pd.DataFrame(),
                        'smoothed_data': pd.DataFrame(), 'interpolated_data': pd.DataFrame() }
        except Exception as e:
            self.logger.critical(f"Critical error in preprocessing for {camera} - {date_span}: {e}", exc_info=True)
            quality_metrics['error'] = str(e)
            return { 'camera': camera, 'date_span': date_span, 'quality_metrics': quality_metrics,
                    'raw_data': pd.DataFrame(), 'resampled_data': pd.DataFrame(),
                    'smoothed_data': pd.DataFrame(), 'interpolated_data': pd.DataFrame() }

    def _calculate_missing_percentage(self, df, expected_rows):
        """Calculate percentage of missing rows in a dataframe."""
        if expected_rows == 0: return 0.0
        actual_rows = len(df.dropna(how='all'))
        missing_rows = expected_rows - actual_rows
        return (missing_rows / expected_rows) * 100 if expected_rows > 0 else 0.0
        
    def _calculate_max_consecutive_nans(self, series):
        """Calculate the maximum consecutive NaN values in a series."""
        if series.empty: return 0
        # Handle potential all-NaN series after resampling/smoothing
        if series.isnull().all():
            return len(series) if len(series) > 0 else 0
        return series.isnull().astype(int).groupby(series.notnull().astype(int).cumsum()).cumsum().max()
        
    def _apply_smoothing(self, data):
        """Apply configured smoothing method to data."""
        smoothed_data = data.copy()
        method = self.config['smoothing_method']
        strength = self.config['smoothing_strength']
        effective_length = 0
        if 'posture_diff' in data.columns: effective_length = data['posture_diff'].count()
        if effective_length < 3:
            self.logger.warning(f"Dataset too small for smoothing (length {effective_length}). Skipping smoothing.")
            return smoothed_data
        if method == 'rolling' and strength > 0: # Check strength > 0 for rolling
             adjusted_strength = min(strength, effective_length)
             if adjusted_strength < 2 : adjusted_strength = 2 # Ensure min window is 2 if possible
             if adjusted_strength != strength: self.logger.warning(f"Adjusted rolling window from {strength} to {adjusted_strength}")
             smoothed_data = data.rolling(window=adjusted_strength, center=True, min_periods=1).mean()
        elif method == 'ewm' and strength > 0:
            try:
                adjusted_strength = min(strength, effective_length)
                if adjusted_strength != strength: self.logger.warning(f"Adjusted EWM span from {strength} to {adjusted_strength}")
                smoothed_data = data.ewm(span=adjusted_strength, min_periods=1).mean()
            except Exception as e:
                self.logger.warning(f"EWM smoothing failed: {e}. Falling back.")
                window = min(3, effective_length)
                smoothed_data = data.rolling(window=window, center=True, min_periods=1).mean()
        elif method == 'savgol' and strength > 0:
            from scipy.signal import savgol_filter
            try:
                window_length = min(strength, effective_length - (effective_length % 2 == 0)) # Odd and <= N-1 approx
                window_length = max(3, window_length if window_length % 2 != 0 else window_length -1) # Ensure odd and >=3
                if window_length > effective_length : window_length = effective_length if effective_length % 2 !=0 else effective_length -1 # Final check

                if window_length < 3:
                    self.logger.warning(f"Effective length {effective_length} too small for Savitzky-Golay. Using moving average.")
                    window = min(3, effective_length)
                    return data.rolling(window=window, center=True, min_periods=1).mean()

                polyorder = min(2, window_length - 1)
                if window_length != strength: self.logger.warning(f"Adjusted Savitzky-Golay window from {strength} to {window_length}")

                for col in ['posture_diff', 'num_tails_upright', 'num_tails_hanging']:
                    if col in data.columns:
                        col_data = data[col].copy()
                        mask = ~col_data.isna()
                        if mask.sum() >= window_length:
                            valid_values = col_data[mask].values
                            smoothed_values = savgol_filter(valid_values, window_length, polyorder, mode='interp') # Use interp mode
                            col_data.loc[mask] = smoothed_values
                            smoothed_data[col] = col_data
                        else: self.logger.debug(f"Not enough points ({mask.sum()}) for Savgol on {col}, skipping.")
            except Exception as e:
                self.logger.warning(f"Savitzky-Golay failed: {e}. Falling back.")
                window = min(3, effective_length)
                smoothed_data = data.rolling(window=window, center=True, min_periods=1).mean()
        elif method == 'loess' and strength > 0:
             try:
                 from statsmodels.nonparametric.smoothers_lowess import lowess
                 if effective_length < 10:
                    self.logger.warning(f"Dataset too small ({effective_length}) for LOESS. Using moving average.")
                    window = min(3, effective_length)
                    return data.rolling(window=window, center=True, min_periods=1).mean()
                 frac = min(0.8, max(0.1, strength / 100.0, 3.0 / effective_length)) # Use floats
                 for col in ['posture_diff', 'num_tails_upright', 'num_tails_hanging']:
                     if col in data.columns:
                         col_data = data[col].copy()
                         mask = ~col_data.isna()
                         if mask.sum() > 3:
                             valid_indices = np.where(mask)[0]
                             filtered = lowess(col_data[mask].values, valid_indices, frac=frac, return_sorted=False)
                             smoothed_values = np.full(len(col_data), np.nan)
                             smoothed_values[valid_indices] = filtered
                             smoothed_data[col] = smoothed_values
                         else: self.logger.debug(f"Not enough points ({mask.sum()}) for LOESS on {col}, skipping.")
             except ImportError:
                 self.logger.warning("statsmodels not found, cannot use LOESS. Falling back.")
                 window = min(3, effective_length)
                 smoothed_data = data.rolling(window=window, center=True, min_periods=1).mean()
             except Exception as e:
                 self.logger.warning(f"LOESS failed: {e}. Falling back.")
                 window = min(3, effective_length)
                 smoothed_data = data.rolling(window=window, center=True, min_periods=1).mean()
        return smoothed_data
        
    def _calculate_rolling_slope(self, series, window):
        """Calculate rolling slope for time series."""
        if window < 2:
            self.logger.warning(f"Window size {window} too small for slope calculation.")
            return pd.Series(np.nan, index=series.index)
        slope_values = np.full(len(series), np.nan) # Use numpy array
        values = series.values # Work with numpy array
        indices = np.arange(len(values))
        for i in range(window - 1, len(values)):
            start_idx = i - window + 1
            y = values[start_idx : i + 1]
            x = indices[start_idx : i + 1]
            valid_mask = ~np.isnan(y)
            n_valid = np.sum(valid_mask)
            if n_valid >= 2:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                try:
                    slope, _ = np.polyfit(x_valid, y_valid, 1)
                    slope_values[i] = slope
                except (np.linalg.LinAlgError, ValueError) as e:
                    self.logger.warning(f"Error calculating slope at index {i}: {e}")
        return pd.Series(slope_values, index=series.index)