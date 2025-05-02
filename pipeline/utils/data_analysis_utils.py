import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

from pipeline.utils.path_manager import PathManager
from pipeline.utils.general import load_json_data


import logging
logger = logging.getLogger("data_analysis_utils") # Add logger for utils

def get_expected_dates(date_span_str):
    """
    Generates a list of expected dates within a datespan string.

    Parameters:
    - date_span_str (str): The span of dates in 'yymmdd_yymmdd' format.

    Returns:
    list: A list of datetime.date objects for each expected day.
          Returns an empty list if the format is invalid.
    """
    try:
        start_str, end_str = date_span_str.split('_')
        start_date = datetime.strptime(start_str, '%y%m%d').date()
        end_date = datetime.strptime(end_str, '%y%m%d').date()

        expected_dates = []
        current_date = start_date
        while current_date <= end_date:
            expected_dates.append(current_date)
            current_date += timedelta(days=1)
        return expected_dates
    except ValueError:
        logger.warning(f"Invalid date_span format: {date_span_str}. Cannot determine expected dates.")
        return []

def parse_date_from_filename(filename):
    """
    Extracts the date from a CSV filename assuming format like
    KameraX_YYYY_MM_DD_...csv or KX_YYYY_MM_DD_...csv.

    Returns:
    datetime.date or None if parsing fails.
    """
    try:
        parts = os.path.basename(filename).split('_')
        # Find the first part that looks like a year (YYYY)
        year_index = -1
        for i, part in enumerate(parts):
            if len(part) == 4 and part.isdigit() and int(part) > 2000:
                 # Check if the next two parts are plausible month/day
                 if i + 2 < len(parts) and len(parts[i+1]) == 2 and parts[i+1].isdigit() and \
                    len(parts[i+2]) == 2 and parts[i+2].isdigit():
                     year_index = i
                     break
        if year_index != -1:
            date_str = f"{parts[year_index]}{parts[year_index+1]}{parts[year_index+2]}"
            return datetime.strptime(date_str, '%Y%m%d').date()
        else:
            # Fallback for older yy format if primary fails (less reliable)
            for i, part in enumerate(parts):
                if len(part) == 6 and part.isdigit() : # yymmdd
                     return datetime.strptime(part,'%y%m%d').date()
            logger.warning(f"Could not parse date from filename: {filename}")
            return None

    except (IndexError, ValueError) as e:
        logger.warning(f"Error parsing date from filename {filename}: {e}")
        return None


def load_monitoring_pipeline_results(dir, path_to_configs, compartment="piglet_rearing", camera=None, as_df_list=True):
    """
    Traverses a directory structure to extract and collect information
    about monitored data results, detects missing daily files, and stores this info.

    Adds 'found_dates', 'missing_dates', 'total_expected_days' to the returned dict.

    Parameters:
    dir (str): The path to the main directory that contains the camera directories.
    path_to_configs (str): path to configuration files.
    compartment (str): The specific compartment to look into. Defaults to "piglet_rearing".
    camera (str, optional): The specific camera name to load results for. If None, load results for all cameras.
    as_df_list (bool): If True, load the .csv files into pandas DataFrames.

    Returns:
    data_list (list of dict): A list of dictionaries for each camera/datespan, including missing data info.
    """
    data_list = []
    dir = os.path.join(dir, compartment) # Path to compartment

    # Load batch info file (adjust path as needed)
    if compartment == "piglet_rearing":
        config_file = "rearing_batch_info.json" # Or piglet_rearing_info.json? Check your naming
    elif compartment == "fattening":
        config_file = "fattening_batch_info.json"
    else:
        logger.error(f"Unknown compartment: {compartment}")
        return []

    path_to_batch_info = os.path.join(path_to_configs, config_file)
    if not os.path.exists(path_to_batch_info):
         # Try the alternative name based on your class init PathManager usage
         path_to_batch_info = PathManager().path_to_piglet_rearing_info if compartment == "piglet_rearing" else None # Add equivalent for fattening if needed
         if not path_to_batch_info or not os.path.exists(path_to_batch_info):
              logger.error(f"Batch info file not found at {path_to_batch_info} or alternative path.")
              # Decide: either return [] or proceed without type info from json
              # return [] # Stricter option
              batch_info_data = [] # Proceed without type info
         else:
             batch_info_data = load_json_data(path_to_batch_info)
    else:
         batch_info_data = load_json_data(path_to_batch_info)


    # Convert JSON data for faster lookup (optional, depends on json structure)
    video_timeframes = {item["camera"] + "_" + item["datespan"]: item for item in batch_info_data}

    # Handle specific camera or all
    if camera:
        camera_dirs = [c for c in os.listdir(dir) if c == camera and os.path.isdir(os.path.join(dir, c))]
    else:
        camera_dirs = [c for c in os.listdir(dir) if os.path.isdir(os.path.join(dir, c))]

    for camera_name in camera_dirs:
        camera_dir_path = os.path.join(dir, camera_name)
        try:
            date_span_dirs = [d for d in os.listdir(camera_dir_path) if os.path.isdir(os.path.join(camera_dir_path, d))]
        except FileNotFoundError:
            logger.warning(f"Camera directory not found or not accessible: {camera_dir_path}")
            continue

        for date_span in date_span_dirs:
            date_span_dir_path = os.path.join(camera_dir_path, date_span)
            camera_dict = {
                'camera': camera_name,
                'date_span': date_span,
                'data_paths': [],
                'dataframes': [],
                'type': 'unknown',
                'found_dates': [],
                'missing_dates': [],
                'total_expected_days': 0
            }

            # Get type from JSON if available
            search_key = f"{camera_name}_{date_span}"
            if search_key in video_timeframes:
                camera_dict['type'] = video_timeframes[search_key].get('type', 'unknown') # Use .get for safety

            # --- Missing Day Detection ---
            expected_dates = get_expected_dates(date_span)
            camera_dict['total_expected_days'] = len(expected_dates)
            found_dates_in_span = set()

            try:
                files_in_dir = sorted(os.listdir(date_span_dir_path))
            except FileNotFoundError:
                 logger.warning(f"Datespan directory not found: {date_span_dir_path}")
                 continue # Skip this datespan if dir doesn't exist

            csv_files = []
            for f in files_in_dir:
                if f.endswith('.csv'):
                    full_path = os.path.join(date_span_dir_path, f)
                    csv_files.append(full_path)
                    file_date = parse_date_from_filename(f)
                    if file_date:
                        found_dates_in_span.add(file_date)
                # Optionally log non-csv files if unexpected
                # else:
                #    logger.debug(f"Ignoring non-csv file: {f} in {date_span_dir_path}")


            camera_dict['data_paths'] = csv_files
            camera_dict['found_dates'] = sorted(list(found_dates_in_span))

            if expected_dates: # Only calculate missing if we could parse the datespan
                 missing_dates_set = set(expected_dates) - found_dates_in_span
                 camera_dict['missing_dates'] = sorted(list(missing_dates_set))
                 if camera_dict['missing_dates']:
                     logger.warning(f"Missing {len(camera_dict['missing_dates'])} day(s) for {camera_name}/{date_span}: {camera_dict['missing_dates']}")

            # --- Load DataFrames ---
            if as_df_list and csv_files:
                try:
                    # Make sure load_data_from_list handles potential errors in load_data_as_dataframe
                    df_list = load_data_from_list(csv_files)
                    # Filter out None values if load_data_as_dataframe returns None on error
                    camera_dict['dataframes'] = [df for df in df_list if df is not None and not df.empty]
                    if len(camera_dict['dataframes']) != len(csv_files):
                         logger.warning(f"Loaded {len(camera_dict['dataframes'])} non-empty dataframes out of {len(csv_files)} found CSVs for {camera_name}/{date_span}.")

                except Exception as e:
                     logger.error(f"Error loading dataframes for {camera_name}/{date_span}: {e}")
                     camera_dict['dataframes'] = [] # Ensure it's empty on error
            elif not csv_files:
                 logger.warning(f"No CSV files found for {camera_name}/{date_span}.")


            data_list.append(camera_dict)

    # Apply sorting key after collecting all data
    # Make sure sorting_key handles potential missing keys gracefully if needed
    try:
        data_list.sort(key=sorting_key)
    except KeyError as e:
        logger.error(f"Failed to sort monitoring results, missing key: {e}")


    return data_list

# Modify load_data_as_dataframe to be more robust
def load_data_as_dataframe(path_to_dataframe):
    """
    Loads a DataFrame from a .csv file and transforms it. Returns None on error.
    Includes check for required columns after loading.
    """
    required_cols_raw = ['start_timestamp', 'end_timestamp', 'num_tail_detections', 'num_tails_upright', 'num_tails_hanging'] # Add others if needed
    try:
        df = pd.read_csv(path_to_dataframe, parse_dates=['start_timestamp', 'end_timestamp'])

        # Check for required columns immediately after loading
        missing_raw_cols = [col for col in required_cols_raw if col not in df.columns]
        if missing_raw_cols:
             logger.error(f"Missing required columns {missing_raw_cols} in raw file: {path_to_dataframe}. Skipping this file.")
             return None

         # Check for empty dataframe
        if df.empty:
             logger.warning(f"Empty dataframe loaded from: {path_to_dataframe}. Skipping this file.")
             return None

        # Proceed with transformations...
        if not df['start_timestamp'].is_monotonic_increasing:
            df = df.sort_values('start_timestamp').reset_index(drop=True)

        df["datetime"] = df['start_timestamp']
        df['start_date'] = df['start_timestamp'].dt.date
        df['start_time'] = df['start_timestamp'].dt.time
        df['end_time'] = df['end_timestamp'].dt.time
        df.drop(['start_timestamp', 'end_timestamp'], axis=1, inplace=True)

        # Define new order of columns - ensure all exist or handle missing ones
        # Check optional columns exist before including them in the reorder list
        optional_cols = ['start_frame', 'end_frame', 'num_pig_detections', 'num_pigs_lying', 'num_pigs_notLying', 'activity']
        base_order = ['datetime', 'start_date', 'start_time', 'end_time', 'num_tail_detections', 'num_tails_upright', 'num_tails_hanging']
        present_optional_cols = [col for col in optional_cols if col in df.columns]
        new_order = base_order + present_optional_cols

        # Check if all columns in new_order are actually in df before reordering
        final_order = [col for col in new_order if col in df.columns]
        df = df[final_order]

        # Filter time (08:00:00 to 16:30:00)
        df = df[(df['start_time'] >= pd.to_datetime('08:00:00').time()) &
                (df['end_time'] <= pd.to_datetime('16:30:00').time())]

        # Check again if filtering made it empty
        if df.empty:
             logger.warning(f"Dataframe became empty after time filtering: {path_to_dataframe}. Skipping.")
             return None

        df.drop_duplicates(subset='datetime', keep='first', inplace=True)

        # Fill intra-day gaps
        df = fill_dataframe_with_nan(df) # fill_dataframe_with_nan needs to be robust

        df.reset_index(drop=True, inplace=True)

        # Final check for emptiness after processing
        if df.empty:
             logger.warning(f"Dataframe became empty after all processing: {path_to_dataframe}. Skipping.")
             return None

        return df

    except pd.errors.EmptyDataError:
        logger.error(f"EmptyDataError: No data in file: {path_to_dataframe}")
        return None
    except FileNotFoundError:
         logger.error(f"FileNotFoundError: File not found: {path_to_dataframe}")
         return None
    except Exception as e:
        logger.error(f"Failed to load or process dataframe {path_to_dataframe}: {e}", exc_info=True) # Log traceback
        return None


def fill_dataframe_with_nan(df):
    """ Robust version of fill_dataframe_with_nan """
    if df.empty:
        logger.warning("Attempted to fill empty DataFrame with NaNs. Returning empty DataFrame.")
        return df

    try:
        # Ensure datetime is index
        if not isinstance(df.index, pd.DatetimeIndex):
             if 'datetime' in df.columns:
                  df['datetime'] = pd.to_datetime(df['datetime'])
                  df.set_index('datetime', inplace=True)
             else:
                  logger.error("Cannot fill NaNs: DataFrame has no 'datetime' column or index.")
                  return df # Or raise error

        # Check for valid index
        if df.index.empty or pd.isna(df.index.min()) or pd.isna(df.index.max()):
             logger.warning("Cannot fill NaNs: DataFrame index is empty or contains NaNs.")
             return df # Or attempt to drop rows with NaT index: df = df.loc[df.index.dropna()]

        start_date = df.index.min().date()
        start_time = pd.Timestamp(start_date) + pd.Timedelta(hours=8)
        # Use the actual max time from data, but ensure it's within the day, or use fixed 16:30
        # end_time_data = df.index.max()
        # end_time_fixed = pd.Timestamp(start_date) + pd.Timedelta(hours=16, minutes=29, seconds=59)
        # end_time = min(end_time_data, end_time_fixed) # Choose the earlier end time if data stops early
        end_time = pd.Timestamp(start_date) + pd.Timedelta(hours=16, minutes=29, seconds=59) # Keep fixed end time

        # Ensure start_time is not after end_time
        if start_time > end_time:
            logger.warning(f"Calculated start_time {start_time} is after end_time {end_time}. Check data range.")
            # Use data min/max perhaps? Or just return df?
            # For now, stick to the fixed range if possible
            start_time = min(start_time, end_time) # Adjust start if needed, though unlikely

        full_time_range = pd.date_range(start=start_time, end=end_time, freq='S')

        # Use reindex instead of merge for potentially better performance and handling
        df_filled = df.reindex(full_time_range)

        # Reindex might drop non-index columns, check if this happens.
        # If using join:
        # df_template = pd.DataFrame(index=full_time_range)
        # df_filled = df_template.join(df, how='left') # Join original data onto the template

        df_filled.index.name = 'datetime' # Ensure index has name
        df_filled.reset_index(inplace=True) # Move index back to column

        return df_filled

    except Exception as e:
        logger.error(f"Error in fill_dataframe_with_nan: {e}", exc_info=True)
        return df # Return original df on error


def load_data_from_list(paths_to_data_list):
    """
    Loads multiple dataframes from a list of paths and returns a list of transformed DataFrames.

    This function iterates over a list of paths to .csv files, calls the function `load_data_as_dataframe` 
    on each path, and appends the resulting DataFrame to a list. It returns this list of DataFrames.

    Parameters:
    paths_to_data_list (list of str): A list of paths to .csv files to be loaded into DataFrames.

    Returns:
    df_list (list of pandas.DataFrame): A list of transformed DataFrames ready for analysis.
    """
    df_list = [load_data_as_dataframe(path_to_df)
               for path_to_df in paths_to_data_list]
    return df_list


def get_mvg_avg_from_data_dict(data_dict, rolling_window):
    """
    Calculate moving averages for multiple dataframes specified in a dictionary.

    This function takes a dictionary as input, where the key 'dataframes' corresponds 
    to a non-empty list of dataframes. For each dataframe in the list, the function 
    calculates the moving average for certain columns using a specified rolling window.

    The function uses the 'get_mvg_avg' function to calculate the moving averages.

    Parameters:
    data_dict (dict): A dictionary containing a non-empty list of dataframes under 
                      the 'dataframes' key.
    rolling_window (int): The size of the moving window used for calculation in 
                          the 'get_mvg_avg' function.

    Returns:
    list: A list of dataframes with calculated moving averages for the specified columns.

    Raises:
    AssertionError: If the input is not a dictionary, if the dictionary does not contain 
                    a 'dataframes' key, or if the value for this key is not a non-empty list.
    """
    assert isinstance(data_dict, dict), "Input should be a dictionary."
    assert isinstance(data_dict["dataframes"], list) and data_dict["dataframes"], \
        "The value for the 'dataframes' key should be a non-empty list."
    assert all(isinstance(df, pd.DataFrame) for df in data_dict["dataframes"]), \
        "All elements in the 'dataframes' list should be pandas DataFrame instances."
    data_list = data_dict["dataframes"]
    data_list_mvg_avg = [get_mvg_avg(df, rolling_window) for df in data_list]
    return data_list_mvg_avg


def get_mvg_avg(df, rolling_window):
    """
    Compute the moving average for several columns in the input DataFrame.

    This function uses a specified rolling window to calculate the mean value for 
    'num_tails_upright', 'num_tails_hanging', 'num_pigs_lying', 'num_pigs_notLying',
    and 'activity' columns in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    rolling_window (int): The size of the moving window used for calculation.

    Returns:
    pandas.DataFrame: A DataFrame containing the moving averages for the specified columns.
    """

    # Select the relevant columns from the input DataFrame and calculate their moving average.
    # The 'window' parameter defines the size of the moving window.
    # 'min_periods' is set to 1 to make sure that we get output even if there are missing values.
    df_mvg_avg = df[['num_tails_upright', 'num_tails_hanging', 'num_pigs_lying',
                    'num_pigs_notLying', 'activity']].rolling(window=rolling_window, min_periods=1).mean()
    df_mvg_avg["datetime"] = df["datetime"]
    return df_mvg_avg


def get_filtered_data_from_data_dict(data_dict, ratio_notLying):
    """
    Filters rows in multiple dataframes specified in a dictionary based on a given ratio.

    This function takes a dictionary as input, where the key 'dataframes' corresponds 
    to a non-empty list of dataframes. For each dataframe in the list, the function 
    filters rows where the ratio of 'num_pigs_notLying' to 'num_pigs_lying' is greater than 
    or equal to the specified ratio_notLying value.

    The function uses the 'filter_data_at_posture_ratio' function to filter the dataframes.

    Parameters:
    data_dict (dict): A dictionary containing a non-empty list of dataframes under 
                      the 'dataframes' key.
    ratio_notLying (float): The threshold ratio used for filtering the DataFrame.

    Returns:
    list: A list of filtered dataframes where each dataframe only includes rows with a 
          'num_pigs_notLying' to 'num_pigs_lying' ratio greater than or equal to ratio_notLying.

    Raises:
    AssertionError: If the input is not a dictionary, if the dictionary does not contain 
                    a 'dataframes' key, if the value for this key is not a non-empty list,
                    or if the elements of the list are not pandas DataFrame instances.
    """
    assert isinstance(data_dict, dict), "Input should be a dictionary."
    assert isinstance(data_dict["dataframes"], list) and data_dict["dataframes"], \
        "The value for the 'dataframes' key should be a non-empty list."
    assert all(isinstance(df, pd.DataFrame) for df in data_dict["dataframes"]), \
        "All elements in the 'dataframes' list should be pandas DataFrame instances."
    data_list = data_dict["dataframes"]
    data_list_mvg_avg = [filter_data_at_posture_ratio(
        df, ratio_notLying) for df in data_list]
    return data_list_mvg_avg


def filter_data_at_posture_ratio(df, ratio_notLying):
    """
    Filter the rows in the input DataFrame based on the ratio of 'num_pigs_notLying' to 'num_pigs_lying'.

    This function returns a DataFrame that only includes rows where the ratio of the 
    'num_pigs_notLying' column to the 'num_pigs_lying' column is greater than or equal to 
    the specified ratio_notLying value.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with at least two columns 'num_pigs_notLying' 
    and 'num_pigs_lying' containing numeric data.
    ratio_notLying (float): The threshold ratio used for filtering the DataFrame.

    Returns:
    pandas.DataFrame: A filtered DataFrame where each row has a 'num_pigs_notLying' to 
    'num_pigs_lying' ratio greater than or equal to ratio_notLying.
    """
    # Calculate the ratio of 'num_pigs_notLying' to 'num_pigs_lying' for each row in the DataFrame
    # and only keep rows where this ratio is greater than or equal to ratio_notLying
    df_filtered = df[df["num_pigs_notLying"] /
                     df["num_pigs_lying"] >= ratio_notLying]
    return df_filtered


def plot_aggregated_data_per_day(list_of_experiments, aggregation_type, figsize, tails_upright=True, tails_hanging=True, activity=True, return_fig=False, legend=True, xticks_interval=1):
    """
    Generate a grid of subplots displaying the aggregated values of different features per day for a list of experiments.

    For each experiment, the function plots the aggregated (mean or median) values of 'num_tails_hanging', 
    'num_tails_upright' and 'activity' against date. The number of subplot rows is determined by the 
    number of experiments. Each row has three subplots, and more rows are added as needed.

    Parameters:
    list_of_experiments (list): List of experiments. Each experiment is a list of pandas DataFrames, 
                                each DataFrame represents a day and contains columns 'num_tails_hanging', 
                                'num_tails_upright', 'activity', and 'datetime'.
    aggregation_type (str): The type of aggregation operation to apply. It must be either "avg" (average) or "median".
    tails_upright (bool, optional): If True, include the 'num_tails_upright' data in the plot. Default is True.
    tails_hanging (bool, optional): If True, include the 'num_tails_hanging' data in the plot. Default is True.
    activity (bool, optional): If True, include the 'activity' data in the plot. Default is True.

    Raises:
    ValueError: If the 'aggregation_type' is not either "avg" or "median".

    Returns:
    None: The function generates a matplotlib plot.
    """

    if aggregation_type not in ["avg", "median"]:
        raise ValueError("Invalid operation type. Expected 'avg' or 'median'.")

    if len(list_of_experiments) == 1:
        rows = 1
        cols = 1

        fig, ax = plt.subplots(
            rows, cols, figsize=(figsize[0], figsize[1]*rows))

    else:

        # Calculate the number of rows for the subplots
        rows = len(list_of_experiments) // 3 + \
            (len(list_of_experiments) % 3 > 0)

        fig, ax = plt.subplots(
            rows, len(list_of_experiments), figsize=(figsize[0], figsize[1]*rows))

        # Reshape ax to a 2D array if it's not
        if ax.ndim == 1:
            ax = ax.reshape(rows, -1)

    for i, exp in enumerate(list_of_experiments):
        holder_hanging = []
        holder_upright = []
        holder_activity = []
        holder_date = []
        for df in exp:
            if aggregation_type == "avg":
                avg_hanging = df['num_tails_hanging'].mean()
                avg_upright = df['num_tails_upright'].mean()
                avg_activity = df['activity'].mean()
                date = df['datetime'].iloc[int(len(df)/2)]

                holder_hanging.append(avg_hanging)
                holder_upright.append(avg_upright)
                holder_activity.append(avg_activity)
                holder_date.append(date)
            else:
                median_hanging = df['num_tails_hanging'].median()
                median_upright = df['num_tails_upright'].median()
                median_activity = df['activity'].median()
                date = df['datetime'].iloc[int(len(df)/2)]

                holder_hanging.append(median_hanging)
                holder_upright.append(median_upright)
                holder_activity.append(median_activity)
                holder_date.append(date)

        if len(list_of_experiments) == 1:
            # Plot holder_hanging and holder_upright on the first y-axis (left side)
            if tails_hanging:
                ax.plot(holder_date, holder_hanging, color="red",
                        label='tails_hanging', linewidth=4)
            if tails_upright:
                ax.plot(holder_date, holder_upright, color="green",
                        label='tails_upright', linewidth=4)

            ax.xaxis.set_major_locator(plt.MaxNLocator(
                nbins=len(holder_date)//xticks_interval))

            # Set the label for the first y-axis
            ax.set_ylabel('Number of Tails', fontsize=14)

            if activity:
                # Create a second y-axis (right side)
                ax2 = ax.twinx()
                # Plot holder_activity on the second y-axis (right side)
                ax2.plot(holder_date, holder_activity, color="blue",
                         label='activity', linewidth=4)
                # Set the label for the second y-axis
                ax2.set_ylabel('Activity', fontsize=14)

            # Combine the legends from both axes
            handles1, labels1 = ax.get_legend_handles_labels()
            if activity:
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles = handles1 + handles2
                labels = labels1 + labels2
            else:
                labels = labels1
                handles = handles1

            if legend:
                ax.legend(handles, labels, loc='upper right')

            # set x label
            ax.set_xlabel('Date', fontsize=14)

            # Rotate x-axis tick labels by 45 degrees for better readability
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        else:
            current_ax = ax[i//3, i % 3]
            if tails_upright:
                current_ax.plot(holder_date, holder_hanging, color="red",
                                label='tails_hanging', linewidth=3)
            if tails_hanging:
                current_ax.plot(holder_date, holder_upright, color="green",
                                label='tails_upright', linewidth=3)
            if activity:
                ax_activity = current_ax.twinx()
                ax_activity.plot(holder_date, holder_activity,
                                 color="blue", label='activity', linewidth=3)

                # Set 'Activity' label on the y-axis for every 3rd subplot
                if i % 3 == 2:
                    # label for second y-axis
                    ax_activity.set_ylabel('Activity')

            if i == 2:
                # Get handles and labels for ax[i] and ax_activity
                handles1, labels1 = current_ax.get_legend_handles_labels()
                if activity:
                    handles2, labels2 = ax_activity.get_legend_handles_labels()
                else:
                    handles2, labels2 = [], []

                # Combine handles and labels
                handles = handles1 + handles2
                labels = labels1 + labels2

                if legend:
                    # Call legend with combined handles and labels
                    current_ax.legend(handles, labels, loc='upper right')

            # Set 'Number of Tails' label on the y-axis for every first subplot
            if i % 3 == 0:
                current_ax.set_ylabel('Number of Tails')

            current_ax.set_title('Experiment ' + str(i+1))

            # set x label
            current_ax.set_xlabel('Date', fontsize=14)

            current_ax.xaxis.set_major_locator(
                plt.MaxNLocator(nbins=len(holder_date)//xticks_interval))

            # Rotate x-axis tick labels by 45 degrees for the first subplot
            current_ax.tick_params(axis='x', rotation=45, labelsize=12)
            current_ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()  # adjust the layout
    plt.show()
    if return_fig:
        return fig


def aggregate_pipeline_montioring_results(monitoring_result, path_to_result_data_aggregations=None, resample_freq="D", compartment="piglet_rearing", normalize=False, rolling_window=None, save_data=False):
    """
    Transforms the tail posture data from the monitoring result according to specified parameters and saves the transformed data to a CSV file.

    Parameters:
    - monitoring_result (dict): Contains dataframes with tail posture data.
    - resample_freq (str): Frequency for data resampling (default is "D" for daily).
    - normalize (bool): If True, normalizes the tail posture data before processing.
    - rolling_window (int or None): Specifies the window size for rolling average calculation. If None, no rolling average is applied.
    - path_to_result_data_aggregations (str): Path to save the transformed data CSV file.

    Returns:
    None. The transformed data is saved as a CSV file.
    """

    if path_to_result_data_aggregations:
        full_path = os.path.join(path_to_result_data_aggregations, compartment,
                                 monitoring_result["camera"], monitoring_result["date_span"])

        # Ensure the directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Concatenate dataframes from the dictionary
    data_all = pd.concat(monitoring_result['dataframes'])

    # Create a copy and set datetime index
    data_all_copy = data_all.copy()
    data_all_copy["datetime"] = pd.to_datetime(data_all_copy["datetime"])
    data_all_copy.set_index("datetime", inplace=True)

    # Normalize if the option is set
    if normalize:
        data_all_copy["num_tails_hanging"] = data_all_copy["num_tails_hanging"] / \
            data_all_copy["num_tail_detections"]
        data_all_copy["num_tails_upright"] = data_all_copy["num_tails_upright"] / \
            data_all_copy["num_tail_detections"]

    avg_data = data_all_copy.resample(resample_freq).mean()
    avg_data['posture_diff'] = avg_data['num_tails_upright'] - \
        avg_data['num_tails_hanging']

    # Apply rolling/moving average if specified
    if rolling_window:
        avg_data = avg_data.rolling(window=rolling_window).mean()

    # Perform interpolation
    interpolated_data = avg_data.resample("H").interpolate(method='linear')

    if save_data == True:
        # Save to CSV
        interpolated_data.to_csv(full_path)
        print(f"Data transformed and saved to {full_path}")


def get_pen_info(camera, date_span, json_data):
    """
    Retrieve information about a pen based on camera and date_span from provided JSON data.

    Parameters:
    - camera (str): The camera identifier for which pen info is being retrieved.
    - date_span (str): The span of dates associated with the pen data, used to match the correct entry in JSON data.
    - json_data (list of dicts): A list containing dictionaries, each representing data about a pen, including camera, datespan, and pen characteristics such as type, culprit removal date, peak day, and low day.

    Returns:
    Tuple[str, str, Optional[int], Optional[int]]: A tuple containing the pen type, culprit removal date and ground truth datespan.
    If no matching entry is found in the json_data, returns ("Unknown", "Unknown", "Unknown").

    The 'culprit_removal' and 'datespan_gt' are returned as "Unknown" if not specified.
    """
    for entry in json_data:
        if entry["camera"] == camera and entry["datespan"] == date_span:
            pen_type = entry["type"]
            culprit_removal = entry.get("culpritremoval", "Unknown")
            datespan_gt = entry.get("datespan_gt", "Unknown")
            return pen_type, culprit_removal, datespan_gt
    return "Unknown", "Unknown", "Unknown"


def format_datespan(datespan):
    """
    Convert a date span string into a more human-readable format.

    Parameters:
    - datespan (str): The span of dates in the format 'yymmdd_yymmdd', where the first part is the start date and the second part is the end date.

    Returns:
    str: The formatted date span as 'dd.mm.yyyy - dd.mm.yyyy'.

    Example:
    Given a datespan '220101_220131', it returns '01.01.2022 - 31.01.2022'.
    """
    start_date, end_date = datespan.split('_')
    start_date_formatted = datetime.strptime(
        start_date, '%y%m%d').strftime('%d.%m.%Y')
    end_date_formatted = datetime.strptime(
        end_date, '%y%m%d').strftime('%d.%m.%Y')
    return f"{start_date_formatted} - {end_date_formatted}"


def sorting_key(pen_data):
    """
    Extracts a sorting key from a dictionary containing information about a pen.

    Parameters:
    - pen_data (dict): A dictionary containing at least two keys, 'camera' and 'date_span'. The
      'camera' key should have a string value in the format "KameraX" where X is the pen number.
      The 'date_span' key should have a string value in the format 'yymmdd_yymmdd' representing
      the start and end dates.

    Returns:
    tuple: A tuple containing two elements:
        - int: The pen number extracted and converted from the 'camera' string.
        - datetime.datetime: The start date extracted from the 'date_span' string and converted to a datetime object.
    """
    # Extract pen number from the 'camera' string
    pen_number = int(pen_data['camera'].replace("Kamera", ""))
    # Extract start date from the 'date_span' string
    start_date = datetime.strptime(
        pen_data['date_span'].split('_')[0], '%y%m%d')
    return pen_number, start_date

def analyze_csv_stats(base_dir):
    """
    Analyze CSV files in the directory structure and return basic statistics
    about the number of entries (rows) in each CSV.
    
    Args:
        base_dir (str): The base directory containing the Kamera subdirectories
        
    Returns:
        dict: Statistics about the CSV files
    """
    # Dictionary to store entry counts for each camera
    camera_entries = defaultdict(list)
    # List to store all entry counts
    all_entries = []
    # Counter for total files processed
    total_files = 0
    
    print(f"Analyzing CSV files in {base_dir}...")
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        csv_files = [f for f in files if f.endswith('.csv')]
        total_files += len(csv_files)
        
        for file in csv_files:
            file_path = os.path.join(root, file)
            
            # Extract camera name from the path
            path_parts = file_path.split(os.sep)
            camera = next((part for part in path_parts if part.startswith('Kamera')), None)
            
            if camera:
                # Read the CSV and count the number of entries
                try:
                    df = pd.read_csv(file_path)
                    num_entries = len(df)
                    camera_entries[camera].append(num_entries)
                    all_entries.append(num_entries)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {total_files} CSV files")
    
    # Calculate statistics for each camera
    camera_stats = {}
    for camera, entries in sorted(camera_entries.items()):
        if entries:
            non_zero_entries = [e for e in entries if e > 0]
            min_value = min(non_zero_entries) if non_zero_entries else "No entries > 0"
            
            camera_stats[camera] = {
                'max_entries': max(entries),
                'min_entries_above_zero': min_value,
                'median_entries': np.median(entries),
                'avg_entries': np.mean(entries),
                'total_csvs': len(entries)
            }
    
    # Calculate overall statistics
    overall_stats = {}
    if all_entries:
        non_zero_entries = [e for e in all_entries if e > 0]
        min_value = min(non_zero_entries) if non_zero_entries else "No entries > 0"
        
        overall_stats = {
            'max_entries': max(all_entries),
            'min_entries_above_zero': min_value,
            'median_entries': np.median(all_entries),
            'avg_entries': np.mean(all_entries),
            'total_csvs': len(all_entries),
            'total_files_processed': total_files
        }
    
    return {
        'overall': overall_stats,
        'by_camera': camera_stats
    }
