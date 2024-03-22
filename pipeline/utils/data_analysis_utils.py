import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from pipeline.utils import path_manager
from pipeline.utils.general import load_json_data


def load_data_as_dataframe(path_to_dataframe):
    """
    Loads a DataFrame from a .csv file and transforms the DataFrame for further analysis. 

    The transformation includes:
    - Sorting the DataFrame based on 'start_timestamp' if not already sorted
    - Creating 'start_date', 'start_time', and 'end_time' columns from the 'start_timestamp' 
      and 'end_timestamp' columns
    - Dropping the original 'start_timestamp' and 'end_timestamp' columns
    - Reordering the DataFrame columns
    - Filtering entries between 08:00:00 and 16:30:00 based on 'start_time' and 'end_time'

    Parameters:
        path_to_dataframe (str): The path to the .csv file to be loaded into a DataFrame

    Returns:
        df (pandas.DataFrame): The transformed DataFrame ready for analysis
    """
    # load dataframe
    df = pd.read_csv(path_to_dataframe, parse_dates=[
                     'start_timestamp', 'end_timestamp'])

    # Check if the dataframe is sorted by 'start_timestamp', if not, sort it
    if not df['start_timestamp'].is_monotonic_increasing:
        df = df.sort_values('start_timestamp')

        # Reset the index and drop the old one
        df.reset_index(drop=True, inplace=True)

    # Create 'start_date', 'start_time', 'end_date' and 'end_time' columns
    df["datetime"] = df['start_timestamp']
    df['start_date'] = df['start_timestamp'].dt.date
    df['start_time'] = df['start_timestamp'].dt.time
    df['end_time'] = df['end_timestamp'].dt.time

    # Drop the 'start_timestamp' and 'end_timestamp' columns
    df.drop(['start_timestamp', 'end_timestamp'], axis=1, inplace=True)

    # Define new order of columns
    new_order = ['datetime', 'start_date', 'start_time', 'end_time',
                 'start_frame', 'end_frame', 'num_tail_detections',
                 'num_tails_upright', 'num_tails_hanging',
                 'num_pig_detections', 'num_pigs_lying',
                 'num_pigs_notLying', 'activity']

    # Reorder dataframe columns
    df = df[new_order]

    # Filter entries between 08:00:00 and 16:30:00
    df = df[(df['start_time'] >= pd.to_datetime('08:00:00').time()) &
            (df['end_time'] <= pd.to_datetime('16:30:00').time())]

    # Removing duplicates
    df.drop_duplicates(subset='datetime', keep='first', inplace=True)

    # fill dataframe with NaN values so that every dataframe has values for timeframe 08:00:00 - 16:30:00
    df = fill_dataframe_with_nan(df)

    # Reset the index and drop the old one
    df.reset_index(drop=True, inplace=True)

    return df


def fill_dataframe_with_nan(df):
    """
    Fills missing timestamps in a dataframe with NaN values, based on a desired time range of 08:00:00 to 16:30:00.

    Parameters:
        df (DataFrame): The dataframe to be filled with NaN values.

    Returns:
        df_filled (DataFrame): The filled dataframe with NaN values.
    """

    # Convert the 'datetime' column to a datetime type if it's not already
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Set the 'datetime' column as the index of the dataframe
    df.set_index('datetime', inplace=True)

    # Get the date from the first row in the dataframe
    start_date = df.index[0].date()

    # Create the start and end timestamps for the desired time range
    start_time = pd.Timestamp(start_date) + pd.Timedelta(hours=8)
    end_time = pd.Timestamp(start_date) + \
        pd.Timedelta(hours=16, minutes=29, seconds=59)

    # Create a new dataframe with the desired time range, with a frequency of 1 second
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='S')
    df_filled = pd.DataFrame(index=full_time_range)

    # Merge the original dataframe with the filled dataframe using a left join, keeping the existing values
    df_merged = df_filled.join(df, how='left')

    # Reset the index of the merged dataframe
    df_merged.reset_index(inplace=True)

    # Rename index column back to datetime
    df_merged.rename(columns={'index': 'datetime'}, inplace=True)

    return df_merged


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


def load_monitoring_pipeline_results(dir, path_to_configs, compartment="piglet_rearing", camera=None, as_df_list=True):
    """
    Traverses a directory structure to extract and collect information 
    about monitored data results from specified or all cameras over various time spans.

    The directory structure should follow this pattern:
    - main directory
        - camera directory
            - time-span directory
                - .csv data files

    The function returns a list of dictionaries. Each dictionary corresponds to a time-span 
    directory under a camera directory and contains the following keys:
    - 'camera': the name of the camera (derived from the camera directory name)
    - 'date_span': the time span for which data was collected (derived from the time-span directory name)
    - 'data_paths': a list of file paths to all .csv data files located in the time-span directory
    - 'dataframes': a list of pandas DataFrames, where each DataFrame is the result of loading and
                    transforming a .csv file from 'data_paths' using the 'load_data_as_dataframe' function,
                    only populated if as_df_list is set to True, otherwise it's an empty list.

    Parameters:
        dir (str): The path to the main directory that contains the camera directories.
        path_to_configs (str): path to cinfiguration files.
        compartment (str): The specific compartment to look into. Defaults to "piglet_rearing".
        camera (str, optional): The specific camera name to load results for. If None, load results for all cameras.
        as_df_list (bool): If True, load the .csv files into pandas DataFrames and store them in the 'dataframes'
                        key of the dictionary. If False, leave the 'dataframes' key as an empty list.

    Returns:
        data_list (list of dict): A list of dictionaries. Each dictionary contains information 
        about a time-span directory under a camera directory, including the camera name, 
        the time span, the file paths to the data files, and potentially a list of DataFrames.
    """
    data_list = []

    # Define the path to the compartment within the main directory
    dir = os.path.join(dir, compartment)

    # load batch info file for respective compartment
    if compartment == "piglet_rearing":
        path_to_configs = os.path.join(
            path_to_configs, "rearing_batch_info.json")
    else:
        path_to_configs = os.path.join(
            path_to_configs, "fattening_batch_info.json")

    batch_info_data = load_json_data(path_to_configs)

    # Convert JSON data to a searchable structure
    video_timeframes = {item["camera"] + "_" +
                        item["datespan"]: item for item in batch_info_data}

    # Get all directories in the main directory (camera directories)
    camera_dirs = os.listdir(dir)
    if camera:
        # If a specific camera is specified, filter out other cameras
        camera_dirs = [c for c in camera_dirs if c == camera]

    for camera_name in camera_dirs:
        camera_dir_path = os.path.join(dir, camera_name)

        # Check if it's a directory
        if os.path.isdir(camera_dir_path):
            # Get all directories in the camera directory (time-span directories)
            for date_span in os.listdir(camera_dir_path):
                date_span_dir_path = os.path.join(camera_dir_path, date_span)

                # Check if it's a directory
                if os.path.isdir(date_span_dir_path):
                    camera_dict = {
                        'camera': camera_name,
                        'date_span': date_span,
                        'data_paths': [],
                        'dataframes': [],
                        'type': 'unknown'  # Default type, in case it's not found in the JSON data
                    }

                    # Construct the key to search in the video_timeframes
                    search_key = f"{camera_name}_{date_span}"
                    if search_key in video_timeframes:
                        camera_dict['type'] = video_timeframes[search_key]['type']

                    # Get all .csv files in the time-span directory
                    csv_files = [os.path.join(date_span_dir_path, f) for f in sorted(
                        os.listdir(date_span_dir_path)) if f.endswith('.csv')]
                    camera_dict['data_paths'] = csv_files

                    if as_df_list:
                        # Ensure this function is defined elsewhere
                        df_list = load_data_from_list(csv_files)
                        camera_dict['dataframes'] = df_list

                data_list.append(camera_dict)

    return data_list


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
        monitoring_result (dict): Contains dataframes with tail posture data.
        resample_freq (str): Frequency for data resampling (default is "D" for daily).
        normalize (bool): If True, normalizes the tail posture data before processing.
        rolling_window (int or None): Specifies the window size for rolling average calculation. If None, no rolling average is applied.
        path_to_result_data_aggregations (str): Path to save the transformed data CSV file.

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
        camera (str): The camera identifier for which pen info is being retrieved.
        date_span (str): The span of dates associated with the pen data, used to match the correct entry in JSON data.
        json_data (list of dicts): A list containing dictionaries, each representing data about a pen, including camera, 
        datespan, and pen characteristics such as type, culprit removal date, peak day, and low day.

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
        datespan (str): The span of dates in the format 'yymmdd_yymmdd', where the first part is the start date and the second part is the end date.

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
        pen_data (dict): A dictionary containing at least two keys, 'camera' and 'date_span'. The
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


