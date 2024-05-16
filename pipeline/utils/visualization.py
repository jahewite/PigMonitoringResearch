from calendar import c
import os
import cv2
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from datetime import datetime
from scipy.signal import find_peaks

from pipeline.model_pipeline import ModelPipeline
from pipeline.utils.path_manager import PathManager
from pipeline.utils.image_preprocessor import ImagePreprocessor
from pipeline.tail_detection_processor import TailDetectionProcessor
from pipeline.utils.data_analysis_utils import format_datespan, get_pen_info
from pipeline.utils.general import load_label_file, load_json_data, save_array_as_image


path_manager = PathManager()

labels_posture_classification, labels_posture_classification_reverse = load_label_file(
    path_manager.path_to_pig_posture_classification_label_file)
labels_tail_posture_detection, labels_tail_posture_detection_reverse = load_label_file(
    path_manager.path_to_tail_posture_detection_label_file)


def inference_on_single_image(input_image, filter_lying_pigs=False, prefix=None, save_img=False):
    """
    Performs inference on a single image, applying a model pipeline for pig and tail detection,
    and optionally saves the result with a unique name.

    Parameters:
        input_image (str or numpy.ndarray): The path to an image or an image object.
        filter_lying_pigs (bool, optional): Whether to apply filtering for lying pigs. Defaults to False.
        prefix (str, optional): Prefix for preprocessing the image. Defaults to None.
        save_img (bool, optional): Flag to save the image with results plotted. If True, the image is saved with a unique name. Defaults to False.

    Raises:
        ValueError: If the input_image is not a string or a numpy.ndarray.
    """

    if isinstance(input_image, str):
        # If input is a string, read the image from the path
        img = cv2.imread(input_image)
        img_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(input_image)}"

    elif isinstance(input_image, np.ndarray):
        # If input is a NumPy array, use it directly
        img = input_image
        img_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    else:
        raise ValueError("Input must be an image path (string) or an image object (NumPy array).")

    # load label files
    labels_posture_classification, labels_posture_classification_reverse = load_label_file(
        path_manager.path_to_pig_posture_classification_label_file)
    labels_tail_posture_detection, labels_tail_posture_detection_reverse = load_label_file(
        path_manager.path_to_tail_posture_detection_label_file)

    if prefix:
        # load config file (contains info for cutting images and additional info for areas to analyze during activity derivation)
        config = load_json_data(path_manager.path_to_config)
        # Initialize the pipeline, activity analyzer, tail detection processor and logger
        image_preprocessor = ImagePreprocessor(
            config, prefix)
    else:
        pass

    # Create Pipeline class instance
    pipeline = ModelPipeline(
        crop_size=224, filter_lying_pigs=filter_lying_pigs)

    # Create TailDetectionProcessor instance
    tail_detections_processor = TailDetectionProcessor(
        labels_posture_classification_reverse, labels_tail_posture_detection_reverse)

    # perform inference
    pig_detections, posture_classifications, tail_posture_detections = pipeline(
        img)

    # post processing tail detections
    tail_detections_holder = tail_detections_processor.process_tail_detections(
        pig_detections, posture_classifications, tail_posture_detections)
    
    if save_img:
        path_to_save_image = os.path.join(path_manager.path_to_test_plots, img_name)
        print(f"Saving image to: {path_to_save_image}")
        
        # generate plot
        plot = plot_results(np.flip(img, axis=-1), pig_detections, posture_classifications, tail_detections_holder,
                            labels_posture_classification_reverse,
                            return_img=True)

        # save img
        save_array_as_image(plot, path_to_save_image)
    
    else:
        plot_results(np.flip(img, axis=-1), pig_detections, posture_classifications, tail_detections_holder,
                            labels_posture_classification_reverse,
                            return_img=False)

def plot_results(img, pig_detections, posture_classifications, tail_detections_reduced,
                 labels_posture_classification_reverse,
                 plot_postures=True,
                 plot_tails=True,
                 return_img=False):
    """
    Plots the results of pig and tail detections on the given image.

    Parameters:
        img (numpy.ndarray): The input image.
        pig_detections (list): List of pig detections.
        posture_classifications (list): List of posture classifications.
        tail_detections_reduced (list): List of reduced tail detections.
        labels_posture_classification_reverse (dict): Mapping for reversing the encoded postures.
        plot_postures (bool, optional): Flag to control whether to plot pig postures. Defaults to True.
        plot_tails (bool, optional): Flag to control whether to plot tail detections. Defaults to True.
        return_img (bool, optional): Flag to control whether to return the image with plotted results. If False, the image is displayed using plt.show(). Defaults to False.

    Returns:
        numpy.ndarray: The input image with detections plotted on it, only if return_img is set to True. Otherwise, None.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds_detection = [pred["box"] for pred in pig_detections]

    # Define label, color, patch mappings
    mapping = {
        "pigLying": {"color": (255, 255, 255), "patch": mpatches.Patch(color='white', label='pig lying')},
        "pigNotLying": {"color": (0, 0, 0), "patch": mpatches.Patch(color='black', label='pig notLying')},
        "upright": {"color": (0, 255, 0), "patch": mpatches.Patch(color='green', label='tail upright')},
        "hanging": {"color": (255, 0, 0), "patch": mpatches.Patch(color='red', label='tail hanging')}
    }

    if plot_postures:
        for i, pred in enumerate(preds_detection):
            x1, y1, x2, y2 = map(int, pred[:4])
            class_label_classification = labels_posture_classification_reverse[int(
                posture_classifications[i])]
            color = mapping[class_label_classification]["color"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if plot_tails:
        for el in tail_detections_reduced:
            box_scaled = el["box_scaled"]
            if box_scaled is not None:
                x1, y1, x2, y2 = map(int, box_scaled)
                label = el["label"]
                color = mapping[label]["color"]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    if return_img:
        img_array = np.array(img)
        return img_array
    else:
        plt.figure(figsize=(15, 10))
        patches = [v["patch"] for v in mapping.values()]
        plt.legend(handles=patches)
        plt.imshow(img)
        plt.axis("off")
        plt.show()

# Activity Index


def save_img_with_grid(frame, grid_info, output_dir, filename):
    """
    This function saves an image with a defined grid overlay.

    Parameters:
    frame: numpy array
        An image represented as a numpy array.
    grid_info: dict
        A dictionary containing the grid layout on an image. This includes 
        the number of rows and columns in the grid, and the bounding box 
        (in pixels) for each cell in the grid.
    output_dir: str
        The directory where the image with the grid overlay should be saved.
    filename: str
        The name of the file (without extension) to be saved.

    Returns:
    None
    """
    # Copy the frame to avoid modifying the original image
    img_with_grid = frame.copy()

    img_with_grid = cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2RGB)

    # Draw the grid on the copied image
    for _, bbox in grid_info['img_tiles'].items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_with_grid, (x1, y1), (x2, y2),
                      (255, 255, 255), 4)

    # Construct the file path
    filepath = os.path.join(output_dir, filename + '.png')

    # Save the image
    cv2.imwrite(filepath, img_with_grid)


def analyze_activity_changes(monitoring_result, json_path, ax=None, resample_freq="D", normalize=True, add_labels=True, rolling_window=None, fix_y_axis_max=False, save_path=None):
    """
    Plots the difference in monitored tail postures over specified intervals for a single monitoring result. The function
    handles data normalization, applies a rolling average if specified, and performs linear interpolation on the data.
    It also colors areas under the curve based on significant events related to tail biting, including pre-outbreak periods
    as indicated by the culprit removal dates, if provided. The plot visually represents the change in tail posture behaviors
    over time, highlighting critical pre-outbreak intervals (based on Larhmann et al. 2018).

    Parameters:
    - monitoring_result (dict): Contains camera, date_span, data_paths, and dataframes with tail posture data.
    - json_path (str): Path to the JSON file with metadata to determine pen type, culprit removal date as well as ground truth datespansee.
    - resample_freq (str): Frequency for data resampling (default is "D" for daily).
    - normalize (bool): If True, normalizes the tail posture data before plotting.
    - rolling_window (int or None): Specifies the window size for rolling average calculation. If None, no rolling average is applied.
    - fix_y_axis_max (bool): If True, fixes the y-axis maximum value to 1.
    - save_path (str or None): If specified, saves the generated plot to the given path.

    Returns:
    None. A plot is displayed showing the tail posture difference over time, with additional visual cues for significant periods.
    """

    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))

    subtitle_fontsize = 12  # Adjust font size for subtitles
    axis_label_fontsize = 11  # Adjust font size for axis labels
    ticks_fontsize = 11  # Adjust font size for axis ticks

    # Load JSON data from the file
    json_data = load_json_data(json_path)

    # Extracting pen info
    pen_type, culprit_removal, datespan_gt = get_pen_info(
        monitoring_result['camera'], monitoring_result['date_span'], json_data)

    camera_label = monitoring_result['camera'].replace("Kamera", "Pen ")
    formatted_datespan = format_datespan(monitoring_result['date_span'])
    formatted_datespan_gt = format_datespan(datespan_gt)

    # Concatenate dataframes from the dictionary
    data_all = pd.concat(monitoring_result['dataframes'])

    # Create a copy and set datetime index
    data_all_copy = data_all.copy()
    data_all_copy["datetime"] = pd.to_datetime(data_all_copy["datetime"])
    data_all_copy.set_index("datetime", inplace=True)

    avg_data = data_all_copy.resample(resample_freq).mean()

    # Apply rolling/moving average if specified
    if rolling_window:
        avg_data = avg_data.rolling(window=rolling_window).mean()

    # Normalize if the option is set
    if normalize:
        avg_data['activity'] = (avg_data['activity'] - avg_data['activity'].min()) / (
            avg_data['activity'].max() - avg_data['activity'].min())

    # Perform interpolation
    interpolated_data = avg_data.resample("H").interpolate(method='linear')

    ax.plot(interpolated_data.index, interpolated_data['activity'],
            label='Activity', color="black", linewidth=2)

    ax.fill_between(interpolated_data.index, interpolated_data['activity'], 0,
                    where=(interpolated_data['activity'] >= 0), color="grey", alpha=0.1)

    # Check if the pen type is "tail biting"
    if pen_type == "tail biting":
        plot_culprit_removal_dates(interpolated_data, culprit_removal, ax)

    # shade the pre-outbreak period if the pen type is "tail biting" and culprit_removal is specified
    if pen_type == "tail biting" and culprit_removal is not None:
        # 7 days pre outbreak (Lahrmann et al. 2018)
        shade_pre_outbreak_period(
            interpolated_data, culprit_removal, 7, ax, color="orange", alpha=0.2)
        # 3 days pre outbreak (Lahrmann et al. 2018)
        shade_pre_outbreak_period(
            interpolated_data, culprit_removal, 3, ax, color="red", alpha=0.2)

    ax.set_title(f"{camera_label} | {pen_type} | Full Timespan: {formatted_datespan_gt} | Analyzed Timespan: {formatted_datespan}",
                 fontsize=subtitle_fontsize)

    if add_labels:
        ax.set_xlabel('Date', fontsize=axis_label_fontsize)
        ax.set_ylabel('Acticity', fontsize=axis_label_fontsize)

    # Check if the y-axis max should be fixed
    if fix_y_axis_max:
        ax.set_ylim(top=0.95)  # Set the maximum y-axis value to 1

    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    # Remove plt.show() if ax is provided; it will be handled externally
    if ax is None:
        plt.show()

def heatmap_average_activity(dataframes_list, resample_freq='10T', days_index=None, normalize=False,
                             rolling_window=None, ewm_window=None, save_path=None):
    """
    Generates and displays a heatmap of average activity over specified intervals for selected days from multiple dataframes.

    This function processes the provided dataframes by resampling, normalizing, and optionally applying rolling or exponential
    weighted moving averages to smooth the data. The heatmap visualizes the activity levels over different times of the day
    across multiple days.

    Parameters:
    - dataframes_list (list of pd.DataFrame): List of dataframes where each dataframe represents a day's data.
    - resample_freq (str, optional): Resampling frequency for averaging the data over time intervals. Defaults to "10T" for 10-minute intervals.
    - days_index (list of int, optional): Indices specifying which days from the list to plot. If None, plots all days provided in the dataframes_list.
    - normalize (bool, optional): If True, normalizes the activity data to a range between 0 and 1.
    - rolling_window (int, optional): Size of the rolling average window. If specified, applies a rolling average to the resampled data.
    - ewm_window (int, optional): Size of the exponential weighted moving window. If specified, applies an exponential weighted mean to smooth the data.
    - save_path (str, optional): File path where the generated heatmap will be saved. If not specified, the plot is not saved.

    Returns:
    None. A heatmap is displayed, and optionally saved to the specified path.

    Side Effects:
    - Displays a heatmap using matplotlib and seaborn.
    - Optionally saves the heatmap to a file if a path is provided.
    """

    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

    # Select the dataframes based on the provided index
    selected_dataframes = [dataframes_list[i] for i in days_index]

    # Concatenate dataframes
    data_all = pd.concat(selected_dataframes)

    # Ensure 'datetime' is a datetime type and set it as the index (if it's not already)
    data_all["datetime"] = pd.to_datetime(data_all["datetime"])
    data_all.set_index("datetime", inplace=True)

    # Resample data if a frequency is provided
    if resample_freq:
        avg_data = data_all.resample(resample_freq).mean()
    else:
        avg_data = data_all.copy()  # use raw data

    # Apply rolling/moving average if specified
    if rolling_window:
        avg_data = avg_data.rolling(window=rolling_window).mean()

    # Apply exponential moving average if specified
    if ewm_window:
        avg_data['activity'] = avg_data['activity'].ewm(span=ewm_window).mean()

    # Normalize if specified
    if normalize:
        avg_data['activity'] = (avg_data['activity'] - avg_data['activity'].min()) / (
            avg_data['activity'].max() - avg_data['activity'].min())

    # Pivot the data for heatmap
    # create a new column combining hour and minute
    avg_data['hour_minute'] = avg_data.index.strftime('%H:%M')
    avg_data['date'] = avg_data.index.date
    heatmap_data = avg_data.pivot_table(
        values='activity', index='date', columns='hour_minute')

    # Convert resample_freq to a more readable format for the title
    if resample_freq.endswith('T'):
        interval = int(resample_freq[:-1])
        if interval == 60:
            readable_freq = '1 hour'
        else:
            readable_freq = f'{interval} min'
    else:
        readable_freq = resample_freq

    # Plot heatmap
    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(heatmap_data, cmap='viridis', annot=False, fmt=".2f",
                     cbar_kws={'label': 'Activity Level',
                               'orientation': 'vertical'},
                     annot_kws={"size": 14})  # Adjust size for annotations font inside the heatmap cells

    # Title and axis label modifications
    plt.title(
        f'Average Activity Across Days ({readable_freq} intervals)', fontsize=20)
    plt.xlabel('Time', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Date', fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    plt.show()

def plot_average_activity(dataframes_list, resample_freqs=None, days_index=None, normalize=False, rolling_window=None,
                          ewm_window=None, save_path=None, with_pig_posture=False, show_peaks=False):
    """
    Plot the average activity from a list of dataframes containing time-stamped data.

    This function processes and visualizes activity data by performing concatenation, resampling, normalization,
    and rolling average or exponential weighted mean computations as specified. It supports plotting raw and
    resampled data, optionally including pig posture data and marking peaks in activity.

    Parameters:
    - dataframes_list (list of pd.DataFrame): A list of dataframes each containing a 'datetime' column and other
      activity-related columns.
    - resample_freqs (list of str, optional): A list of strings denoting the resampling frequencies (e.g., 'D' for daily).
      If None, plots raw data.
    - days_index (list of int, optional): Indices of the days to plot. If None, all days are plotted.
    - normalize (bool, optional): If True, normalize the activity data to a range of 0 to 1.
    - rolling_window (int, optional): Window size for the rolling average. None if not used.
    - ewm_window (int, optional): Span for exponential weighted mean calculation. None if not used.
    - save_path (str, optional): Path to save the generated plot. If None, the plot is not saved.
    - with_pig_posture (bool, optional): If True, includes pig posture data in the plot.
    - show_peaks (bool, optional): If True, marks peaks in the activity data.

    Returns:
    None. Displays a plot and optionally saves it to a file.

    Side Effects:
    - A plot is displayed using matplotlib with optional saving to disk.
    - Prints peak activity data points if show_peaks is True.
    """

    # Concatenate dataframes
    data_all = pd.concat(dataframes_list)

    # Ensure 'datetime' is a datetime type and set it as the index (if it's not already)
    data_all["datetime"] = pd.to_datetime(data_all["datetime"])
    data_all.set_index("datetime", inplace=True)

    # When resample_freqs is None, set it to ['raw'] to denote raw data plotting
    if resample_freqs is None:
        resample_freqs = ['raw']

    # Resample the data if required
    if 'raw' not in resample_freqs:
        data_all = data_all.resample(resample_freqs[0]).mean()

    # Apply rolling/moving average if specified
    if rolling_window:
        data_all = data_all.rolling(window=rolling_window).mean()

    if ewm_window:
        data_all['activity'] = data_all['activity'].ewm(span=ewm_window).mean()

    # Calculate the proportions
    if normalize:
        data_all['activity'] = (data_all['activity'] - data_all['activity'].min()) / (
            data_all['activity'].max() - data_all['activity'].min())

    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

    # Select the dataframes based on the provided index AFTER all calculations are done
    selected_dataframes = [data_all[data_all.index.date ==
                                    dataframes_list[i]["datetime"].iloc[0].date()] for i in days_index]

    # Concatenate selected dataframes
    avg_data_all = pd.concat(selected_dataframes)

    # Create subplots based on the number of frequencies specified
    fig, axes = plt.subplots(nrows=len(resample_freqs),
                             ncols=1, figsize=(18, 6 * len(resample_freqs)))

    # Determine columns to plot based on with_pig_posture
    columns_to_plot = ['activity']
    if with_pig_posture:
        columns_to_plot.extend(['num_pigs_lying', 'num_pigs_notLying'])

    # If there's only one frequency, make axes an array for consistent indexing
    if len(resample_freqs) == 1:
        axes = [axes]

    # Get the unique dates for x-ticks
    # unique_dates = avg_data_all.index.normalize().unique()[::3]

    for i, freq in enumerate(resample_freqs):
        # For raw data, use the avg_data_all directly
        if freq == 'raw':
            title_freq = "Raw Activity"
            avg_data = avg_data_all.copy()
        else:
            if resample_freqs[i] == 'D':
                title_freq = "Average Daily Activity"
            elif resample_freqs[i] == 'H':
                title_freq = "Average Hourly Activity"
            elif resample_freqs[i] == 'T':
                title_freq = "Average Minutly Activity"
            avg_data = avg_data_all.copy()

        # Creating a secondary y-axis if plotting with pig postures
        if with_pig_posture:
            ax2 = axes[i].twinx()
            color_idx = 0
            colors = ["green", "blue"]
        else:
            ax2 = None

        # Plotting each column
        for col in columns_to_plot:

            # If this is the 'activity' column, and show_peaks is True, identify and mark peaks
            if col == 'activity' and show_peaks:
                peaks, _ = find_peaks(avg_data[col].values, prominence=0.3)
                axes[i].plot(avg_data.index[peaks],
                             avg_data[col].iloc[peaks], 'rx')

                for peak in peaks:
                    print(
                        f"Highpoint Datetime: {avg_data.index[peak]}, Value: {avg_data[col].iloc[peak]}")

                # Plotting data on the primary y-axis
                axes[i].plot(avg_data.index, avg_data[col],
                             linewidth="5", color="black", label="activity")
            elif col == 'activity':
                axes[i].plot(avg_data.index, avg_data[col],
                             linewidth="5", color="black", label="activity")

            # Handle the posture columns
            if with_pig_posture and col in ['num_pigs_lying', 'num_pigs_notLying']:
                ax2.plot(avg_data.index, avg_data[col], label=col)
                ax2.legend(loc='upper right', fontsize=16)

        # Add titles, legends, labels, etc.
        axes[i].set_title(title_freq, fontsize=20)
        axes[i].set_xlabel('Date', fontsize=18)
        axes[i].set_ylabel('Activity Level', fontsize=16)
        if ax2:
            ax2.set_ylabel('Number of Pigs', fontsize=16)

    # Common X label for the last subplot
    axes[-1].set_xlabel('Date', fontsize=18)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()  # Ensure the plots are spaced nicely

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()



# Data analysis (monitoring pipeline results)


def get_plot_title(resample_freq, normalize=False, rolling_window=None):
    """
    Generate a descriptive plot title based on resampling frequency, normalization, and rolling window parameters.

    Parameters:
    - resample_freq (str): The resampling frequency used to aggregate the data. Common values might include "10T" for 10 minutes, "1H" for 1 hour, and "D" for daily.
    - normalize (bool, optional): Indicates whether the data was normalized. Defaults to False.
    - rolling_window (int or None, optional): Specifies the size of the moving average window, if applied. None means no rolling window was applied. 

    Returns:
    str: A descriptive title suitable for use in plots.

    Example:
    If resample_freq="10T", normalize=True, and rolling_window=3, the returned title would be "10 Minute Average with Rolling Window (3 periods) and Normalization".
    """

    # Mapping of resample frequencies to titles
    freq_map = {
        "10T": "10 Minute Average",
        "30T": "30 Minute Average",
        "1H": "1 Hour Average",
        "3H": "3 Hour Average",
        "D": "Daily Average"
    }

    # Use the resampling frequency itself as default if not found in the map
    title_freq = freq_map.get(resample_freq, resample_freq)

    extras = []

    # Only add rolling window to title if freq is not "D"
    if rolling_window and resample_freq != "D":
        extras.append(f"Moving Average ({rolling_window})")

    if normalize:
        extras.append("Normalization")

    if extras:
        title_freq += " with " + " and ".join(extras)

    return title_freq


def plot_average_tail_postures(dataframes_list, resample_freqs=["10T"], days_index=None, normalize=False, rolling_window=None, save_path=None):
    """
    Plot average tail postures over specified intervals.

    Parameters:
    - dataframes_list (list of pd.DataFrame): List of dataframes, each representing a day's data.
    - days_index (list of int, optional): List of indices specifying which days from the list to plot. Defaults to None, which plots all days.
    - normalize (bool, optional): If True, normalizes the data based on the number of detected tails.
    - resample_freqs (list of str, optional): List of resampling frequencies. Defaults to ["10T"] for 10 minutes.
    - rolling_window (int, optional): Size of the moving average window. If provided, applies moving average smoothing to the resampled data.

    Returns:
    None. Displays a multi-subplot figure.
    """
    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

        # Select the dataframes based on the provided index
    selected_dataframes = [dataframes_list[i] for i in days_index]

    # Concatenate dataframes
    data_all = pd.concat(selected_dataframes)

    # Ensure 'datetime' is a datetime type and set it as the index (if it's not already)
    data_all["datetime"] = pd.to_datetime(data_all["datetime"])
    data_all.set_index("datetime", inplace=True)

    # Create subplots based on the number of frequencies specified
    fig, axes = plt.subplots(nrows=len(resample_freqs),
                             ncols=1, figsize=(18, 6 * len(resample_freqs)))

    # If there's only one frequency, make axes an array for consistent indexing
    if len(resample_freqs) == 1:
        axes = [axes]

    # Get the unique dates for x-ticks
    unique_dates = data_all.index.normalize().unique()[::3]

    # Calculate the proportions before resampling
    if normalize:
        data_all["num_tails_hanging"] = data_all["num_tails_hanging"] / \
            data_all["num_tail_detections"]
        data_all["num_tails_upright"] = data_all["num_tails_upright"] / \
            data_all["num_tail_detections"]

    # Loop through each frequency and plot accordingly
    for i, freq in enumerate(resample_freqs):

        # get cleaned freq for title
        title_freq = get_plot_title(freq, normalize, rolling_window)

        # now, this will give the average proportion for the resampled period
        avg_data = data_all.resample(freq).mean()

        # Apply rolling/moving average if specified
        if rolling_window:
            if freq == "D":
                pass
            else:
                avg_data = avg_data.rolling(window=rolling_window).mean()

        # Plotting data
        axes[i].plot(avg_data.index, avg_data["num_tails_hanging"],
                     color="red", label=f'AVG tails hanging ({freq})')
        axes[i].plot(avg_data.index, avg_data["num_tails_upright"],
                     color="green", label=f'AVG tails upright ({freq})')

        axes[i].set_title(title_freq)
        axes[i].legend()
        axes[i].set_ylabel(
            'Tail Postures (Normalized)' if normalize else 'Tail Postures')

        # Set x-ticks for each day
        axes[i].set_xticks(unique_dates)
        axes[i].set_xticklabels([date.strftime('%Y-%m-%d')
                                for date in unique_dates], rotation=45)

    # Common X label for the last subplot
    axes[-1].set_xlabel('Datum')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Ensure the plots are spaced nicely

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


def plot_tail_posture_difference(dataframes_list, days_index=None, resample_freqs=["10T"], normalize=False, rolling_window=None, save_path=None):
    """
    Plot the difference in monitored tail postures over specified intervals.

    Parameters:
    - dataframes_list (list of pd.DataFrame): List of dataframes, each representing a day's data.
    - days_index (list of int, optional): List of indices specifying which days from the list to plot. Defaults to None, which plots all days.
    - resample_freqs (list of str, optional): List of resampling frequencies. Defaults to ["10T"] for 10 minutes.
    - normalize (bool, optional): Normalize the tail postures data based on total detections.
    - rolling_window (int, optional): Size of the moving average window. If provided, applies moving average smoothing to the resampled data.

    Returns:
    None. Displays a multi-subplot figure showing the difference between 'num_tails_upright' and 'num_tails_hanging'.
    """

    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

    # Select the dataframes based on the provided index
    selected_dataframes = [dataframes_list[i] for i in days_index]

    # Concatenate selected dataframes
    data_all = pd.concat(selected_dataframes)

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

    # Create subplots based on the number of frequencies specified
    fig, axes = plt.subplots(nrows=len(resample_freqs),
                             ncols=1, figsize=(18, 6 * len(resample_freqs)))

    # If there's only one frequency, make axes an array for consistent indexing
    if len(resample_freqs) == 1:
        axes = [axes]

    # Generate x-ticks: dates spaced every 3 days within the range of your data
    xticks = pd.date_range(start=data_all_copy.index.min(),
                           end=data_all_copy.index.max(), freq='3D')

    for i, freq in enumerate(resample_freqs):

        # get cleaned freq for title
        title_freq = get_plot_title(freq, normalize, rolling_window)

        avg_data = data_all_copy.resample(freq).mean()
        avg_data['posture_diff'] = avg_data['num_tails_upright'] - \
            avg_data['num_tails_hanging']

        # Apply rolling/moving average if specified
        if rolling_window:
            if freq == "D":
                pass
            else:
                avg_data = avg_data.rolling(window=rolling_window).mean()

        # Plotting
        axes[i].plot(avg_data.index, avg_data['posture_diff'])
        axes[i].set_title(str(title_freq) +
                          ": Difference between Upright and Hanging Tails")
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].set_ylabel(
            'Posture Difference (Normalized)' if normalize else 'Posture Difference')
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticks.strftime('%Y-%m-%d'), rotation=45)

    # Common X label for the last subplot
    axes[-1].set_xlabel('Date and Time')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


def plot_rate_of_change(dataframes_list, resample_freqs=["10T"], days_index=None, normalize=False, rolling_window=None, save_path=None):
    """
    Plot rate of change in tail postures over specified intervals.

    Parameters:
    - dataframes_list (list of pd.DataFrame): List of dataframes, each representing a day's data.
    - days_index (list of int, optional): List of indices specifying which days from the list to plot. Defaults to None, which plots all days.
    - normalize (bool, optional): If True, normalizes the data based on the number of detected tails.
    - resample_freqs (list of str, optional): List of resampling frequencies. Defaults to ["10T"] for 10 minutes.

    Returns:
    None. Displays a multi-subplot figure.
    """

    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

    # Select the dataframes based on the provided index
    selected_dataframes = [dataframes_list[i] for i in days_index]

    # Concatenate dataframes
    data_all = pd.concat(selected_dataframes)
    data_all["datetime"] = pd.to_datetime(data_all["datetime"])
    data_all.set_index("datetime", inplace=True)

    # Normalize if the option is set
    if normalize:
        data_all["num_tails_hanging"] = data_all["num_tails_hanging"] / \
            data_all["num_tail_detections"]
        data_all["num_tails_upright"] = data_all["num_tails_upright"] / \
            data_all["num_tail_detections"]

    # Create subplots based on the number of frequencies specified
    fig, axes = plt.subplots(nrows=len(resample_freqs),
                             ncols=1, figsize=(18, 6 * len(resample_freqs)))

    # If there's only one frequency, make axes an array for consistent indexing
    if len(resample_freqs) == 1:
        axes = [axes]

    # Loop through each frequency and plot accordingly
    for i, freq in enumerate(resample_freqs):

        title_freq = get_plot_title(freq, normalize)

        avg_data = data_all.resample(freq).mean()

        # Apply rolling/moving average if specified
        if rolling_window:
            if freq == "D":
                pass
            else:
                avg_data = avg_data.rolling(window=rolling_window).mean()

        # Compute the difference for rate of change
        avg_data['upright_diff'] = avg_data['num_tails_upright'].diff()
        avg_data['hanging_diff'] = avg_data['num_tails_hanging'].diff()

        # Plotting
        axes[i].plot(avg_data.index, avg_data['upright_diff'],
                     color='green', label='Rate of Change (Upright)')
        axes[i].plot(avg_data.index, avg_data['hanging_diff'],
                     color='red', label='Rate of Change (Hanging)')

        axes[i].set_title(title_freq)
        axes[i].axhline(0, color='blue', linestyle='--')
        axes[i].legend()
        axes[i].set_ylabel('Rate of Change')

        unique_dates = avg_data.index.normalize().unique()[::3]
        axes[i].set_xticks(unique_dates)
        axes[i].set_xticklabels([date.strftime('%Y-%m-%d')
                                for date in unique_dates], rotation=45)

    # Common X label for the last subplot
    axes[-1].set_xlabel('Datum')
    plt.tight_layout()  # Ensure the plots are spaced nicely

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


def plot_rate_of_change_tail_posture_difference(dataframes_list, resample_freqs=["10T"], days_index=None, normalize=False, rolling_window=None, save_path=None):
    """
    Plot rate of change in the difference between tail postures over specified intervals.

    Parameters:
    - dataframes_list (list of pd.DataFrame): List of dataframes, each representing a day's data.
    - days_index (list of int, optional): List of indices specifying which days from the list to plot. Defaults to None, which plots all days.
    - normalize (bool, optional): If True, normalizes the data based on the number of detected tails.
    - resample_freqs (list of str, optional): List of resampling frequencies. Defaults to ["10T"] for 10 minutes.

    Returns:
    None. Displays a multi-subplot figure.
    """

    # If no specific index is provided, plot all days
    if days_index is None:
        days_index = range(len(dataframes_list))

    # Select the dataframes based on the provided index
    selected_dataframes = [dataframes_list[i] for i in days_index]

    # Concatenate dataframes
    data_all = pd.concat(selected_dataframes)
    data_all["datetime"] = pd.to_datetime(data_all["datetime"])
    data_all.set_index("datetime", inplace=True)

    # Normalize if the option is set
    if normalize:
        data_all["num_tails_hanging"] = data_all["num_tails_hanging"] / \
            data_all["num_tail_detections"]
        data_all["num_tails_upright"] = data_all["num_tails_upright"] / \
            data_all["num_tail_detections"]

    # Compute the difference in tail postures
    data_all['posture_diff'] = data_all['num_tails_upright'] - \
        data_all['num_tails_hanging']

    # Create subplots based on the number of frequencies specified
    fig, axes = plt.subplots(nrows=len(resample_freqs),
                             ncols=1, figsize=(18, 6 * len(resample_freqs)))

    # If there's only one frequency, make axes an array for consistent indexing
    if len(resample_freqs) == 1:
        axes = [axes]

    # Loop through each frequency and plot accordingly
    for i, freq in enumerate(resample_freqs):

        title_freq = get_plot_title(freq, normalize)

        avg_data = data_all.resample(freq).mean()

        # Apply rolling/moving average if specified
        if rolling_window:
            if freq == "D":
                pass
            else:
                avg_data = avg_data.rolling(window=rolling_window).mean()

        # Compute the rate of change for the posture difference
        avg_data['posture_diff_rate'] = avg_data['posture_diff'].diff()

        # Plotting
        axes[i].plot(avg_data.index, avg_data['posture_diff_rate'],
                     color='blue', label='Rate of Change (Difference)')
        axes[i].set_title(
            title_freq + ": Rate of Change for Difference between Upright and Hanging Tails")
        axes[i].axhline(0, color='red', linestyle='--')
        axes[i].legend()
        axes[i].set_ylabel('Rate of Change for Posture Difference')

        unique_dates = avg_data.index.normalize().unique()[::3]
        axes[i].set_xticks(unique_dates)
        axes[i].set_xticklabels([date.strftime('%Y-%m-%d')
                                for date in unique_dates], rotation=45)

    # Common X label for the last subplot
    axes[-1].set_xlabel('Datum')
    plt.tight_layout()  # Ensure the plots are spaced nicely

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


def plot_culprit_removal_dates(interpolated_data, culprit_removal, ax):
    """
    Plots vertical lines on a given Matplotlib Axes (ax) to indicate the dates when culprits were removed 
    in tail biting pens. This function is designed to visually mark significant events directly on time series plots,
    aiding in the analysis of temporal patterns relative to these events.

    Parameters:
    - interpolated_data (pd.DataFrame): The DataFrame containing the interpolated time series data. 
      This data is used to ensure that the vertical lines for culprit removal are aligned with the data's time axis. 
      Note: The DataFrame is expected to have a datetime index.
    - culprit_removal (str, list of str, or None): The date(s) when culprits were removed. 
      Can be a single date string (YYYY-MM-DD), a list of such strings, or None if there's no culprit removal to plot.
      If "Unknown", no line is plotted.
    - ax (matplotlib.axes.Axes): The Matplotlib Axes object where the vertical lines will be plotted. 
      This allows the function to integrate with figures that contain multiple subplots.

    Returns:
    None: This function directly modifies the Matplotlib Axes object passed as a parameter by adding vertical lines 
    to it and does not return any value.
    """
    if isinstance(culprit_removal, list):
        for removal_date in culprit_removal:
            removal_date = pd.to_datetime(removal_date)
            ax.axvline(x=removal_date, linestyle='dotted',
                       linewidth=3, color="grey")
    elif culprit_removal != "Unknown":
        culprit_removal_date = pd.to_datetime(culprit_removal)
        ax.axvline(x=culprit_removal_date, linestyle='dotted',
                   linewidth=3, color="grey")


def shade_pre_outbreak_period(interpolated_data, culprit_removal_date, num_days, ax, type="posture_diff", color="red", alpha=0.2):
    """
    Shades the area on a plot to highlight the period leading up to a specified event, typically used
    to mark a pre-outbreak period in tail biting studies. This visualization aid helps in identifying 
    changes in the monitored data leading up to significant events.

    Parameters:
    - interpolated_data (pd.DataFrame): The DataFrame containing the interpolated time series data. 
      This data is used to ensure that the shading aligns with the data's time axis. 
      The DataFrame is expected to have a datetime index.
    - culprit_removal_date (str or list of str): The date of the first culprit removal, 
      specified in "YYYY-MM-DD" format. If a list of dates is provided, only the first date is used. 
      This date marks the end of the pre-outbreak period to be shaded.
    - num_days (int): The number of days before the culprit_removal_date to start shading. 
      This defines the length of the pre-outbreak period.
    - ax (matplotlib.axes.Axes): The Matplotlib Axes object where the shading will be applied. 
      This allows the function to integrate with figures that contain multiple subplots.
    - color (str, optional): The color of the shaded area. Defaults to "red".
    - alpha (float, optional): The transparency level of the shaded area. Defaults to 0.2.

    Returns:
    None: This function directly modifies the Matplotlib Axes object passed as a parameter by adding 
    a shaded area to it and does not return any value.
    """
    if isinstance(culprit_removal_date, list) and len(culprit_removal_date) > 0:
        culprit_removal_date = culprit_removal_date[0]
    
    if culprit_removal_date == "Unknows":
        pass

    if culprit_removal_date is not None:
        culprit_removal_datetime = pd.to_datetime(culprit_removal_date)
        start_date = culprit_removal_datetime - pd.Timedelta(days=num_days)
        mask = (interpolated_data.index >= start_date) & (
            interpolated_data.index <= culprit_removal_datetime)
        try:
            ax.fill_between(interpolated_data.index, interpolated_data['posture_diff'], 0,
                            where=mask, color=color, alpha=alpha)
        except:
            ax.fill_between(interpolated_data.index, interpolated_data['activity'], 0,
                where=mask, color=color, alpha=alpha)


def analyze_tail_posture_changes(monitoring_result, json_path, ax=None, resample_freq="D", normalize=False, add_labels=True, rolling_window=None, fix_y_axis_max=False, save_path=None):
    """
    Plots the difference in monitored tail postures over specified intervals for a single monitoring result. The function
    handles data normalization, applies a rolling average if specified, and performs linear interpolation on the data.
    It also colors areas under the curve based on significant events related to tail biting, including pre-outbreak periods
    as indicated by the culprit removal dates, if provided. The plot visually represents the change in tail posture behaviors
    over time, highlighting critical pre-outbreak intervals (based on Larhmann et al. 2018).

    Parameters:
    - monitoring_result (dict): Contains camera, date_span, data_paths, and dataframes with tail posture data.
    - json_path (str): Path to the JSON file with metadata to determine pen type, culprit removal date as well as ground truth datespansee.
    - resample_freq (str): Frequency for data resampling (default is "D" for daily).
    - normalize (bool): If True, normalizes the tail posture data before plotting.
    - rolling_window (int or None): Specifies the window size for rolling average calculation. If None, no rolling average is applied.
    - fix_y_axis_max (bool): If True, fixes the y-axis maximum value to 1.
    - save_path (str or None): If specified, saves the generated plot to the given path.

    Returns:
    None. A plot is displayed showing the tail posture difference over time, with additional visual cues for significant periods.
    """

    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))

    subtitle_fontsize = 12  # Adjust font size for subtitles
    axis_label_fontsize = 11  # Adjust font size for axis labels
    ticks_fontsize = 11  # Adjust font size for axis ticks

    # Load JSON data from the file
    json_data = load_json_data(json_path)

    # Extracting pen info
    pen_type, culprit_removal, datespan_gt = get_pen_info(
        monitoring_result['camera'], monitoring_result['date_span'], json_data)

    camera_label = monitoring_result['camera'].replace("Kamera", "Pen ")
    formatted_datespan = format_datespan(monitoring_result['date_span'])
    formatted_datespan_gt = format_datespan(datespan_gt)

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

    ax.plot(interpolated_data.index, interpolated_data['posture_diff'],
            label='Tail Posture Difference', color="black", linewidth=2)
    
    # Set the formatter for the x-axis to display dates as 'Month-Day'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    ax.fill_between(interpolated_data.index, interpolated_data['posture_diff'], 0,
                    where=(interpolated_data['posture_diff'] >= 0), color="grey", alpha=0.1)
    ax.fill_between(interpolated_data.index, interpolated_data['posture_diff'], 0,
                    where=(interpolated_data['posture_diff'] < 0), color="grey", alpha=0.2)

    ax.axhline(0, color='red', linestyle='--')  # Zero line

    # Check if the pen type is "tail biting"
    if pen_type == "tail biting":
        plot_culprit_removal_dates(interpolated_data, culprit_removal, ax)

    # shade the pre-outbreak period if the pen type is "tail biting" and culprit_removal is specified
    if pen_type == "tail biting" and culprit_removal is not None:
        # 7 days pre outbreak (Lahrmann et al. 2018)
        shade_pre_outbreak_period(
            interpolated_data, culprit_removal, 7, ax, color="orange", alpha=0.2)
        # 3 days pre outbreak (Lahrmann et al. 2018)
        shade_pre_outbreak_period(
            interpolated_data, culprit_removal, 3, ax, color="red", alpha=0.2)

    ax.set_title(f"{camera_label} | {pen_type} | Full Timespan: {formatted_datespan_gt} | Analyzed Timespan: {formatted_datespan}",
                 fontsize=subtitle_fontsize)

    if add_labels:
        ax.set_xlabel('Date', fontsize=axis_label_fontsize)
        ax.set_ylabel('Tail Posture Difference', fontsize=axis_label_fontsize)

    # Check if the y-axis max should be fixed
    if fix_y_axis_max:
        ax.set_ylim(top=0.95)  # Set the maximum y-axis value to 1

    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    # Remove plt.show() if ax is provided; it will be handled externally
    if ax is None:
        plt.show()


def generate_tail_posture_subplots(pipeline_monitoring_results, defined_cameras, json_path, resample_freq="D", skip_datespans=None, normalize=False, rolling_window=None, save_path=None):
    """
    Generates subplots for each camera and its monitoring results, organizing the plots in a grid
    where each column represents a camera and each row represents a datespan for that camera.

    Parameters:
    - pipeline_monitoring_results (list): List of dicts, each dict containing monitoring results for a camera.
    - defined_cameras (list): List of camera names to be plotted.
    - json_path (str): Path to the JSON file with metadata.
    - skip_datespans (list, optional): Datespans to skip. Can be a single datespan or a list of datespans.
    - Other parameters are passed directly to analyze_tail_posture_changes.
    """

    num_datespans = 8 if not skip_datespans else 8 - len(skip_datespans)
    num_cameras = len(defined_cameras)

    fig, axs = plt.subplots(num_datespans, num_cameras, figsize=(
        6 * num_cameras, 3 * num_datespans), squeeze=False)

    # Adjust the fontsize here for the titles
    camera_title_fontsize = 16
    common_label_fontsize = 16

    # Calculate the x position for the camera titles
    # Distribute the titles evenly across the width of the figure
    title_positions = [(0.55/num_cameras) + (i * (1/num_cameras))
                       for i in range(num_cameras)]

    # Define letters for subplot labeling
    letters = string.ascii_lowercase

    for i, camera in enumerate(defined_cameras):
        camera_results = [
            result for result in pipeline_monitoring_results if result['camera'] == camera]

        # Sort the results by date_span to ensure the plots are in chronological order
        camera_results = sorted(camera_results, key=lambda x: x['date_span'])

        # Transform camera name
        batch_label = camera.replace("Kamera", "Pen ")

        # Optionally, add the camera name above each column
        fig.text(title_positions[i], 0.926,
                 f"{batch_label}", ha='center', va='top', fontsize=camera_title_fontsize)

        j_adjustment = 0  # Initialize the adjustment for skipped elements

        for j, result in enumerate(camera_results):

            # Extract additional info needed for the subplot titles
            # Assuming this function is defined elsewhere and loads the JSON data
            json_data = load_json_data(json_path)
            # Extracting pen info
            pen_type, culprit_removal, datespan_gt = get_pen_info(
                result['camera'], result['date_span'], json_data)

            # Skip the result if its datespan matches any in skip_datespans
            if skip_datespans and datespan_gt in skip_datespans:
                j_adjustment += 1  # Increase the adjustment
                num_datespans -= 1  # Update the number of datespans
                continue

            # Adjust the j index by the current adjustment
            j -= j_adjustment

            ax = axs[j, i]  # j for datespan, i for camera

            formatted_datespan = format_datespan(result['date_span'])
            formatted_datespan_gt = format_datespan(datespan_gt)

            analyze_tail_posture_changes(
                result, json_path, ax=ax, resample_freq=resample_freq, normalize=normalize, add_labels=False, fix_y_axis_max=True, rolling_window=rolling_window)

            # Set subplot title with pen_type, formatted_datespan_gt, and formatted_datespan
            subplot_label = f"{j+1}{letters[i]}"
            if j == 0:
                # For the first row, camera name is aleady on top, just include the detailed info
                ax.set_title(
                    f"{subplot_label} | {pen_type} | {formatted_datespan_gt} | {formatted_datespan}", fontsize=10)
            else:
                ax.set_title(
                    f"{subplot_label} |  {pen_type} | {formatted_datespan_gt} | {formatted_datespan}", fontsize=10)

    # Set common axis labels for the entire subplot grid
    fig.text(0.5, -0.0025, 'Date', ha='center', va='center',
             fontsize=16, transform=fig.transFigure)
    fig.supylabel('Tail Posture Difference', fontsize=common_label_fontsize)

    # Adjust layout to make room for the top titles and the main title
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        if not save_path.endswith('.pdf'):
            save_path += '.pdf'
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

        import pandas as pd

def tail_posture_changes_to_df(pipeline_monitoring_results, json_path, resample_freq="D", normalize=False, rolling_window=None, filter_cameras=None):
    """
    Analyzes tail posture changes from a set of monitoring results and returns the analysis as a pandas DataFrame.
    
    This function processes monitoring results related to tail posture, filtering for specific cameras if required, 
    and calculates the difference in tail posture over time. It specifically looks for "tail biting" pens and 
    calculates tail posture difference scores at the day of culprit removal and 7 days prior to it. Results for each 
    camera are compiled into a DataFrame.
    
    Parameters:
    - pipeline_monitoring_results (list): A list of dictionaries, where each dictionary contains monitoring results
      for a camera. Each dictionary must have keys 'camera', 'date_span', and 'dataframes', where 'dataframes' is a list
      of pandas DataFrames with columns including 'datetime', 'num_tails_hanging', and 'num_tail_detections'.
    - json_path (str): Path to the JSON file containing metadata to determine pen type, culprit removal dates, and other 
      relevant information.
    - resample_freq (str, optional): Frequency for data resampling. Default is "D" for daily.
    - normalize (bool, optional): If set to True, normalizes the tail posture data before plotting. Default is False.
    - rolling_window (int or None, optional): Size of the rolling window to apply for a moving average calculation.
      If None, no rolling average is applied. Default is None.
    - filter_cameras (list of str or None, optional): A list of camera names to include in the analysis. If None, 
      all cameras in the input data are included. Default is None.
    
    Returns:
    - pandas.DataFrame: A DataFrame containing the analysis results. Columns include 'camera', 'date_span,culprit_removal_date', 
      'score_at_removal', 'score_7_days_prior', 'num_tails_upright_at_removal', 'num_tails_hanging_at_removal', 
      'num_tails_upright_7_days_prior', 'num_tails_hanging_7_days_prior', 'days_from_negative_posture_difference_to_culprit_removal'
      Each row corresponds to a camera included 
      in the analysis.
    
    Note:
    - The function only includes pens identified as "tail biting" in the final DataFrame.
    - If the score at the day of culprit removal is not available, the last available 'posture_diff' value is used.
    """
    results_to_save = []

    for monitoring_result in pipeline_monitoring_results:
        if filter_cameras is not None and monitoring_result['camera'] not in filter_cameras:
            continue
        
        json_data = load_json_data(json_path)
        pen_type, culprit_removal, _ = get_pen_info(monitoring_result['camera'], monitoring_result['date_span'], json_data)

        if pen_type != "tail biting":
            continue  # Process only "tail biting" pens

        data_all = pd.concat(monitoring_result['dataframes'])
        data_all["datetime"] = pd.to_datetime(data_all["datetime"])
        data_all.set_index("datetime", inplace=True)

        if normalize:
            data_all["num_tails_hanging"] /= data_all["num_tail_detections"]
            data_all["num_tails_upright"] /= data_all["num_tail_detections"]

        avg_data = data_all.resample(resample_freq).mean()
        avg_data['posture_diff'] = avg_data['num_tails_upright'] - avg_data['num_tails_hanging']

        if rolling_window:
            avg_data = avg_data.rolling(window=rolling_window).mean()

        interpolated_data = avg_data.resample("H").interpolate(method='linear')

        record = {
            'camera': monitoring_result['camera'],
            'date_span': monitoring_result['date_span'],
            'culprit_removal_date': None,
            'score_at_removal': None,
            'score_7_days_prior': None,
            'num_tails_upright_at_removal': None,  
            'num_tails_hanging_at_removal': None,
            'num_tails_upright_7_days_prior': None,
            'num_tails_hanging_7_days_prior': None,
            'days_from_negative_posture_difference_to_culprit_removal': None
        }

        if culprit_removal:
            culprit_removal_date = pd.to_datetime(culprit_removal, format='%Y-%m-%d')

            if isinstance(culprit_removal_date, pd.DatetimeIndex):
                culprit_removal_date = culprit_removal_date[0]

            record['culprit_removal_date'] = culprit_removal_date.strftime('%Y-%m-%d')

            # Find the first day with a negative posture difference before the removal date
            before_removal = interpolated_data[interpolated_data.index < culprit_removal_date]
            first_negative_posture_diff = before_removal[before_removal['posture_diff'] < 0].first_valid_index()
            # Calculate the day before first_negative_posture_diff
            day_before_first_negative_posture_diff = first_negative_posture_diff - pd.Timedelta(days=1)

            if first_negative_posture_diff is not None:
                days_diff = (culprit_removal_date - day_before_first_negative_posture_diff).days
                record['days_from_negative_posture_difference_to_culprit_removal'] = days_diff

            if culprit_removal_date in interpolated_data.index:
                record['score_at_removal'] = interpolated_data.loc[culprit_removal_date, 'posture_diff']
                record['num_tails_upright_at_removal'] = interpolated_data.loc[culprit_removal_date, 'num_tails_upright']
                record['num_tails_hanging_at_removal'] = interpolated_data.loc[culprit_removal_date, 'num_tails_hanging']
            else:
                last_valid_index = interpolated_data['posture_diff'].last_valid_index()
                if last_valid_index is not None:
                    record['score_at_removal'] = interpolated_data.loc[last_valid_index, 'posture_diff']
                    # Assuming you want to capture these values only if the score_at_removal is available
                    record['num_tails_upright_at_removal'] = interpolated_data.loc[last_valid_index, 'num_tails_upright']
                    record['num_tails_hanging_at_removal'] = interpolated_data.loc[last_valid_index, 'num_tails_hanging']

            seven_days_prior = culprit_removal_date - pd.Timedelta(days=7)
            if seven_days_prior in interpolated_data.index:
                record['score_7_days_prior'] = interpolated_data.loc[seven_days_prior, 'posture_diff']
                record['num_tails_upright_7_days_prior'] = interpolated_data.loc[seven_days_prior, 'num_tails_upright']
                record['num_tails_hanging_7_days_prior'] = interpolated_data.loc[seven_days_prior, 'num_tails_hanging']

            results_to_save.append(record)
    
    return pd.DataFrame(results_to_save)