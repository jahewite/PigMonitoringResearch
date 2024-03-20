import os
import cv2
import json
import glob
import datetime
import subprocess
import numpy as np

from datetime import datetime, timedelta
from pipeline.utils.path_manager import PathManager

path_manager = PathManager()


def get_video_files(directory):
    """
    Returns all video files in a directory and its subdirectories.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        list: List of paths to video files ending in .mp4 or .avi.
    """

    # Join the directory with the patterns '**/*.mp4' and '**/*.avi' to search for video files recursively
    mp4_files = sorted(glob.glob(os.path.join(
        directory, '**/*.mp4'), recursive=True))
    avi_files = sorted(glob.glob(os.path.join(
        directory, '**/*.avi'), recursive=True))

    # Concatenate the lists of mp4 and avi files and return the result
    return mp4_files + avi_files


def get_time_of_video_capture(path_to_video):
    """
    Returns time of video capture in the format "2022-01-01T00:00:00.000000Z".
    Handles cases where 'creation_time' metadata might not be available.

    Args:
        path_to_video (str): Path to the video file

    Returns:
        datetime: Datetime object representing the creation time of the video or None
    """
    try:
        # Run ffprobe command to extract video metadata in json format
        command = f'ffprobe -v quiet -print_format json -show_format {path_to_video}'
        result = subprocess.check_output(command, shell=True)

        # Parse the json result to extract creation time
        metadata = json.loads(result)['format']
        if 'creation_time' in metadata['tags']:
            time_of_video_capture = metadata['tags']['creation_time']
            format_str = '%Y-%m-%dT%H:%M:%S.%fZ'
            return datetime.strptime(time_of_video_capture, format_str)
        else:
            return None
    except Exception as e:
        return None


def get_frame_of_video(videofile, frame_num=0, save_img=False):
    '''
    Returns the specified frame of a video.

    Args:
        videofile (str): Path to the video file
        frame_num (int): Frame number to extract (default is 0 for the first frame)

    Returns:
        img (numpy array): The specified frame as a numpy array, or None if unsuccessful
    '''
    # Open the video file
    cap = cv2.VideoCapture(videofile)

    # Set the position of the next frame to read
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Read the specified frame
    success, frame = cap.read()

    # If successful, flip the frame colors from BGR to RGB, release the video file, and return the frame
    if success:
        img = np.flip(frame, axis=-1)

        if save_img:
            output_dir = os.path.join(path_manager.path_to_config_files, "example_images_from_every_camera/other")

            img_filename = f"{os.path.splitext(os.path.basename(videofile))[0]}_frame{frame_num}.png"
            img_filepath = os.path.join(output_dir, img_filename)

            cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Image saved to {img_filepath}")
        cap.release()
        return img
    else:
        print(f"Failed to read frame {frame_num}")
        cap.release()
        return None


def get_total_number_of_frames_of_videocap(path_to_video):
    """
    Returns the total number of frames in a video file, with exception handling.

    Args:
        path_to_video (str): Path to the video file.

    Returns:
        int: Total number of frames in the video, or None if an error occurs.
    """
    try:
        command = f'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {path_to_video}'
        result = subprocess.check_output(command, shell=True)
        return int(result)
    except Exception as e:
        print(f"[ERROR] [Video] [METADATA] Failed to count frames for {path_to_video}: {e}")
        return None


def format_pipeline_result_filename(path_to_video_dir):
    """
    Format the pipeline result filename based on the videos in the directory.

    Args:
        path_to_video_dir (str): Path to the directory containing the videos.

    Returns:
        str: Formatted filename for the pipeline result.
    """

    # Get a list of all video files in the directory
    videos = get_video_files(path_to_video_dir)

    # Extract the filenames from the full file paths
    video_names = [video.split("/")[-1] for video in videos]

    # Identify the first and last videos in the list (sorted by filename)
    first_video = video_names[0]
    last_video = video_names[-1]

    # Split the video filenames into parts
    first_video_parts = first_video.split("-")
    last_video_parts = last_video.split("-")

    # Extract the camera name, date, and start and end times from the filenames
    camera = first_video_parts[0]
    date = first_video_parts[1]
    start_time = first_video_parts[2]
    end_time = last_video_parts[2]

    # Convert the date, and times to the desired formats
    # defaults to original name if not in mapping
    formatted_date = f'{date[:4]}_{date[4:6]}_{date[6:]}'
    formatted_start_time = f'{start_time[:2]}_{start_time[2:4]}_{start_time[4:]}'
    formatted_end_time = f'{end_time[:2]}_{end_time[2:4]}_{end_time[4:]}'

    # Add one hour to the end time
    formatted_end_time = add_hour(formatted_end_time)

    # Construct the pipeline result filename
    pipeline_result_filename = f'{camera}_{formatted_date}_{formatted_start_time}-{formatted_end_time}.csv'

    return pipeline_result_filename


def add_hour(time_string):
    """
    Add an hour to a time string.

    Args:
        time_string (str): Time string in the format 'HH:MM:SS'.

    Returns:
        str: Time string with an hour added, in the format 'HH:MM:SS'.
    """

    # Parse the time string into a datetime object
    time = datetime.strptime(time_string, "%H_%M_%S")

    # Add an hour to the time
    new_time = time + timedelta(hours=1)

    # Format the new time back into a string
    new_time_string = new_time.strftime("%H_%M_%S")

    return new_time_string