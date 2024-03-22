import os
import argparse
import datetime
import imageio
import logging

from tqdm import tqdm
from pipeline.model_pipeline import ModelPipeline
from pipeline.utils.path_manager import PathManager
from pipeline.result_logger import PipelineResultsLogger
from pipeline.activity_monitoring import PigActivityAnalyzer
from pipeline.utils.image_preprocessor import ImagePreprocessor
from pipeline.tail_detection_processor import TailDetectionProcessor

from pipeline.utils.general import load_label_file, load_json_data, setup_logging
from pipeline.utils.video_file_utils import get_frame_of_video, get_time_of_video_capture, format_pipeline_result_filename

# Init logging
setup_logging()

# Create parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--path_to_videos",
                    help="Path to Directory which contains atleast one video.", required=True)
parser.add_argument("--path_to_pipeline_outputs",
                    help="Path to Directory where to store the results.", required=False)
parser.add_argument("--log_activity", action="store_true",
                    help="Log activity based on activity derivation function in 'activity_monitoring.py' (WIP)")


# Parse Arguments
args = parser.parse_args()


def analyze_videos(video_dir):
    """
    Analyzes all videos in the provided directory using various components (like ModelPipeline,
    PigActivityAnalyzer, TailDetectionProcessor, etc.) and saves the results in a CSV file.

    Args:
        video_dir (str): Path to the directory that contains the videos.

    The function works as follows:
    - Initializes the PathManager, which manages paths for various outputs and labels.
    - Loads the label files for posture classification and tail posture detection.
    - Retrieves a list of all .mp4 and .avi files in the specified directory.
    - Loads the first frame of the first video file for grid info initialization.
    - Initializes the pipeline, activity analyzer, tail detection processor, and logger.
    - Analyzes each video file using the initialized components.
    - Collects the results and saves them in a CSV file.

    The results file is named using the format_pipeline_result_filename function, and saved 
    in the output path specified by the PathManager.

    Note: This function assumes that the directory contains at least one video file, and that
    the first video file can be successfully loaded. If these conditions are not met, the function
    may not behave as expected.
    """

    # Initialize path manager
    path_manager = PathManager()

    # define output path
    if args.path_to_pipeline_outputs:
        output_path = args.path_to_pipeline_outputs
    else:
        # path to root output
        output_path = path_manager.path_to_outputs

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load config file (contains info for cutting images during inference and additional info for areas to analyze during activity derivation)
    config = load_json_data(path_manager.path_to_config)

    # load label file
    labels_posture_classification, labels_posture_classification_reverse = load_label_file(
        path_manager.path_to_pig_posture_classification_label_file)
    labels_tail_posture_detection, labels_tail_posture_detection_reverse = load_label_file(
        path_manager.path_to_tail_posture_detection_label_file)

    # Get a list of all files in the directory
    files = os.listdir(video_dir)

    # Filter the list to only include video files
    video_files = sorted([f for f in files if f.endswith(
        '.mp4') or f.endswith('.avi') or f.endswith('.mkv')])

    # load first frame of video sequence for grid_info initialization
    first_frame = get_frame_of_video(
        os.path.join(video_dir, video_files[0]), 0)

    # Initialize the pipeline, activity analyzer, tail detection processor and logger
    try:
        image_preprocessor = ImagePreprocessor(
            config, os.path.join(video_dir, video_files[0]))
    except:
        logging.info(
            f"[GENERAL] No image preprocessor for this video format. Proceeding without...")
    model_pipeline = ModelPipeline(crop_size=224)
    activity_analyzer = PigActivityAnalyzer(
        first_frame, n_grids_row=3, n_grids_col=4)
    tail_detection_processor = TailDetectionProcessor(
        labels_posture_classification_reverse, labels_tail_posture_detection_reverse)
    result_logger = PipelineResultsLogger()

    # Process each video file
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(video_dir, video_file)
        video_name = video_path.split("/")[-1]
        logging.info(
            f"[Video {i+1}] [ANALYSIS] Processing video {video_name} ...")
        analyze_video(video_path, image_preprocessor, model_pipeline, activity_analyzer,
                      tail_detection_processor, result_logger)

    # Create dataframe with results
    results = result_logger.get_results()

    # Get formatted filename for results
    results_filename = format_pipeline_result_filename(video_dir)

    # Define path to save results
    path_to_save_results = os.path.join(output_path, results_filename)

    # Save results as .csv
    results.to_csv(path_to_save_results, index=False)


def analyze_video(video_path, image_preprocessor, model_pipeline, activity_analyzer, tail_detection_processor, result_logger):
    """
    Analyzes a single video file using the given model pipeline, activity analyzer, tail detection processor, and logger.

    The function reads the video file frame by frame. For each frame, it performs model inference, processes tail detections,
    calculates activity, and logs results.

    Args:
        video_path (str): The path to the video file to be analyzed.
        model_pipeline (ModelPipeline): The model pipeline for inferencing.
        activity_analyzer (PigActivityAnalyzer): The analyzer for pig activity.
        tail_detection_processor (TailDetectionProcessor): The processor for tail detections.
        result_logger (PipelineResultsLogger): The logger for logging the results.

    Returns:
        None
    """
    # load video capture using imagio
    cap = imageio.get_reader(video_path,  'ffmpeg')

    # Get FPS of video capture
    fps = int(cap.get_meta_data()["fps"])

    # Get duration of video capture
    duration = int(cap.get_meta_data()["duration"])

    # Calculate total number of frames
    total_frames = int(fps * duration)

    # get time of video capture
    time_of_video_cap = get_time_of_video_capture(video_path)

    # If 'creation_time' metadata is not found, use a default timestamp:
    if time_of_video_cap is None:
        logging.warning(f"[ERROR] [Video] [METADATA] No creation time found for {video_path}. Using file's last modification time as a fallback.")
        last_mod_time = os.path.getmtime(video_path)
        time_of_video_cap = datetime.datetime.fromtimestamp(last_mod_time)

    # declare frame counter
    frame_counter = 0

    # Read the first frame before the loop
    frame_t = cap.get_data(0)
    start_timestamp = time_of_video_cap

    # preprocess frame if preprocessor available
    try:
        frame_t = image_preprocessor.cut_img_portion(frame_t)
    except:
        pass

    # model pipeline inference for first frame
    pig_detections_t, posture_classifications_t, tail_posture_detections_t = model_pipeline(
        frame_t)

    # process tail detections for first frame
    tail_posture_detections_t = tail_detection_processor.process_tail_detections(pig_detections_t,
                                                                                 posture_classifications_t,
                                                                                 tail_posture_detections_t)

    # get end timestamp
    end_timestamp = time_of_video_cap + datetime.timedelta(seconds=1)

    # log results for first frame - activity = None since 2 frames are needed for activity calculation
    result_logger.log_frame(start_timestamp, end_timestamp, posture_classifications_t,
                            tail_posture_detections_t, None)

    prev_pig_detections = pig_detections_t
    prev_posture_classifications = posture_classifications_t
    prev_end_timestamp = end_timestamp

    for i, frame_t in enumerate(tqdm(cap, total=total_frames, desc="Processing video")):
        frame_counter += 1
        # If the frame number is not divisible by fps, skip it
        if frame_counter % fps != 0:
            continue

        # get prev values
        start_timestamp = prev_end_timestamp
        end_timestamp = prev_end_timestamp + datetime.timedelta(seconds=1)

        try:
            # preprocess frame if preprocessor available
            try:
                frame_t = image_preprocessor.cut_img_portion(frame_t)
            except:
                pass

            # model pipeline inference
            pig_detections_t, posture_classifications_t, tail_posture_detections_t = model_pipeline(
                frame_t)

            # preprocess frame
            frame_t = image_preprocessor.cut_img_portion(frame_t)

            # model pipeline inference
            pig_detections_t, posture_classifications_t, tail_posture_detections_t = model_pipeline(
                frame_t)

            # process tail detections
            tail_posture_detections_t = tail_detection_processor.process_tail_detections(pig_detections_t,
                                                                                         posture_classifications_t,
                                                                                         tail_posture_detections_t)
            if args.log_activity:
                # get activity between t and prev frame
                activity = activity_analyzer.get_activity(
                    prev_pig_detections, prev_posture_classifications, pig_detections_t, posture_classifications_t)
            else:
                activity = None

            # log results
            result_logger.log_frame(start_timestamp, end_timestamp, posture_classifications_t,
                                    tail_posture_detections_t, activity)

            prev_pig_detections = pig_detections_t
            prev_posture_classifications = posture_classifications_t
            prev_end_timestamp = end_timestamp

        except Exception as e:
            pass
            logging.info(
                f"[GENERAL] [ERROR] Failed to process frame {i}. Error: {str(e)}.")


# run code
analyze_videos(args.path_to_videos)
