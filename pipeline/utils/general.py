import os
import sys
import json
import time
import torch
import logging
import matplotlib.pyplot as plt


def print_ascii_art():
    """
    Print an ASCII art banner with the text "Monitoring Pipeline".

    This function is intended to be used as a decorative banner at the start
    of the execution of the monitoring pipeline to provide a visual indication
    to the user and to enhance the CLI experience.
    """
    print("""
  __  __             _ _             _               _____ _            _ _            
 |  \/  |           (_) |           (_)             |  __ (_)          | (_)           
 | \  / | ___  _ __  _| |_ ___  _ __ _ _ __   __ _  | |__) | _ __   ___| |_ _ __   ___ 
 | |\/| |/ _ \| '_ \| | __/ _ \| '__| | '_ \ / _` | |  ___/ | '_ \ / _ \ | | '_ \ / _ \\
 | |  | | (_) | | | | | || (_) | |  | | | | | (_| | | |   | | |_) |  __/ | | | | |  __/
 |_|  |_|\___/|_| |_|_|\__\___/|_|  |_|_| |_|\__, | |_|   |_| .__/ \___|_|_|_| |_|\___|
                                              __/ |         | |                        
                                             |___/          |_|                        
          """)


def setup_logging():
    """
    Configure logging for the application.
    Sets the logging level and format for log messages.
    """
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)s] %(message)s')


def load_label_file(json_path):
    """
    Loads the json annotation files and return a dict with the annotation infos as well as a reverse version.
    """
    with open(json_path) as json_file:
        labels = json.load(json_file)
    labels_reverse = dict((value, key) for key, value in labels.items())
    return labels, labels_reverse


def load_json_data(file_path):
    """
    Load data from a JSON file.

    Args:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    dict: The data loaded from the JSON file.

    Raises:
    SystemExit: If an error occurs while reading the file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        sys.exit(1)


def save_json_data(file_path, data):
    """
    Save data to a JSON file.

    Args:
    file_path (str): The file path where the JSON data should be saved.
    data (dict): The data to be saved as JSON.

    Raises:
    SystemExit: If an error occurs while writing to the file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        logging.error(f"Error writing JSON file: {e}")
        sys.exit(1)


def measure_inference_time(pipeline, img):
    """
    Measures inference time when using GPU.

    Args:
        pipeline (function): Inference pipeline function.
        img (any): Input data for inference.

    Returns:
        inference_time (float): Time in seconds it took for the inference to complete.
        output (any): Output of the inference function.
    """
    # Save the start time
    start_time = time.time()

    # Perform inference
    output = pipeline(img)

    # Wait for all kernels in all streams on a CUDA device to complete
    torch.cuda.synchronize()

    # Save the end time
    end_time = time.time()

    # Calculate the duration of inference
    inference_time = end_time - start_time

    # Print the inference time
    print("Inference time: {:.4f} seconds".format(inference_time))

    # Return the inference time and the output of the inference function
    return inference_time


def save_array_as_image(array, file_path, uuid=False):
    """
    Saves a numpy array as an image using matplotlib.

    Args:
        array (numpy.ndarray): A 2D or 3D numpy array.
        filepath (str): The path where the image will be saved, including the file name and extension.

    Returns:
        None
    """
    if uuid:
        # Generate a random UUID
        file_name = str(uuid.uuid4())

        # define image name
        img_name = os.path.join(file_path, file_name)

    # Create a plot with the array as an image
    fig, ax = plt.subplots()
    ax.imshow(array)

    # remove axis
    ax.axis('off')

    # Save the figure
    fig.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=300)

    # Close the figure to prevent it from displaying in the notebook
    plt.close(fig)


def get_all_camera_names(json_file_path):
    """
    Extracts 'camera' values from a JSON file.

    Parameters:
    - json_file_path (str): The file path of the JSON file.

    Returns:
    - list: A list of camera values.
    """
    # Read and parse the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract camera values
    camera_names = sorted(set(item['camera'] for item in data))

    return camera_names
