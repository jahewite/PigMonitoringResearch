import pandas as pd

from collections import Counter
from pipeline.utils.general import load_label_file
from pipeline.utils.path_manager import PathManager

path_manager = PathManager()
labels_posture_classification, labels_posture_classification_reverse = load_label_file(path_manager.path_to_pig_posture_classification_label_file)

class PipelineResultsLogger:
    """
    A utility class for logging detailed results of pig posture and tail detection analyses
    across a sequence of images. It tracks the occurrence of different postures ('lying' and 'notLying')
    and tail positions ('upright' and 'hanging') per frame, as well as activity between frames.
    
    This logger facilitates the aggregation of detection results into a structured format,
    particularly a pandas DataFrame, which can then be used for further analysis 
    or visualization. The DataFrame includes information about each analyzed frame or image,
    such as the time stamps, frame numbers, counts of detected postures, tail postures,
    and activity.

    Attributes:
        results_df (pandas.DataFrame): The DataFrame that accumulates the logging results. 
                                       Columns include 'start_timestamp', 'end_timestamp', 'start_frame', 'end_frame',
                                       'num_tails_upright', 'num_tails_hanging', 'num_pigs_lying', 'num_pigs_notLying',
                                       'activity', 'num_pig_detections', 'num_tail_detections'.
        frame_num (int): A counter to keep track of the number of frames processed.

    Methods:
        __init__(self): Initializes a new instance of the PipelineResultsLogger, setting up the results DataFrame.
        log_frame(self, start_timestamp, end_timestamp, pig_posture_classifications, tail_detections, activity=None):
            Logs the results of analyzing a single frame or image into the results DataFrame.
        get_results(self): Returns the accumulated results DataFrame.
    """
    def __init__(self):
        """
        Initializes the PipelineResultsLogger instance, creating an empty DataFrame with predefined columns
        to store the results of the pig posture and tail detection analyses.
        """
        # Create an empty DataFrame to store the results
        self.results_df = pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'start_frame', 'end_frame',
                                                'num_tail_detections', 'num_tails_upright', 'num_tails_hanging', 
                                                'num_pig_detections', 'num_pigs_lying', 'num_pigs_notLying',
                                                'activity'])
        self.frame_num = 1
        
    def log_frame(self, start_timestamp, end_timestamp, pig_posture_classifications, tail_detections, activity=None):
        """
        Logs the detection results for a single frame or image, including posture classifications, tail detections,
        and any observed activity, into the results DataFrame.

        Args:
            start_timestamp (datetime): The start timestamp of the frame being analyzed.
            end_timestamp (datetime): The end timestamp of the frame being analyzed.
            pig_posture_classifications (list): A list of classifications for each detected pig in the frame,
                                                 typically indicating 'lying' or 'notLying' postures.
            tail_detections (list): A list of detections for pig tails in the frame, typically classified as
                                    'upright' or 'hanging'.
            activity (str, optional): An optional description of the activity observed between frames. Defaults to None.

        The method updates the internal DataFrame with the provided information and increments the frame counter.
        """
        frame_df = pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'start_frame', 'end_frame',
                                         'num_tail_detections', 'num_tails_upright', 'num_tails_hanging', 
                                         'num_pig_detections', 'num_pigs_lying', 'num_pigs_notLying',
                                         'activity'])
        # List of '0' and '1', needs to be mapped to label class
        pig_posture_classifications = [labels_posture_classification_reverse[i.item()] 
                                       for i in pig_posture_classifications]

        # Tail posture list
        counter_tail_postures = Counter(d['label'] for d in tail_detections)
        counter_pig_postures = Counter(pig_posture_classifications)

        num_pig_detections = len(pig_posture_classifications)
        num_tail_detections = len(tail_detections)

        frame_df = pd.DataFrame({'start_timestamp': start_timestamp,
                                 'end_timestamp': end_timestamp,
                                 'start_frame': self.frame_num - 1, 
                                 'end_frame': self.frame_num,
                                 'num_tail_detections': [num_tail_detections],
                                 'num_tails_upright': [counter_tail_postures["upright"]],
                                 'num_tails_hanging': [counter_tail_postures["hanging"]],
                                 'num_pig_detections': [num_pig_detections],
                                 'num_pigs_lying': [counter_pig_postures['pigLying']],
                                 'num_pigs_notLying': [counter_pig_postures['pigNotLying']],
                                 'activity': [activity],
                                 })
        self.frame_num += 1
        self.results_df = pd.concat([self.results_df, frame_df], ignore_index=True)
        
    def get_results(self):
        """
        Retrieves the accumulated results of the posture and tail detections across all logged frames or images.

        Returns:
            pandas.DataFrame: The DataFrame containing the detailed logging results, including timestamps, frame numbers,
                              counts of different postures and tail positions, and any noted activity.
        """
        return self.results_df