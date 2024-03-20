import pandas as pd

from collections import Counter
from pipeline.utils.general import load_label_file
from pipeline.utils.path_manager import PathManager

path_manager = PathManager()
labels_posture_classification, labels_posture_classification_reverse = load_label_file(path_manager.path_to_pig_posture_classification_label_file)

class PipelineResultsLogger:
    '''
    A Logger that logs the frequency of 'upright' and 'hanging' tails as well as the frequency of 
    'lying' and 'notLying' pigs for each given image of a sequence. Also logs activity between frames.
    Returns a pandas Dataframe with columns 'start_timestamp', 'end_timestamp', 'start_frame', 'end_frame', 
    'num_tails_upright', 'num_tails_hanging', 'num_pigs_lying', 'num_pigs_notLying', 'activity', 'num_pig_detections', 
    'num_tail_detections'.
    '''
    def __init__(self):
        # Create an empty DataFrame to store the results
        self.results_df = pd.DataFrame(columns=['start_timestamp', 'end_timestamp', 'start_frame', 'end_frame',
                                                'num_tail_detections', 'num_tails_upright', 'num_tails_hanging', 
                                                'num_pig_detections', 'num_pigs_lying', 'num_pigs_notLying',
                                                'activity'])
        self.frame_num = 1
        
    def log_frame(self, start_timestamp, end_timestamp, pig_posture_classifications, tail_detections, activity=None):
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
        return self.results_df