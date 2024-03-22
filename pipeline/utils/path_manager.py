import os


class PathManager:
    """
    Manages and provides easy access to various file and directory paths used throughout the project.
    
    This class centralizes the paths to important resources such as model weights, label files, configuration files,
    and output directories. It ensures that all components of the project refer to the correct file locations,
    facilitating consistency and ease of access when working with files and directories.

    Attributes:
        root_dir (str): The root directory of the project, calculated relative to the location of this script.
        path_to_pig_detection_weights (str): Path to the pig detection model weights.
        path_to_tail_posture_detection_weights (str): Path to the tail posture detection model weights.
        path_to_posture_classification_weights (str): Path to the posture classification model weights.
        path_to_pig_detection_label_file (str): Path to the pig detection label file.
        path_to_pig_posture_classification_label_file (str): Path to the pig posture classification label file.
        path_to_tail_posture_detection_label_file (str): Path to the tail posture detection label file.
        path_to_outputs (str): Root path for output data generated by the pipeline.
        path_to_pipeline_results_aggregations (str): Path to store aggregated results data.
        path_to_output_plots (str): Path to store plots generated by the pipeline.
        path_to_monitoring_pipeline_outputs (str): Path to store outputs from the monitoring pipeline specifically.
        path_to_test_plots (str): Path to store test plots generated for visual inspection or testing.
        path_to_test_images (str): Path to a directory containing test images.
        path_to_config_files (str): Path to the directory containing configuration files and other assets.
        path_to_config (str): Path to the main configuration file for the project.
        path_to_piglet_rearing_info (str): Path to the file containing information about piglet rearing batches.
        path_to_piglet_rearing_timespans (str): Path to the file containing timespan data for piglet rearing batches.
        path_to_fattening_timepsans (str): Path to the file containing timespan data for fattening batches.
        path_to_test_clip (str): Path to a test video clip for development or testing purposes.

    Methods:
        __init__(self): Initializes the PathManager instance by setting up all the paths relative to the project root.
    """

    def __init__(self):
        """
        Initializes the PathManager instance, setting up paths to important resources used throughout the project.
        The paths are set relative to the project root directory to ensure they are correctly resolved regardless
        of the execution context.
        """
        # root path
        self.root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        self.root_dir = os.path.abspath(os.path.join(self.root_dir, '..'))

        # model paths
        self.path_to_pig_detection_weights = os.path.join(
            self.root_dir, 'assets/models/detection/pig_detection/yolo/pig_detection.pt')
        self.path_to_tail_posture_detection_weights = os.path.join(
            self.root_dir, 'assets/models/detection/tail_detection/yolo/tail_posture_detection.pt')
        self.path_to_posture_classification_weights = os.path.join(
            self.root_dir, "assets/models/classification/posture_classification/efficientnetv2/posture_classification.pth")

        # label file paths
        self.path_to_pig_detection_label_file = os.path.join(
            self.root_dir, 'assets/label_files/pig_detection/labels_pig.json')
        self.path_to_pig_posture_classification_label_file = os.path.join(
            self.root_dir, 'assets/label_files/pig_posture_classification/labels_lying_notLying.json')
        self.path_to_tail_posture_detection_label_file = os.path.join(
            self.root_dir, 'assets/label_files/tail_posture_detection/labels_tail_upright_hanging.json')

        # output paths
        self.path_to_outputs = os.path.join(self.root_dir, 'pipeline_outputs')
        self.path_to_pipeline_results_aggregations = os.path.join(
            self.root_dir, 'pipeline_outputs/data_aggregation')
        self.path_to_output_plots = os.path.join(
            self.root_dir, 'pipeline_outputs/plots')
        self.path_to_monitoring_pipeline_outputs = os.path.join(
            self.root_dir, 'pipeline_outputs/monitoring_pipeline/')
        self.path_to_test_plots = os.path.join(
            self.root_dir, 'pipeline_outputs/plots/test_plots')

        # path to test images
        self.path_to_test_images = os.path.join(
            self.root_dir, 'assets/test_images')

        # path to assets
        self.path_to_config_files = os.path.join(
            self.root_dir, 'assets')
        self.path_to_config = os.path.join(
            self.root_dir, 'assets/configuration_files/config.json')
        self.path_to_piglet_rearing_info = os.path.join(
            self.root_dir, 'assets/rearing_batch_info.json')
        self.path_to_piglet_rearing_timespans = os.path.join(
            self.root_dir, 'assets/rearing_batch_timespans.json')
        self.path_to_fattening_timepsans = os.path.join(
            self.root_dir, 'assets/fattening_batch_timespans.json')
        self.path_to_test_clip = os.path.join(
            self.root_dir, 'assets/test_clips/test_1.mp4')