import os


class PathManager:
    def __init__(self):
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