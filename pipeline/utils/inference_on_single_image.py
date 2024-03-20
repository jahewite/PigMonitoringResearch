import os
import cv2
import argparse

from pipeline.model_pipeline import ModelPipeline
from pipeline.utils.path_manager import PathManager
from pipeline.utils.visualization import plot_results
from pipeline.utils.image_preprocessor import ImagePreprocessor
from pipeline.tail_detection_processor import TailDetectionProcessor
from pipeline.utils.general import load_label_file, save_array_as_image, load_json_data


def main():

    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Inference of model pipeline on a single image.")
    parser.add_argument("path_to_image", type=str, help="Path to image.")
    parser.add_argument("--filter_lying_pigs", action='store_true')
    parser.add_argument("--prefix", type=str, default=None,
                        help="Prefix for image preprocessing.")

    # Parse the arguments
    args = parser.parse_args()

    # init path manager
    path_manager = PathManager()

    # load label files
    labels_posture_classification, labels_posture_classification_reverse = load_label_file(
        path_manager.path_to_pig_posture_classification_label_file)
    labels_tail_posture_detection, labels_tail_posture_detection_reverse = load_label_file(
        path_manager.path_to_tail_posture_detection_label_file)

    if args.prefix:
        # load config file (contains info for cutting images and additional info for areas to analyze during activity derivation)
        config = load_json_data(path_manager.path_to_config)
        # Initialize the pipeline, activity analyzer, tail detection processor and logger
        image_preprocessor = ImagePreprocessor(
            config, args.prefix)
    else:
        pass

    # Create Pipeline class instance
    pipeline = ModelPipeline(
        crop_size=224, filter_lying_pigs=args.filter_lying_pigs)

    # Create TailDetectionProcessor instance
    tail_detections_processor = TailDetectionProcessor(
        labels_posture_classification_reverse, labels_tail_posture_detection_reverse)

    # load test image
    img = cv2.imread(args.path_to_image)
    img_name = args.path_to_image.split("/")[-1]

    # path to save plot
    path_to_save_image = os.path.join(
        path_manager.path_to_test_plots, img_name)

    # perform inference
    pig_detections, posture_classifications, tail_posture_detections = pipeline(
        img)

    # post processing tail detections
    tail_detections_holder = tail_detections_processor.process_tail_detections(
        pig_detections, posture_classifications, tail_posture_detections)

    # plot results
    result = plot_results(img, pig_detections, posture_classifications, tail_detections_holder,
                          labels_posture_classification_reverse,
                          return_img=True)

    save_array_as_image(result, path_to_save_image)


if __name__ == "__main__":
    main()
