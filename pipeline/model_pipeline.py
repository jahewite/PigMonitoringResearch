import torch
import numpy as np
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics.models.yolo import YOLO
from pipeline.utils.path_manager import PathManager
from ultralytics.utils.plotting import save_one_box


class ModelPipeline(nn.Module):
    """
    A model pipeline that integrates YOLOv8 for pig detection and EfficientNetV2 for posture classification,
    with an additional YOLOv8 model for tail detection. This comprehensive model is designed for analyzing
    pig images to detect pigs, classify their postures as 'lying' or 'not lying', and detect tail postures.
    The model supports filtering to exclude lying pigs from tail detection analysis, focusing on standing pigs.

    Attributes:
        crop_size (int): The size to which detected pig images are cropped for posture classification.
        filter_lying_pigs (bool): Flag to determine whether lying pigs are excluded from tail detection.
        path_manager (PathManager): Instance of PathManager to manage paths for model weights.
        pig_detection (YOLO): YOLO model for detecting pigs in images.
        tail_posture_detection (YOLO): YOLO model for detecting tail postures in cropped pig images.
        posture_classification (EfficientNetV2): Model for classifying pig postures based on cropped images.
        device (torch.device): The device (CPU/GPU) on which the models are loaded and inference is performed.
        crop_transform (transforms.Compose): Transformations applied to cropped pig images for posture classification.

    Methods:
        __init__(self, crop_size, filter_lying_pigs=True): Initializes the model pipeline with specified configurations.
        format_parser(self, x, predictions): Formats YOLOv8 predictions to a standardized structure.
        forward(self, x): Processes an input image through the pipeline, performing detection, classification, and analysis.
    """

    def __init__(self, crop_size, filter_lying_pigs=True):
        """
        Initializes the ModelPipeline class with specified crop size and an option to filter lying pigs during tail detection.

        Args:
            crop_size (int): The size to which images are cropped for posture classification.
            filter_lying_pigs (bool, optional): Whether to exclude lying pigs from tail detection. Defaults to True.
        """
        super(ModelPipeline, self).__init__()

        self.filter_lying_pigs = filter_lying_pigs

        # get model weight paths
        self.path_manager = PathManager()
        self.path_to_pig_detection_weights = self.path_manager.path_to_pig_detection_weights
        self.path_to_posture_classification_weights = self.path_manager.path_to_posture_classification_weights
        self.path_to_tail_posture_detection_weights = self.path_manager.path_to_tail_posture_detection_weights

        # load detection models
        self.pig_detection = YOLO(self.path_to_pig_detection_weights)
        self.tail_posture_detection = YOLO(
            self.path_to_tail_posture_detection_weights)

        # load classification model with map_location for CPU
        self.posture_classification = torch.load(
            self.path_to_posture_classification_weights, map_location=torch.device('cpu'))
        self.posture_classification = self.posture_classification.module
        self.crop_size = crop_size

        # Send model to device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.pig_detection.to(self.device)
        self.posture_classification.to(self.device)
        self.tail_posture_detection.to(self.device)

        self.posture_classification.eval()
        self.crop_transform = transforms.Compose([
            ResizeAspectRatioPad(img_size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def format_parser(self, x, predictions):
        """
        Formats the predictions from the YOLOv8 model to match the expected structure 
        (Legacy code since pipeline was originally developed using YOLOv5), including converting bounding boxes
        and extracting cropped images for further analysis.

        Args:
            x (Tensor): The input image tensor.
            predictions (List[Dict]): Predictions from the YOLOv8 model, including bounding boxes and other details.

        Returns:
            List[Dict]: Formatted predictions with standardized structure for ease of processing.
        """
        holder = []
        for pred in predictions[0]:
            parser_holder = {
                'box': [bbox for bbox in pred.boxes.xyxy[0]],
                'conf': pred.boxes.conf[0],
                'cls': pred.boxes.cls[0],
                'label': None,
                'im': save_one_box(pred.boxes.xyxy[0], x, save=False)
            }
            holder.append(parser_holder)
        return holder

    def forward(self, x):
        """
        Conducts a forward pass through the pipeline, executing pig detection, posture classification, and tail detection
        in sequence. Optionally filters out lying pigs based on posture classification before tail detection.

        Args:
            x (Tensor): The input image tensor.

        Returns:
            Tuple[List[Dict], Tensor, List[Dict]]: A tuple containing lists of pig detection results, posture classification
            results, and tail detection results, respectively.
        """
        with torch.no_grad():
            # Pig detection
            pig_detection_outputs = self.pig_detection(x, verbose=False)
            pig_detection_outputs = self.format_parser(
                x, pig_detection_outputs)

            crops_posture_classification = []
            tail_detection_outputs = []
            crops_tail_detection = []

            # Iterate over pig detections for pig posture classification
            for crop in pig_detection_outputs:
                crop = crop["im"]
                crop = np.ascontiguousarray(crop)

                # Prepare posture Classification
                crop = Image.fromarray(crop)
                crop = self.crop_transform(crop)
                crops_posture_classification.append(crop)

            # check if there are any pig detections
            if len(crops_posture_classification) > 0:
                # Stack the cropped images for posture classification
                crops_posture_classification = torch.stack(
                    crops_posture_classification).to(self.device)

                # Perform posture classification
                posture_classification_outputs = self.posture_classification(
                    crops_posture_classification)
                _, posture_classification_outputs = torch.max(
                    posture_classification_outputs.data, 1)

            else:
                # No pig detections. Set empty list.
                posture_classification_outputs = []

            # iterate over pig detections for tail posture detection
            for i, crop in enumerate(pig_detection_outputs):
                crop = crop["im"]
                crop = np.ascontiguousarray(crop)

                # check if filter_lying_pigs is True
                if self.filter_lying_pigs:

                    # Check the posture classification of the current crop
                    posture = posture_classification_outputs[i].item()

                    if posture == 1:
                        # Tail posture detection on crop
                        tail_detection_output = self.tail_posture_detection(
                            crop, verbose=False)
                        tail_detection_output = self.format_parser(
                            x, tail_detection_output)
                        tail_detection_outputs.append(tail_detection_output)
                        crops_tail_detection.append(crop)
                    else:
                        tail_detection_outputs.append([])

                # if filter_lying_pigs False, apply tail posture detection to every crop
                else:
                    # Tail posture detection on crop
                    tail_detection_output = self.tail_posture_detection(
                        crop, verbose=False)
                    tail_detection_output = self.format_parser(
                        x, tail_detection_output)
                    tail_detection_outputs.append(tail_detection_output)
                    crops_tail_detection.append(crop)

            return pig_detection_outputs, posture_classification_outputs, tail_detection_outputs


class ResizeAspectRatioPad(object):
    """
    Transformation class for resizing and padding images to a specified size while maintaining the original aspect ratio.

    This class ensures that images are processed in a uniform size for model input, using padding as necessary to
    preserve aspect ratio, which is critical for maintaining image integrity during analysis.

    Args:
        img_size (int): Desired size (width and height) for the output image after resizing and padding.

    Example:
        resize_transform = ResizeAspectRatioPad(img_size=224)
        transformed_image = resize_transform(original_image)
    """

    def __init__(self, img_size):
        """
        Initializes the ResizeAspectRatioPad transformation with the specified target image size.

        Args:
            img_size (int): The target size for both width and height of the image after transformation.
        """
        self.img_size = img_size

    def resize_and_pad(self, img, min_size=None):
        '''
        Resizes and pads the input image to the specified size while maintaining the aspect ratio.

        Args:
            img (PIL Image): The input image to resize and pad.
            min_size (int, optional): The minimum size for the output image. If None, `img_size` is used.

        Returns:
            PIL Image: The resized and padded image.
        '''
        width, height = img.size
        size = max(min_size, width, height)
        # if width < height, padding needs to be on the left and right border
        # if width > height, padding needs to be on the top and bottom
        # both paddings need to be evenly spaced on both of the respective sides
        if width < height:
            padding = int((size - width) / 2)
            img = ImageOps.expand(img, border=(
                padding, 0, padding, 0), fill='black')
        else:
            padding = int((size - height) / 2)
            img = ImageOps.expand(img, border=(
                0, padding, 0, padding), fill='black')
        img = img.resize((min_size, min_size))
        return img

    def __call__(self, img):
        """
        Applies the resizing and padding transformation to an input image, maintaining aspect ratio.

        Args:
            img (PIL Image): The input image to be resized and padded.

        Returns:
            PIL Image: The transformed image with maintained aspect ratio, resized and padded to the target size.
        """
        img = self.resize_and_pad(img, min_size=self.img_size)
        return img
