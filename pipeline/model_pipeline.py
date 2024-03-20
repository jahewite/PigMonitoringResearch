import torch
import numpy as np
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics.models.yolo import YOLO
from pipeline.utils.path_manager import PathManager
from ultralytics.utils.plotting import save_one_box


class ModelPipeline(nn.Module):
    '''
    Combination of YOLOv5 and EfficientNetV2.
    This class represents a model that combines pig detection, posture classification, and tail detection tasks.
    The model takes an input image, detects pigs using the YOLOv8 model, and passes the predicted bounding boxes
    to the EfficientNetV2 model for posture classification into 'lying' and 'not lying' postures.
    The tail detection is also performed on the cropped image of the pig detection.
    Optionally, you can filter out lying pigs during the tail detection process to monitor only the tail posture
    of standing pigs.
    '''

    def __init__(self, crop_size, filter_lying_pigs=True):
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
        '''
        Formats the predictions from the YOLOv8 model into YOLOv5 format.

        Args:
            x (Tensor): Input image tensor.
            predictions (List[Dict]): List of prediction dictionaries containing bounding box,
                                    confidence, class, label, and cropped image.

        Returns:
            List[Dict]: Formatted predictions.
        '''
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
        '''
        Forward pass of the Pipeline class.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tuple[List[Dict], Tensor, List[Dict]]: Pig detection outputs, posture classification outputs,
                                                and tail detection outputs.
        '''
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
    '''
    Resizes and pads images while maintaining the aspect ratio.
    This class is used as a transformation for resizing and padding images to a specific size.
    It ensures that the aspect ratio of the original image is preserved while resizing and padding.

    Args:
        img_size (int): The desired size (width and height) for the output image.

    Example usage:
        resize_transform = ResizeAspectRatioPad(img_size=224)
        output_image = resize_transform(input_image)
    '''

    def __init__(self, img_size):
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
        img = self.resize_and_pad(img, min_size=self.img_size)
        return img
