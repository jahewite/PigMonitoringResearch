import torch
import torchvision

from collections import defaultdict


class TailDetectionProcessor:
    """
    Processes and filters tail detections in pig images, specifically focusing on handling overlapping 
    detections and filtering based on pig posture. The processor scales tail detections to the original image size, 
    applies criteria to filter out detections based on posture, and utilizes Non-Maximum Suppression (NMS) to select 
    the best bounding box for tail detections.

    Note:
        Currently, the tail detections within a 'pigLying' bounding box are filtered out, which might lead to discarding correctly 
        classified bounding boxes of standing pigs. A special consideration is given to 'pigNotLying' bounding boxes that overlap 
        with multiple 'pigLying' bounding boxes and contain the detected tail object in their overlapping area.

    Attributes:
        labels_posture_classification_reverse (dict): A mapping from numeric labels to their string representations for pig postures.
        labels_tail_upright_hanging_detection_reverse (dict): A mapping from numeric labels to their string representations for tail positions (upright or hanging).

    Methods:
        scale_tail_detections(pig_detections, posture_classifications, tail_detections): Scales the tail detection bounding boxes to the original image size.
        calculate_overlap_percentage(big_box, small_box): Calculates the overlap percentage of one bounding box within another.
        filter_tail_detections(tail_detections_holder): Filters overlapping tail detections and tail detections for pigs in a lying posture.
        is_bbox_inside(box_1, box_2): Checks if one bounding box is completely inside another.
        apply_nms(predictions, iou_threshold): Applies Non-Maximum Suppression (NMS) to reduce overlapping bounding boxes based on their confidence scores.
        select_best_detections(tail_detections_holder): Selects the detection with the highest confidence score for each unique pig detection.
        process_tail_detections(pig_detections, posture_classifications, tail_detections): Orchestrates the tail detection processing, including scaling, filtering, and applying NMS.
    """

    def __init__(self, labels_posture_classification_reverse, labels_tail_upright_hanging_detection_reverse):
        """
        Initializes the TailDetectionProcessor with mappings for posture and tail position classifications.
        """
        self.labels_posture_classification_reverse = labels_posture_classification_reverse
        self.labels_tail_upright_hanging_detection_reverse = labels_tail_upright_hanging_detection_reverse

    def scale_tail_detections(self, pig_detections, posture_classifications, tail_detections):
        """
        Scales tail detection bounding boxes to the original image size and packages detection information.

        Parameters:
            pig_detections (list): Detected pig bounding boxes.
            posture_classifications (list): Posture classification for each detected pig.
            tail_detections (list): Detected tail bounding boxes within pig detections.

        Returns:
            List[Dict]: Tail detections with scaled bounding boxes and additional information.
        """
        tail_detections_holder = []

        for i, detection in enumerate(tail_detections):
            if not detection:
                continue  # skip iteration if detection is empty

            for j, tail in enumerate(detection):
                pig_box = pig_detections[i]["box"]
                pig_box_width = pig_box[2] - pig_box[0]
                pig_box_height = pig_box[3] - pig_box[1]
                pig_crop = pig_detections[i]["im"]
                diff_x = pig_crop.shape[1] - pig_box_width
                diff_y = pig_crop.shape[0] - pig_box_height

                tail_box = tail["box"]
                pig_posture = self.labels_posture_classification_reverse[int(
                    posture_classifications[i])]

                # Scale coords of tail box to original image input size
                scaled_x1, scaled_y1 = pig_box[0] + \
                    tail_box[0], pig_box[1] + tail_box[1]
                scaled_x2, scaled_y2 = pig_box[0] + \
                    tail_box[2], pig_box[1] + tail_box[3]
                tail_box_scaled = [
                    scaled_x1-(diff_x/2), scaled_y1-(diff_y/2), scaled_x2-(diff_x/2), scaled_y2-(diff_y/2)]
                tail_box_scaled_area = (
                    scaled_x2 - scaled_x1 + 1) * (scaled_y2 - scaled_y1 + 1)

                tail_detections_holder.append({
                    "idx_pig_detection": i,
                    "idx_tail_detection": j,
                    "posture": pig_posture,
                    "box": tail["box"],
                    "box_scaled": tail_box_scaled,
                    "conf_score": tail["conf"],
                    "area": tail_box_scaled_area,
                    'im': tail["im"],
                    'label': self.labels_tail_upright_hanging_detection_reverse[int(tail["cls"])]
                })
        return tail_detections_holder

    def calculate_overlap_percentage(self, big_box, small_box):
        """
        Calculates the overlap percentage between a smaller box and a bigger box.

        Parameters:
            big_box (list): Bounding box coordinates of the bigger box.
            small_box (list): Bounding box coordinates of the smaller box.

        Returns:
            overlap_percentage (float): Percentage of overlap of the small_box within the big_box.
        """
        xmin_intersection = max(big_box[0], small_box[0])
        ymin_intersection = max(big_box[1], small_box[1])
        xmax_intersection = min(big_box[2], small_box[2])
        ymax_intersection = min(big_box[3], small_box[3])

        intersection_area = max(0, xmax_intersection - xmin_intersection) * \
            max(0, ymax_intersection - ymin_intersection)
        small_box_area = (small_box[2] - small_box[0]) * \
            (small_box[3] - small_box[1])

        return intersection_area / small_box_area * 100

    def filter_tail_detections(self, tail_detections_holder):
        """
        Filters out overlapping tail detections. This function applies more specific filtering criteria:
        If the bounding boxes of two detections overlap by more than 90%, the tail detection that belongs
        to a pig with 2 or more tail detections is removed. If both detections belong to a pig with 2 or
        more tail detections, the detection with the smaller area is removed.

        Parameters:
            pig_detections (list): List of pig detections. Each detection is a dictionary containing 
                                information about the detection, such as bounding box coordinates, 
                                confidence score, etc.
            tail_detections_holder (list): List of tail detections. Each detection is a dictionary 
                                            containing information about the detection, such as bounding 
                                            box coordinates, confidence score, associated pig detection, etc.

        Returns:
            final_tail_detections (list): Filtered list of tail detections. Detections that do not meet 
                                        the criteria (overlapping more than 90% and associated with a 
                                        pig that has 2 or more tail detections) are removed.
        """
        n = len(tail_detections_holder)
        to_keep = list(range(n))  # Start by keeping all detections

        # Create a dictionary that maps each pig detection to a list of its tail detections
        pig_to_tails = defaultdict(list)
        for i in range(n):
            pig_to_tails[tail_detections_holder[i]['idx_pig_detection']].append(i)

        for i in range(n):
            for j in range(i + 1, n):
                box_1, box_2 = tail_detections_holder[i]["box_scaled"], tail_detections_holder[j]["box_scaled"]

                if box_1 and box_2:
                    area_box_1, area_box_2 = tail_detections_holder[
                        i]["area"], tail_detections_holder[j]["area"]
                    overlap_percentage = self.calculate_overlap_percentage(
                        box_1, box_2) if area_box_1 > area_box_2 else self.calculate_overlap_percentage(box_2, box_1)

                    if overlap_percentage > 90:
                        # Get the pig detections that box_1 and box_2 belong to
                        pig_1 = tail_detections_holder[i]['idx_pig_detection']
                        pig_2 = tail_detections_holder[j]['idx_pig_detection']

                        # Check the number of tail detections for each pig
                        num_tails_1 = len(pig_to_tails[pig_1])
                        num_tails_2 = len(pig_to_tails[pig_2])

                        # Apply the new condition
                        if num_tails_1 >= 2 and area_box_1 < area_box_2:
                            if i in to_keep:
                                to_keep.remove(i)
                        elif num_tails_2 >= 2:
                            if j in to_keep:
                                to_keep.remove(j)

        # Only keep the selected detections
        return [tail_detections_holder[i] for i in to_keep]

    def is_bbox_inside(self, box_1, box_2):
        """
        Checks if one bounding box is entirely within another.

        Parameters:
            box_1 (list): Bounding box coordinates of the first box.
            box_2 (list): Bounding box coordinates of the second box.

        Returns:
            is_inside (bool): True if one bounding box is inside the other, False otherwise.
        """
        is_inside = all(box_2[i] <= box_1[i] <= box_2[i+2] for i in range(2)) or \
            all(box_1[i] <= box_2[i] <= box_1[i+2] for i in range(2))
        return is_inside

    def apply_nms(self, predictions, iou_threshold):
        sorted_predictions = sorted(
            predictions, key=lambda x: x['conf_score'], reverse=True)
        filtered_predictions = []

        while len(sorted_predictions) > 0:
            current_prediction = sorted_predictions[0]
            filtered_predictions.append(current_prediction)

            remaining_predictions = []

            for i in range(1, len(sorted_predictions)):
                iou = torchvision.ops.box_iou(
                    torch.tensor([current_prediction['box_scaled']]),
                    torch.tensor([sorted_predictions[i]['box_scaled']])
                )[0][0]

                if iou <= iou_threshold:
                    remaining_predictions.append(sorted_predictions[i])

            sorted_predictions = remaining_predictions

        return filtered_predictions

    def select_best_detections(self, tail_detections_holder):
        """
        Selects the best detection for each unique pig based on the highest confidence score.

        Parameters:
            tail_detections_holder (list): A list of processed and potentially overlapping tail detections.

        Returns:
            List[Dict]: A list containing the best detection for each unique pig.
        """

        # Create a dictionary where the keys are the 'idx_pig_detection' values and the
        # values are lists of detections.
        detections = defaultdict(list)

        # Categorize all detections based on 'idx_pig_detection'
        for detection in tail_detections_holder:
            detections[detection['idx_pig_detection']].append(detection)

        # Find the tail posture detection with the highest confidence score for each pig detection
        for pig_detection, tail_detections in detections.items():
            if len(tail_detections) > 1:  # If there are multiple detections,
                # Sort the list of detections by 'conf_score' in descending order
                tail_detections.sort(
                    key=lambda x: x['conf_score'], reverse=True)
                # Keep only the first element of the list (the one with the highest 'conf_score')
                detections[pig_detection] = [tail_detections[0]]

        # Convert it back to a list:
        tail_detections_holder = [detection for tail_detections in detections.values(
        ) for detection in tail_detections]

        return tail_detections_holder

    def process_tail_detections(self, pig_detections, posture_classifications, tail_detections):
        """
        Orchestrates the processing of tail detections, including scaling, filtering based on posture, and applying 
        Non-Maximum Suppression (NMS) to refine the detection results.

        Parameters:
            pig_detections (list): Detected pig bounding boxes from the image.
            posture_classifications (list): Classification results indicating the posture of each detected pig.
            tail_detections (list): Detected tail bounding boxes within the pig detections.

        Returns:
            List[Dict]: Processed and filtered tail detections, with each detection including details such as the scaled bounding box, 
            confidence score, and the label indicating tail posture.

        """
        scaled_tail_detections = self.scale_tail_detections(
            pig_detections, posture_classifications, tail_detections)
        filtered_tail_detections = self.filter_tail_detections(scaled_tail_detections)
        nms_tail_detections = self.apply_nms(filtered_tail_detections,  0.45)
        tail_detections_final = self.select_best_detections(
            nms_tail_detections)
        return tail_detections_final
