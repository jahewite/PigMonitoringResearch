class PigActivityAnalyzer:
    """
    This class is used to analyze the activity of pigs in a video frame, with the frame divided into a grid. 
    The class can calculate the Intersection over Union (IoU) of two bounding boxes, count the number of pigs 
    in each grid cell given bounding box detections and posture classifications, calculate the difference in 
    pig count between two consecutive frames, and calculate an activity metric for a video frame.

    Attributes:
        frame: numpy array
            An image represented as a numpy array.
        n_grids_row: int
            The desired number of rows in the grid.
        n_grids_col: int
            The desired number of columns in the grid.
        grid_info: dict
            A dictionary containing the grid layout on an image. This includes 
            the number of rows and columns in the grid, and the bounding box 
            (in pixels) for each cell in the grid.

    Methods:
        get_grid_info(img): Returns a dictionary containing grid information.
        calculate_iou(box1, box2): Calculates and returns the Intersection over Union (IoU) between two bounding boxes.
        count_pigs_and_total(pig_detections, posture_classifications): Counts and returns the number of pigs in each cell of a grid.
        get_grid_count_difference(grid_count_holder_t, grid_count_holder_t_plus_1): Calculates and returns the difference in the number of pigs in each grid cell between two consecutive frames.
        calculate_activity(grid_count_holder_t, grid_count_holder_t_plus_1): Calculates and returns an activity metric for a video frame.
        get_activity(pig_detections_t, posture_classifications_t, pig_detections_t_plus_1, posture_classifications_t_plus_1): Returns the activity metric calculated for given pig detections and posture classifications for two consecutive frames.
    """

    def __init__(self, frame, n_grids_row, n_grids_col):
        self.frame = frame
        self.n_grids_row = n_grids_row
        self.n_grids_col = n_grids_col
        self.grid_info = self.get_grid_info(frame)

    def get_grid_info(self, img):
        """
        This function generates a grid layout on a given image.

        Parameters:
        img: numpy array
            An image represented as a numpy array.
        n_grids_row: int
            The desired number of rows in the grid.
        n_grids_col: int
            The desired number of columns in the grid.

        Returns:
        grid_info: dict
            A dictionary containing the number of rows and columns in the grid 
            and the bounding box (in pixels) for each cell in the grid. Each 
            bounding box is represented by the upper-left and lower-right coordinates 
            (x1, y1, x2, y2). The keys for each bounding box in the dictionary 
            are strings representing the row and column indices of the cell.

        Example:
        If n_grids_row = 2 and n_grids_col = 2, the 'img_tiles' part of the output 
        might look like this:
            'img_tiles': {'00': [0, 0, 50, 50],
                        '01': [0, 50, 50, 100],
                        '10': [50, 0, 100, 50],
                        '11': [50, 50, 100, 100]}
        """
        # Initialize a dictionary to hold the grid information.
        grid_info = dict(
            rows=self.n_grids_row,  # The number of rows in the grid
            cols=self.n_grids_col,  # The number of columns in the grid
            img_tiles={}        # This will hold the coordinates for each cell in the grid
        )

        # Get the height and width of the image
        height, width = img.shape[:2]

        # Calculate the height and width of each cell in the grid
        step_h = height / self.n_grids_row
        step_w = width / self.n_grids_col

        # Loop over each cell in the grid
        for i in range(self.n_grids_col):
            for j in range(self.n_grids_row):
                # Calculate the bounding box coordinates for the current grid cell
                bbox = [step_w * i, step_h * j,
                        step_w * (i + 1), step_h * (j + 1)]

                # Add the bounding box to the grid information dictionary,
                # using the cell coordinates as the key (e.g., "00" for the first cell)
                grid_info["img_tiles"][f"{i}{j}"] = bbox

        # Return the grid information dictionary
        return grid_info

    def calculate_iou(self, box1, box2):
        """
        This function calculates the Intersection over Union (IoU) between two bounding boxes.

        Parameters:
        box1, box2: list or tuple of four numbers
            The bounding boxes to compare, each represented as a list or tuple 
            in the format (x1, y1, x2, y2), where (x1, y1) is the upper-left 
            coordinate of the box and (x2, y2) is the lower-right coordinate.

        Returns:
        iou: float
            The Intersection over Union of the two boxes. This is a measure of 
            the overlap between the two bounding boxes. An IoU of 0 indicates 
            no overlap, and an IoU of 1 indicates that the bounding boxes are 
            identical.

        Example:
        If box1 = [0, 0, 50, 50] and box2 = [25, 25, 75, 75], then
        calculate_iou(box1, box2) would return 0.25.
        """
        # determine the (x, y)-coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # compute the intersection over union
        iou = inter_area / float(box1_area + box2_area - inter_area)

        return iou

    def count_pigs_and_total(self, pig_detections, posture_classifications):
        """
        This function counts the number of pigs in each cell of a grid, given 
        bounding box detections and posture classifications for the pigs. If a 
        pig's bounding box overlaps with multiple grid cells, the cell with the 
        highest Intersection over Union (IoU) with the bounding box is incremented.
        In addition to the per-cell counts, this function also calculates the 
        total counts of lying and not lying pigs across all grid cells in the image.

        Parameters:
        grid_info: dict
            A dictionary describing the grid layout on an image. Generated from 
            the 'get_grid_info' function.
        pig_detections: list of dict
            A list of dictionaries, each representing a detected pig. Each dictionary 
            includes a 'box' key, which maps to a list of four numbers representing 
            the bounding box of the detected pig in the format (x1, y1, x2, y2).
        posture_classifications: list
            A list of classifications for each detected pig, where 0 represents 
            'lying down' and 1 represents 'not lying down'. The i-th element in 
            this list corresponds to the i-th element in 'pig_detections'.

        Returns:
        result: dict
            A dictionary containing two keys: 'per_grid_cell', mapping to the per-cell 
            counts, and 'total', mapping to the total counts of each posture across 
            all grid cells.

        Example:
        Assuming 'grid_info' is the output of get_grid_info(img, 3, 4),
        'pig_detections' is [{'box': [750, 300, 1000, 550], 'conf': 0.95, 'cls': 0, 'label': None, ...}, ...],
        and 'posture_classifications' is [0, 1, 1, ...], the output might look something like this:

        {
            'per_grid_cell': {
                '00': {'pigLying': 1, 'pigNotLying': 0, 'summed': 1},
                '01': {'pigLying': 0, 'pigNotLying': 0, 'summed': 0},
                ...
                '32': {'pigLying': 0, 'pigNotLying': 1, 'summed': 1}
            },
            'total': {'totalPigLying': 1, 'totalPigNotLying': 1, 'totalSummed': 2}
        }
        """
        # Initialize counter dictionary
        count_dict = {key: {'pigLying': 0, 'pigNotLying': 0, 'summed': 0}
                      for key in self.grid_info['img_tiles'].keys()}
        total_counts = {'totalPigLying': 0,
                        'totalPigNotLying': 0, 'totalSummed': 0}

        for pig, posture in zip(pig_detections, posture_classifications):
            # Convert tensor to python scalar for each bounding box coordinate
            x1, y1, x2, y2 = [coordinate.item() for coordinate in pig['box']]

            # Initialize variables for tracking the best match (highest IoU)
            best_iou = -1
            best_key = None

            # Check if the detected pig bounding box falls inside a certain grid
            for key, val in self.grid_info['img_tiles'].items():
                x1_grid, y1_grid, x2_grid, y2_grid = val
                iou = self.calculate_iou(
                    [x1, y1, x2, y2], [x1_grid, y1_grid, x2_grid, y2_grid])

                # If this box has a higher IoU than the current best, update the best match
                if iou > best_iou:
                    best_iou = iou
                    best_key = key

            # Update the counter for the best match
            if best_key is not None:
                if posture.item() == 0:
                    count_dict[best_key]['pigLying'] += 1
                    total_counts['totalPigLying'] += 1
                elif posture.item() == 1:
                    count_dict[best_key]['pigNotLying'] += 1
                    total_counts['totalPigNotLying'] += 1

                count_dict[best_key]['summed'] += 1
                total_counts['totalSummed'] += 1

        return {'per_grid_cell': count_dict, 'total': total_counts}

    def sum_pigs_in_sectors(count_dict, sectors):
        """
        This function calculates the total number of pigs in each given sector,
        as well as the total number of pigs that are lying down and not lying down.

        Parameters:
        count_dict: dict
            A dictionary containing the counts of pigs for each sector. It should have the 
            structure described in the 'count_pigs_and_total' function.
        sectors: list
            A list of sectors to consider. Each sector is a string corresponding to a key
            in the 'per_grid_cell' dictionary of the 'count_dict'.

        Returns:
        total_counts: dict
            A dictionary containing the total count of pigs, the total count of pigs lying down,
            and the total count of pigs not lying down, in the given sectors.
        """
        pigs_in_sectors = {'totalPigs': 0,
                           'totalPigLying': 0, 'totalPigNotLying': 0}

        for sector in sectors:
            sector_counts = count_dict['per_grid_cell'].get(
                sector, {'pigLying': 0, 'pigNotLying': 0, 'summed': 0})
            pigs_in_sectors['totalPigs'] += sector_counts['summed']
            pigs_in_sectors['totalPigLying'] += sector_counts['pigLying']
            pigs_in_sectors['totalPigNotLying'] += sector_counts['pigNotLying']

        return pigs_in_sectors

    def get_grid_count_difference(self, grid_count_holder_t, grid_count_holder_t_plus_1):
        """
        This function calculates the difference in the number of pigs in each grid cell between two consecutive frames.

        Args:
            grid_count_holder_t (dict): Grid count dictionary for frame at time t, where each key is a grid cell ID and 
                its value is another dictionary with the number of lying and not lying pigs, and the total.
                e.g., 
                {
                    'per_grid_cell': {
                        '00': {'pigLying': 1, 'pigNotLying': 0, 'summed': 1},
                        '01': {'pigLying': 0, 'pigNotLying': 0, 'summed': 0},
                        ...
                        '32': {'pigLying': 0, 'pigNotLying': 1, 'summed': 1}
                    },
                    'total': {'totalPigLying': 1, 'totalPigNotLying': 1, 'totalSummed': 2}
                }

            grid_count_holder_t_plus_1 (dict): Grid count dictionary for frame at time t+1 (same format as grid_count_holder_t).

        Returns:
            diff_dict (dict): Dictionary showing the difference in the number of pigs in each grid cell between the two frames.
        """
        diff_dict = {}
        for key in grid_count_holder_t['per_grid_cell'].keys():
            diff_dict[key] = max(0, grid_count_holder_t_plus_1['per_grid_cell'][key]
                                 ['summed'] - grid_count_holder_t['per_grid_cell'][key]['summed'])
        return diff_dict

    def calculate_activity(self, grid_count_holder_t, grid_count_holder_t_plus_1):
        """
        This function calculates an activity metric for a video frame.

        Args:
            grid_count_diff (dict): Dictionary showing the difference in the number of pigs in each grid cell between two consecutive frames.
                e.g.,
                {
                    '00': 1,
                    '01': 0,
                    ...
                    '32': 1
                }

            grid_count_holder_t_plus_1 (dict): Grid count dictionary for frame at time t+1, where each key is a grid cell ID and 
                its value is another dictionary with the number of lying and not lying pigs, and the total.
                e.g., 
                {
                    'per_grid_cell': {
                        '00': {'pigLying': 1, 'pigNotLying': 0, 'summed': 1},
                        '01': {'pigLying': 0, 'pigNotLying': 0, 'summed': 0},
                        ...
                        '32': {'pigLying': 0, 'pigNotLying': 1, 'summed': 1}
                    },
                    'total': {'totalPigLying': 1, 'totalPigNotLying': 1, 'totalSummed': 2}
                }

        Returns:
            activity (float): An activity metric for the frame, calculated as the sum of the grid cell differences multiplied by the 
            ratio of not lying to lying pigs.
        """

        total_not_lying = grid_count_holder_t_plus_1['total']['totalPigNotLying']
        total_lying = grid_count_holder_t_plus_1['total']['totalPigLying']

        # Check if there are no pigs at all
        if total_not_lying == 0 and total_lying == 0:
            return 0

        # Get the difference in grid counts between the two frames
        diff_counts = self.get_grid_count_difference(
            grid_count_holder_t, grid_count_holder_t_plus_1)

        # Sum all the differences to get the total change in grid counts
        total_change = sum(diff_counts.values())

        # Calculate the ratio of notLying pigs to lying pigs
        ratio = total_not_lying / max(1, total_lying)
        ratio = ratio / (ratio + 1)

        # Multiply the total change in grid counts by the ratio to get the activity index
        activity = total_change * ratio

        return activity

    def get_activity(self, pig_detections_t, posture_classifications_t, pig_detections_t_plus_1, posture_classifications_t_plus_1):
        """
        Calculates the activity level based on pig detections and posture classifications at two time points.

        Parameters:
            pig_detections_t (list): List of pig detections at time point t.
            posture_classifications_t (list): List of posture classifications at time point t.
            pig_detections_t_plus_1 (list): List of pig detections at time point t+1.
            posture_classifications_t_plus_1 (list): List of posture classifications at time point t+1.

        Returns:
            activity (float): Activity level at the given time points, rounded to 2 decimal places.
        """

        grid_count_holder_t = self.count_pigs_and_total(
            pig_detections_t, posture_classifications_t)
        grid_count_holder_t_plus_1 = self.count_pigs_and_total(
            pig_detections_t_plus_1, posture_classifications_t_plus_1)

        activity = self.calculate_activity(
            grid_count_holder_t, grid_count_holder_t_plus_1)
        activity = round(activity, 2)

        return activity
