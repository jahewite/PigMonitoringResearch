import os

class ImagePreprocessor:
    """
    A class used to preprocess images based on the configuration and video path.
    ...
    Attributes
    ----------
    config : dict
        A dictionary containing the configuration settings loaded from a JSON file.
    camera_name : str
        The name of the camera parsed from the video file path or given prefix.
    cut_ratios : dict
        A dictionary containing the cut ratios for the top, right, left, and bottom of the image.

    Methods
    -------
    cut_img_portion(img):
        Cuts the portions of an image based on the cut ratios.
    """

    def __init__(self, config, video_path=None, prefix=None):
        self.config = config

        # If prefix is given, use it as the camera name
        if prefix:
            self.camera_name = prefix
        else:
            # Else, parse the camera name from the video path
            filename = os.path.basename(video_path)  # Get the filename
            # The camera name is the part before the first '-'
            self.camera_name = filename.split('-')[0]

        # Assuming it's the first dict in the list
        self.cut_ratios = self.config[self.camera_name]['cut_ratio'][0]

    def cut_img_portion(self, img):
        """
        This function cuts the specified portions of an image based on the cut ratios.

        Args:
            img (np.ndarray): The image array.

        Returns:
            cut_img (np.ndarray): The resultant image after cutting the specified portions.
        """

        # Get the height and width of the image
        height, width = img.shape[:2]

        # Define the start and end points of the portion to cut for each edge
        start_row = int(height * self.cut_ratios['top'])
        end_row = int(height * (1 - self.cut_ratios['bottom']))

        start_col = int(width * self.cut_ratios['left'])
        end_col = int(width * (1 - self.cut_ratios['right']))

        # Use slicing to cut out the required portion of the image
        cut_img = img[start_row:end_row, start_col:end_col]

        return cut_img