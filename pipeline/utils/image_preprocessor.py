import os

class ImagePreprocessor:
    """
    Preprocesses images for analysis by cutting specified portions based on predefined cut ratios. 
    This preprocessor is designed to work with images from specific cameras, where each camera 
    might require different areas of the image to be excluded from analysis due to variations 
    in camera setup, field of view, or specific areas of interest.

    Attributes:
        config (dict): Configuration settings loaded from a JSON file, containing preprocessing details
                       for one or more cameras. This includes cut ratios for each camera to specify
                       what portions of the image should be removed during preprocessing.
        camera_name (str): The name of the camera, determined either directly from a provided prefix
                           or parsed from the video file path. This name is used to retrieve camera-specific
                           preprocessing settings from the config.
        cut_ratios (dict): Cut ratios for the current camera, dictating the portions of the image to be cut
                           from the top, right, left, and bottom edges. Ratios are expressed as fractions of
                           the total image height or width.

    Methods:
        cut_img_portion(img):
            Cuts the specified portions from an image according to the cut ratios. This method modifies
            the input image to focus on the area of interest by removing unnecessary or distracting
            parts of the image that do not contribute to the analysis.

    Parameters:
        config (dict): A dictionary containing the preprocessing configuration for each camera.
        video_path (str, optional): The path to the video file from which the camera name should be
                                    parsed. Required if 'prefix' is not provided.
        prefix (str, optional): An optional prefix to directly specify the camera name. Takes precedence
                                over 'video_path' for determining the camera name.
    """

    def __init__(self, config, video_path=None, prefix=None):
        """
        Initializes the ImagePreprocessor with configuration settings and determines the camera name
        based on either a provided video path or prefix.
        """
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
        Cuts specified portions from the input image based on the preconfigured cut ratios for the camera.

        Args:
            img (np.ndarray): The input image array to be preprocessed.

        Returns:
            np.ndarray: The resultant image after cutting specified portions from the top, right,
                        bottom, and left, according to the cut ratios.
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