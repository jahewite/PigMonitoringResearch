# Configuration File Documentation

The configuration file is written in JSON format and holds configuration details for each camera, identified by the camera name. 

The configuration details consist of two main elements: `cut_ratio` and `shapes`.

## cut_ratio

`cut_ratio` is a decimal number between 0 and 1 that indicates the percentage of the image's bottom part to be cut off during image preprocessing. For example, a `cut_ratio` of 0.15 implies that the bottom 15% of the image will be removed.

## shapes

`shapes` is a list of dictionaries. Each dictionary represents an object of interest, such as a feeder or a drinker, present in the camera view. The dictionary is composed of the following keys:

### label

The `label` field indicates the type of object the shape represents. The values can be 'feeder' or 'drinker'.

### points

The `points` field describes the bounding box of the object in the image. The points are represented by a list of two lists, each containing the x and y coordinates of the top left and bottom right corners of the bounding box, respectively.