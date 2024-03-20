# Outputs Directory Structure

This directory contains the results of various processes related to the pig detection and tail posture monitoring system. Below is an explanation of the contents and structure of each subdirectory within `outputs`.

## Directory Contents

### images
Contains cropped images from pig detection results. These images are the output of the detection algorithm, which identifies and isolates instances of pigs in the monitored environment.

### monitoring_pipeline
Holds the results from the tail posture monitoring pipeline. The subdirectories are named `Kamera1` through `Kamera6`, corresponding to the camera from which the data was collected. Each `Kamera` directory contains subdirectories labeled with the timespans (YYYYMMDD_YYYYMMDD) that have been analyzed.

#### Kamera Folders Structure
Each `Kamera` folder follows the same structure:
- `YYYYMMDD_YYYYMMDD`: This represents the date range for which the monitoring data is available.

### plots
Includes various test plots or other visualizations based on the results from the monitoring pipeline. These could be used for presentations, reports, or further analysis.

#### Notable Plot Files
- `activity_heatmap_YYYYMMDD_YYYYMMDD`: Heatmaps representing activity over a specific period.
- `avg_activity_YYYYMMDD_1m_with_posture.svg`: An average activity plot over a month with posture data.

## Naming Conventions
Files and directories follow a consistent naming convention for ease of identification:
- `YYYYMMDD`: Date in the format of year, month, and day.
- `HHMMSS`: Time in the format of hours, minutes, and seconds.

Files are typically in `.png`, `.jpg`, or `.svg` format, providing both raster and vector options for various use cases.

## Additional Notes
Please ensure that you have the necessary permissions to access and manipulate the data within these directories. The data is structured for easy access, but confidentiality and data protection guidelines must be followed at all times.
