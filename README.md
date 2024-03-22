# Overview

This repository contains the code and associated materials for a series of publications on image-based monitoring techniques in pig farming. The projects aim to leverage image processing and machine learning to improve monitoring practices for pig behavior and welfare. Included works are:

- [Image-based Activity Monitoring of Pigs](https://www.researchgate.net/publication/378804050_Image-based_activity_monitoring_of_pigs)
- [Image-based Tail Posture Monitoring of Pigs](https://www.researchgate.net/publication/374156938_Image-based_Tail_Posture_Monitoring_of_Pigs)
- [ECPLF Publication]

It also includes links to the datasets and models cited in the publications. For details on data usage and restrictions, see the [License](#license) section. For a detailed description of the dataset, refer to the [Dataset](#dataset) section.

Check out the [data exploration](data_exploration) directory for more details.

## Prerequisites

- Python 3.6 or higher
- Git
- FFmpeg

## Setup

This section guides you through setting up the project environment, including downloading the necessary models and data files.

### 1. Clone the Repository

First, clone this repository:

```
git clone https://github.com/jahewite/PigMonitoringResearch.git
```

### 2. Install dependencies

After cloning the repository, navigate into the project directory and install the project dependencies:

```
cd PigMonitoringResearch

# recommended: create venv
python3 -m venv *env_name*
# activate env
source *env_name*/bin/activate

# install packages
pip install -r requirements.txt
```

### 3. Download necessary files

- **Models**: Download the model files from [here](https://cloudstorage.uni-oldenburg.de/s/HMbifg6gtb8EaCY).
  - Place the **posture_classification.pth** file in the 'assets/models/classification/posture_classification/efficientnetv2' directory.
  - Place the **pig_detection.pt** file in the 'assets/models/detection/pig_detection/yolo/pig_detection.pt' directory.
  - Place the **tail_posture_detection.pt** file in the 'assets/models/detection/tail_detection/yolo/tail_posture_detection.pt' directory.
- **Test video clip**: Download the test video clip from here and place it in the 'assets/test_clips' directory.
- **Pipeline result files**: Download the .csv files containing the pipeline results for each analyzed pen from [here](https://cloudstorage.uni-oldenburg.de/s/a9B9QgYxqx6zFwf) and place them in the 'pipeline_outputs/monitoring_pipeline/piglet_rearing' directory.
- **Optional**: Download the aggregated pipeline results .csv files from [here](https://cloudstorage.uni-oldenburg.de/s/bnsFTT5MGk7eKCt) and place them in the 'pipeline_outputs/data_aggregation' directory for individual use.

### (Optional) Installing FFmpeg

This project requires FFmpeg for video processing capabilities. Please follow the instructions below to install FFmpeg on your system:

#### Windows
1. Download the FFmpeg build from https://ffmpeg.org/download.html.
2. Extract the files to a location on your computer.
3. Add the path to the FFmpeg bin folder (e.g., `C:\path\to\ffmpeg\bin`) to your system's Environment Variables under `Path`.

#### macOS
Use Homebrew to install FFmpeg:
```
brew install ffmpeg
```

#### Linux
Use apt-get to install FFmpeg:
```
sudo apt-get update
sudo apt-get install ffmpeg
```

## Example usage

If downloaded, use the test clip for example usage. From the root directory, execute the following command:
```
python3 -m pipeline.analyze_video --path_to_videos assets/test_clips/test_1.mp4
```

## Datasets

This project includes all the datasets used in the research papers. The data can be downloaded here.

### Data sources

The datasets include many different data sources. It contains iamges from:
- **Psota et al.**: [Multi-Pig Part Detection and Association with a Fully-Convolutional Network](https://www.mdpi.com/1424-8220/19/4/852)
- **Alameer et al.**: [Automated recognition of postures and drinking behaviour for the detection of compromised health in pigs](https://www.nature.com/articles/s41598-020-70688-6#data-availability)
- **Riekert et al.**: [Automatically detecting pigposition and posture by 2D camera imaging anddeep learning](https://www.sciencedirect.com/science/article/pii/S0168169918318283)
- **InnoPig project**: Funded by the Federal Ministry of Food and Agriculture (BMEL), grant number: 2817205413
- **KoVeSch project**: Funded by the Federal Office for Agriculture and Food (BLE), grant number: 2819109817

### Pig detection dataset

- Format: YOLO
- Contents: Contains X images annotated with the class 'Pig'.
- Purpose: Designed for detecting pigs within various environmental settings.

### Posture classification dataset

- Format: Standard image classification format
- Contents: Comprises X images classified into 'lying' and 'notLying'.
- Purpose: Facilitates the classification of pig postures to distinguish between lying and not lying behaviors.

### Tail posture detection datasets

There are two datasets focused on the detection of pig tail postures, annotated with the classes 'upright' and 'hanging':

1. **Full Image Dataset**:
- Format: YOLO
 - Contents: Includes X images where tail postures are annotated directly on the input images.
 - Purpose: Aims to identify tail postures within unaltered farm environment images.
1. **Cropped Pig Detections Dataset**:
- Format: YOLO
 - Contents: Features X images of cropped pig detections, annotated for tail posture. These crops are derived using the pig detection model.
 - Purpose: Enhances the focus on tail postures by using cropped images that highlight the area of interest.
 - Additional Information: For detailed insights into the methodology behind cropped pig detections and tail posture annotation, refer to the research papers.

## License

This project is licensed under multiple licenses:

- The code is licensed under the GNU Affero General Public License version 3 (AGPL-3.0), as found in the [LICENSE_CODE.txt](./LICENSE_CODE.txt) file. The AGPL-3.0 license offers strong copyleft provisions, requiring that any modifications and derived works also be available under the same license, and extends these requirements to networked use of the software.
- The dataset provided in this project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license, as detailed in the [LICENSE_DATA](./LICENSE_DATA.txt) file. This license allows others to remix, tweak, and build upon the work non-commercially, as long as they credit the contributors and license their new creations under the identical terms.

Please see the respective license files for more detailed information.

## Acknowledgements
This project was developed as part of the [DigiSchwein](https://www.lwk-niedersachsen.de/lwk/news/35309_DigiSchwein) project.

The project is supported (was supported) by fundsof the Federal Ministry of Food and Agriculture (BMEL) based on a decision of the Parliament of theFederal Republic of Germany. The Federal Office for Agriculture and Food (BLE) provides (provided)coordinating support for digitalization in agriculture asfunding organization, grantnumber 28DE109A18.
