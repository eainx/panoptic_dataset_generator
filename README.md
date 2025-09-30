# 3D-Dynamic-Scene-Processing-Pipeline

This repository contains a comprehensive pipeline for processing multi-view video data to generate dynamic 3D point clouds and filter camera trajectories. 
It is to preprocess data to execute Dynamic 3D Gaussians (https://github.com/JonathonLuiten/Dynamic3DGaussians) and post-process to get point trajectories, according to TAPVid-3D (https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d).

## 🚀 Key Features

* **Multi-view Image Processing**: Handles distortion correction, resizing, and segmentation mask generation for HD images from multiple cameras.
* **3D Point Cloud Generation**: Creates detailed 3D point clouds from Kinect depth data and projects them into HD camera views.
* **Metadata Generation**: Produces JSON metadata files containing camera intrinsics, extrinsics, and file paths for training.
* **Trajectory Filtering**: Filters 3D trajectories based on visibility and motion criteria to select dynamic points of interest.
* **Data Post-processing**: Splits the filtered data into 'easy', 'medium', and 'hard' camera sets for evaluation.
* **Visualization**: Generates videos visualizing the filtered 2D tracks on the corresponding image sequences.
* **Configurable Experiments**: Easily define and manage different experiments and data sequences through a structured configuration system.

## ⚙️ Project Structure

The project is organized into the following key modules:

* `main.py`: The main entry point for running the different processing steps of the pipeline.
* `config.py`: Contains the configuration classes for different experiments and datasets. This is where you define file paths, camera parameters, and frame ranges.
* `data_processing/`: A directory containing the core processing modules:
    * `image_processor.py`: Handles image-related tasks like resizing, undistortion, and segmentation.
    * `pointcloud_generator.py`: Generates the initial 3D point cloud from depth data.
    * `metadata_generator.py`: Creates the metadata files required for training.
    * `trajectory_filter.py`: Implements the logic for filtering 3D trajectories.
    * `post_processor.py`: Splits the output data based on camera difficulty.
    * `visualize.py`: Contains the `Visualizer` class for creating trajectory visualizations.
* `utils/`: A directory with utility functions for camera operations and common tasks:
    * `camera_utils.py`: Includes functions for unprojecting depth, finding nearest cameras, and interpolating colors.
    * `common_utils.py`: Provides helper functions, such as reading depth data.

## Workflow

The data processing pipeline follows these sequential steps:

1.  **Image Processing** (`image`):
    * Reads the original HD images from the Panoptic dataset.
    * Corrects for lens distortion.
    * Resizes images to the target dimensions.
    * Generates segmentation masks for persons and other specified objects using a Mask R-CNN model.
    * Saves the processed images and masks to the output directory.

2.  **Point Cloud Generation** (`pointcloud`):
    * Loads calibration and synchronization data.
    * For a specified initial frame, it reads depth data from all Kinect sensors.
    * Unprojects the depth images into 3D points in the world coordinate system.
    * Assigns colors to the 3D points by projecting them into the nearest HD camera view and interpolating the pixel colors.
    * Saves the final colored point cloud as a `.ply` file and in `.npz` format.

3.  **Metadata Generation** (`meta`):
    * Reads the camera calibration data.
    * Generates `train_meta.json` which contains:
        * Image dimensions (`w`, `h`).
        * Camera intrinsic matrices (`k`).
        * World-to-camera transformation matrices (`w2c`).
        * A list of file names (`fn`) for each frame and camera.

4.  **Trajectory Filtering** (`filter`):
    * Loads the 3D Gaussian Splatting results and the generated metadata.
    * Selects initial query points from the segmented regions in the first frame of a reference camera.
    * Computes the 2D tracks and visibility of these points across all frames for all cameras.
    * Filters the tracks based on two main criteria:
        1.  **Foreground Presence**: The track must be visible in the foreground (based on segmentation masks) for a minimum ratio of the total frames.
        2.  **Motion**: The track must exhibit a minimum amount of motion in the 2D image plane.
    * Saves the filtered trajectories, visibility information, and corresponding video frames into a single `annotation.pkl` file.

5.  **Post-processing** (`postprocess`):
    * Loads the stacked `annotation.pkl` file.
    * Splits the data based on the `EASY_CAMS`, `MEDIUM_CAMS`, and `HARD_CAMS` lists defined in the configuration.
    * Saves separate `annotation.pkl` files for each difficulty level.

6.  **Visualization** (`visualize`):
    * Loads the stacked `annotation.pkl` file.
    * For each camera, it renders a video that overlays the filtered 2D trajectories onto the corresponding RGB image sequence.
    * Saves the output videos as `.mp4` files.

## 🛠️ Prerequisites

Before running the pipeline, ensure you have the following dependencies installed:

* Python 3.x
* OpenCV
* NumPy
* Open3D
* PyTorch
* TorchVision
* Matplotlib
* tqdm
* SciPy

You can install the required Python packages using pip:
```bash
pip install opencv-python numpy open3d torch torchvision matplotlib tqdm scipy
