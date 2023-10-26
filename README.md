# Image Segmentation and Colorization with Unsupervised Learning

## Overview

![Mon_Mar_14_18-05-55_2022_1](https://user-images.githubusercontent.com/95425179/166098314-4a28304d-e234-47a3-82b6-2e12886acfcd.jpg)

This repository contains a Python project that focuses on image processing, unsupervised learning, and the automatic segmentation and colorization of images. The primary goal is to take an input image and divide it into meaningful regions while automatically applying colorization based on these regions. This is accomplished without the need for manual annotation or labeling.

![Mon_Mar_14_18-05-55_2022_0](https://user-images.githubusercontent.com/95425179/166098311-eae87f04-c355-47a6-8b8c-c6421936de7d.jpg)

The project comprises several key components:

- **Data Preprocessing**: Initially, the project selects a specified number of pixels from an input image. These pixels are randomly sampled, and various attributes such as color values and surrounding neighborhood information are extracted for each selected pixel. The resulting dataset holds essential information about these sampled pixels.

- **Unsupervised Learning**: Principal Component Analysis (PCA) is applied to the dataset. PCA reduces the dimensionality of the data, enabling efficient analysis and visualization. It helps identify patterns and structure in the image, grouping pixels with similar attributes together. This process is crucial for the subsequent segmentation and colorization steps.

- **Image Masking**: Based on the regions identified through unsupervised learning, masks are created. These masks serve as guidelines to separate the image into distinct segments, each corresponding to a specific region. The mask for each region highlights the pixel locations belonging to that region, serving as a visual representation of the segmentation.

- **Colorization**: Once the image is segmented into different regions, the next step is to colorize each region. Colors are applied to the segmented regions based on the information learned from the dataset. This step transforms the segmented image into a visually meaningful result, allowing for better understanding and interpretation.

- **Visualization**: The final output of the project includes various visual components. It displays the original image with segmentation masks overlaid, a colorized version of the image that highlights different regions, and the individual masks that represent each region. This visualization helps users understand how the unsupervised learning process has partitioned and colorized the image.

## Getting Started

To use this project, you'll need Python installed, along with the following libraries:

- NumPy
- OpenCV
- Matplotlib

Make sure you've downloaded the provided image dataset or use your own images for experimentation. To start processing and analyzing images, you can follow the provided Python scripts:

1. `image_processing.py`: Contains essential functions for data preprocessing, unsupervised learning, and segmentation.
2. `image_segmentation.py`: Implements the image segmentation using unsupervised learning techniques.
3. `image_colorization.py`: Handles the colorization of segmented regions.
4. `show_images.py`: Provides functions for displaying the results, saving them, and visualizing the segmentation and colorization.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the required dependencies installed.
3. Add your images to the project directory or specify the image file path in the code.
4. Run the scripts to process and analyze the images.

## Examples

You can find example images in the `exo2_test` folder. Use these examples to test and explore the capabilities of the project.


