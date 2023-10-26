# Partitionnement_image

Le partitionnement d’image, la catégorisation de données ou la régression de nuage de points ont en commun une solution simple à voir à l’œil nu. 
Comme sur une image de parking ou notre cerveau  différencie très facilement une voiture du reste du cliché, la question est de traduire cette intuition numériquement.
Nous avons écrit un programme d’apprentissage non superviser qui partitionnement des images pour en  extraire les masques des différents objets que le code reconnaitra.




# Unsupervised Image Processing with Python

This Python project demonstrates the use of unsupervised image processing techniques, including dimensionality reduction with Principal Component Analysis (PCA) and clustering with K-means, to partition and enhance images. The project allows you to preprocess images, perform unsupervised learning, and apply color masking to create artistic visual effects.

![Mon_Mar_14_18-05-55_2022_1](https://user-images.githubusercontent.com/95425179/166098314-4a28304d-e234-47a3-82b6-2e12886acfcd.jpg)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How it Works](#how-it-works)
- [Preprocessing](#preprocessing)
- [Unsupervised Learning](#unsupervised-learning)
- [Image Processing](#image-processing)
- [Examples](#examples)
- [Key Functions](#key-functions)

## Overview

This project is an image processing application that utilizes unsupervised learning techniques to segment and colorize images. The primary objective is to automatically divide an image into meaningful regions and apply colorization based on these regions without the need for manual annotation or labeling. Here's a breakdown of the project components:

![Mon_Mar_14_18-05-55_2022_0](https://user-images.githubusercontent.com/95425179/166098311-eae87f04-c355-47a6-8b8c-c6421936de7d.jpg)

**Data Preprocessing:** One of the initial steps in the project involves selecting a specified number of pixels from an input image. These pixels are chosen randomly from the image. For each selected pixel, various attributes are extracted, such as its color values and information about its surrounding neighborhood. The result is a dataset that contains information about these sampled pixels.

**Unsupervised Learning:** To make sense of the data, Principal Component Analysis (PCA) is applied to the dataset. PCA reduces the dimensionality of the data, enabling efficient analysis and visualization. It helps identify patterns and structure in the image, grouping pixels with similar attributes together. This process is essential for the subsequent segmentation and colorization steps.

**Image Masking:** Based on the regions identified through unsupervised learning, masks are created. These masks serve as guidelines to separate the image into distinct segments, each corresponding to a specific region. The mask for each region highlights the pixel locations that belong to that region, which can be seen as a visual representation of the segmentation.

**Colorization:** Once the image is segmented into different regions, the next step is to colorize each region. The colors are applied to the segmented regions based on the information learned from the dataset. This step transforms the segmented image into a visually meaningful result, allowing for better understanding and interpretation.

**Visualization:** The final output of the project includes various visual components. It displays the original image with segmentation masks overlaid, a colorized version of the image that highlights different regions, and the individual masks that represent each region. This visualization helps users understand how the unsupervised learning process has partitioned and colorized the image.

This project offers a unique perspective on image analysis by combining unsupervised learning with image processing to automatically interpret and segment images. The resulting colorization and visualizations provide valuable insights and aesthetic enhancements for a wide range of applications, from computer vision to art and design.

## Key Features

- Image Preprocessing: Select a region of interest from an image and extract data points.
- Unsupervised Learning: Perform Principal Component Analysis (PCA) to reduce dimensionality and apply K-means clustering.
- Color Masking: Generate color masks based on clustering results and apply them to the image.
- Customizable Parameters: Adjust the number of data points, zoom factor, and weights for PCA.

## Usage

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Ensure you have Python installed on your system and install the required libraries by running the following command:

```bash
pip install numpy matplotlib opencv-python
```

3. **Run the Project**: Execute the main script `unsupervised_image_processing.py`.

```bash
python unsupervised_image_processing.py
```

4. **Explore Different Images**: Experiment with different images by changing the `img_path` variable in the script. You can use the provided image filenames in the `img_path` list.

## Getting Started

### Prerequisites

Before running the project, make sure you have the following prerequisites:

- Python 3
- NumPy
- Matplotlib
- OpenCV (opencv-python)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/unsupervised-image-processing.git
cd unsupervised-image-processing
```

2. Install the required dependencies as mentioned in the Prerequisites section.

## How it Works

The project follows a series of steps to process images:

1. **Preprocessing**: Choose a region of interest from the input image and extract data points. Adjust parameters like the number of pixels and zoom factor.

2. **Unsupervised Learning**: Apply Principal Component Analysis (PCA) to reduce dimensionality. Use the K-means algorithm to cluster the data into categories.

3. **Image Processing**: Generate color masks based on the clustering results and apply them to the image.

4. **Examples**: The project provides sample images to showcase the results of unsupervised image processing.


## Key Functions:

- `esperance(Xi)`: Calculates the unbiased estimate of the expectation of a vector `Xi`.
- `variance(Xi)`: Computes the asymptotically biased estimate of the variance of a vector `Xi`.
- `centre_red(R)`: Centers and scales data vectors to have an expectation of 0 and a variance of 1.
- `ACP(X, q, w)`: Performs Principal Component Analysis (PCA) on a data matrix `X` and reduces it to `q` dimensions.
- `Kmoy(A, k, err)`: Implements the k-means clustering algorithm for categorizing data points into `k` categories.
- `load_image(file_name)`: Loads an image from a file.
- `data_pixels(img, I, J, radius)`: Extracts pixel data from an image, including color values and nearby color averages.
- `ACP_img(data_img, w)`: Applies PCA to image data and returns the transformed data along with the Kaiser rule value.
- `Kmoy_img(data_img, kaiser, err)`: Utilizes k-means clustering on image data based on PCA results.
- `masque(img, S, color)`: Generates a color mask for an image based on clustering results.
- `remplissage_masque(img_masque_ponctuel)`: Fills in color masks with color diffusion.
- `true_color_masque(img_masque, img, kaiser)`: Creates independent images for each category defined by k-means clustering.
- `show(mosaic, img_masque, save)`: Displays and optionally saves the processed images.

## Examples

The project includes various example images that demonstrate the capabilities of unsupervised image processing. These examples highlight the creative effects that can be achieved with the techniques used in the project.




