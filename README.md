# K-MEANS CLUSTERING

This script clusters images using the K-means algorithm. It first preprocesses the images by resizing them to 16x16 and converting them to grayscale. Then, it normalizes the image features using MinMaxScaler and applies PCA to reduce the dimensionality. The optimal number of clusters is found by evaluating the Silhouette Score for a range of possible values. The final clustering is saved to a text file, and an HTML file is created for visualizing the clusters.

## Method Description

The method used in this script is as follows:

1. Preprocess images by resizing them to 16x16 and converting them to grayscale.
2. Normalize the image features using MinMaxScaler.
3. Apply PCA to reduce the dimensionality of the image features.
4. Determine the optimal number of clusters (k) by evaluating the Silhouette Score for a range of k values.
5. Apply the K-means clustering algorithm with the optimal k value.
6. Save the clusters to a text file and create an HTML file for visualization.
   
The distance metric used for clustering and calculating the Silhouette Score is the Euclidean distance.

## Expected Duration of Execution

Execution time depends on the number of images, their sizes, and the range of cluster values ​​considered. For 7600 images, the script should take about 2 minutes to run.

## Custom Options

The script accepts the following command-line arguments:

1. 'input_file': Path to the input file containing the image file paths.
2. 'output_file': Path to the output file containing the clustered image file names.
3. 'output_html': Path to the output HTML file for visualizing the clusters.

## How to Run

Detailed instructions for running the code on Linux and Windows systems can be found in the project files named 'how_to_run_linux.txt' and 'how_to_run_windows.txt', respectively.