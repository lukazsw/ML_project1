import os
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import argparse

def get_file_list(input_file):
    with open(input_file, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    return file_list

def preprocess_images(file_list):
    resized_images = []
    for file in file_list:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        resized_images.append(img)
    return np.array(resized_images)

def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def apply_pca(features):
    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(features)
    return pca_features

def cluster_images_kmeans(images, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(images)
    return labels

def calculate_silhouette_score(distance_matrix, labels):
    try:
        silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
        return silhouette_avg
    except ValueError:
        print("Silhouette Score calculation failed. This might be due to having only one sample in a cluster.")
        return None

def save_clusters_to_file(labels, file_list, output_file):
    clusters = {label: [] for label in np.unique(labels)}

    for file, label in zip(file_list, labels):
        clusters[label].append(os.path.basename(file))

    with open(output_file, 'w') as f:
        for cluster in clusters.values():
            f.write(" ".join(cluster))
            f.write("\n")

def create_html_visualization(labels, file_list, output_html):
    clusters = {label: [] for label in np.unique(labels)}

    for file, label in zip(file_list, labels):
        clusters[label].append(file)

    with open(output_html, 'w') as f:
        for cluster in clusters.values():
            for image_file in cluster:
                f.write(f'<img src="{image_file}" width="12" height="12">')
            f.write('<hr>')

def optimal_k(images, min_clusters, max_clusters):
    best_silhouette_avg = -1
    best_k = 0

    for k in range(min_clusters, max_clusters+1):
        labels = cluster_images_kmeans(images, k)
        distance_matrix = pairwise_distances(images)
        silhouette_avg = calculate_silhouette_score(distance_matrix, labels)

        if silhouette_avg is not None and silhouette_avg > best_silhouette_avg:
            best_silhouette_avg = silhouette_avg
            best_k = k

    return best_k, best_silhouette_avg

def main(input_file, output_file, output_html):
    file_list = get_file_list(input_file)
    images = preprocess_images(file_list)
    reshaped_images = images.reshape(images.shape[0], -1)
    normalized_images = normalize_features(reshaped_images)
    pca_features = apply_pca(normalized_images)
    
    min_clusters = 16
    max_clusters = 36
    best_k, best_silhouette_avg = optimal_k(pca_features, min_clusters, max_clusters)
    print(f"Best number of clusters (k): {best_k}")
    print(f"Best Silhouette Score: {best_silhouette_avg}")

    labels = cluster_images_kmeans(pca_features, n_clusters=best_k)
    save_clusters_to_file(labels, file_list, output_file)
    create_html_visualization(labels, file_list, output_html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster characters in images')
    parser.add_argument('input_file', help='Input file containing image file paths')
    parser.add_argument('output_file', help='Output file containing clustered image file names')
    parser.add_argument('output_html', help='Output HTML file to visualize clusters')
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.output_html)