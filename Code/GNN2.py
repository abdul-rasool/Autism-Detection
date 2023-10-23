import os
import cv2
import numpy as np
from skimage.segmentation import slic
import networkx as nx


dataset_path = "F:/images/"

# Function to load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    labels = []
    for class_folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, class_folder)):
            class_label = class_folder
            for image_file in os.listdir(os.path.join(folder_path, class_folder)):
                if image_file.endswith(".png"):
                    image_path = os.path.join(folder_path, class_folder, image_file)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (224, 224))  # Resize to a common dimension
                    images.append(img)
                    labels.append(class_label)
    return np.array(images), labels

# Load and preprocess images
images, labels = load_and_preprocess_images(dataset_path)
# Process a single image for demonstration
demo_image = images[0]

# Performing superpixel segmentation using SLIC
superpixels = slic(demo_image, n_segments=100, compactness=10)

# Creating a graph
G = nx.Graph()

# Treating each superpixel as a node and adding it to the graph
for i in range(np.max(superpixels) + 1):
    G.add_node(i, label=i)



# Finding adjacent superpixels and adding edges between them
for i in range(np.max(superpixels) + 1):
    # Get pixel coordinates of superpixel i
    coords_i = np.column_stack(np.where(superpixels == i))

    # Checking if superpixel i has no pixels; if true, skip to the next iteration
    if coords_i.size == 0:
        continue

    # Comparing superpixel i with all subsequent superpixels
    for j in range(i + 1, np.max(superpixels) + 1):
        # Get pixel coordinates of superpixel j
        coords_j = np.column_stack(np.where(superpixels == j))

        # Checking if superpixel j has no pixels; if true, skip to the next iteration
        if coords_j.size == 0:
            continue

        # Calculating pairwise distances between all pixels in superpixels i and j
        dists = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=2)

        # If the minimum distance is below a threshold, superpixels are adjacent
        if np.min(dists) < 1.5:  # threshold = 1.5 pixels allows for direct neighbors
            G.add_edge(i, j)
import matplotlib.pyplot as plt

def visualize_superpixels_and_edges(image, superpixels, graph):
    # Overlay superpixel boundaries on the image
    boundaries = (superpixels == -1)
    image_boundaries = image.copy()
    image_boundaries[boundaries, :] = [0, 0, 255]

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(image_boundaries)

    # Draw edges between superpixel centers
    for edge in graph.edges():
        superpixel_a = edge[0]
        superpixel_b = edge[1]

        center_a = np.mean(np.column_stack(np.where(superpixels == superpixel_a)), axis=0)
        center_b = np.mean(np.column_stack(np.where(superpixels == superpixel_b)), axis=0)

        ax.plot([center_a[1], center_b[1]], [center_a[0], center_b[0]], 'r-')

    plt.show()

# Calling the function
visualize_superpixels_and_edges(demo_image, superpixels, G)
