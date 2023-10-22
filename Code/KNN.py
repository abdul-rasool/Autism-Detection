import os
import cv2
import numpy as np
from skimage import io, color, segmentation, feature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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

# preprocessing
images, labels = load_and_preprocess_images(dataset_path)

# Extracting superpixel features and labels
def extract_superpixel_features(image, label):
    gray_image = color.rgb2gray(image) # grayscale images
    segments = segmentation.slic(image, n_segments=100, compactness=10) # superpixels
    superpixel_features = [] #features
    superpixel_labels = [] #labels

    for segment_id in np.unique(segments):
        mask = (segments == segment_id)
        segment = gray_image * mask

        mean_intensity = np.mean(segment) # Compute superpixel features 
        area = np.sum(mask)
        superpixel_features.append([mean_intensity, area])
        superpixel_labels.append(label)

    return superpixel_features, superpixel_labels

# features extraction
all_superpixel_features = []
all_superpixel_labels = []

for i in range(len(images)):
    image_features, image_labels = extract_superpixel_features(images[i], labels[i])
    all_superpixel_features.extend(image_features)
    all_superpixel_labels.extend(image_labels)

# Splitting data
X = all_superpixel_features
y = all_superpixel_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test) # predictions

# Report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
