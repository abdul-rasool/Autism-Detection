import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import cv2
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report

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

images, labels = load_and_preprocess_images(dataset_path)

# Converting labels to numerical values
labels = [0 if label == 'TSImages' else 1 for label in labels]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# one-hot encoding
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two classes (TSImages and TCImages)
])

# Compiling
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Model evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predictions
y_pred = model.predict(X_test)

# the one-hot encoding conversion
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Report
report = classification_report(y_test_labels, y_pred_labels, target_names=["TSImages", "TCImages"])
print("Classification Report:")
print(report)


