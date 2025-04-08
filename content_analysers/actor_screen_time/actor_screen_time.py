import os
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# Define paths with new directory structure
base_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\content_analysers\actor_screen_time\model_trainer"
positive_dir = os.path.join(base_dir, "positives")
negative_dir = os.path.join(base_dir, "negatives")
test_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\downloads\tiktok_shawnlemuel_20250408_235452_urls"

# Set target size for all images
target_size = (224, 224)

# Load positive examples (class 1)
positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.jpg')]
X_pos = np.array([resize(plt.imread(f), preserve_range=True, output_shape=target_size)
                  for f in positive_files])
y_pos = np.ones(len(positive_files))  # All positives are class 1

# Load negative examples (class 0)
negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.jpg')]
X_neg = np.array([resize(plt.imread(f), preserve_range=True, output_shape=target_size)
                  for f in negative_files])
y_neg = np.zeros(len(negative_files))  # All negatives are class 0

# Combine positive and negative examples
X = np.concatenate((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

# Preprocess images (additional preprocessing for VGG16)
X = tf.keras.applications.vgg16.preprocess_input(X)
dummy_y = tf.keras.utils.to_categorical(y)

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

# Load VGG16 base model
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract features using VGG16
X_train_features = base_model.predict(X_train)
X_valid_features = base_model.predict(X_valid)

# Flatten features
X_train = X_train_features.reshape(X_train_features.shape[0], -1)
X_valid = X_valid_features.reshape(X_valid_features.shape[0], -1)

# Normalize features
train_max = X_train.max()
X_train = X_train / train_max
X_valid = X_valid / train_max

# Define and compile the model
model = keras.models.Sequential([
    keras.layers.InputLayer((X_train.shape[1],)),
    keras.layers.Dense(units=1024, activation='sigmoid'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))
print("Training Completed!")


# --- Testing Phase ---
def process_video(video_path, output_dir="test_frames", threshold=0.8):
    """Processes a video, extracts frames, and returns predictions."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / frame_rate

    count = 0
    frames = []
    while cap.isOpened():
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % math.floor(frame_rate) == 0:
            filename = os.path.join(output_dir, f"test{count}.jpg")
            cv2.imwrite(filename, frame)
            frames.append(filename)
            count += 1
    cap.release()

    # Process test frames
    test_images = np.array([resize(plt.imread(f), preserve_range=True, output_shape=target_size)
                            for f in frames])
    test_images = tf.keras.applications.vgg16.preprocess_input(test_images)

    # Extract features
    test_features = base_model.predict(test_images)
    test_features = test_features.reshape(test_features.shape[0], -1) / train_max

    # Make predictions
    probabilities = model.predict(test_features)
    predictions = (probabilities[:, 1] > threshold).astype(int)
    return predictions, total_seconds


def process_directory(directory, results):
    """Processes videos in a directory and appends results to the list."""
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            video_path = os.path.join(directory, filename)
            video_id = os.path.splitext(filename)[0]
            predictions, total_seconds = process_video(video_path)
            shawn_seconds = predictions[predictions == 1].shape[0]
            results.append([video_id, shawn_seconds, total_seconds])


# Process test directory
results = []
process_directory(test_dir, results)

# Save results in the specified output directory
output_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\content_analysers\actor_screen_time"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_csv = os.path.join(output_dir, "video_results.csv")
df_results = pd.DataFrame(results, columns=["Video ID", "Shawn Seconds", "Total Seconds"])
df_results.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
print(df_results)