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
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

# ====== Configuration ======
BASE_DIR = r"C:\Users\shann\PycharmProjects\capstone2025V2\content_analysers\actor_screen_time"
MODEL_TRAINER_DIR = os.path.join(BASE_DIR, "model_trainer")
POSITIVE_DIR = os.path.join(MODEL_TRAINER_DIR, "positives")
NEGATIVE_DIR = os.path.join(MODEL_TRAINER_DIR, "negatives")
DEFAULT_VIDEO_DIR = r"C:\Users\shann\PycharmProjects\capstone2025V2\downloads"
OUTPUT_DIR = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\content_analysers\actor_screen_time"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

# ====== Global Variables ======
train = None
max_feature_value = None
video_dir = DEFAULT_VIDEO_DIR  # Initialize with default, will be updated by user
base_model = None
screen_time_model = None


def initialize_screen_time_model():
    global base_model, screen_time_model
    print("Loading screen time model architecture...")
    base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    screen_time_model = keras.models.Sequential([
        keras.layers.InputLayer((25088,)),
        keras.layers.Dense(units=1024, activation='sigmoid'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])


def load_training_data():
    positive_files = [os.path.join(POSITIVE_DIR, f) for f in os.listdir(POSITIVE_DIR) if f.endswith('.jpg')]
    negative_files = [os.path.join(NEGATIVE_DIR, f) for f in os.listdir(NEGATIVE_DIR) if f.endswith('.jpg')]

    X_pos = np.array([plt.imread(f) for f in positive_files])
    X_neg = np.array([plt.imread(f) for f in negative_files])

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    X_processed = np.array([resize(img, (224, 224), preserve_range=True).astype(int) for img in X])
    X_processed = tf.keras.applications.vgg16.preprocess_input(X_processed)
    y_categorical = tf.keras.utils.to_categorical(y)

    return train_test_split(X_processed, y_categorical, test_size=0.3, random_state=42)


def extract_features(model, data):
    features = model.predict(data)
    return features.reshape(features.shape[0], -1)


def train_screen_time_model():
    global train, max_feature_value, base_model, screen_time_model
    print("\nTraining screen time detection model...")
    X_train, X_valid, y_train, y_valid = load_training_data()

    print("Extracting features from training data...")
    X_train_features = extract_features(base_model, X_train)
    print("Extracting features from validation data...")
    X_valid_features = extract_features(base_model, X_valid)

    max_feature_value = X_train_features.max()
    train = X_train_features / max_feature_value
    X_valid_norm = X_valid_features / max_feature_value

    screen_time_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    screen_time_model.fit(train, y_train, epochs=100, validation_data=(X_valid_norm, y_valid),
                          verbose=1)  # Keep verbose for training progress
    print("Screen time model training complete!")


def process_video_for_screen_time(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = total_frames / frame_rate

        frames = []
        with tqdm(total=total_frames, unit="frame",
                  desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
            while cap.isOpened():
                frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % math.floor(frame_rate) == 0:
                    frames.append(frame)
                pbar.update(1)
        cap.release()

        if len(frames) > 0:
            test_image = np.array(frames)
            test_image_resized = np.array([resize(img, (224, 224), preserve_range=True) for img in test_image])
            test_image_processed = tf.keras.applications.vgg16.preprocess_input(test_image_resized)
            test_image_features = base_model.predict(test_image_processed)
            test_image_normalized = test_image_features.reshape(test_image_features.shape[0], -1) / max_feature_value
            test_image_probabilities = screen_time_model.predict(test_image_normalized)
            predictions = (test_image_probabilities[:, 1] > 0.8).astype(int)
            return predictions, total_seconds
        return np.array([]), total_seconds
    except Exception as e:
        print(f"Error processing screen time for {os.path.basename(video_path)}: {e}")
        return np.array([]), 0


def select_video_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(initialdir=DEFAULT_VIDEO_DIR, title="Select Video Directory")
    return folder_selected


def get_output_csv_path(directory):
    """Generates the output CSV path based on the chosen directory name."""
    directory_name = os.path.basename(directory.rstrip(os.sep))
    output_filename = f"actor_screen_time_results_{directory_name}.csv"
    return os.path.join(OUTPUT_DIR, output_filename)


def main():
    global video_dir, base_model, screen_time_model
    initialize_screen_time_model()

    video_dir = select_video_directory()
    if not video_dir:
        print("No video directory selected. Exiting.")
        return
    print(f"Selected video directory: {video_dir}")

    train_screen_time_model()

    output_csv_path = get_output_csv_path(video_dir)

    # Load existing results if available
    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        processed_videos = set(str(vid) for vid in existing_df['Video ID'].tolist())
        print(f"Found {len(processed_videos)} already processed videos in {output_csv_path}")
    else:
        existing_df = pd.DataFrame()
        processed_videos = set()
        print(f"No existing results file found at {output_csv_path}, starting fresh")

    all_videos = [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    videos_to_process = sorted(list(set(all_videos) - processed_videos))
    num_videos_to_process = len(videos_to_process)

    if not videos_to_process:
        print("All videos in the selected directory have already been processed.")
        return

    print(
        f"\nFound {num_videos_to_process} new videos to process out of {len(all_videos)} total videos in the directory.")

    results = []
    for i, video_id in enumerate(videos_to_process, 1):
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        print(f"\nProcessing video {i}/{num_videos_to_process}: {video_filename}")

        try:
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                continue

            # Process screen time
            predictions, total_seconds = process_video_for_screen_time(video_path)
            shawn_seconds = np.sum(predictions)

            # Add result
            results.append({
                'Video ID': video_id,
                'Total Seconds': total_seconds,
                'Shawn Seconds': shawn_seconds,
                'Shawn Percentage': (shawn_seconds / total_seconds) * 100 if total_seconds > 0 else 0,
            })

            # Save incremental results
            updated_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
            updated_df.to_csv(output_csv_path, index=False)
            print(f"Successfully processed and saved results for {video_filename} to {output_csv_path}")

        except Exception as e:
            print(f"An unexpected error occurred while processing {video_filename}: {e}")

    # Final summary
    print(f"\nProcessing complete. Results saved to {output_csv_path}")
    print(f"Successfully processed {len(results)}/{num_videos_to_process} new videos.")
    total_processed = len(existing_df) + len(results)
    print(f"Total videos in the results file: {total_processed}")


if __name__ == "__main__":
    main()
