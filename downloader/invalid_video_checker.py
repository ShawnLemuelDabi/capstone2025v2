import os
import cv2
import numpy as np
import moviepy.editor as mp
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog

def is_black_frame(frame, threshold=30):
    """Check if a frame is completely black"""
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray)[0] < threshold


def has_audio(video_path):
    """Check if video has any audio track"""
    try:
        clip = mp.VideoFileClip(video_path)
        has_audio = clip.audio is not None
        clip.close()
        return has_audio
    except Exception:
        return False


def is_invalid_video(video_path, sample_frames=10):
    """Check if video is invalid (all black frames and no audio)"""
    # First check for audio
    if has_audio(video_path):
        return False

    # Then check frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle case where video has no frames
    if total_frames <= 0:
        cap.release()
        return True

    # Calculate frame indices to sample
    sample_count = min(sample_frames, total_frames)
    if sample_count <= 0:
        cap.release()
        return True

    # Get evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    all_black = True
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if not is_black_frame(frame):
            all_black = False
            break

    cap.release()
    return all_black


def clean_video_directory(directory):
    """Scan directory and delete invalid videos"""
    deleted_files = []
    video_files = [f for f in os.listdir(directory)
                   if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm'))]

    if not video_files:
        print("No video files found in the directory.")
        return deleted_files

    print(f"Scanning {len(video_files)} videos in {directory}...")

    for filename in tqdm(video_files, desc="Checking videos"):
        filepath = os.path.join(directory, filename)

        try:
            if is_invalid_video(filepath):
                try:
                    os.remove(filepath)
                    deleted_files.append(filename)
                    print(f"\nDeleted invalid video: {filename}")
                except Exception as e:
                    print(f"\nError deleting {filename}: {str(e)}")
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            continue

    print("\nCleanup complete!")
    print(f"Total videos scanned: {len(video_files)}")
    print(f"Deleted invalid videos: {len(deleted_files)}")

    if deleted_files:
        print("\nDeleted files:")
        for f in deleted_files:
            print(f"- {f}")

    return deleted_files


def select_directory():
    """Opens a directory selection dialog and returns the selected path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(initialdir=r"C:\Users\shann\PycharmProjects\capstone2025V2\downloads")
    return directory


def main():
    selected_dir = select_directory()
    if selected_dir:
        print(f"Selected directory: {selected_dir}")
        clean_video_directory(selected_dir)
    else:
        print("No directory selected.")


if __name__ == "__main__":
    main()