import os
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path


class FramesExtractor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main tkinter window

        # Set default directory
        self.default_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\content_analysers\actor_screen_time\model_trainer"

        # Output directories mapping
        self.output_mapping = {
            "positive_videos": "positives",
            "negative_videos": "negatives"
        }

        # Default starting frame numbers for each type
        self.default_starting_frames = {
            "positives": 50,  # Default for positives
            "negatives": 410  # Default for negatives
        }

    def select_directory(self):
        """Select a directory containing videos"""
        dir_path = filedialog.askdirectory(initialdir=self.default_dir)
        if not dir_path:
            print("No directory selected. Exiting.")
            return None
        return dir_path

    def get_output_dir(self, input_dir):
        """Determine the appropriate output directory based on input directory"""
        dir_name = os.path.basename(input_dir)
        if dir_name in self.output_mapping:
            output_dir = os.path.join(os.path.dirname(input_dir), self.output_mapping[dir_name])
            os.makedirs(output_dir, exist_ok=True)
            return output_dir, self.output_mapping[dir_name]
        return None, None

    def get_starting_frame_number(self, output_dir, output_type):
        """Get the starting frame number either from user input or existing files"""
        # First try to get user input
        user_input = simpledialog.askinteger(
            "Starting Frame Number",
            f"Enter starting frame number for {output_type} (or cancel to auto-detect):",
            parent=self.root,
            minvalue=0
        )

        if user_input is not None:
            return user_input

        # If user cancelled, check existing files
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('frame') and f.endswith('.jpg')]
        if existing_files:
            existing_numbers = [int(f[5:-4]) for f in existing_files]
            return max(existing_numbers) + 1

        # If no existing files, use default
        return self.default_starting_frames.get(output_type, 0)

    def extract_frames(self, video_path, output_dir, starting_frame_num, frame_interval=5):
        """Extract frames from a video file with given frame interval"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        next_frame_num = starting_frame_num

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every 5th frame
            if frame_count % frame_interval == 0:
                output_path = os.path.join(output_dir, f"frame{next_frame_num}.jpg")
                cv2.imwrite(output_path, frame)
                next_frame_num += 1
                saved_count += 1

            frame_count += 1

        cap.release()
        return saved_count

    def process_videos(self):
        """Main method to process all videos in selected directory"""
        input_dir = self.select_directory()
        if not input_dir:
            return

        output_dir, output_type = self.get_output_dir(input_dir)
        if not output_dir:
            print("Selected directory doesn't match expected video directories (positive_videos/negative_videos)")
            return

        # Get starting frame number from user or existing files
        starting_frame_num = self.get_starting_frame_number(output_dir, output_type)
        print(f"Starting frame number: {starting_frame_num}")

        # Get all video files in directory
        video_files = [f for f in os.listdir(input_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

        if not video_files:
            print("No video files found in selected directory")
            return

        total_frames_saved = 0

        for video_file in video_files:
            video_path = os.path.join(input_dir, video_file)
            print(f"Processing {video_file}...")
            frames_saved = self.extract_frames(video_path, output_dir, starting_frame_num)
            total_frames_saved += frames_saved
            starting_frame_num += frames_saved  # Update for next video
            print(f"Saved {frames_saved} frames from {video_file}")

        print(f"\nCompleted! Saved {total_frames_saved} frames total to {output_dir}")


# For standalone use or importing
def extract_frames():
    extractor = FramesExtractor()
    extractor.process_videos()


if __name__ == "__main__":
    extract_frames()