import os
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tqdm import tqdm

# ====== Configuration ======
DEFAULT_VIDEO_DIR = r"C:\Users\shann\PycharmProjects\capstone2025V2\downloads"
OUTPUT_DIR = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\content_analysers\sentiment_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# ====== Global Variables ======
whisper_model = None
emotion_analyzer = None
video_dir = DEFAULT_VIDEO_DIR # Initialize with default, will be updated by user


def initialize_sentiment_models():
 global whisper_model, emotion_analyzer
 print("Loading sentiment analysis models...")
 try:
     whisper_model = whisper.load_model("base")
     emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
     print("Sentiment analysis models loaded successfully.")
 except Exception as e:
     print(f"Error loading sentiment analysis models: {e}")
     exit()


def select_video_directory():
 root = tk.Tk()
 root.withdraw()  # Hide the main window
 folder_selected = filedialog.askdirectory(initialdir=DEFAULT_VIDEO_DIR, title="Select Video Directory")
 return folder_selected


def get_output_csv_path(directory):
 """Generates the output CSV path based on the chosen directory name."""
 directory_name = os.path.basename(directory.rstrip(os.sep))
 output_filename = f"sentiment_results_{directory_name}.csv"
 return os.path.join(OUTPUT_DIR, output_filename)


def process_video_for_sentiment(video_path):
 audio_path = os.path.join(video_dir, f"temp_{os.path.basename(video_path)}.wav")
 try:
     video = VideoFileClip(video_path)
     video.audio.write_audiofile(audio_path)
     result = whisper_model.transcribe(audio_path)
     text = result["text"]

     if len(text) > 500:
         text = text[:500] + "..."

     emotions = emotion_analyzer(text, truncation=True)
     dominant_emotion = max(emotions, key=lambda x: x['score'])
     return {
         'emotion_label': dominant_emotion['label'],
         'emotion_confidence': dominant_emotion['score']
     }
 except Exception as e:
     print(f"Error processing sentiment for {os.path.basename(video_path)}: {e}")
     return {'emotion_label': 'N/A', 'emotion_confidence': 0.0}
 finally:
     if os.path.exists(audio_path):
         os.remove(audio_path)


def main():
 global video_dir, whisper_model, emotion_analyzer

 video_dir = select_video_directory()
 if not video_dir:
     print("No video directory selected. Exiting.")
     return
 print(f"Selected video directory: {video_dir}")

 initialize_sentiment_models()

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
     print("All videos in the selected directory have already been processed for sentiment.")
     return

 print(f"\nFound {num_videos_to_process} new videos to process for sentiment out of {len(all_videos)} total videos in the directory.")

 results = []
 for i, video_id in enumerate(videos_to_process, 1):
     video_filename = f"{video_id}.mp4"
     video_path = os.path.join(video_dir, video_filename)
     print(f"\nProcessing video {i}/{num_videos_to_process}: {video_filename}")

     try:
         if not os.path.exists(video_path):
             print(f"Error: Video file not found: {video_path}")
             continue

         # Process sentiment
         sentiment_data = process_video_for_sentiment(video_path)

         # Add result
         results.append({
             'Video ID': video_id,
             'Dominant Emotion': sentiment_data['emotion_label'],
             'Emotion Confidence': sentiment_data['emotion_confidence']
         })

         # Save incremental results
         updated_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
         updated_df.to_csv(output_csv_path, index=False)
         print(f"Successfully processed and saved sentiment results for {video_filename} to {output_csv_path}")

     except Exception as e:
         print(f"An unexpected error occurred while processing sentiment for {video_filename}: {e}")

 # Final summary
 print(f"\nSentiment analysis complete. Results saved to {output_csv_path}")
 print(f"Successfully processed sentiment for {len(results)}/{num_videos_to_process} new videos.")
 total_processed = len(existing_df) + len(results)
 print(f"Total videos in the sentiment results file: {total_processed}")


if __name__ == "__main__":
 main()