import os
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# ====== Global Variables ======
whisper_model = None
emotion_analyzer = None

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

def get_output_csv_path(video_directory, output_directory):
    """Generates the output CSV path based on the chosen directory name."""
    directory_name = os.path.basename(video_directory.rstrip(os.sep))
    output_filename = f"sentiment_results_{directory_name}.csv"
    return os.path.join(output_directory, output_filename)

def process_video_for_sentiment(video_path, video_dir):
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
    global whisper_model, emotion_analyzer

    # Get user inputs for directories
    video_dir = input("Enter the path to the video directory: ").strip()
    if not os.path.isdir(video_dir):
        print(f"Directory not found: {video_dir}")
        return

    output_dir = input("Enter the path to the output directory: ").strip()
    os.makedirs(output_dir, exist_ok=True)

    initialize_sentiment_models()
    output_csv_path = get_output_csv_path(video_dir, output_dir)

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

    print(f"\nFound {num_videos_to_process} new videos to process for sentiment out of {len(all_videos)} total videos.")

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
            sentiment_data = process_video_for_sentiment(video_path, video_dir)

            # Add result
            results.append({
                'Video ID': video_id,
                'Dominant Emotion': sentiment_data['emotion_label'],
                'Emotion Confidence': sentiment_data['emotion_confidence']
            })

            # Save incremental results
            updated_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
            updated_df.to_csv(output_csv_path, index=False)
            print(f"Successfully processed and saved results for {video_filename}")

        except Exception as e:
            print(f"An unexpected error occurred processing {video_filename}: {e}")

    # Final summary
    print(f"\nAnalysis complete. Results saved to {output_csv_path}")
    print(f"Successfully processed {len(results)}/{num_videos_to_process} new videos.")
    total_processed = len(existing_df) + len(results)
    print(f"Total videos in results file: {total_processed}")

if __name__ == "__main__":
    main()