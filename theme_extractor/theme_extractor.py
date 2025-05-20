import pandas as pd
from transformers import pipeline
import time
from tqdm import tqdm

# Define the topics
TOPICS = [
    "motivation",
    "lee kuan yew",
    "lechon",
    "bak kut teh",
    "japanese food",
    "filipino food",
    "halal",
    "travel",
    "product sales video",
    "dessert",
    "food (general)"
]

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def assign_topic(caption):
    """
    Assigns a topic to a caption using zero-shot classification
    """
    if pd.isna(caption) or not caption.strip():
        return "unknown"

    try:
        # Classify the caption against our topics
        result = classifier(caption, TOPICS, multi_label=False)
        return result['labels'][0]  # Return the top predicted topic
    except Exception as e:
        print(f"Error processing caption: {caption}. Error: {e}")
        return "unknown"


def process_dataset(input_file, output_file):
    """
    Processes the dataset by reading captions and assigning topics
    """
    # Read the dataset
    df = pd.read_csv(input_file)

    # Initialize progress bar
    tqdm.pandas(desc="Assigning topics to captions")

    # Assign topics to each caption
    df['Topic'] = df['Caption'].progress_apply(assign_topic)

    # Save the results
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")

    return df


if __name__ == "__main__":
    # File paths
    input_file = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_for_predictive_modelling.csv"
    output_file = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\theme_extractor\refined_combined_with_topics.csv"

    # Process the dataset
    processed_df = process_dataset(input_file, output_file)

    # Print some statistics
    print("\nTopic Distribution:")
    print(processed_df['Topic'].value_counts())