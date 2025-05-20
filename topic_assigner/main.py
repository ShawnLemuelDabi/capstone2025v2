import pandas as pd
import re
from pathlib import Path


def assign_topics_to_videos():
    """
    Assigns topics to TikTok videos based on caption analysis using keyword matching.
    Uses specific file paths provided by the user.
    """
    # Define paths using raw strings (r prefix) to handle Windows paths
    input_csv_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_for_predictive_modelling.csv"
    output_csv_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_with_topics.csv"

    try:
        # 1. Read the CSV file
        print(f"Reading input file from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Check if DataFrame is empty
        if df.empty:
            raise pd.errors.EmptyDataError("The input CSV file is empty.")
        print(f"Successfully read {len(df)} video entries.")

        # 2. Define topic categories and their keywords with improved matching
        topic_keywords = {
            'Motivation': [
                r'\bmotivat', r'\binspir', r'\bsuccess', r'\bgoal', r'\bachieve',
                r'endorphin', r'💪', r'work\s*out', r'\bgym\b', r'\bfitness\b',
                r'\bdiscipline\b', r'\bgrowth\b', r'\bimprove', r'\bhabits\b',
                r'\bpush\b', r'\bgrind\b', r'\bdetermin', r'\bwinner'
            ],
            'Lechon': [
                r'\blechon\b', r'roast\s*pig', r'\bpork\b', r'\bcrispy\b', r'suckling\s*pig',
                r'filipino\s*food', r'\bcebu\b', r'\bCebu\b', r'\bLechon\b'
            ],
            'Japanese Food': [
                r'\bjapan\b', r'\bjapanese\b', r'\bsushi\b', r'\bramen\b', r'\btempura\b',
                r'\bwagyu\b', r'\budon\b', r'\bsashimi\b', r'\bmiso\b', r'\btonkatsu\b',
                r'\bbento\b', r'🍣', r'🍜'
            ],
            'Chinese Food': [
                r'\bchina\b', r'\bchinese\b', r'\bdumpling\b', r'\bnoodle\b', r'dim\s*sum',
                r'\bbao\b', r'xiaolongbao', r'\bhotpot\b', r'peking\s*duck', r'\bwonton\b',
                r'🥟', r'🍜'
            ],
            'Lee Kuan Yew': [
                r'lee\s*kuan\s*yew', r'\blky\b', r'singapore\s*politics',
                r'founder\s*of\s*singapore', r'pioneer\s*generation',
                r'modern\s*singapore', r'\bLKY\b', r'\bMM\b\s*Lee'
            ]
        }

        # 3. Enhanced function to assign topic based on caption
        def assign_topic(caption):
            if not isinstance(caption, str):
                return 'Other'

            caption = caption.lower()

            # First check for exact matches (whole words)
            for topic, keywords in topic_keywords.items():
                for pattern in keywords:
                    if re.search(pattern, caption, flags=re.IGNORECASE):
                        return topic

            return 'Other'

        # 4. Apply topic assignment and track distribution
        print("Assigning topics to videos...")
        df['Assigned Topic'] = df['Caption'].apply(assign_topic)

        # Show topic distribution
        topic_counts = df['Assigned Topic'].value_counts()
        print("\nTopic Assignment Summary:")
        print(topic_counts)

        # 5. Save the modified DataFrame
        output_dir = Path(output_csv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully saved output with topics to: {output_csv_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv_path}")
    except pd.errors.EmptyDataError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    assign_topics_to_videos()