import pandas as pd
from datetime import datetime
import re

def refine_csv(input_filepath, output_filepath):
    """
    Refines a CSV file by removing specified columns, adding new calculated columns,
    and saving the result to a new CSV file. Handles view counts with 'K'.

    Args:
        input_filepath (str): The path to the input CSV file.
        output_filepath (str): The path to save the refined CSV file.
    """
    try:
        df = pd.read_csv(input_filepath)

        # Drop specified columns
        columns_to_drop = ['Hashtags', 'Promoted']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Convert 'Date Posted' to datetime objects
        df['Date Posted'] = pd.to_datetime(df['Date Posted'])

        # Add 'Day Uploaded' column
        df['Day Uploaded'] = df['Date Posted'].dt.date

        # Function to convert view counts like '55K' to numbers
        def convert_views(view_str):
            if isinstance(view_str, str):
                view_str = view_str.upper()
                if 'K' in view_str:
                    return float(re.sub(r'[^\d.]', '', view_str)) * 1000
                elif 'M' in view_str:
                    return float(re.sub(r'[^\d.]', '', view_str)) * 1000000
            try:
                return float(view_str)
            except (ValueError, TypeError):
                return 0

        # Apply the conversion function to the 'Views' column
        df['Views'] = df['Views'].apply(convert_views)

        # Convert 'Likes' to numeric, handling potential errors
        df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce').fillna(0)

        # Add 'Screen Time Percentage' column
        df['Screen Time Percentage'] = ((df['Shawn Seconds'] / df['Total Seconds']) * 100).round(2)

        # Add 'Engagement Rate' column
        df['Engagement Rate'] = ((df['Likes'] + df['Comments']) / df['Views'] * 100).fillna(0).round(2)

        # Add 'Like-to-Comment Ratio' column
        df['Like-to-Comment Ratio'] = (df['Likes'] / df['Comments']).replace([float('inf'), -float('inf')], 0).fillna(0).round(2)

        # Add 'Weekday vs. Weekend Upload' column
        df['Weekday vs. Weekend Upload'] = df['Date Posted'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        # Add 'Views per Day' column
        now = datetime.now()
        df['Time Since Upload'] = (now - df['Date Posted']).dt.days
        df['Views per Day'] = (df['Views'] / (df['Time Since Upload'] + 1)).round(2)

        # Save the refined DataFrame to a new CSV file
        df.to_csv(output_filepath, index=False)

        print(f"Successfully refined '{input_filepath}' and saved to '{output_filepath}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_csv_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\combined_for_predictive_modelling.csv"  # Corrected input path
    output_csv_path = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger\refined_combined_for_predictive_modelling.csv"
    refine_csv(input_csv_path, output_csv_path)