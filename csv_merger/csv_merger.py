import pandas as pd
from pathlib import Path
import re
from datetime import datetime


def process_date(date_str):
    try:
        # Clean the string - remove special characters and normalize
        date_str = str(date_str).strip()
        date_str = re.sub(r'[\u200b\u202f]', ' ', date_str)  # Remove special spaces

        # Handle the specific problematic format "Apr 19, date_time 2025"
        if "date_time" in date_str:
            date_str = date_str.replace("date_time", "").strip()

        # Handle cases where year is missing
        if "," in date_str and not any(year in date_str for year in ["2024", "2025"]):
            # Format like "Apr 10, 6:36 PM"
            if re.match(r'^[A-Za-z]{3} \d{1,2}, \d{1,2}:\d{2} [AP]M$', date_str):
                date_str = date_str + " 2025"
            # Format like "Jan 3, 10:08 PM"
            elif re.match(r'^[A-Za-z]{3} \d{1,2}, \d{1,2}:\d{2} [AP]M$', date_str):
                date_str = date_str + " 2025"

        # Standardize the date format before parsing
        date_str = date_str.replace(" ", " ")  # Replace special space with normal space

        # Try parsing with different formats
        try:
            dt = pd.to_datetime(date_str, format='%b %d, %I:%M %p %Y')
        except:
            try:
                dt = pd.to_datetime(date_str, format='%b %d, %Y, %I:%M %p')
            except:
                dt = pd.to_datetime(date_str, format='mixed', dayfirst=False)

        # Ensure the year is 2025 for dates without year
        if pd.notna(dt) and dt.year == 1900:  # Default year when not specified
            dt = dt.replace(year=2025)

        return dt
    except Exception as e:
        print(f"Could not parse date: {date_str} - Error: {str(e)}")
        return pd.NaT


def main():
    # Define file paths
    base_dir = Path("C:/Users/shann/PycharmProjects/capstone2025V2")
    tiktok_file = base_dir / "outputs/tiktok_shawnlemuel_20250417_141303.csv"
    video_results_file = base_dir / "outputs/content_analysers/actor_screen_time/video_results.csv"
    sentiment_file = base_dir / "outputs/content_analysers/sentiment_analysis/sentiment_results_tiktok_downloads.csv"
    output_dir = base_dir / "outputs/csv_merger"
    output_file = output_dir / "combined_for_predictive_modelling.csv"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the CSV files
    tiktok_df = pd.read_csv(tiktok_file, header=None)
    video_df = pd.read_csv(video_results_file)
    sentiment_df = pd.read_csv(sentiment_file)

    # Assign column names to tiktok_df
    tiktok_df.columns = [
        "Video ID", "Video URL", "Date Posted", "Promoted", "Views",
        "Likes", "Comments", "Caption", "Hashtags"
    ]

    # Convert Video ID to string in all dataframes
    tiktok_df["Video ID"] = tiktok_df["Video ID"].astype(str)
    video_df["Video ID"] = video_df["Video ID"].astype(str)
    sentiment_df["Video ID"] = sentiment_df["Video ID"].astype(str)

    # Process the dates
    tiktok_df["Date Posted"] = tiktok_df["Date Posted"].apply(process_date)

    # Check for any dates that couldn't be parsed
    if tiktok_df["Date Posted"].isna().any():
        na_count = tiktok_df["Date Posted"].isna().sum()
        print(f"Warning: Could not parse {na_count} dates")
        if na_count > 0:
            print("Problematic dates:")
            problematic = tiktok_df[tiktok_df["Date Posted"].isna()].copy()
            problematic["Original Date"] = tiktok_df.loc[tiktok_df["Date Posted"].isna(), "Date Posted"]
            print(problematic[["Video ID", "Original Date"]].to_string())

    # Merge the dataframes
    merged_df = pd.merge(video_df, sentiment_df, on="Video ID", how="inner")
    final_df = pd.merge(merged_df, tiktok_df, on="Video ID", how="left")

    # Format the datetime to show as "2025-04-10 18:36:00"
    final_df["Date Posted"] = final_df["Date Posted"].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Reorder columns
    column_order = [
        "Video ID", "Video URL", "Date Posted", "Caption", "Hashtags",
        "Promoted", "Views", "Likes", "Comments",
        "Shawn Seconds", "Total Seconds",
        "Dominant Emotion", "Emotion Confidence"
    ]
    final_df = final_df[column_order]

    # Save the final merged dataframe
    final_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully merged data saved to {output_file}")


if __name__ == "__main__":
    main()