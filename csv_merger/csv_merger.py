import pandas as pd
import os
from datetime import datetime

def main():
    try:
        # --- File Paths ---
        tiktok_file = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\tiktok_shawnlemuel_20250408_235452.csv"
        screen_time_file = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\content_analysers\actor_screen_time\video_results.csv"
        sentiment_file = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\content_analysers\sentiment_analysis\sentiment_results_tiktok_shawndowhat_20250328_125539_urls.csv"
        output_dir = r"C:\Users\shann\PycharmProjects\capstone2025V2\outputs\csv_merger"
        output_file = os.path.join(output_dir, "combined_for_predictive_modelling.csv")

        # --- Validate Input Files ---
        print("Checking input files...")
        for file in [tiktok_file, screen_time_file, sentiment_file]:
            if not os.path.exists(file):
                print(f"❌ Error: File not found - {file}")
                return
            else:
                print(f"✅ Found: {file}")

        # --- Read Data ---
        print("\nReading CSV files...")
        df_tiktok = pd.read_csv(tiktok_file)
        df_screen = pd.read_csv(screen_time_file)
        df_sentiment = pd.read_csv(sentiment_file)

        # Debug: Show column names
        print("\nTikTok columns:", df_tiktok.columns.tolist())
        print("Screen Time columns:", df_screen.columns.tolist())
        print("Sentiment columns:", df_sentiment.columns.tolist())

        # --- Standardize Columns ---
        df_tiktok = df_tiktok.rename(columns={"video_id": "Video ID"})

        # --- Clean Dates ---
        print("\nCleaning dates...")
        df_tiktok['date_time'] = df_tiktok['date_time'].replace('Pinned', pd.NaT)
        df_tiktok['date_time'] = pd.to_datetime(
            df_tiktok['date_time'],
            format="%b %d, %I:%M\u202f%p",
            errors='coerce'
        ).dt.strftime("%m-%d %H:%M")  # Remove year

        # --- Merge Data ---
        print("\nMerging DataFrames...")
        df_merged = df_tiktok.merge(df_screen, on="Video ID", how="left")
        df_merged = df_merged.merge(df_sentiment, on="Video ID", how="left")
        print("Merged DataFrame shape:", df_merged.shape)

        if df_merged.empty:
            print("❌ Merged DataFrame is empty!")
            return

        # --- Process Final Data ---
        print("\nCalculating metrics...")
        df_final = df_merged[[
            "Video ID", "date_time", "views", "likes", "comments",
            "Shawn Seconds", "Total Seconds", "Dominant Emotion", "Emotion Confidence"
        ]].copy()

        # Convert views (e.g., "32K" → 32000)
        df_final['views'] = df_final['views'].apply(
            lambda x: int(float(x.replace('K', '')) * 1000 if 'K' in str(x) else int(x))
        )

        # Day/week analysis
        temp_dates = pd.to_datetime(df_final['date_time'], format="%m-%d %H:%M", errors='coerce')
        df_final['day_of_week'] = temp_dates.dt.dayofweek
        df_final['is_weekend'] = df_final['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Other metrics
        df_final['Engagement Rate'] = ((df_final['likes'] + df_final['comments']) / df_final['views'] * 100).round(2)
        # Caption Length is removed as 'caption' column is not in df_final anymore

        # --- Save Output ---
        os.makedirs(output_dir, exist_ok=True)
        df_final.to_csv(output_file, index=False)
        print(f"\n✅ Success! Output saved to {output_file}")
        print("Sample output:")
        print(df_final.head())

    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()