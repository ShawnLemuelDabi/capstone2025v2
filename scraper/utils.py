import csv
import os
from datetime import datetime


def save_results(data, account_name=None):
    os.makedirs("outputs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    account_suffix = f"_{account_name.replace(' ', '_').lower()}" if account_name else ""
    filename = f"outputs/scraper/tiktok{account_suffix}_{timestamp}.csv"

    with open(filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_id',
            'video_url',  # NEW FIELD
            'date_time',
            'pinned',
            'views',
            'likes',
            'comments',
            'caption',
            'privacy'
        ])
        writer.writeheader()
        writer.writerows(data)

    print(f"💾 Saved {len(data)} videos to {filename}")
    return filename