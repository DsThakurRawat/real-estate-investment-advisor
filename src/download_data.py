"""
Dataset Downloader for Real Estate Investment Advisor
Downloads india_housing_prices.csv (41.1 MB) from Google Drive.
"""

import os
import subprocess
import sys

DRIVE_FILE_ID = "1ys25Eaqo2n8IeHhyI9s0kmJBgnNzxQHX"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "india_housing_prices.csv")


def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 1000000:
        print(f"Dataset already exists at {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.1f} MB).")
        return

    print("Downloading india_housing_prices.csv from Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    cmd = [sys.executable, "-m", "gdown", url, "-O", OUTPUT_FILE]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Failed to download using gdown: {e}")
        print(f"Direct download link: https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing")


if __name__ == "__main__":
    download_dataset()
