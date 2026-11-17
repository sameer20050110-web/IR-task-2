"""
download_dataset.py
-------------------
Utility script to download the official SMS Spam Collection dataset from the
UCI Machine Learning Repository (with fallback mirrors).

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_RAW_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
OUTPUT_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
CSV_OUTPUT_FILE = os.path.join(DATA_DIR, "sms_spam_collection.csv")
INFO_FILE = os.path.join(DATA_DIR, "dataset_info.json")

def download_and_extract():
    print(f"[*] Ensuring data directory exists at: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    downloaded = False
    
    # Attempt 1: Official UCI ZIP
    print(f"[*] Attempting download from official UCI repository: {ZIP_URL}")
    try:
        req = urllib.request.Request(ZIP_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            zip_content = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                # Look for SMSSpamCollection
                for filename in z.namelist():
                    if "SMSSpamCollection" in filename:
                        extracted_data = z.read(filename)
                        with open(OUTPUT_FILE, "wb") as f_out:
                            f_out.write(extracted_data)
                        print(f"[+] Successfully extracted {filename} to {OUTPUT_FILE}")
                        downloaded = True
                        break
    except Exception as e:
        print(f"[-] Official UCI download failed: {e}")
        
    # Attempt 2: Fallback Raw TSV mirror
    if not downloaded:
        print(f"[*] Attempting fallback download from: {FALLBACK_RAW_URL}")
        try:
            req = urllib.request.Request(FALLBACK_RAW_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as response:
                content = response.read()
                with open(OUTPUT_FILE, "wb") as f_out:
                    f_out.write(content)
            print(f"[+] Successfully downloaded raw dataset to {OUTPUT_FILE}")
            downloaded = True
        except Exception as e:
            print(f"[-] Fallback download failed: {e}")
            raise RuntimeError(f"Could not download SMS Spam Collection dataset: {e}")

    # Process into clean CSV and verify records
    print("[*] Validating and saving structured CSV...")
    df = pd.read_csv(
        OUTPUT_FILE,
        sep='\t',
        header=None,
        names=['label', 'message'],
        quoting=3,
        encoding='utf-8'
    )
    
    # Basic cleanup: drop any null rows
    df = df.dropna().reset_index(drop=True)
    df.to_csv(CSV_OUTPUT_FILE, index=False)
    
    class_counts = df['label'].value_counts().to_dict()
    total_records = len(df)
    
    info = {
        "dataset_name": "SMS Spam Collection v.1",
        "source": "UCI Machine Learning Repository / Almeida et al.",
        "description": "A collection of SMS tagged messages that have been collected for SMS Spam research.",
        "total_records": total_records,
        "class_distribution": class_counts,
        "class_percentages": {k: round((v / total_records) * 100, 2) for k, v in class_counts.items()},
        "columns": list(df.columns),
        "data_path": CSV_OUTPUT_FILE
    }
    
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        
    print(f"[+] Dataset validation complete:")
    print(f"    Total records: {total_records}")
    print(f"    Class distribution: {class_counts}")
    print(f"    Saved structured CSV to: {CSV_OUTPUT_FILE}")
    print(f"    Saved metadata to: {INFO_FILE}")

if __name__ == "__main__":
    download_and_extract()

