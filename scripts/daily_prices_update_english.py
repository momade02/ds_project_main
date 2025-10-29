"""
daily_prices_update.py
----------------------
Automated script for updating the price list (prices)
from the official Tankerkönig Azure repository

Functions:
1. Automatically downloads the previous day's price CSV from the official Tankerkönig Azure repository
2. Checks column names and structure
3. Performs type conversions for numeric columns
4. Cleans missing values
5. Uploads the cleaned data into the Supabase table "prices_test"

Inputs:
    - None (the date is automatically determined as "yesterday")

Outputs:
    - CSV file in the ./data/ folder
    - Upload of the data to Supabase

Required environment variables:
    SUPABASE_URL
    SUPABASE_SECRET_KEY
"""

import os
import requests
import pandas as pd
from datetime import date, timedelta
from supabase import create_client
from pathlib import Path

# === 1. Determine the date ===
# Determines yesterday's date (the price files are always provided for the previous day)
yesterday = date.today() - timedelta(days=1)
date_str = yesterday.strftime("%Y-%m-%d")
year = yesterday.strftime("%Y")
month = yesterday.strftime("%m")

# === 2. Prepare the local storage location for the CSV ===
# This script lives in the /scripts subfolder, so we move one level up (project directory)
base_dir = Path(__file__).resolve().parent.parent
save_dir = base_dir / "data"
save_dir.mkdir(exist_ok=True)  # creates the "data" folder if it does not already exist
save_path = save_dir / f"{date_str}-prices.csv"

# === 3. Generate the download URL for the Azure raw data ===
# Tankerkönig hosts daily CSV files in the Azure DevOps repository
# The URL is assembled dynamically based on year, month, and date
url = (
    "https://dev.azure.com/tankerkoenig/362e70d1-bafa-4cf7-a346-1f3613304973/"
    "_apis/git/repositories/0d6e7286-91e4-402c-af56-fa75be1f223d/items"
    f"?path=/prices/{year}/{month}/{date_str}-prices.csv"
    "&versionDescriptor%5BversionOptions%5D=0"
    "&versionDescriptor%5BversionType%5D=0"
    "&versionDescriptor%5Bversion%5D=master"
    "&resolveLfs=true&%24format=octetStream"
    "&api-version=5.0&download=true"
)

# === 4. Download the CSV file ===
print(f"\nDownloading Tankerkönig data for {date_str} ...")

# Sends an HTTP GET request to the URL and waits for a response
resp = requests.get(url, timeout=60)

# Checks whether the download was successful (status code 200 = OK)
if resp.status_code != 200:
    raise SystemExit(
        f"Error {resp.status_code}: The file could not be downloaded.\nURL: {url}"
    )

# Saves the response content (resp.content = CSV binary data) locally as a file
with open(save_path, "wb") as f:
    f.write(resp.content)

print(f"File saved to: {save_path}")

# === 5. Load the CSV file and check its structure ===
df = pd.read_csv(save_path)  # read the CSV
df.columns = [c.strip().lower() for c in df.columns]  # standardize column names

print(f"\nCSV loaded: {len(df):,} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Checks whether the downloaded CSV file contains all expected columns
# -> expected_cols = set of all column names that must be present in the dataset
# -> set(df.columns) = creates a set of the actual column names found in the CSV
# -> expected_cols - set(df.columns) returns the missing columns
# -> If anything is missing: exit immediately with SystemExit to prevent uploading faulty data later on

expected_cols = {
    "date", "station_uuid", "diesel", "e5", "e10",
    "dieselchange", "e5change", "e10change"
}
missing = expected_cols - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns in CSV: {missing}")
else:
    print("All expected columns present.")

# === 6. Type conversions ===
# Date values are kept exactly as in the CSV (e.g. "2025-10-26 00:00:49+02").
# Currently there is no timezone conversion -> it is stripped during import, meaning "2025-10-26 00:00:49+02" becomes "2025-10-26 00:00:49" (final in the database)
df["date"] = df["date"].astype(str).str.strip()

# Converts the change fields dieselchange, e5change, e10change to small integers (int16) for space-efficient storage
# -> otherwise they would be uploaded to Supabase as text datatypes
for col in ["dieselchange", "e5change", "e10change"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

# Replace NaN with None (Supabase compatible)
df = df.where(pd.notnull(df), None)

# === 7. Establish the Supabase connection ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing environment variables: SUPABASE_URL or SUPABASE_SECRET_KEY.")

# Creates a Supabase client using the environment variables
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === 8. Upload the data to Supabase ===
records = df.to_dict(orient="records")  # convert to JSON-compatible structure (list of dictionaries)
chunk_size = 25000  # number of rows per upload batch

print(f"\nStarting upload of {len(records):,} rows to Supabase ...")

# Upload the CSV rows in chunks to avoid timeouts
for i in range(0, len(records), chunk_size):
    chunk = records[i:i + chunk_size]
    supabase.table("prices_test").insert(chunk).execute()
    print(f"{i + len(chunk):,} / {len(records):,} uploaded")

print("\nUpload complete.")
