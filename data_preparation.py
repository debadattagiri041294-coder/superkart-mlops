"""
data_preparation.py
--------------------
Step 1 of the SuperKart Sales Forecast MLOps pipeline.

What this script does:
1. Loads the raw SuperKart dataset from the Hugging Face Hub dataset space.
2. Cleans the data (handles missing values, fixes inconsistent categories,
   drops columns that don't add predictive value).
3. Splits the cleaned data into train/test sets.
4. Saves train/test CSVs locally under ./data.
5. Uploads the train/test CSVs back to the Hugging Face dataset space.

Usage:
    python data_preparation.py

Environment variables required:
    HF_TOKEN        - Hugging Face access token with write permission
    HF_DATASET_REPO - e.g. "your-username/superkart-sales-data"
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download, login

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "your-username/superkart-sales-data")
RAW_FILENAME = "SuperKart4.csv"
LOCAL_DATA_DIR = "data"
TRAIN_PATH = os.path.join(LOCAL_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(LOCAL_DATA_DIR, "test.csv")
TARGET_COL = "Product_Store_Sales_Total"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from the Hugging Face dataset space.
    Falls back to the local copy if HF_TOKEN isn't set (useful for local dev)."""
    if HF_TOKEN:
        login(token=HF_TOKEN)
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=RAW_FILENAME,
            repo_type="dataset",
        )
        return pd.read_csv(local_path)

    print("HF_TOKEN not set — loading local CSV instead of Hugging Face Hub.")
    return pd.read_csv(os.path.join(LOCAL_DATA_DIR, RAW_FILENAME))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the raw dataframe."""
    df = df.copy()

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Standardize sugar content labels (dataset commonly has 'LF', 'low fat', etc.)
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = (
            df["Product_Sugar_Content"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Lf": "Low Sugar", "Reg": "Regular"})
        )

    # Fill missing product weight with the median (numeric, skew-robust)
    if "Product_Weight" in df.columns:
        df["Product_Weight"] = df["Product_Weight"].fillna(df["Product_Weight"].median())

    # Fill missing store size with the mode (categorical)
    if "Store_Size" in df.columns and df["Store_Size"].isna().any():
        df["Store_Size"] = df["Store_Size"].fillna(df["Store_Size"].mode()[0])

    # Derive store age instead of raw establishment year (more useful signal,
    # and avoids the model treating "year" as an unbounded numeric feature)
    if "Store_Establishment_Year" in df.columns:
        current_year = pd.Timestamp.now().year
        df["Store_Age"] = current_year - df["Store_Establishment_Year"]
        df = df.drop(columns=["Store_Establishment_Year"])

    # Drop identifier columns that carry no generalizable signal
    id_cols = [c for c in ["Product_Id", "Store_Id"] if c in df.columns]
    df = df.drop(columns=id_cols)

    # Drop rows missing the target
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    """Split into train/test and save locally."""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved train ({train_df.shape}) -> {TRAIN_PATH}")
    print(f"Saved test  ({test_df.shape}) -> {TEST_PATH}")
    return train_df, test_df


def upload_to_hf():
    """Upload train/test CSVs to the Hugging Face dataset space."""
    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping upload to Hugging Face Hub.")
        return

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True)

    for path, name in [(TRAIN_PATH, "train.csv"), (TEST_PATH, "test.csv")]:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
        )
        print(f"Uploaded {name} -> {HF_DATASET_REPO}")


def main():
    print("Loading raw data...")
    raw_df = load_raw_data()
    print(f"Raw shape: {raw_df.shape}")

    print("Cleaning data...")
    clean_df = clean_data(raw_df)
    print(f"Cleaned shape: {clean_df.shape}")

    print("Splitting and saving locally...")
    split_and_save(clean_df)

    print("Uploading to Hugging Face Hub...")
    upload_to_hf()

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
