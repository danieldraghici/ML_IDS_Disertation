"""
CIC-IDS2017 data preparation pipeline.

Loads the raw MachineLearningCSV files, reports the dataset's known quality
issues, cleans them, and saves a processed dataset ready for model training.

Usage:
    python prepare_data.py --explore          # inspect only, write nothing
    python prepare_data.py                    # full clean, save processed set
    python prepare_data.py --sample 300000    # clean + stratified subsample

Expected layout:
    data/raw/*.csv          <- unzipped MachineLearningCSV
    data/processed/         <- output written here
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "cicids2017_clean.parquet")

# The label column after whitespace stripping
LABEL_COL = "Label"


def load_raw(raw_dir=RAW_DIR):
    """Load and concatenate every CSV in the raw directory.

    CIC-IDS2017 ships as 8 files (one per capture session). Some of them
    contain non-UTF-8 bytes, so we fall back to latin-1 rather than crashing.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{raw_dir}'. "
            "Unzip MachineLearningCSV.zip into that folder first."
        )

    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")

        # Strip the leading spaces CIC-IDS2017 has in most column names
        # (e.g. " Destination Port" -> "Destination Port")
        df.columns = df.columns.str.strip()

        print(f"  loaded {os.path.basename(path):<55} {df.shape[0]:>9,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  combined shape: {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    return combined


def report_quality(df):
    """Print the data quality issues that need to be handled.

    Documenting these numbers is worth doing: they belong in the methodology
    chapter as evidence that the cleaning decisions were deliberate.
    """
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- Class distribution: this is where the imbalance becomes obvious ---
    print(f"\nClass distribution ({LABEL_COL}):")
    counts = df[LABEL_COL].value_counts()
    total = len(df)
    for label, count in counts.items():
        print(f"  {label:<28} {count:>9,}  ({count / total * 100:6.3f}%)")

    # --- Infinite values: Flow Bytes/s and Flow Packets/s are the usual culprits ---
    numeric = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric).sum()
    inf_cols = inf_mask[inf_mask > 0]
    print(f"\nInfinite values: {int(inf_mask.sum()):,} total")
    for col, count in inf_cols.items():
        print(f"  {col:<28} {count:>9,}")

    # --- Missing values ---
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    print(f"\nMissing values: {int(na_counts.sum()):,} total")
    for col, count in na_cols.items():
        print(f"  {col:<28} {count:>9,}")

    # --- Duplicates ---
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate rows: {dup_count:,} ({dup_count / total * 100:.2f}%)")

    # --- Constant columns carry no signal and can be dropped ---
    constant = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
    print(f"\nConstant columns ({len(constant)}): {constant if constant else 'none'}")

    print("=" * 70 + "\n")


def clean(df):
    """Apply the cleaning steps, reporting how many rows each one removes."""
    print("CLEANING")
    print("-" * 70)
    start_rows = len(df)

    # 1. Replace infinities with NaN so a single dropna handles both
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 2. Drop rows with missing values. They are a small fraction of the
    #    dataset, so dropping is cleaner than imputing values we can't justify.
    before = len(df)
    df = df.dropna()
    print(f"  dropped {before - len(df):>9,} rows with NaN/inf")

    # 3. Drop exact duplicates. These inflate accuracy and leak between
    #    train and test splits if left in.
    before = len(df)
    df = df.drop_duplicates()
    print(f"  dropped {before - len(df):>9,} duplicate rows")

    # 4. Drop constant columns - zero variance means zero predictive value
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
    if constant:
        df = df.drop(columns=constant)
        print(f"  dropped {len(constant):>9} constant columns")

    # 5. Normalise label text (some files use non-breaking hyphens/spaces)
    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .str.strip()
        .str.replace("\u2013", "-", regex=False)
        .str.replace("\u00a0", " ", regex=False)
    )

    # 6. Add a binary target alongside the multi-class label, so both
    #    binary and multi-class experiments can reuse this file.
    df["is_attack"] = (df[LABEL_COL] != "BENIGN").astype(int)

    removed = start_rows - len(df)
    print("-" * 70)
    print(f"  {len(df):,} rows remain ({removed:,} removed, "
          f"{removed / start_rows * 100:.2f}%)\n")
    return df


def stratified_sample(df, n_rows, random_state=42):
    """Take a stratified subsample that keeps every class represented.

    Development iterations get slow on the full ~2.8M rows. Sampling
    proportionally keeps the class ratios intact, and rare attack classes
    are kept in full so they don't vanish entirely.
    """
    if n_rows >= len(df):
        print(f"Sample size {n_rows:,} >= dataset size, skipping.\n")
        return df

    fraction = n_rows / len(df)
    parts = []
    for label, group in df.groupby(LABEL_COL):
        # Keep small classes whole; sample proportionally from large ones
        take = len(group) if len(group) < 1000 else max(int(len(group) * fraction), 1000)
        take = min(take, len(group))
        parts.append(group.sample(n=take, random_state=random_state))

    sampled = pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Stratified sample: {len(sampled):,} rows\n")
    return sampled


def save(df, path=OUTPUT_FILE):
    """Save as Parquet - far smaller and faster to reload than CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        # pyarrow not installed - fall back to CSV
        path = path.replace(".parquet", ".csv")
        df.to_csv(path, index=False)
        print("(pyarrow missing, saved as CSV instead - `pip install pyarrow` for Parquet)")

    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"Saved -> {path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare the CIC-IDS2017 dataset.")
    parser.add_argument("--explore", action="store_true",
                        help="only print the quality report, write nothing")
    parser.add_argument("--sample", type=int, default=None,
                        help="stratified subsample size, e.g. 300000")
    parser.add_argument("--raw-dir", default=RAW_DIR, help="raw CSV directory")
    args = parser.parse_args()

    print("\nLoading raw CSV files...")
    df = load_raw(args.raw_dir)

    report_quality(df)

    if args.explore:
        print("Explore mode - nothing written. Drop --explore to clean and save.\n")
        return

    df = clean(df)

    if args.sample:
        df = stratified_sample(df, args.sample)

    # Final class distribution after cleaning
    print("Final class distribution:")
    for label, count in df[LABEL_COL].value_counts().items():
        print(f"  {label:<28} {count:>9,}")
    print()

    save(df)


if __name__ == "__main__":
    main()