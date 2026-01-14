#!/usr/bin/env python3
"""
Check column count consistency across merged_segment CSV files.

Usage:
  python check_column_consistency.py \
    --ref /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic/sub_1/merged_segment_0000.csv \
    --root /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic \
    --out  /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic/column_mismatch

It reads the reference file to get the expected column count, checks all CSVs
under the root's sub_* folders, and copies any file with a different column
count into the output folder.
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd


def count_columns(csv_path: Path) -> int:
    """Return number of columns by reading only the header."""
    df = pd.read_csv(csv_path, nrows=0)
    return len(df.columns)


def main():
    parser = argparse.ArgumentParser(description="Check merged_segment CSV column consistency")
    parser.add_argument("--ref", required=True, type=Path, help="Reference CSV file path")
    parser.add_argument("--root", required=True, type=Path, help="Root folder containing sub_* folders")
    parser.add_argument("--out", required=True, type=Path, help="Folder to store mismatched CSV copies")
    args = parser.parse_args()

    ref_path = args.ref
    root = args.root
    out_dir = args.out

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Root folder not found: {root}")

    expected_cols = count_columns(ref_path)
    print(f"Reference columns: {expected_cols} ({ref_path})")

    out_dir.mkdir(parents=True, exist_ok=True)

    mismatches = []
    # Search sub_* folders for CSVs named merged_segment_*.csv
    for csv_path in root.glob("sub_*/merged_segment_*.csv"):
        col_count = count_columns(csv_path)
        if col_count != expected_cols:
            mismatches.append((csv_path, col_count))
            dest = out_dir / csv_path.name
            shutil.copy2(csv_path, dest)

    if mismatches:
        print(f"Found {len(mismatches)} mismatched files. Copied to: {out_dir}")
        for path, cnt in mismatches:
            print(f" - {path} (cols={cnt})")
    else:
        print("All files match the reference column count.")


if __name__ == "__main__":
    main()