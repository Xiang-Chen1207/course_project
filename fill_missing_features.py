#!/usr/bin/env python3
"""
Check merged_segment_*.csv files under an output directory, find missing EEG
feature columns, recompute them from the original HDF5 dataset, and fill the
values in-place.

Example (directory of many H5 files → per-file subfolders of CSVs):
    python fill_missing_features.py \
        --input-h5-dir /mnt/dataset2/hdf5_datasets/Workload_MATB \
        --output-dir /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic \
        --merge-count 1 --preset basic --microstate-segs 20

Layout assumptions (now flexible):
- Default: output-dir contains subfolders (any name) each holding merged_segment_*.csv
    created from one H5 whose stem matches that subfolder name (e.g., sub_1 → sub_1.h5).
- If no subfolders are found, but output-dir itself has merged_segment_*.csv, we treat
    that directory as a single subject and pick the H5 automatically (single H5 in the
    input directory or one matching the output folder name).
- Only columns listed in REQUIRED_COLUMNS are checked/fixed. Existing values are
    preserved; only missing/NaN cells are filled from recomputation.
- A rebuilt all_merged_features.csv is written at the end (optional).
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# Local imports: ensure project root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from merged_segment_extraction import (  # type: ignore
    Config,
    FeatureSelectionConfig,
    MergedSegmentExtractor,
    apply_preset,
)

# Required columns supplied by the user (order preserved)
REQUIRED_COLUMNS: List[str] = [
    "trial_ids",
    "segment_ids",
    "session_id",
    "primary_label",
    "labels",
    "start_time",
    "end_time",
    "total_time_length",
    "merge_count",
    "source_segments",
    "Microstate_0_meandurs",
    "Microstate_0_occurrence",
    "Microstate_0_timecov",
    "Microstate_0_mean_corr",
    "Microstate_0_gev",
    "Microstate_1_meandurs",
    "Microstate_1_occurrence",
    "Microstate_1_timecov",
    "Microstate_1_mean_corr",
    "Microstate_1_gev",
    "Microstate_2_meandurs",
    "Microstate_2_occurrence",
    "Microstate_2_timecov",
    "Microstate_2_mean_corr",
    "Microstate_2_gev",
    "Microstate_3_meandurs",
    "Microstate_3_occurrence",
    "Microstate_3_timecov",
    "Microstate_3_mean_corr",
    "Microstate_3_gev",
    "theta_alpha_ratio",
    "frontal_beta_ratio",
    "cognitive_load_estimate",
    "alertness_estimate",
    "relaxation_index",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "low_gamma_power",
    "high_gamma_power",
    "delta_relative_power",
    "theta_relative_power",
    "alpha_relative_power",
    "beta_relative_power",
    "gamma_relative_power",
    "low_gamma_relative_power",
    "high_gamma_relative_power",
    "peak_frequency",
    "spectral_entropy",
    "spectral_centroid",
    "individual_alpha_frequency",
    "theta_beta_ratio",
    "delta_theta_ratio",
    "low_high_power_ratio",
    "aperiodic_exponent",
    "mean_total_power",
    "wavelet_energy_entropy",
    "higuchi_fd",
    "katz_fd",
    "petrosian_fd",
    "mean_interchannel_correlation",
    "mean_alpha_coherence",
    "interhemispheric_alpha_coherence",
    "alpha_beta_band_power_correlation",
    "hemispheric_alpha_asymmetry",
    "frontal_occipital_alpha_ratio",
    "plv_theta_mean",
    "plv_alpha_mean",
    "plv_beta_mean",
    "plv_gamma_mean",
    "plv_theta_interhemispheric",
    "plv_alpha_interhemispheric",
    "de_delta",
    "de_theta",
    "de_alpha",
    "de_beta",
    "de_gamma",
    "de_low_gamma",
    "de_high_gamma",
    "dasm_delta",
    "dasm_theta",
    "dasm_alpha",
    "dasm_beta",
    "dasm_gamma",
    "rasm_delta",
    "rasm_theta",
    "rasm_alpha",
    "rasm_beta",
    "rasm_gamma",
    "dcau_delta",
    "dcau_theta",
    "dcau_alpha",
    "dcau_beta",
    "dcau_gamma",
    "faa_f3f4",
    "faa_f7f8",
    "faa_fp1fp2",
    "faa_mean",
    "mean_abs_amplitude",
    "mean_channel_std",
    "mean_peak_to_peak",
    "mean_rms",
    "mean_zero_crossing_rate",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "network_clustering_coefficient",
    "network_characteristic_path_length",
    "network_global_efficiency",
]

META_COLS: List[str] = REQUIRED_COLUMNS[:10]


def _subject_id_from_name(name: str) -> Optional[str]:
    """Extract subject id from subdirectory name like sub_1."""
    if name.startswith("sub_") and len(name.split("_")) == 2:
        return name.split("_")[1]
    return None


def _candidate_h5_paths(input_dir: Path, subject_dir: Path) -> List[Path]:
    """Return likely H5 paths for a subject directory name."""
    sid = _subject_id_from_name(subject_dir.name)
    candidates: List[Path] = []
    if sid:
        candidates.append(input_dir / f"sub_{sid}.h5")
        candidates.append(input_dir / f"{sid}.h5")
    candidates.append(input_dir / f"{subject_dir.name}.h5")
    return candidates


def _load_recomputed_df(
    h5_path: Path,
    tmp_out: Path,
    merge_count: int,
    preset: str,
    microstate_segs: Optional[int],
    n_jobs: Optional[int],
    no_gpu: bool,
    use_parallel: bool,
) -> pd.DataFrame:
    cfg = Config()
    cfg.use_gpu = not no_gpu
    selection_cfg = apply_preset(preset)
    extractor = MergedSegmentExtractor(
        config=cfg,
        selection_config=selection_cfg,
        merge_count=merge_count,
        cross_trial=False,
        n_jobs=n_jobs,
        microstate_segments_per_trial=microstate_segs,
    )
    tmp_out.mkdir(parents=True, exist_ok=True)
    df_new = extractor.process_h5_file(
        str(h5_path), str(tmp_out), verbose=False, use_parallel=use_parallel
    )
    return df_new


def _find_missing_columns(df: pd.DataFrame) -> List[str]:
    missing: List[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)
        else:
            if df[col].isna().any():
                missing.append(col)
    return missing


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any required columns that are absent, filled with NA."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in REQUIRED_COLUMNS if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    return df[ordered + tail]


def _fill_file(
    csv_path: Path,
    df_new_by_source: Dict[str, pd.Series],
) -> Tuple[bool, List[str]]:
    """
    Fill missing/NaN columns in one CSV using recomputed features.

    Returns (changed, missing_cols_detected).
    """
    df_old = pd.read_csv(csv_path)
    df_old = _ensure_columns(df_old)
    missing_cols = _find_missing_columns(df_old)
    if not missing_cols:
        return False, []

    changed = False
    for idx, row in df_old.iterrows():
        src_key = str(row["source_segments"])
        if src_key not in df_new_by_source:
            warnings.warn(f"source_segments not found in recomputed data: {src_key}")
            continue
        recomputed = df_new_by_source[src_key]
        for col in missing_cols:
            if col not in recomputed:
                continue
            if pd.isna(row[col]):
                df_old.at[idx, col] = recomputed[col]
                changed = True
    if changed:
        df_old = _reorder_columns(df_old)
        df_old.to_csv(csv_path, index=False, encoding="utf-8")
    return changed, missing_cols


def _discover_subject_dirs(output_dir: Path) -> List[Path]:
    """Find subdirectories that contain merged_segment CSVs."""
    subject_dirs = []
    for sub in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        if any(sub.glob("merged_segment_*.csv")):
            subject_dirs.append(sub)
    return subject_dirs


def rebuild_summary(output_dir: Path) -> None:
    csv_files = sorted(output_dir.glob("sub_*/merged_segment_*.csv"))
    if not csv_files:
        return
    frames = [pd.read_csv(p) for p in csv_files]
    if not frames:
        return
    summary = pd.concat(frames, ignore_index=True)
    summary = _ensure_columns(summary)
    summary = _reorder_columns(summary)
    summary_path = output_dir / "all_merged_features.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Rebuilt summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing EEG features in merged CSVs")
    parser.add_argument("--input-h5-dir", type=str, required=True, help="Directory containing source HDF5 files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory with merged_segment CSV outputs")
    parser.add_argument("--merge-count", type=int, default=1, help="Merge count used originally")
    parser.add_argument("--preset", type=str, default="basic", help="Preset used originally")
    parser.add_argument("--microstate-segs", type=int, default=20, help="Microstate segments per trial")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for recomputation")
    parser.add_argument("--no-parallel", action="store_true", help="Force sequential recomputation")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU during recomputation")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary recomputed CSVs")
    parser.add_argument("--skip-summary", action="store_true", help="Do not rebuild all_merged_features.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_h5_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    use_parallel = not args.no_parallel

    subjects = _discover_subject_dirs(output_dir)
    # Fallback: if no subdirectories, but CSVs exist directly in output_dir, treat it as one subject
    if not subjects and any(output_dir.glob("merged_segment_*.csv")):
        subjects = [output_dir]

    if not subjects:
        print(f"No merged_segment CSVs found under {output_dir}")
        sys.exit(1)

    total_changed = 0
    for subject_dir in subjects:
        candidates = _candidate_h5_paths(input_dir, subject_dir)
        h5_path = next((c for c in candidates if c.exists()), None)

        if h5_path is None:
            # If no direct match, but there is exactly one H5 in input_dir, use it.
            h5_files = list(input_dir.glob("*.h5"))
            if len(h5_files) == 1:
                h5_path = h5_files[0]
            else:
                print(f"Skipping {subject_dir.name}: H5 file not found in {input_dir}")
                continue

        print(f"\nProcessing {subject_dir.name} using {h5_path.name}")
        tmp_out = subject_dir / "_tmp_recompute"
        df_new = _load_recomputed_df(
            h5_path=h5_path,
            tmp_out=tmp_out,
            merge_count=args.merge_count,
            preset=args.preset,
            microstate_segs=args.microstate_segs,
            n_jobs=args.n_jobs,
            no_gpu=args.no_gpu,
            use_parallel=use_parallel,
        )
        if df_new.empty:
            print("  Recomputed DataFrame is empty, skipping.")
            continue

        # Index recomputed rows by source_segments string for quick lookup
        df_new = _ensure_columns(df_new)
        df_new = _reorder_columns(df_new)
        df_new_by_source: Dict[str, pd.Series] = {
            str(row["source_segments"]): row for _, row in df_new.iterrows()
        }

        subject_changed = 0
        csv_files = sorted(subject_dir.glob("merged_segment_*.csv"))
        for csv_path in csv_files:
            changed, missing_cols = _fill_file(csv_path, df_new_by_source)
            if changed:
                subject_changed += 1
                print(f"  Filled {csv_path.name}; columns fixed: {missing_cols}")
        total_changed += subject_changed

        if not args.keep_temp and tmp_out.exists():
            import shutil

            try:
                shutil.rmtree(tmp_out)
            except Exception:
                warnings.warn(f"Failed to remove temp directory: {tmp_out}")

    if not args.skip_summary:
        rebuild_summary(output_dir)

    print(f"\nDone. Files updated: {total_changed}")


if __name__ == "__main__":
    main()
