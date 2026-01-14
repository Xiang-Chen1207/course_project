#!/usr/bin/env python3
"""
Check merged_segment_*.csv files under an output directory, find missing EEG
feature columns (NaN values), recompute them from the original HDF5 dataset,
and fill the values in-place.

Key improvements over original version:
- Only recomputes missing features (not all features)
- Longer timeout for recomputation (configurable, default 90s per feature)
- Retry mechanism for timed-out features
- Detailed logging of which features are missing

Example:
    python fill_missing_features.py \
        --input-h5-dir /mnt/dataset2/hdf5_datasets/Workload_MATB \
        --output-dir /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic \
        --merge-count 1 --preset basic --microstate-segs 20

Layout assumptions:
- output-dir contains subfolders (any name) each holding merged_segment_*.csv
    created from one H5 whose stem matches that subfolder name (e.g., sub_1 -> sub_1.h5).
- If no subfolders are found, but output-dir itself has merged_segment_*.csv, we treat
    that directory as a single subject.
"""
import argparse
import sys
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports: ensure project root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eeg_feature_extraction.config import Config, FrequencyBands, ChannelGroups
from eeg_feature_extraction.psd_computer import PSDComputer, PSDResult
from eeg_feature_extraction.data_loader import EEGDataLoader, SegmentData
from eeg_feature_extraction.features.base import BaseFeature, FeatureRegistry
from eeg_feature_extraction.features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    ComplexityFeatures,
    ConnectivityFeatures,
    NetworkFeatures,
    CompositeFeatures,
    DEFeatures,
    MicrostateFeatures,
    MicrostateAnalyzer,
)
from selective_feature_extraction import (
    FEATURE_GROUPS,
    FEATURE_TO_GROUP,
    PRESETS,
    FeatureSelectionConfig,
    apply_preset,
)

# Supplement microstate group
if 'microstate' not in FEATURE_GROUPS:
    FEATURE_GROUPS['microstate'] = MicrostateFeatures.feature_names.copy()
    for feat in MicrostateFeatures.feature_names:
        FEATURE_TO_GROUP[feat] = 'microstate'


# Required columns (all expected columns in output CSVs)
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

# Default timeout for feature recomputation (seconds)
DEFAULT_FEATURE_TIMEOUT = 90
# Number of retry attempts for timed-out features
DEFAULT_RETRY_COUNT = 2


@dataclass
class MergedSegmentData:
    """Merged segment data structure"""
    eeg_data: np.ndarray
    trial_ids: List[int]
    segment_ids: List[int]
    session_id: int
    labels: List[int]
    primary_label: int
    start_time: float
    end_time: float
    total_time_length: float
    merge_count: int
    source_segments: List[str]


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


def _find_missing_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that are missing or have NaN values."""
    missing: List[str] = []
    for col in REQUIRED_COLUMNS:
        if col in META_COLS:
            continue  # Skip metadata columns
        if col not in df.columns:
            missing.append(col)
        else:
            if df[col].isna().any():
                missing.append(col)
    return missing


def _find_missing_features_in_row(row: pd.Series) -> List[str]:
    """Find feature columns that have NaN values in a specific row."""
    missing: List[str] = []
    for col in REQUIRED_COLUMNS:
        if col in META_COLS:
            continue
        if col not in row.index:
            missing.append(col)
        elif pd.isna(row[col]):
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


def _get_groups_for_features(feature_names: List[str]) -> Set[str]:
    """Get the set of feature groups required to compute the given features."""
    groups = set()
    for feat in feature_names:
        if feat in FEATURE_TO_GROUP:
            groups.add(FEATURE_TO_GROUP[feat])
    return groups


class TargetedFeatureExtractor:
    """Feature extractor that only computes specified features with longer timeout."""

    def __init__(
        self,
        config: Config,
        target_features: Set[str],
        timeout_sec: int = DEFAULT_FEATURE_TIMEOUT,
        retry_count: int = DEFAULT_RETRY_COUNT,
    ):
        self.config = config
        self.target_features = target_features
        self.timeout_sec = timeout_sec
        self.retry_count = retry_count

        # PSD computer
        self.psd_computer = PSDComputer(
            sampling_rate=config.sampling_rate,
            use_gpu=config.use_gpu,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            nfft=config.nfft
        )

        # Initialize only required feature computers
        self.feature_computers: Dict[str, BaseFeature] = {}
        required_groups = _get_groups_for_features(list(target_features))
        all_feature_classes = FeatureRegistry.get_all_feature_classes()
        for group_name in required_groups:
            if group_name in all_feature_classes:
                self.feature_computers[group_name] = all_feature_classes[group_name](config)

    def _run_with_timeout(self, fn, timeout_sec: int, *args, **kwargs):
        """Run function with timeout using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                raise TimeoutError("feature computation timed out")

    def extract_features(
        self,
        eeg_data: np.ndarray,
        microstate_analyzer: Optional[MicrostateAnalyzer] = None
    ) -> Dict[str, Any]:
        """Extract only the target features with retry mechanism."""
        # Compute PSD once
        psd_result = self.psd_computer.compute_psd(
            eeg_data,
            bands=self.config.freq_bands.get_all_bands()
        )

        all_features: Dict[str, Any] = {}

        for group_name, computer in self.feature_computers.items():
            # Filter to only features we need from this group
            group_features = [f for f in computer.get_feature_names() if f in self.target_features]
            if not group_features:
                continue

            for feat_name in group_features:
                # Try multiple times with longer timeout
                value = None
                last_error = None

                for attempt in range(self.retry_count + 1):
                    try:
                        if group_name == 'microstate':
                            def _task():
                                res = computer.compute(
                                    eeg_data, psd_result=psd_result,
                                    microstate_analyzer=microstate_analyzer
                                )
                                return res.get(feat_name, None)
                        else:
                            def _task():
                                res = computer.compute(eeg_data, psd_result=psd_result)
                                return res.get(feat_name, None)

                        value = self._run_with_timeout(_task, self.timeout_sec)
                        if value is not None:
                            break
                    except TimeoutError as e:
                        last_error = f"Timeout (attempt {attempt + 1}/{self.retry_count + 1})"
                        if attempt < self.retry_count:
                            continue
                    except Exception as e:
                        last_error = str(e)
                        break

                if value is not None:
                    all_features[feat_name] = value
                else:
                    if last_error:
                        warnings.warn(f"Failed to compute '{feat_name}': {last_error}")

        return all_features


def _compute_microstate_template(
    loader: EEGDataLoader,
    microstate_segs: Optional[int],
    verbose: bool = True
) -> MicrostateAnalyzer:
    """Generate microstate template from segments."""
    if verbose:
        if microstate_segs and microstate_segs > 0:
            print(f"  Generating microstate template (sampling {microstate_segs} segments per trial)...")
        else:
            print("  Generating microstate template (using all segments)...")

    analyzer = MicrostateAnalyzer(n_states=4)
    all_peak_maps: List[np.ndarray] = []
    n_segments_used = 0
    n_trials = 0

    trial_names = loader.get_trial_names()
    for trial_name in trial_names:
        segment_names = loader.get_segment_names(trial_name)
        n_trials += 1

        if microstate_segs and microstate_segs > 0 and len(segment_names) > microstate_segs:
            rng = np.random.default_rng(seed=42 + n_trials)
            selected_indices = rng.choice(len(segment_names), size=microstate_segs, replace=False)
            selected_segment_names = [segment_names[i] for i in sorted(selected_indices)]
        else:
            selected_segment_names = segment_names

        for seg_name in selected_segment_names:
            segment = loader.get_segment(trial_name, seg_name)
            data = segment.eeg_data
            gfp = analyzer.compute_gfp(data)
            peak_indices = analyzer.find_gfp_peaks(gfp)
            peak_maps = data[:, peak_indices].T
            if peak_maps.size > 0:
                all_peak_maps.append(peak_maps)
            n_segments_used += 1

    if n_segments_used == 0 or len(all_peak_maps) == 0:
        raise ValueError("No valid segment peak maps for microstate template generation")

    combined_maps = np.vstack(all_peak_maps)
    analyzer.centroids = analyzer._polarity_invariant_kmeans(combined_maps)

    if verbose:
        print(f"  Microstate template generated: {n_trials} trials, {n_segments_used} segments used")

    return analyzer


def _discover_subject_dirs(output_dir: Path) -> List[Path]:
    """Find subdirectories that contain merged_segment CSVs."""
    subject_dirs = []
    for sub in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        if any(sub.glob("merged_segment_*.csv")):
            subject_dirs.append(sub)
    return subject_dirs


def _parse_source_segments(source_str: str) -> List[str]:
    """Parse source_segments string like "['trial_0/seg_0', 'trial_0/seg_1']" """
    try:
        import ast
        return ast.literal_eval(source_str)
    except Exception:
        return []


def _load_segment_data(
    loader: EEGDataLoader,
    source_segments: List[str],
    merge_count: int,
) -> Optional[MergedSegmentData]:
    """Load and merge segment data based on source_segments list."""
    segments_to_merge = []

    for src in source_segments:
        parts = src.split('/')
        if len(parts) != 2:
            continue
        trial_name, seg_name = parts
        try:
            seg = loader.get_segment(trial_name, seg_name)
            segments_to_merge.append((trial_name, seg_name, seg))
        except Exception:
            return None

    if not segments_to_merge:
        return None

    # Merge EEG data
    eeg_arrays = [seg.eeg_data for _, _, seg in segments_to_merge]
    merged_eeg = np.concatenate(eeg_arrays, axis=1)

    trial_ids = [seg.trial_id for _, _, seg in segments_to_merge]
    segment_ids = [seg.segment_id for _, _, seg in segments_to_merge]
    labels = [seg.label for _, _, seg in segments_to_merge]
    src_segments = [f"{t}/{s}" for t, s, _ in segments_to_merge]

    first_seg = segments_to_merge[0][2]
    last_seg = segments_to_merge[-1][2]
    total_time = sum(seg.time_length for _, _, seg in segments_to_merge)

    return MergedSegmentData(
        eeg_data=merged_eeg,
        trial_ids=trial_ids,
        segment_ids=segment_ids,
        session_id=first_seg.session_id,
        labels=labels,
        primary_label=labels[0],
        start_time=first_seg.start_time,
        end_time=last_seg.end_time,
        total_time_length=total_time,
        merge_count=len(segments_to_merge),
        source_segments=src_segments,
    )


def _fill_csv_file(
    csv_path: Path,
    loader: EEGDataLoader,
    config: Config,
    merge_count: int,
    microstate_analyzer: Optional[MicrostateAnalyzer],
    timeout_sec: int,
    retry_count: int,
    verbose: bool,
) -> Tuple[bool, List[str], List[str]]:
    """
    Fill missing features in a single CSV file.

    Returns:
        (changed, missing_cols_before, still_missing_cols)
    """
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    # Check if there are any missing features
    missing_cols = _find_missing_columns(df)
    if not missing_cols:
        return False, [], []

    if verbose:
        print(f"    {csv_path.name}: {len(missing_cols)} missing features: {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")

    changed = False
    still_missing = set(missing_cols)

    for idx, row in df.iterrows():
        row_missing = _find_missing_features_in_row(row)
        if not row_missing:
            continue

        # Get source segments and load data
        src_str = str(row.get("source_segments", "[]"))
        source_segs = _parse_source_segments(src_str)
        if not source_segs:
            warnings.warn(f"Cannot parse source_segments: {src_str}")
            continue

        merged_data = _load_segment_data(loader, source_segs, merge_count)
        if merged_data is None:
            warnings.warn(f"Cannot load segment data for: {src_str}")
            continue

        # Create targeted extractor for only missing features
        target_features = set(row_missing)
        extractor = TargetedFeatureExtractor(
            config=config,
            target_features=target_features,
            timeout_sec=timeout_sec,
            retry_count=retry_count,
        )

        # Extract features
        try:
            computed = extractor.extract_features(
                merged_data.eeg_data,
                microstate_analyzer=microstate_analyzer
            )
        except Exception as e:
            warnings.warn(f"Feature extraction error: {e}")
            computed = {}

        # Fill in computed values
        for feat_name, value in computed.items():
            if pd.isna(row[feat_name]) and value is not None:
                df.at[idx, feat_name] = value
                changed = True
                if feat_name in still_missing:
                    still_missing.discard(feat_name)

    # Save updated CSV
    if changed:
        df = _reorder_columns(df)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return changed, missing_cols, list(still_missing)


def rebuild_summary(output_dir: Path) -> None:
    """Rebuild all_merged_features.csv from individual CSVs."""
    csv_files = sorted(output_dir.glob("sub_*/merged_segment_*.csv"))
    if not csv_files:
        csv_files = sorted(output_dir.glob("merged_segment_*.csv"))
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
    parser = argparse.ArgumentParser(
        description="Fill missing EEG features in merged CSVs with targeted recomputation"
    )
    parser.add_argument(
        "--input-h5-dir", type=str, required=True,
        help="Directory containing source HDF5 files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory with merged_segment CSV outputs"
    )
    parser.add_argument("--merge-count", type=int, default=1, help="Merge count used originally")
    parser.add_argument("--preset", type=str, default="basic", help="Preset used originally")
    parser.add_argument("--microstate-segs", type=int, default=20, help="Microstate segments per trial")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_FEATURE_TIMEOUT,
        help=f"Timeout per feature in seconds (default: {DEFAULT_FEATURE_TIMEOUT})"
    )
    parser.add_argument(
        "--retry", type=int, default=DEFAULT_RETRY_COUNT,
        help=f"Number of retry attempts for timed-out features (default: {DEFAULT_RETRY_COUNT})"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU during recomputation")
    parser.add_argument("--skip-summary", action="store_true", help="Do not rebuild all_merged_features.csv")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    input_dir = Path(args.input_h5_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("=" * 60)
    print("Fill Missing Features - Targeted Recomputation")
    print("=" * 60)
    print(f"Input H5 directory: {input_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Timeout per feature: {args.timeout}s")
    print(f"Retry attempts: {args.retry}")
    print("-" * 60)

    # Discover subject directories
    subjects = _discover_subject_dirs(output_dir)
    if not subjects and any(output_dir.glob("merged_segment_*.csv")):
        subjects = [output_dir]

    if not subjects:
        print(f"No merged_segment CSVs found under {output_dir}")
        sys.exit(1)

    print(f"Found {len(subjects)} subject(s) to process")

    total_files_changed = 0
    total_features_fixed = 0

    for subject_dir in subjects:
        # Find matching H5 file
        candidates = _candidate_h5_paths(input_dir, subject_dir)
        h5_path = next((c for c in candidates if c.exists()), None)

        if h5_path is None:
            h5_files = list(input_dir.glob("*.h5"))
            if len(h5_files) == 1:
                h5_path = h5_files[0]
            else:
                print(f"Skipping {subject_dir.name}: H5 file not found")
                continue

        print(f"\nProcessing {subject_dir.name} using {h5_path.name}")

        # Load data
        loader = EEGDataLoader(str(h5_path))
        subject_info = loader.get_subject_info()

        # Create config
        config = Config()
        config.use_gpu = not args.no_gpu
        config.update_from_electrode_names(subject_info.channel_names)
        config.sampling_rate = subject_info.sampling_rate

        # Generate microstate template if needed
        microstate_analyzer = None
        selection_cfg = apply_preset(args.preset)
        if 'microstate' in selection_cfg.get_required_groups():
            try:
                microstate_analyzer = _compute_microstate_template(
                    loader, args.microstate_segs, verbose=args.verbose
                )
            except Exception as e:
                warnings.warn(f"Failed to generate microstate template: {e}")

        # Process each CSV file
        csv_files = sorted(subject_dir.glob("merged_segment_*.csv"))
        subject_changed = 0
        subject_features_fixed = 0

        if args.verbose:
            csv_iter = tqdm(csv_files, desc=f"  Processing {subject_dir.name}")
        else:
            csv_iter = csv_files

        for csv_path in csv_iter:
            try:
                changed, missing_before, still_missing = _fill_csv_file(
                    csv_path=csv_path,
                    loader=loader,
                    config=config,
                    merge_count=args.merge_count,
                    microstate_analyzer=microstate_analyzer,
                    timeout_sec=args.timeout,
                    retry_count=args.retry,
                    verbose=args.verbose,
                )
                if changed:
                    subject_changed += 1
                    features_fixed = len(missing_before) - len(still_missing)
                    subject_features_fixed += features_fixed
                    if args.verbose and still_missing:
                        print(f"      Still missing after retry: {still_missing}")
            except Exception as e:
                warnings.warn(f"Error processing {csv_path.name}: {e}")
                if args.verbose:
                    traceback.print_exc()

        total_files_changed += subject_changed
        total_features_fixed += subject_features_fixed
        print(f"  {subject_dir.name}: {subject_changed} files updated, {subject_features_fixed} features fixed")

    # Rebuild summary
    if not args.skip_summary:
        rebuild_summary(output_dir)

    print("\n" + "=" * 60)
    print(f"Done. Total files updated: {total_files_changed}")
    print(f"Total features fixed: {total_features_fixed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
