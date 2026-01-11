"""
Quick diagnostic for band coverage / PSD grid / data quality on an EEG HDF5 file.

Usage:
    python scripts/debug_band_coverage.py --h5 /path/to/file.h5 [--max-segments 1]

What it reports:
- Sampling rate, nperseg actually used, freq range and resolution of Welch PSD
- For each default band (delta/theta/alpha/beta/gamma): number of freq bins inside the band
- Band power stats (min/max/median across channels) and whether any NaN/inf
- Channel std stats to catch near-constant channels
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.data_loader import EEGDataLoader
from eeg_feature_extraction.psd_computer import PSDComputer


def analyze_file(h5_path: str, max_segments: int = 1):
    cfg = Config()
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()

    fs = float(getattr(subject_info, "sampling_rate", cfg.sampling_rate))
    cfg.sampling_rate = fs

    # Use default PSD params but cap nperseg by segment length later
    psd_comp = PSDComputer(
        sampling_rate=cfg.sampling_rate,
        use_gpu=False,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        nfft=cfg.nfft,
    )

    bands = cfg.freq_bands.get_all_bands()

    print(f"File: {h5_path}")
    print(f"Sampling rate (fs): {fs} Hz")
    print(f"Default nperseg: {cfg.nperseg}, nfft: {cfg.nfft}")

    seg_count = 0
    for trial_name in loader.get_trial_names():
        for seg_name in loader.get_segment_names(trial_name):
            trial_seg = loader.get_segment(trial_name, seg_name)
            data = trial_seg.eeg_data  # shape (n_channels, n_samples)
            n_ch, n_samp = data.shape

            # Adjust nperseg for this segment
            psd_comp.nperseg = min(cfg.nperseg, n_samp)
            psd_comp.noverlap = min(cfg.noverlap if cfg.noverlap is not None else cfg.nperseg // 2, n_samp // 2)

            # Channel std to catch flat channels
            ch_std = np.std(data, axis=1)
            near_zero = np.sum(ch_std < 1e-8)

            freqs, psd = psd_comp._compute_psd_cpu(data)
            freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else np.nan

            print(f"\nTrial {trial_name} / {seg_name}: shape={data.shape}, nperseg_used={psd_comp.nperseg}, noverlap_used={psd_comp.noverlap}")
            print(f"Freq range: {freqs[0]:.4f} - {freqs[-1]:.4f} Hz, resolution ~{freq_res:.4f} Hz")
            print(f"Channel std: min={ch_std.min():.4e}, max={ch_std.max():.4e}, near_zero(<1e-8)={near_zero}/{n_ch}")

            for band_name, (lo, hi) in bands.items():
                mask = (freqs >= lo) & (freqs < hi)
                n_bins = int(np.sum(mask))
                if n_bins == 0:
                    print(f"  Band {band_name}: bins=0 (band {lo}-{hi} Hz not covered)")
                    continue
                band_power = np.trapz(psd[:, mask], dx=freq_res, axis=1)
                finite = band_power[np.isfinite(band_power)]
                any_nan = np.isnan(band_power).any()
                any_inf = np.isinf(band_power).any()
                if finite.size == 0:
                    stats = "all NaN/inf"
                else:
                    stats = f"min={finite.min():.4e}, max={finite.max():.4e}, median={np.median(finite):.4e}"
                print(f"  Band {band_name}: bins={n_bins}, {stats}, NaN={any_nan}, Inf={any_inf}")

            seg_count += 1
            if seg_count >= max_segments:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to an EEG HDF5 file")
    parser.add_argument("--max-segments", type=int, default=1, help="How many segments to inspect")
    args = parser.parse_args()

    analyze_file(args.h5, max_segments=args.max_segments)


if __name__ == "__main__":
    main()
