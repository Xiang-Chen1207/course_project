#!/usr/bin/env python3
"""
EEG 特征计算性能基准测试脚本

功能：
- 测试单个被试的一个 trial 的一个或多个 segment 中，每个特征的计算耗时
- 支持多个 segment 合并数据的特征计算性能测试
- 支持使用真实 EEG 数据或模拟数据
- 生成详细的性能报告
- 识别计算瓶颈

使用方法：
python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SEED/sub_2.h5  --num-segments 1 --benchmark-presets --preset-names basic --skip-default-suite
    python benchmark_features.py [--data-path PATH] [--no-gpu] [--iterations N] [--num-segments M]
  python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/Workload_MATB/sub_4.h5  --num-segments 1 --benchmark-presets --preset-names basic --skip-default-suite
  python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SleepEDF/sub_2.h5  --num-segments 1 --benchmark-presets --preset-names basic --skip-default-suite
示例：
示例：
    # 使用真实数据测试单个 segment
    python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SleepEDF/sub_2.h5

    # 使用真实数据测试 5 个 segment 合并
    python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SEED/sub_2.h5 --num-segments 1
    python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/Workload_MATB/sub_4.h5 --num-segments 1
    python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SleepEDF/sub_3.h5 --num-segments 1

    # 使用真实数据测试不同 segment 数量的性能对比
    python benchmark_features.py --data-path /path/to/subject.h5 --num-segments 1,2,5,10

    # 使用模拟数据测试（不指定 --data-path）
    python benchmark_features.py --segment-length 2.0 --num-segments 1,2,5,10
    only
    python benchmark_features.py --data-path /mnt/dataset2/hdf5_datasets/SleepEDF/sub_3.h5 --num-segments 1 --benchmark-microstate --microstate-template-per-trial 2000 --skip-default-suite
    python benchmark_features.py --data-path /path/to/sub_2.h5 --num-segments 1 --benchmark-presets --preset-names fast --skip-default-suite

"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import signal
import copy

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.psd_computer import PSDComputer
from eeg_feature_extraction.data_loader import EEGDataLoader
from eeg_feature_extraction.features.microstate import MicrostateAnalyzer, MicrostateFeatures
from selective_feature_extraction import (
    FEATURE_GROUPS,
    PRESETS,
    FeatureSelectionConfig,
    SelectiveFeatureExtractor,
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    feature_name: str
    category: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float


# ====================
# 工具：单特征超时保护
# ====================
def _run_with_timeout(fn, timeout_sec: int, *args, **kwargs):
    """在给定超时时间内运行函数，超时抛 TimeoutError。"""
    def _handler(signum, frame):
        raise TimeoutError("feature benchmark timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_synthetic_eeg(config: Config, num_segments: int = 1) -> np.ndarray:
    """
    生成模拟 EEG 数据（支持多个 segment 合并）

    Args:
        config: 配置对象
        num_segments: 要生成的 segment 数量（合并后的总长度 = segment_length * num_segments）

    Returns:
        模拟 EEG 数据, shape: (n_channels, n_timepoints * num_segments)
    """
    n_channels = config.n_channels
    n_timepoints_per_segment = int(config.segment_length * config.sampling_rate)
    total_timepoints = n_timepoints_per_segment * num_segments
    fs = config.sampling_rate

    # 生成包含多个频段成分的模拟信号
    t = np.arange(total_timepoints) / fs
    eeg_data = np.zeros((n_channels, total_timepoints))

    for ch in range(n_channels):
        # Delta (0.5-4 Hz)
        delta = 20 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2*np.pi))
        # Theta (4-8 Hz)
        theta = 15 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
        # Alpha (8-13 Hz)
        alpha = 25 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        # Beta (13-30 Hz)
        beta = 10 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
        # Gamma (30-100 Hz)
        gamma = 5 * np.sin(2 * np.pi * 40 * t + np.random.uniform(0, 2*np.pi))
        # 噪声
        noise = 5 * np.random.randn(total_timepoints)

        eeg_data[ch] = delta + theta + alpha + beta + gamma + noise

    return eeg_data


def build_microstate_template(data_path: str, segments_per_trial: Optional[int] = None,
                              verbose: bool = True) -> MicrostateAnalyzer:
    """从被试所有 trial 的若干 segment 构建微状态模板。

    Args:
        data_path: HDF5 文件路径
        segments_per_trial: 每个 trial 采样的 segment 数量；None/0 表示使用全部
        verbose: 是否打印进度
    Returns:
        训练好的 MicrostateAnalyzer
    """
    loader = EEGDataLoader(data_path)
    analyzer = MicrostateAnalyzer(n_states=4)

    if verbose:
        if segments_per_trial and segments_per_trial > 0:
            print(f"正在生成 microstate 模板（每个 trial 随机选 {segments_per_trial} 个 segments）...")
        else:
            print("正在生成 microstate 模板（使用所有 segments）...")

    all_peak_maps: List[np.ndarray] = []
    n_segments_used = 0
    n_segments_total = 0
    n_peak_maps = 0
    n_trials = 0

    trial_names = loader.get_trial_names()
    for trial_idx, trial_name in enumerate(trial_names):
        segment_names = loader.get_segment_names(trial_name)
        n_trials += 1
        n_segments_total += len(segment_names)

        if segments_per_trial and segments_per_trial > 0 and len(segment_names) > segments_per_trial:
            rng = np.random.default_rng(seed=42 + trial_idx)
            selected_indices = rng.choice(len(segment_names), size=segments_per_trial, replace=False)
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
                n_peak_maps += peak_maps.shape[0]
            n_segments_used += 1

    if n_segments_used == 0 or n_peak_maps == 0:
        raise ValueError("没有有效的 segment 峰值地形图用于微状态模板生成")

    combined_maps = np.vstack(all_peak_maps)
    analyzer.centroids = analyzer._polarity_invariant_kmeans(combined_maps)

    if verbose:
        print(f"  微状态模板完成：trials={n_trials}, segments_used={n_segments_used}/{n_segments_total}, peak_maps={n_peak_maps}")

    return analyzer


def load_real_eeg_data(data_path: str, num_segments: int,
                       trial_name: Optional[str] = None) -> Tuple[np.ndarray, float, int, Any]:
    """
    从 HDF5 文件加载真实 EEG 数据并合并多个 segment

    Args:
        data_path: HDF5 数据文件路径
        num_segments: 要合并的 segment 数量
        trial_name: 指定的 trial 名称，如果为 None 则使用第一个 trial

    Returns:
        (合并后的 EEG 数据, segment 长度, 实际合并的 segment 数量, subject_info)
        EEG 数据 shape: (n_channels, n_timepoints * num_segments)
    """
    loader = EEGDataLoader(data_path)
    subject_info = loader.get_subject_info()

    # 获取 trial 列表
    trial_names = loader.get_trial_names()
    if not trial_names:
        raise ValueError(f"数据文件中没有找到 trial: {data_path}")

    # 选择 trial
    if trial_name is None:
        trial_name = trial_names[0]
    elif trial_name not in trial_names:
        raise ValueError(f"未找到指定的 trial: {trial_name}, 可用的 trial: {trial_names}")

    # 获取 segment 列表
    segment_names = loader.get_segment_names(trial_name)
    if not segment_names:
        raise ValueError(f"Trial {trial_name} 中没有找到 segment")

    # 确定要加载的 segment 数量
    available_segments = len(segment_names)
    actual_num_segments = min(num_segments, available_segments)

    if actual_num_segments < num_segments:
        print(f"  警告: 请求 {num_segments} 个 segment，但只有 {available_segments} 个可用，"
              f"将使用 {actual_num_segments} 个")

    # 加载并合并 segment 数据
    segments_data = []
    segment_length = None

    for i in range(actual_num_segments):
        segment = loader.get_segment(trial_name, segment_names[i])
        segments_data.append(segment.eeg_data)
        if segment_length is None:
            segment_length = segment.time_length

    # 沿时间轴合并
    merged_data = np.concatenate(segments_data, axis=1)

    print(f"  加载数据: {trial_name}, 合并 {actual_num_segments} 个 segment")
    print(f"  数据形状: {merged_data.shape}")
    print(f"  采样率: {subject_info.sampling_rate} Hz")

    return merged_data, segment_length, actual_num_segments, subject_info


def benchmark_time_domain(eeg_data: np.ndarray, config: Config,
                          iterations: int = 10) -> List[BenchmarkResult]:
    """测试时域特征"""
    from eeg_feature_extraction.features.time_domain import TimeDomainFeatures

    results = []
    computer = TimeDomainFeatures(config)

    # 测试整体
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        features = computer.compute(eeg_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="[all_time_domain_features]",
        category="time_domain",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    # 测试各个子功能
    feature_funcs = {
        'mean_abs_amplitude': lambda: np.mean(np.abs(eeg_data)),
        'mean_channel_std': lambda: np.mean(np.std(eeg_data, axis=1)),
        'mean_peak_to_peak': lambda: np.mean(np.max(eeg_data, axis=1) - np.min(eeg_data, axis=1)),
        'mean_rms': lambda: np.mean(np.sqrt(np.mean(eeg_data ** 2, axis=1))),
        'mean_zero_crossing_rate': lambda: computer._compute_zero_crossing_rate_cpu(eeg_data),
        'hjorth_params (3)': lambda: computer._compute_hjorth_params_cpu(eeg_data),
    }

    for name, func in feature_funcs.items():
        times = []
        timed_out = False
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                _run_with_timeout(func, 30)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except TimeoutError:
                timed_out = True
                print(f"  [跳过] {name} 超过30s，已跳过后续迭代")
                break

        if times:
            results.append(BenchmarkResult(
                feature_name=name,
                category="time_domain",
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times)
            ))
        elif timed_out:
            results.append(BenchmarkResult(
                feature_name=name,
                category="time_domain",
                mean_time_ms=float('nan'),
                std_time_ms=float('nan'),
                min_time_ms=float('nan'),
                max_time_ms=float('nan')
            ))

    return results


def benchmark_frequency_domain(eeg_data: np.ndarray, psd_result,
                               config: Config, iterations: int = 10) -> List[BenchmarkResult]:
    """测试频域特征"""
    from eeg_feature_extraction.features.frequency_domain import FrequencyDomainFeatures

    results = []
    computer = FrequencyDomainFeatures(config)

    # 测试整体（不含PSD计算）
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        features = computer.compute(eeg_data, psd_result)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="[all_frequency_domain_features (no_psd)]",
        category="frequency_domain",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    # 测试各个子功能
    feature_funcs = {
        'band_power (abs/rel, 10)': lambda: None,  # 已在整体中包含
        'peak_frequency': lambda: computer._compute_peak_frequency(psd_result.freqs, psd_result.psd),
        'spectral_entropy': lambda: computer._compute_spectral_entropy(psd_result.psd),
        'spectral_centroid': lambda: computer._compute_spectral_centroid(psd_result.freqs, psd_result.psd),
        'individual_alpha_frequency': lambda: computer._compute_iaf(psd_result.freqs, psd_result.psd),
        'low_high_power_ratio': lambda: computer._compute_low_high_ratio(psd_result),
        'aperiodic_exponent': lambda: computer._compute_aperiodic_exponent(psd_result.freqs, psd_result.psd),
    }

    for name, func in feature_funcs.items():
        if func() is None:
            continue
        times = []
        timed_out = False
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                _run_with_timeout(func, 30)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except TimeoutError:
                timed_out = True
                print(f"  [跳过] {name} 超过30s，已跳过后续迭代")
                break

        if times:
            results.append(BenchmarkResult(
                feature_name=name,
                category="frequency_domain",
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times)
            ))
        elif timed_out:
            results.append(BenchmarkResult(
                feature_name=name,
                category="frequency_domain",
                mean_time_ms=float('nan'),
                std_time_ms=float('nan'),
                min_time_ms=float('nan'),
                max_time_ms=float('nan')
            ))

    return results


def benchmark_complexity(eeg_data: np.ndarray, config: Config,
                         iterations: int = 5) -> List[BenchmarkResult]:
    """测试复杂度特征（迭代次数较少因为耗时长）"""
    from eeg_feature_extraction.features.complexity import ComplexityFeatures

    results = []
    computer = ComplexityFeatures(config)

    # 使用较少的通道来减少测试时间（仅用于单特征测试）
    test_data_single = eeg_data[:3]  # 只测试3个通道

    # 测试各个子功能（单独测试以精确计时）
    feature_funcs = {
        'wavelet_energy_entropy': lambda: computer._compute_wavelet_entropy(eeg_data),
        'sample_entropy (3ch)': lambda: computer._compute_sample_entropy(test_data_single),
        'approx_entropy (3ch)': lambda: computer._compute_approx_entropy(test_data_single),
        'hurst_exponent': lambda: computer._compute_hurst_exponent(eeg_data),
    }

    for name, func in feature_funcs.items():
        times = []
        timed_out = False
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                _run_with_timeout(func, 30)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except TimeoutError:
                timed_out = True
                print(f"  [跳过] {name} 超过30s，已跳过后续迭代")
                break

        if times:
            results.append(BenchmarkResult(
                feature_name=name,
                category="complexity",
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times)
            ))
        elif timed_out:
            results.append(BenchmarkResult(
                feature_name=name,
                category="complexity",
                mean_time_ms=float('nan'),
                std_time_ms=float('nan'),
                min_time_ms=float('nan'),
                max_time_ms=float('nan')
            ))

    # 估算全部62通道的耗时
    sample_entropy_per_ch = results[1].mean_time_ms / 3  # sample_entropy per-channel
    approx_entropy_per_ch = results[2].mean_time_ms / 3  # approx_entropy per-channel

    results.append(BenchmarkResult(
        feature_name="sample_entropy (estimated_62ch)",
        category="complexity",
        mean_time_ms=sample_entropy_per_ch * 62,
        std_time_ms=0,
        min_time_ms=sample_entropy_per_ch * 62,
        max_time_ms=sample_entropy_per_ch * 62
    ))

    results.append(BenchmarkResult(
        feature_name="approx_entropy (estimated_62ch)",
        category="complexity",
        mean_time_ms=approx_entropy_per_ch * 62,
        std_time_ms=0,
        min_time_ms=approx_entropy_per_ch * 62,
        max_time_ms=approx_entropy_per_ch * 62
    ))

    # 测试整体
    print("  测试全部复杂度特征（可能需要较长时间）...")
    times = []
    for i in range(min(iterations, 3)):
        start = time.perf_counter()
        features = computer.compute(eeg_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"    迭代 {i+1}: {times[-1]:.1f} ms")

    results.insert(0, BenchmarkResult(
        feature_name="[all_complexity_features]",
        category="complexity",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    return results


def benchmark_connectivity(eeg_data: np.ndarray, psd_result,
                           config: Config, iterations: int = 5) -> List[BenchmarkResult]:
    """测试连接性特征"""
    from eeg_feature_extraction.features.connectivity import ConnectivityFeatures

    results = []
    computer = ConnectivityFeatures(config)

    # 测试整体
    print("  测试全部连接性特征（相干性计算可能较慢）...")
    times = []
    for i in range(min(iterations, 3)):
        start = time.perf_counter()
        features = computer.compute(eeg_data, psd_result)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"    迭代 {i+1}: {times[-1]:.1f} ms")

    results.append(BenchmarkResult(
        feature_name="[全部连接性特征]",
        category="connectivity",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    # 测试相关系数
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = computer._compute_mean_correlation(eeg_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="mean_interchannel_correlation",
        category="connectivity",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    # 测试相干性计算
    times = []
    for _ in range(min(iterations, 3)):
        start = time.perf_counter()
        _ = computer.psd_computer.compute_coherence(eeg_data, band=(8.0, 13.0))
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="alpha_coherence_matrix_computation",
        category="connectivity",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    return results


def benchmark_network(eeg_data: np.ndarray, config: Config,
                      iterations: int = 5) -> List[BenchmarkResult]:
    """测试网络特征"""
    from eeg_feature_extraction.features.network import NetworkFeatures

    results = []
    computer = NetworkFeatures(config)

    # 测试整体
    print("  测试全部网络特征（包含Floyd-Warshall算法）...")
    times = []
    for i in range(min(iterations, 3)):
        start = time.perf_counter()
        features = computer.compute(eeg_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"    迭代 {i+1}: {times[-1]:.1f} ms")

    results.append(BenchmarkResult(
        feature_name="[all_network_features]",
        category="network",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    # 先计算相干性矩阵
    coherence_matrix = computer.psd_computer.compute_coherence(eeg_data, band=(8.0, 13.0))
    adj_matrix = computer._threshold_matrix(coherence_matrix)

    # 测试各个子功能
    feature_funcs = {
        'network_clustering_coefficient': lambda: computer._compute_clustering_coefficient(adj_matrix),
        'network_characteristic_path_length': lambda: computer._compute_characteristic_path_length(adj_matrix),
        'network_global_efficiency': lambda: computer._compute_global_efficiency(adj_matrix),
    }

    for name, func in feature_funcs.items():
        times = []
        timed_out = False
        for _ in range(iterations):
            try:
                start = time.perf_counter()
                _run_with_timeout(func, 30)
                end = time.perf_counter()
                times.append((end - start) * 1000)
            except TimeoutError:
                timed_out = True
                print(f"  [跳过] {name} 超过30s，已跳过后续迭代")
                break

        if times:
            results.append(BenchmarkResult(
                feature_name=name,
                category="network",
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times)
            ))
        elif timed_out:
            results.append(BenchmarkResult(
                feature_name=name,
                category="network",
                mean_time_ms=float('nan'),
                std_time_ms=float('nan'),
                min_time_ms=float('nan'),
                max_time_ms=float('nan')
            ))

    return results


def benchmark_composite(eeg_data: np.ndarray, psd_result,
                        config: Config, iterations: int = 10) -> List[BenchmarkResult]:
    """测试综合特征（包含拆分后的认知负荷指标）"""
    from eeg_feature_extraction.features.composite import CompositeFeatures

    results = []
    computer = CompositeFeatures(config)

    # 测试整体
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        features = computer.compute(eeg_data, psd_result)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="[all_composite_features]",
        category="composite",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    return results


def benchmark_de_features(eeg_data: np.ndarray, psd_result,
                          config: Config, iterations: int = 5) -> List[BenchmarkResult]:
    """测试微分熵相关特征 (DE, DASM, RASM, DCAU, FAA)"""
    from eeg_feature_extraction.features.de_features import DEFeatures

    results = []
    computer = DEFeatures(config)

    # 测试整体
    print("  测试全部 DE 特征（包括 DASM, RASM, DCAU, FAA）...")
    times = []
    for i in range(min(iterations, 3)):
        start = time.perf_counter()
        features = computer.compute(eeg_data, psd_result)
        end = time.perf_counter()
        times.append((end - start) * 1000)
        print(f"    迭代 {i+1}: {times[-1]:.1f} ms")

    results.append(BenchmarkResult(
        feature_name="[all_de_features]",
        category="de_features",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    return results


def benchmark_plv(eeg_data: np.ndarray, config: Config,
                  iterations: int = 3) -> List[BenchmarkResult]:
    """测试 PLV（相位锁定值）特征"""
    from eeg_feature_extraction.features.connectivity import compute_plv_matrix

    results = []

    # 测试单频段 PLV 计算
    print("  测试 PLV 矩阵计算...")
    bands = {
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'gamma': (30.0, 80.0),
    }

    for band_name, band_range in bands.items():
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = compute_plv_matrix(eeg_data, config.sampling_rate, band_range)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results.append(BenchmarkResult(
            feature_name=f"plv_matrix_{band_name}",
            category="plv",
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times)
        ))
        print(f"    PLV {band_name}: {np.mean(times):.1f} ms")

    return results


def benchmark_microstate(eeg_data: np.ndarray, config: Config,
                         microstate_analyzer: MicrostateAnalyzer,
                         iterations: int = 3) -> List[BenchmarkResult]:
    """测试微状态特征（依赖预先构建的模板）"""

    results: List[BenchmarkResult] = []
    computer = MicrostateFeatures(config)

    times = []
    timed_out = False
    for i in range(iterations):
        try:
            start = time.perf_counter()
            _run_with_timeout(
                computer.compute,
                120,
                eeg_data,
                psd_result=None,
                microstate_analyzer=microstate_analyzer,
            )
            end = time.perf_counter()
            times.append((end - start) * 1000)
            print(f"    微状态迭代 {i+1}: {times[-1]:.1f} ms")
        except TimeoutError:
            timed_out = True
            print("  [跳过] microstate 超过120s，已跳过后续迭代")
            break

    if times:
        results.append(BenchmarkResult(
            feature_name="[all_microstate_features]",
            category="microstate",
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times)
        ))
    elif timed_out:
        results.append(BenchmarkResult(
            feature_name="[all_microstate_features]",
            category="microstate",
            mean_time_ms=float('nan'),
            std_time_ms=float('nan'),
            min_time_ms=float('nan'),
            max_time_ms=float('nan')
        ))

    return results


# ====================
# 预设组合基准测试
# ====================
def _selection_config_from_preset(preset_name: str) -> FeatureSelectionConfig:
    preset = PRESETS[preset_name]
    return FeatureSelectionConfig(
        selected_features=set(preset.get('include_features', [])),
        selected_groups=set(preset.get('groups', [])),
        excluded_features=set(preset.get('exclude_features', [])),
    )


def benchmark_presets(eeg_data: np.ndarray, config: Config,
                      iterations: int = 3, preset_names: Optional[List[str]] = None
                      ) -> List[BenchmarkResult]:
    """测试 selective_feature_extraction 里的预设组合耗时"""

    results: List[BenchmarkResult] = []
    names = preset_names or list(PRESETS.keys())

    for name in names:
        if name not in PRESETS:
            print(f"  [跳过] 未知预设: {name}")
            continue

        selection_config = _selection_config_from_preset(name)
        # 深拷贝 config，避免修改主配置
        preset_config = copy.deepcopy(config)
        extractor = SelectiveFeatureExtractor(preset_config, selection_config)

        selected = extractor.get_selected_features()
        print(f"  预设 {name}: 选中特征 {len(selected)} 个")

        times = []
        timed_out = False
        for i in range(iterations):
            try:
                start = time.perf_counter()
                _run_with_timeout(extractor.extract_features, 120, eeg_data)
                end = time.perf_counter()
                times.append((end - start) * 1000)
                print(f"    迭代 {i+1}: {times[-1]:.1f} ms")
            except TimeoutError:
                timed_out = True
                print(f"  [跳过] preset {name} 超过120s，已跳过后续迭代")
                break

        if times:
            results.append(BenchmarkResult(
                feature_name=f"[preset_{name}]",
                category="preset",
                mean_time_ms=np.mean(times),
                std_time_ms=np.std(times),
                min_time_ms=np.min(times),
                max_time_ms=np.max(times)
            ))
        elif timed_out:
            results.append(BenchmarkResult(
                feature_name=f"[preset_{name}]",
                category="preset",
                mean_time_ms=float('nan'),
                std_time_ms=float('nan'),
                min_time_ms=float('nan'),
                max_time_ms=float('nan')
            ))

    return results


def benchmark_fractal_dimensions(eeg_data: np.ndarray, config: Config,
                                  iterations: int = 5) -> List[BenchmarkResult]:
    """测试分形维数特征 (Higuchi, Katz, Petrosian)"""
    from eeg_feature_extraction.features.complexity import (
        _higuchi_fd_single, _katz_fd_single, _petrosian_fd_single
    )

    results = []

    # 使用单通道测试单个分形维数算法
    test_signal = eeg_data[0]

    fd_funcs = {
        'higuchi_fd (single_ch)': lambda: _higuchi_fd_single(test_signal, kmax=8),
        'katz_fd (single_ch)': lambda: _katz_fd_single(test_signal),
        'petrosian_fd (single_ch)': lambda: _petrosian_fd_single(test_signal),
    }

    for name, func in fd_funcs.items():
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        results.append(BenchmarkResult(
            feature_name=name,
            category="fractal_dimension",
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times)
        ))

    # 估算全部通道耗时
    n_channels = eeg_data.shape[0]
    for name in fd_funcs.keys():
        per_ch_time = results[-1].mean_time_ms if 'petrosian' in name else \
                      results[-2].mean_time_ms if 'katz' in name else \
                      results[-3].mean_time_ms
        results.append(BenchmarkResult(
            feature_name=name.replace('single_ch', f'estimated_{n_channels}ch'),
            category="fractal_dimension",
            mean_time_ms=per_ch_time * n_channels,
            std_time_ms=0,
            min_time_ms=per_ch_time * n_channels,
            max_time_ms=per_ch_time * n_channels
        ))

    return results


def benchmark_psd_computation(eeg_data: np.ndarray, config: Config,
                              iterations: int = 10) -> List[BenchmarkResult]:
    """测试PSD计算"""
    results = []

    psd_computer = PSDComputer(
        sampling_rate=config.sampling_rate,
        use_gpu=config.use_gpu,
        nperseg=config.nperseg,
        noverlap=config.noverlap,
        nfft=config.nfft
    )

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = psd_computer.compute_psd(eeg_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    results.append(BenchmarkResult(
        feature_name="[PSD计算 (Welch)]",
        category="preprocessing",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times)
    ))

    return results


def print_results(results: List[BenchmarkResult], segment_length: float,
                  num_segments: int = 1):
    """打印基准测试结果"""
    total_length = segment_length * num_segments
    total_timepoints = int(total_length * 200)

    print("\n" + "=" * 80)
    print("EEG 特征计算性能基准测试报告")
    print("=" * 80)
    print(f"\n测试配置:")
    print(f"  - 通道数: 62")
    print(f"  - 单个 Segment 长度: {segment_length} 秒")
    print(f"  - Segment 数量: {num_segments}")
    print(f"  - 合并后总长度: {total_length} 秒")
    print(f"  - 采样率: 200 Hz")
    print(f"  - 总时间点数: {total_timepoints}")
    print()

    # 按类别分组
    categories = ['preprocessing', 'time_domain', 'frequency_domain',
                  'complexity', 'connectivity', 'network', 'composite',
                  'de_features', 'fractal_dimension', 'plv', 'microstate', 'preset']
    category_names = {
        'preprocessing': 'PSD 预处理',
        'time_domain': '时域特征',
        'frequency_domain': '频域特征',
        'complexity': '复杂度特征',
        'connectivity': '连接性特征',
        'network': '网络特征',
        'composite': '综合特征',
        'de_features': '微分熵特征 (DE/DASM/RASM/DCAU/FAA)',
        'fractal_dimension': '分形维数 (Higuchi/Katz/Petrosian)',
        'plv': '相位锁定值 (PLV)',
        'microstate': '微状态特征',
        'preset': '预设组合 (selective_feature_extraction)'
    }

    total_time = 0.0
    category_times = {}

    for category in categories:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue

        print("-" * 80)
        print(f"\n{category_names.get(category, category).upper()}")
        print("-" * 80)
        print(f"{'特征名称':<35} {'平均耗时':>12} {'标准差':>10} {'最小':>10} {'最大':>10}")
        print("-" * 80)

        cat_total = 0.0
        for r in cat_results:
            print(f"{r.feature_name:<35} {r.mean_time_ms:>10.2f}ms {r.std_time_ms:>8.2f}ms "
                  f"{r.min_time_ms:>8.2f}ms {r.max_time_ms:>8.2f}ms")
            if r.feature_name.startswith('['):
                cat_total = r.mean_time_ms

        category_times[category] = cat_total
        total_time += cat_total

    # 汇总
    print("\n" + "=" * 80)
    print("性能汇总")
    print("=" * 80)
    print(f"\n{'类别':<25} {'耗时 (ms)':>15} {'占比':>10}")
    print("-" * 50)

    for category in categories:
        if category in category_times and category_times[category] > 0:
            pct = category_times[category] / total_time * 100
            print(f"{category_names.get(category, category):<25} "
                  f"{category_times[category]:>13.2f} {pct:>9.1f}%")

    print("-" * 50)
    print(f"{'总计':<25} {total_time:>13.2f} {'100.0':>9}%")

    # 性能瓶颈识别
    print("\n" + "=" * 80)
    print("性能瓶颈分析 (Top 10)")
    print("=" * 80)

    # 排除汇总项
    individual_results = [r for r in results if not r.feature_name.startswith('[')]
    sorted_results = sorted(individual_results, key=lambda x: x.mean_time_ms, reverse=True)

    print(f"\n{'排名':<5} {'特征名称':<35} {'类别':<15} {'耗时 (ms)':>12}")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"{i:<5} {r.feature_name:<35} {r.category:<15} {r.mean_time_ms:>10.2f}")

    # 优化建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)

    if category_times.get('complexity', 0) > 1000:
        print("\n[!] 复杂度特征耗时过长:")
        print("    - sample_entropy 和 approx_entropy 采用 O(N^2) 算法，建议使用优化库如 antropy")
        print("    - 可考虑降采样或使用近似算法")

    if category_times.get('connectivity', 0) > 1000:
        print("\n[!] 连接性特征耗时过长:")
        print("    - 相干性计算需要62*61/2=1891对，建议使用GPU加速")
        print("    - 可考虑只计算部分通道对")

    if category_times.get('network', 0) > 500:
        print("\n[!] 网络特征耗时过长:")
        print("    - Floyd-Warshall算法为O(N^3)复杂度")
        print("    - 建议使用scipy.sparse.csgraph.shortest_path")
        print("    - 特征路径长度和全局效率重复计算，可合并")


def run_benchmark_for_segments(config: Config, num_segments: int,
                                iterations: int,
                                data_path: Optional[str] = None,
                                benchmark_presets_flag: bool = False,
                                preset_names: Optional[List[str]] = None,
                                benchmark_microstate_flag: bool = False,
                                microstate_segments_per_trial: Optional[int] = None,
                                skip_default_suite: bool = False
                                ) -> Tuple[List[BenchmarkResult], int]:
    """
    对指定数量的 segments 运行基准测试

    Args:
        config: 配置对象
        num_segments: segment 数量
        iterations: 迭代次数
        data_path: HDF5 数据文件路径，如果为 None 则使用模拟数据
        skip_default_suite: 仅运行微状态/预设，跳过默认基准套件

    Returns:
        (基准测试结果列表, 实际使用的 segment 数量)
    """
    print(f"\n{'=' * 80}")
    print(f"测试 {num_segments} 个 Segment 合并数据")
    print(f"{'=' * 80}")

    # 加载数据
    if data_path:
        # 使用真实数据
        print("加载真实 EEG 数据...")
        eeg_data, segment_length, actual_num_segments, subject_info = load_real_eeg_data(
            data_path, num_segments
        )
        # 更新配置：通道、采样率、segment_length、n_timepoints
        config.update_from_electrode_names(subject_info.channel_names)
        config.sampling_rate = subject_info.sampling_rate
        config.segment_length = segment_length
        config.n_timepoints = int(config.segment_length * config.sampling_rate)
    else:
        # 使用模拟数据
        actual_num_segments = num_segments
        print("生成模拟 EEG 数据...")
        eeg_data = generate_synthetic_eeg(config, num_segments)
        print(f"  数据形状: {eeg_data.shape}")

    total_length = config.segment_length * actual_num_segments
    total_timepoints = eeg_data.shape[1]

    print(f"  - 单个 Segment 长度: {config.segment_length}s")
    print(f"  - Segment 数量: {actual_num_segments}")
    print(f"  - 合并后总长度: {total_length:.2f}s ({total_timepoints} 时间点)")
    print()

    if skip_default_suite and not (benchmark_microstate_flag or benchmark_presets_flag):
        print("  [提示] 已跳过默认基准套件，且未选择微状态/预设，当前配置不会运行任何基准测试")

    psd_result = None
    if not skip_default_suite:
        print("\n预计算 PSD...")
        psd_computer = PSDComputer(
            sampling_rate=config.sampling_rate,
            use_gpu=config.use_gpu,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            nfft=config.nfft
        )
        psd_result = psd_computer.compute_psd(eeg_data)
        print("  PSD 计算完成")

    # 运行基准测试
    all_results = []
    microstate_analyzer = None

    if not skip_default_suite:
        print("\n" + "-" * 40)
        print("测试 PSD 计算...")
        all_results.extend(benchmark_psd_computation(eeg_data, config, iterations))

        print("\n" + "-" * 40)
        print("测试时域特征...")
        all_results.extend(benchmark_time_domain(eeg_data, config, iterations))

        print("\n" + "-" * 40)
        print("测试频域特征...")
        all_results.extend(benchmark_frequency_domain(eeg_data, psd_result, config, iterations))

        print("\n" + "-" * 40)
        print("测试复杂度特征...")
        all_results.extend(benchmark_complexity(eeg_data, config, iterations))

        print("\n" + "-" * 40)
        print("测试连接性特征...")
        all_results.extend(benchmark_connectivity(eeg_data, psd_result, config, iterations))

        print("\n" + "-" * 40)
        print("测试网络特征...")
        all_results.extend(benchmark_network(eeg_data, config, iterations))

        print("\n" + "-" * 40)
        print("测试综合特征...")
        all_results.extend(benchmark_composite(eeg_data, psd_result, config, iterations))

        print("\n" + "-" * 40)
        print("测试 DE 特征（微分熵、DASM、RASM、DCAU、FAA）...")
        all_results.extend(benchmark_de_features(eeg_data, psd_result, config, iterations))

        print("\n" + "-" * 40)
        print("测试分形维数特征...")
        all_results.extend(benchmark_fractal_dimensions(eeg_data, config, iterations))

        print("\n" + "-" * 40)
        print("测试 PLV 特征...")
        all_results.extend(benchmark_plv(eeg_data, config, iterations))

    if benchmark_microstate_flag and data_path:
        print("\n" + "-" * 40)
        print("构建微状态模板并测试微状态特征...")
        try:
            microstate_analyzer = build_microstate_template(
                data_path,
                segments_per_trial=microstate_segments_per_trial,
                verbose=True,
            )
            all_results.extend(benchmark_microstate(eeg_data, config, microstate_analyzer, iterations))
        except Exception as e:
            print(f"  [跳过] 微状态模板构建失败: {e}")

    if benchmark_presets_flag:
        print("\n" + "-" * 40)
        print("测试预设组合特征 (selective_feature_extraction)...")
        all_results.extend(benchmark_presets(eeg_data, config, iterations, preset_names))

    # 直接打印本轮所有特征的耗时，顺序与计算顺序一致，便于筛掉慢特征
    print("\n" + "=" * 80)
    print("按计算顺序列出所有特征耗时 (均值ms)，含超时标记")
    print("=" * 80)
    for idx, r in enumerate(all_results, 1):
        status = "TIMEOUT" if np.isnan(r.mean_time_ms) else "OK"
        print(
            f"{idx:03d}. {r.feature_name:<45} | 类别: {r.category:<15} | "
            f"平均耗时: {r.mean_time_ms:8.2f} ms | 状态: {status}"
        )

    # 按 FEATURE_GROUPS 枚举每个特征（即便未单独计时也标记为 NOT_MEASURED）
    print("\n" + "=" * 80)
    print("按 FEATURE_GROUPS 列出每个特征耗时 (均值ms)，含缺失/超时标记")
    print("=" * 80)
    result_map = {r.feature_name: r for r in all_results}
    for group_name, feature_list in FEATURE_GROUPS.items():
        print(f"[组] {group_name}")
        for feat in feature_list:
            r = result_map.get(feat)
            if r is None:
                status = "NOT_MEASURED"
                mean_str = "   n/a"
            else:
                status = "TIMEOUT" if np.isnan(r.mean_time_ms) else "OK"
                mean_str = f"{r.mean_time_ms:8.2f}"
            print(f"  - {feat:<40} | 平均耗时: {mean_str} ms | 状态: {status}")

    return all_results, actual_num_segments


def print_comparison_results(all_segment_results: Dict[int, List[BenchmarkResult]],
                              segment_length: float):
    """
    打印多个 segment 数量的对比结果

    Args:
        all_segment_results: 各 segment 数量的测试结果 {num_segments: results}
        segment_length: 单个 segment 长度
    """
    print("\n" + "=" * 100)
    print("多 Segment 数量性能对比")
    print("=" * 100)

    segment_counts = sorted(all_segment_results.keys())

    # 按类别汇总
    categories = ['preprocessing', 'time_domain', 'frequency_domain',
                  'complexity', 'connectivity', 'network', 'composite', 'microstate', 'preset']
    category_names = {
        'preprocessing': 'PSD 预处理',
        'time_domain': '时域特征',
        'frequency_domain': '频域特征',
        'complexity': '复杂度特征',
        'connectivity': '连接性特征',
        'network': '网络特征',
        'composite': '综合特征',
        'microstate': '微状态特征',
        'preset': '预设组合'
    }

    # 打印表头
    print(f"\n{'特征类别':<20}", end="")
    for n in segment_counts:
        total_len = segment_length * n
        col_header = f"{n}seg ({total_len:.1f}s)"
        print(f" | {col_header:>18}", end="")
    print()
    print("-" * (22 + 21 * len(segment_counts)))

    # 各类别时间对比
    category_totals = {n: {} for n in segment_counts}

    for category in categories:
        row = f"{category_names.get(category, category):<20}"
        for n in segment_counts:
            results = all_segment_results[n]
            cat_results = [r for r in results if r.category == category and r.feature_name.startswith('[')]
            if cat_results:
                time_ms = cat_results[0].mean_time_ms
                category_totals[n][category] = time_ms
                row += f" | {time_ms:>15.2f}ms"
            else:
                row += f" | {'N/A':>15}"
        print(row)

    # 总计
    print("-" * (22 + 21 * len(segment_counts)))
    row = f"{'总计':<20}"
    for n in segment_counts:
        total = sum(category_totals[n].values())
        row += f" | {total:>15.2f}ms"
    print(row)

    # 计算时间增长率
    if len(segment_counts) > 1:
        print("\n" + "=" * 100)
        print("时间增长分析")
        print("=" * 100)

        base_n = segment_counts[0]
        print(f"\n以 {base_n} segment 为基准的耗时倍率:")
        print(f"\n{'特征类别':<20}", end="")
        for n in segment_counts:
            print(f" | {n}seg:>12", end="")
        print()
        print("-" * (22 + 15 * len(segment_counts)))

        for category in categories:
            if category not in category_totals[base_n]:
                continue
            base_time = category_totals[base_n][category]
            if base_time == 0:
                continue

            row = f"{category_names.get(category, category):<20}"
            for n in segment_counts:
                if category in category_totals[n]:
                    ratio = category_totals[n][category] / base_time
                    row += f" | {ratio:>11.2f}x"
                else:
                    row += f" | {'N/A':>11}"
            print(row)

        # 总计增长率
        print("-" * (22 + 15 * len(segment_counts)))
        base_total = sum(category_totals[base_n].values())
        row = f"{'总计':<20}"
        for n in segment_counts:
            total = sum(category_totals[n].values())
            ratio = total / base_total if base_total > 0 else 0
            row += f" | {ratio:>11.2f}x"
        print(row)

        # 数据量增长对比
        print(f"\n数据量增长对比:")
        row = f"{'数据量倍率':<20}"
        for n in segment_counts:
            ratio = n / base_n
            row += f" | {ratio:>11.2f}x"
        print(row)


def main():
    parser = argparse.ArgumentParser(description='EEG 特征计算性能基准测试')
    parser.add_argument('--data-path', type=str, default=None,
                        help='HDF5 数据文件路径，如果不指定则使用模拟数据')
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU 加速')
    parser.add_argument('--iterations', type=int, default=5, help='测试迭代次数')
    parser.add_argument('--segment-length', type=float, default=2.0,
                        help='单个 Segment 长度（秒），仅在使用模拟数据时有效')
    parser.add_argument('--num-segments', type=str, default='1',
                        help='Segment 数量，可以是单个值(如 5)或逗号分隔的多个值(如 1,2,5,10)进行对比测试')
    parser.add_argument('--benchmark-presets', action='store_true',
                        help='额外测试 selective_feature_extraction.py 中预设组合的耗时')
    parser.add_argument('--preset-names', type=str, default=None,
                        help='指定要测试的预设名称，逗号分隔；不指定则测试全部预设')
    parser.add_argument('--benchmark-microstate', action='store_true',
                        help='测试微状态特征，并先构建微状态模板')
    parser.add_argument('--microstate-template-per-trial', type=int, default=None,
                        help='微状态模板构建时每个 trial 使用的 segment 数，未指定或 0 表示用全部')
    parser.add_argument('--skip-default-suite', action='store_true',
                        help='跳过默认基准套件，只运行微状态/预设部分（若指定）')
    args = parser.parse_args()

    # 解析 segment 数量
    try:
        segment_counts = [int(x.strip()) for x in args.num_segments.split(',')]
    except ValueError:
        print(f"错误: --num-segments 参数格式不正确: {args.num_segments}")
        print("请使用单个整数(如 5)或逗号分隔的整数列表(如 1,2,5,10)")
        sys.exit(1)

    # 验证 segment 数量
    for n in segment_counts:
        if n < 1:
            print(f"错误: segment 数量必须大于 0，收到: {n}")
            sys.exit(1)

    # 解析预设名称
    preset_names = None
    if args.preset_names:
        preset_names = [x.strip() for x in args.preset_names.split(',') if x.strip()]

    # 配置
    config = Config()
    config.use_gpu = not args.no_gpu
    config.segment_length = args.segment_length
    config.n_timepoints = int(args.segment_length * config.sampling_rate)

    print("=" * 80)
    print("EEG 特征计算性能基准测试")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 数据源: {'真实数据 (' + args.data_path + ')' if args.data_path else '模拟数据'}")
    print(f"  - GPU: {'启用' if config.use_gpu else '禁用'}")
    print(f"  - 迭代次数: {args.iterations}")
    if not args.data_path:
        print(f"  - 单个 Segment 长度: {args.segment_length}s ({config.n_timepoints} 时间点)")
    print(f"  - 测试 Segment 数量: {segment_counts}")
    if args.benchmark_microstate:
        print(f"  - 微状态模板: 每个 trial 取 {args.microstate_template_per_trial or '全部'} 个 segment")
    if args.skip_default_suite:
        print("  - 跳过默认基准套件，仅运行微状态/预设")
    print()

    # 存储所有测试结果
    all_segment_results: Dict[int, List[BenchmarkResult]] = {}
    actual_segment_counts: Dict[int, int] = {}  # 记录实际使用的 segment 数量

    # 对每个 segment 数量运行测试
    for num_segments in segment_counts:
        results, actual_num = run_benchmark_for_segments(
            config,
            num_segments,
            args.iterations,
            args.data_path,
            args.benchmark_presets,
            preset_names,
            args.benchmark_microstate,
            args.microstate_template_per_trial,
            args.skip_default_suite
        )
        actual_segment_counts[num_segments] = actual_num
        all_segment_results[actual_num] = results

        # 打印单个配置的详细结果
        print_results(results, config.segment_length, actual_num)

    # 如果测试了多个 segment 数量，打印对比结果
    actual_counts = list(set(actual_segment_counts.values()))
    if len(actual_counts) > 1:
        print_comparison_results(all_segment_results, config.segment_length)


if __name__ == '__main__':
    main()
