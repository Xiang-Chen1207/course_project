#!/usr/bin/env python3
"""
动态合并 Segment 的特征提取脚本

功能：
- 将多个连续的 segment 合并成一个更长的信号
- 支持按 trial 内合并或跨 trial 合并
- 如果被试的 segment 数量不足以组成完整的合并单元，则丢弃该部分（不补充）
- 支持多CPU并行处理

使用方法：
  python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/Workload_MATB -o /mnt/dataset4/cx/code/EEG_LLM_text/Workload_basic --merge-count 1 --preset basic --microstate-segs 20
  python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/SEED -o /mnt/dataset4/cx/code/EEG_LLM_text/SEED_basic --merge-count 1 --preset basic --microstate-segs 20
    # 每15个segment合并成1个（默认按trial内合并）
    python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/Workload_MATB -o /mnt/dataset4/cx/code/EEG_LLM_text/Workload_new_full --merge-count 1
    python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/SEED -o /mnt/dataset4/cx/code/EEG_LLM_text/SEED_2s_full --merge-count 1 --preset full
    python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/Workload_MATB -o /mnt/dataset4/cx/code/EEG_LLM_text/Workload_output_30s --merge-count 15 --preset fast
    python merged_segment_extraction.py -i /mnt/dataset2/hdf5_datasets/SleepEDF -o /mnt/dataset4/cx/code/EEG_LLM_text/SleepEDF_basic --merge-count 1 --preset sleep --microstate-segs 500
    # 每3个segment合并，跨trial合并
    python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 3 --cross-trial

    # 使用预设特征配置
    python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2 --preset standard

    # 指定并行进程数
    python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2 --n-jobs 8
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import warnings
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import multiprocessing as mp

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.config import Config, FrequencyBands, ChannelGroups
from eeg_feature_extraction.psd_computer import PSDComputer, PSDResult
from eeg_feature_extraction.data_loader import EEGDataLoader, SegmentData
from eeg_feature_extraction.features.base import BaseFeature, FeatureRegistry

# 导入所有特征模块
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

# 复用 selective_feature_extraction 中的特征配置
from selective_feature_extraction import (
    FEATURE_GROUPS,
    FEATURE_TO_GROUP,
    PRESETS,
    FeatureSelectionConfig,
    apply_preset,
)

# 补充 microstate 组（原 FEATURE_GROUPS 不含 microstate）
if 'microstate' not in FEATURE_GROUPS:
    FEATURE_GROUPS['microstate'] = MicrostateFeatures.feature_names.copy()
    for feat in MicrostateFeatures.feature_names:
        FEATURE_TO_GROUP[feat] = 'microstate'


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MergedSegmentData:
    """合并后的 segment 数据结构"""
    eeg_data: np.ndarray  # shape: (n_channels, n_timepoints * merge_count)
    trial_ids: List[int]  # 包含的 trial IDs
    segment_ids: List[int]  # 包含的 segment IDs
    session_id: int
    labels: List[int]  # 包含的标签
    primary_label: int  # 主标签（第一个segment的标签或众数）
    start_time: float
    end_time: float
    total_time_length: float
    merge_count: int  # 实际合并的segment数量
    source_segments: List[str]  # 源segment名称列表


# =============================================================================
# 并行处理辅助函数
# =============================================================================

def _get_optimal_n_jobs() -> int:
    """获取最优的并行进程数"""
    cpu_count = mp.cpu_count()
    return max(1, int(cpu_count * 0.75))


def _process_merged_segment_task(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    处理单个合并后的 segment 的任务函数（用于并行处理）

    Args:
        args: (merged_segment_dict, config_dict, selection_config_dict, output_path)

    Returns:
        (output_path, features_dict, error_text)
    """
    # 支持附带 microstate centroids 的新格式
    if len(args) == 5:
        merged_segment_dict, config_dict, selection_config_dict, output_path, microstate_centroids = args
    else:
        merged_segment_dict, config_dict, selection_config_dict, output_path = args
        microstate_centroids = None

    try:
        # 重建 Config 对象
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 还原嵌套的 dataclass 配置
        if isinstance(config.freq_bands, dict):
            try:
                config.freq_bands = FrequencyBands(**config.freq_bands)
            except Exception:
                config.freq_bands = FrequencyBands()

        if isinstance(config.channel_groups, dict):
            try:
                config.channel_groups = ChannelGroups(**config.channel_groups)
            except Exception:
                config.channel_groups = ChannelGroups()

        # 重建 FeatureSelectionConfig
        selection_config = FeatureSelectionConfig(
            selected_features=set(selection_config_dict.get('selected_features', [])),
            selected_groups=set(selection_config_dict.get('selected_groups', [])),
            excluded_features=set(selection_config_dict.get('excluded_features', []))
        )

        # 重建 MergedSegmentData
        merged_segment = MergedSegmentData(
            eeg_data=merged_segment_dict['eeg_data'],
            trial_ids=merged_segment_dict['trial_ids'],
            segment_ids=merged_segment_dict['segment_ids'],
            session_id=merged_segment_dict['session_id'],
            labels=merged_segment_dict['labels'],
            primary_label=merged_segment_dict['primary_label'],
            start_time=merged_segment_dict['start_time'],
            end_time=merged_segment_dict['end_time'],
            total_time_length=merged_segment_dict['total_time_length'],
            merge_count=merged_segment_dict['merge_count'],
            source_segments=merged_segment_dict['source_segments'],
        )

        # 创建临时的特征提取器（子进程禁用GPU以保证稳定性）
        config.use_gpu = False
        extractor = MergedSegmentExtractor(
            config=config,
            selection_config=selection_config,
            merge_count=1  # 已经是合并后的数据
        )

        microstate_analyzer = None
        if microstate_centroids is not None:
            microstate_analyzer = MicrostateAnalyzer(n_states=4)
            microstate_analyzer.centroids = microstate_centroids

        # 提取特征
        features = extractor.extract_features(merged_segment.eeg_data, microstate_analyzer=microstate_analyzer)

        # 添加元信息
        features['trial_ids'] = str(merged_segment.trial_ids)
        features['segment_ids'] = str(merged_segment.segment_ids)
        features['session_id'] = merged_segment.session_id
        features['primary_label'] = merged_segment.primary_label
        features['labels'] = str(merged_segment.labels)
        features['start_time'] = merged_segment.start_time
        features['end_time'] = merged_segment.end_time
        features['total_time_length'] = merged_segment.total_time_length
        features['merge_count'] = merged_segment.merge_count
        features['source_segments'] = str(merged_segment.source_segments)

        # 转换为 DataFrame 并保存
        df = pd.DataFrame([features])
        meta_cols = ['trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
                     'start_time', 'end_time', 'total_time_length', 'merge_count', 'source_segments']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols]
        df.to_csv(output_path, index=False, encoding='utf-8')

        return (output_path, features, None)
    except Exception as e:
        return (output_path, None, traceback.format_exc())


# =============================================================================
# 合并 Segment 特征提取器
# =============================================================================

class MergedSegmentExtractor:
    """合并 Segment 的特征提取器，支持多CPU并行处理"""

    FEATURE_TIMEOUT_SEC = 30  # 单个特征组计算超时时间

    def __init__(self, config: Optional[Config] = None,
                 selection_config: Optional[FeatureSelectionConfig] = None,
                 merge_count: int = 2,
                 cross_trial: bool = False,
                 n_jobs: Optional[int] = None,
                 microstate_segments_per_trial: Optional[int] = None):
        """
        初始化

        Args:
            config: EEG 配置
            selection_config: 特征选择配置
            merge_count: 合并的 segment 数量
            cross_trial: 是否跨 trial 合并
            n_jobs: 并行进程数，None 表示自动选择
            microstate_segments_per_trial: 每个 trial 用于生成微状态模板的 segment 数量
                - None 或 0: 使用所有 segments（默认行为）
                - >0: 每个 trial 随机选择指定数量的 segments
        """
        self.config = config or Config()
        self.selection_config = selection_config or FeatureSelectionConfig()
        self.merge_count = merge_count
        self.cross_trial = cross_trial
        self.n_jobs = n_jobs if n_jobs is not None else _get_optimal_n_jobs()
        self.microstate_segments_per_trial = microstate_segments_per_trial

        # PSD 计算器
        self.psd_computer = PSDComputer(
            sampling_rate=self.config.sampling_rate,
            use_gpu=self.config.use_gpu,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft
        )

        # 初始化需要的特征计算器
        self.feature_computers: Dict[str, BaseFeature] = {}
        self._initialize_computers()

    # ====================
    # 工具：特征计算超时保护（跨平台实现）
    # ====================
    def _run_with_timeout(self, fn, timeout_sec: int, *args, **kwargs):
        """在给定超时时间内运行函数，超时则抛出 TimeoutError。

        使用 ThreadPoolExecutor 实现跨平台超时机制，
        适用于 Windows/Linux/macOS。
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout_sec)
            except FuturesTimeoutError:
                raise TimeoutError("feature computation timed out")

    def _initialize_computers(self):
        """初始化需要的特征计算器"""
        required_groups = self.selection_config.get_required_groups()
        all_feature_classes = FeatureRegistry.get_all_feature_classes()

        for group_name in required_groups:
            if group_name in all_feature_classes:
                self.feature_computers[group_name] = all_feature_classes[group_name](self.config)

    def extract_features(self, eeg_data: np.ndarray, microstate_analyzer: Optional[MicrostateAnalyzer] = None) -> Dict[str, float]:
        """
        提取选定的特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)

        Returns:
            特征字典
        """
        final_features = self.selection_config.get_final_features()

        if not final_features:
            return {}

        # 计算 PSD（微状态特征不依赖 PSD，但其余特征需要）
        psd_result = self.psd_computer.compute_psd(
            eeg_data,
            bands=self.config.freq_bands.get_all_bands()
        )

        # 提取特征
        all_features = {}

        # 改为“单特征粒度”的超时控制：每个特征单独执行 compute，并各自套 30s 保护
        for group_name, computer in self.feature_computers.items():
            group_feature_names = [f for f in computer.get_feature_names() if f in final_features]
            if not group_feature_names:
                continue

            for feat_name in group_feature_names:
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

                    value = self._run_with_timeout(_task, self.FEATURE_TIMEOUT_SEC)
                    if value is not None:
                        all_features[feat_name] = value
                except Exception as e:
                    warnings.warn(f"计算特征 '{feat_name}' 时出错或超时: {e}")

        return all_features

    def get_selected_features(self) -> List[str]:
        """获取选定的特征名称列表"""
        return sorted(list(self.selection_config.get_final_features()))

    def _merge_segments_within_trial(self, loader: EEGDataLoader) -> List[MergedSegmentData]:
        """
        按 trial 内合并 segments

        Args:
            loader: 数据加载器

        Returns:
            合并后的 segment 列表
        """
        merged_segments = []

        for trial_name in loader.get_trial_names():
            segment_names = loader.get_segment_names(trial_name)

            # 计算可以组成多少个完整的合并单元
            n_complete_groups = len(segment_names) // self.merge_count

            if n_complete_groups == 0:
                # 该 trial 的 segment 数量不足，丢弃
                continue

            for group_idx in range(n_complete_groups):
                start_idx = group_idx * self.merge_count
                end_idx = start_idx + self.merge_count

                # 收集要合并的 segments
                segments_to_merge = []
                for seg_name in segment_names[start_idx:end_idx]:
                    seg = loader.get_segment(trial_name, seg_name)
                    segments_to_merge.append((trial_name, seg_name, seg))

                # 合并
                merged = self._merge_segment_list(segments_to_merge)
                merged_segments.append(merged)

        return merged_segments

    def _merge_segments_cross_trial(self, loader: EEGDataLoader) -> List[MergedSegmentData]:
        """
        跨 trial 合并 segments（按顺序）

        Args:
            loader: 数据加载器

        Returns:
            合并后的 segment 列表
        """
        # 收集所有 segments
        all_segments = []
        for trial_name, segment_name, segment in loader.iter_segments():
            all_segments.append((trial_name, segment_name, segment))

        merged_segments = []

        # 计算可以组成多少个完整的合并单元
        n_complete_groups = len(all_segments) // self.merge_count

        if n_complete_groups == 0:
            return merged_segments

        for group_idx in range(n_complete_groups):
            start_idx = group_idx * self.merge_count
            end_idx = start_idx + self.merge_count

            segments_to_merge = all_segments[start_idx:end_idx]
            merged = self._merge_segment_list(segments_to_merge)
            merged_segments.append(merged)

        return merged_segments

    def _compute_microstate_template(self, loader: EEGDataLoader, verbose: bool = True) -> MicrostateAnalyzer:
        """从该被试的 segments 生成微状态模板（使用 GFP 峰值地形图节省内存）。

        支持两种模式：
        1. 使用所有 segments（microstate_segments_per_trial=None 或 0）
        2. 每个 trial 随机选择指定数量的 segments（microstate_segments_per_trial > 0）
        """
        segments_per_trial = self.microstate_segments_per_trial

        if verbose:
            if segments_per_trial and segments_per_trial > 0:
                print(f"正在生成 microstate 模板（每个 trial 随机选择 {segments_per_trial} 个 segments）...")
            else:
                print("正在生成 microstate 模板（使用所有 segments）...")

        analyzer = MicrostateAnalyzer(n_states=4)
        all_peak_maps: List[np.ndarray] = []
        n_segments_used = 0
        n_segments_total = 0
        n_peak_maps = 0
        n_trials = 0

        # 按 trial 组织 segments
        trial_names = loader.get_trial_names()

        for trial_name in trial_names:
            segment_names = loader.get_segment_names(trial_name)
            n_trials += 1
            n_segments_total += len(segment_names)

            # 决定使用哪些 segments
            if segments_per_trial and segments_per_trial > 0 and len(segment_names) > segments_per_trial:
                # 随机选择指定数量的 segments
                rng = np.random.default_rng(seed=42 + n_trials)  # 可重复的随机选择
                selected_indices = rng.choice(len(segment_names), size=segments_per_trial, replace=False)
                selected_segment_names = [segment_names[i] for i in sorted(selected_indices)]
            else:
                # 使用所有 segments
                selected_segment_names = segment_names

            for seg_name in selected_segment_names:
                segment = loader.get_segment(trial_name, seg_name)
                data = segment.eeg_data
                gfp = analyzer.compute_gfp(data)
                peak_indices = analyzer.find_gfp_peaks(gfp)
                peak_maps = data[:, peak_indices].T  # (n_peaks, n_channels)
                if peak_maps.size > 0:
                    all_peak_maps.append(peak_maps)
                    n_peak_maps += peak_maps.shape[0]
                n_segments_used += 1

        if n_segments_used == 0 or n_peak_maps == 0:
            raise ValueError("没有有效的 segment 峰值地形图用于微状态模板生成")

        combined_maps = np.vstack(all_peak_maps)
        analyzer.centroids = analyzer._polarity_invariant_kmeans(combined_maps)

        if verbose:
            print(f"  微状态模板生成完成：trials={n_trials}, "
                  f"segments_used={n_segments_used}/{n_segments_total}, peak_maps={n_peak_maps}")

        return analyzer

    def _merge_segment_list(self, segments: List[Tuple[str, str, SegmentData]]) -> MergedSegmentData:
        """
        合并一组 segments

        Args:
            segments: [(trial_name, segment_name, SegmentData), ...]

        Returns:
            MergedSegmentData
        """
        # 提取所有 EEG 数据并沿时间轴拼接
        eeg_arrays = [seg.eeg_data for _, _, seg in segments]
        merged_eeg = np.concatenate(eeg_arrays, axis=1)

        # 收集元信息
        trial_ids = [seg.trial_id for _, _, seg in segments]
        segment_ids = [seg.segment_id for _, _, seg in segments]
        labels = [seg.label for _, _, seg in segments]
        source_segments = [f"{trial_name}/{seg_name}" for trial_name, seg_name, _ in segments]

        # 使用第一个 segment 的标签作为主标签
        primary_label = labels[0]

        # 计算时间信息
        first_seg = segments[0][2]
        last_seg = segments[-1][2]
        start_time = first_seg.start_time
        end_time = last_seg.end_time
        total_time_length = sum(seg.time_length for _, _, seg in segments)

        return MergedSegmentData(
            eeg_data=merged_eeg,
            trial_ids=trial_ids,
            segment_ids=segment_ids,
            session_id=first_seg.session_id,
            labels=labels,
            primary_label=primary_label,
            start_time=start_time,
            end_time=end_time,
            total_time_length=total_time_length,
            merge_count=len(segments),
            source_segments=source_segments,
        )

    def _get_config_dict(self) -> Dict:
        """将配置对象转换为可序列化的字典"""
        config_dict = {}
        for attr in dir(self.config):
            if not attr.startswith('_'):
                value = getattr(self.config, attr)
                if not callable(value):
                    if hasattr(value, '__dict__'):
                        config_dict[attr] = value.__dict__
                    else:
                        try:
                            import pickle
                            pickle.dumps(value)
                            config_dict[attr] = value
                        except Exception:
                            pass
        return config_dict

    def _get_selection_config_dict(self) -> Dict:
        """将选择配置转换为可序列化的字典"""
        return {
            'selected_features': list(self.selection_config.selected_features),
            'selected_groups': list(self.selection_config.selected_groups),
            'excluded_features': list(self.selection_config.excluded_features),
        }

    def process_h5_file(self, h5_path: str, output_dir: str, verbose: bool = True,
                        use_parallel: bool = True) -> pd.DataFrame:
        """
        处理 HDF5 文件

        Args:
            h5_path: HDF5 文件路径
            output_dir: 输出目录
            verbose: 是否显示进度
            use_parallel: 是否使用多进程并行处理

        Returns:
            汇总的 DataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        loader = EEGDataLoader(h5_path)
        subject_info = loader.get_subject_info()

        # 同步当前文件的通道与采样率配置，避免使用默认 62 通道
        self.config.update_from_electrode_names(subject_info.channel_names)
        self.config.sampling_rate = subject_info.sampling_rate
        # 重建 PSD 计算器与特征计算器以匹配新的通道设置
        self.psd_computer = PSDComputer(
            sampling_rate=self.config.sampling_rate,
            use_gpu=self.config.use_gpu,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft
        )
        self.feature_computers = {}
        self._initialize_computers()

        # 获取原始统计信息
        trial_names = loader.get_trial_names()
        total_segments = sum(len(loader.get_segment_names(t)) for t in trial_names)

        if verbose:
            print(f"\n处理被试 {subject_info.subject_id}")
            print(f"原始 Trials: {len(trial_names)}, 原始 Segments: {total_segments}")
            print(f"合并模式: {'跨 Trial' if self.cross_trial else 'Trial 内'}")
            print(f"合并数量: 每 {self.merge_count} 个 segment 合并为 1 个")

        # 执行合并
        if self.cross_trial:
            merged_segments = self._merge_segments_cross_trial(loader)
        else:
            merged_segments = self._merge_segments_within_trial(loader)

        # 如果选择了 microstate 特征，先生成 subject 级模板
        microstate_analyzer = None
        if 'microstate' in self.selection_config.get_required_groups():
            microstate_analyzer = self._compute_microstate_template(loader, verbose=verbose)

        if verbose:
            print(f"合并后 Segments: {len(merged_segments)}")
            discarded = total_segments - len(merged_segments) * self.merge_count
            if discarded > 0:
                print(f"丢弃 Segments: {discarded}（不足以组成完整合并单元）")
            print(f"选定特征数: {len(self.get_selected_features())}")
            if use_parallel and self.n_jobs > 1:
                print(f"使用 {self.n_jobs} 个CPU进程并行处理")

        if len(merged_segments) == 0:
            if verbose:
                print("警告: 没有足够的 segments 可以合并")
            return pd.DataFrame()

        if use_parallel and self.n_jobs > 1:
            df_all = self._process_parallel(merged_segments, output_path, verbose, microstate_analyzer)
        else:
            df_all = self._process_sequential(merged_segments, output_path, verbose, microstate_analyzer)

        if verbose:
            print(f"结果保存至目录: {output_path}")

        return df_all

    def _process_sequential(self, merged_segments: List[MergedSegmentData],
                            output_path: Path, verbose: bool,
                            microstate_analyzer: Optional[MicrostateAnalyzer]) -> pd.DataFrame:
        """顺序处理合并后的 segments"""
        meta_cols = ['trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
                     'start_time', 'end_time', 'total_time_length', 'merge_count', 'source_segments']
        selected_features = self.get_selected_features()

        iterator = enumerate(merged_segments)
        if verbose:
            iterator = tqdm(list(iterator), desc="提取特征")

        all_results = []
        for idx, merged_seg in iterator:
            features = self.extract_features(merged_seg.eeg_data, microstate_analyzer=microstate_analyzer)

            # 添加元信息
            features['trial_ids'] = str(merged_seg.trial_ids)
            features['segment_ids'] = str(merged_seg.segment_ids)
            features['session_id'] = merged_seg.session_id
            features['primary_label'] = merged_seg.primary_label
            features['labels'] = str(merged_seg.labels)
            features['start_time'] = merged_seg.start_time
            features['end_time'] = merged_seg.end_time
            features['total_time_length'] = merged_seg.total_time_length
            features['merge_count'] = merged_seg.merge_count
            features['source_segments'] = str(merged_seg.source_segments)

            all_results.append(features)

            # 保存单个 CSV
            df_seg = pd.DataFrame([features])
            feature_cols = [c for c in selected_features if c in df_seg.columns]
            other_cols = [c for c in df_seg.columns if c not in meta_cols and c not in feature_cols]
            df_seg = df_seg[meta_cols + feature_cols + other_cols]

            csv_path = output_path / f"merged_segment_{idx:04d}.csv"
            df_seg.to_csv(csv_path, index=False, encoding='utf-8')

        if verbose:
            print(f"共生成 {len(all_results)} 个合并 segment CSV")

        # 返回汇总 DataFrame
        if all_results:
            df_all = pd.DataFrame(all_results)
            feature_cols_all = [c for c in selected_features if c in df_all.columns]
            other_cols_all = [c for c in df_all.columns if c not in meta_cols and c not in feature_cols_all]
            df_all = df_all[meta_cols + feature_cols_all + other_cols_all]
        else:
            df_all = pd.DataFrame()

        return df_all

    def _process_parallel(self, merged_segments: List[MergedSegmentData],
                          output_path: Path, verbose: bool,
                          microstate_analyzer: Optional[MicrostateAnalyzer]) -> pd.DataFrame:
        """并行处理合并后的 segments"""
        meta_cols = ['trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels',
                     'start_time', 'end_time', 'total_time_length', 'merge_count', 'source_segments']
        selected_features = self.get_selected_features()

        # 收集所有任务
        tasks = []
        config_dict = self._get_config_dict()
        selection_config_dict = self._get_selection_config_dict()

        microstate_centroids = None
        if microstate_analyzer is not None and microstate_analyzer.centroids is not None:
            microstate_centroids = microstate_analyzer.centroids

        # 多进程并行时强制禁用 GPU
        if config_dict.get('use_gpu', False):
            if verbose:
                warnings.warn(
                    "检测到 use_gpu=True 且启用多进程并行；将对子进程强制 use_gpu=False 以保证稳定性。"
                )
            config_dict['use_gpu'] = False

        for idx, merged_seg in enumerate(merged_segments):
            csv_path = str(output_path / f"merged_segment_{idx:04d}.csv")

            merged_segment_dict = {
                'eeg_data': merged_seg.eeg_data,
                'trial_ids': merged_seg.trial_ids,
                'segment_ids': merged_seg.segment_ids,
                'session_id': merged_seg.session_id,
                'labels': merged_seg.labels,
                'primary_label': merged_seg.primary_label,
                'start_time': merged_seg.start_time,
                'end_time': merged_seg.end_time,
                'total_time_length': merged_seg.total_time_length,
                'merge_count': merged_seg.merge_count,
                'source_segments': merged_seg.source_segments,
            }

            tasks.append((merged_segment_dict, config_dict, selection_config_dict, csv_path, microstate_centroids))

        # 使用进程池并行处理
        if verbose:
            pbar = tqdm(total=len(tasks), desc=f"并行提取特征 ({self.n_jobs} CPUs)")

        all_results = []
        failures: List[Tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_process_merged_segment_task, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    out_csv = result[0] if len(result) > 0 else "<unknown>"
                    features = result[1] if len(result) > 1 else None
                    error_text = result[2] if len(result) > 2 else None

                    if features is not None:
                        all_results.append(features)
                    else:
                        short_err = None
                        if error_text:
                            short_err = error_text.strip().splitlines()[-1]
                        warnings.warn(
                            f"处理失败: {out_csv}" + (f" | {short_err}" if short_err else "")
                        )
                        failures.append((out_csv, error_text or ""))
                except Exception as e:
                    warnings.warn(f"任务执行出错: {traceback.format_exc()}")

                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()

        # 将失败原因写入日志
        if failures:
            fail_log = output_path / "failures.log"
            try:
                with open(fail_log, 'w', encoding='utf-8') as f:
                    for out_csv, err in failures:
                        f.write(out_csv + "\n")
                        if err:
                            f.write(err.rstrip() + "\n")
                        f.write("-" * 80 + "\n")
                if verbose:
                    warnings.warn(f"共有 {len(failures)} 个合并 segment 失败，详见: {fail_log}")
            except Exception:
                warnings.warn(f"写入失败日志出错: {traceback.format_exc()}")

        if verbose:
            print(f"共生成 {len(all_results)} 个合并 segment CSV")

        # 返回汇总 DataFrame
        if all_results:
            df_all = pd.DataFrame(all_results)
            feature_cols_all = [c for c in selected_features if c in df_all.columns]
            other_cols_all = [c for c in df_all.columns if c not in meta_cols and c not in feature_cols_all]
            df_all = df_all[meta_cols + feature_cols_all + other_cols_all]
        else:
            df_all = pd.DataFrame()

        return df_all


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='动态合并 Segment 的特征提取脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 每2个segment合并成1个（默认按trial内合并）
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2

  # 每3个segment合并，跨trial合并
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 3 --cross-trial

  # 使用预设特征配置
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2 --preset standard

  # 指定并行进程数
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2 --n-jobs 8

  # 禁用并行处理
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 2 --no-parallel

  # 微状态模板：每个trial随机选择5个segments生成模板
  python merged_segment_extraction.py -i data.h5 -o ./output --merge-count 1 --preset full --microstate-segs 5

  # 完整示例：使用full预设，每个trial选3个segment生成微状态模板
  python merged_segment_extraction.py -i /path/to/dataset -o ./output --merge-count 1 --preset full --microstate-segs 3
        """
    )

    # 输入输出
    parser.add_argument('-i', '--input', type=str, required=True, help='输入 HDF5 文件路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出目录')

    # 合并选项
    parser.add_argument('--merge-count', type=int, default=2,
                        help='合并的 segment 数量（默认: 2）')
    parser.add_argument('--cross-trial', action='store_true',
                        help='跨 trial 合并（默认: trial 内合并）')

    # 特征选择
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='使用预设配置')
    parser.add_argument('--groups', type=str, nargs='+',
                        choices=list(FEATURE_GROUPS.keys()),
                        help='选择特征组')
    parser.add_argument('--features', type=str, nargs='+',
                        help='选择单个特征')
    parser.add_argument('--exclude', type=str, nargs='+',
                        help='排除特定特征')

    # 并行选项
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='并行进程数（默认：自动选择）')
    parser.add_argument('--no-parallel', action='store_true',
                        help='禁用并行处理（使用串行模式）')

    # 微状态选项
    parser.add_argument('--microstate-segs', type=int, default=None,
                        dest='microstate_segments_per_trial',
                        help='每个 trial 用于生成微状态模板的 segment 数量（默认: 使用所有 segments）')

    # 其他选项
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    # 解析输入路径列表（支持目录、通配符、单文件）
    input_path = Path(args.input)
    if input_path.is_dir():
        h5_files = sorted(str(p) for p in input_path.glob('*.h5'))
    elif '*' in args.input or '?' in args.input:
        h5_files = sorted(str(p) for p in Path().glob(args.input))
    else:
        h5_files = [str(input_path)]

    if not h5_files:
        print(f"错误: 未找到匹配的 HDF5 文件: {args.input}")
        sys.exit(1)

    # 构建特征选择配置
    selection_config = FeatureSelectionConfig()

    # 应用预设
    if args.preset:
        selection_config = apply_preset(args.preset)

    # 添加命令行指定的组
    if args.groups:
        selection_config.selected_groups.update(args.groups)

    # 添加命令行指定的特征
    if args.features:
        selection_config.selected_features.update(args.features)

    # 应用排除
    if args.exclude:
        selection_config.excluded_features.update(args.exclude)

    # 如果没有选择任何特征，使用默认（标准模式）
    if (not selection_config.selected_groups and
        not selection_config.selected_features):
        if not args.quiet:
            print("未选择任何特征，使用 'standard' 预设")
        selection_config = apply_preset('standard')

    use_parallel = not args.no_parallel

    if not args.quiet:
        print("\n" + "=" * 60)
        print("动态合并 Segment 特征提取")
        print("=" * 60)
        print(f"输入: {len(h5_files)} 个 HDF5")
        print(f"输出目录: {args.output}")
        print(f"合并数量: {args.merge_count}")
        print(f"合并模式: {'跨 Trial' if args.cross_trial else 'Trial 内'}")
        print(f"并行: {'启用' if use_parallel else '禁用'}")
        if args.microstate_segments_per_trial:
            print(f"微状态模板: 每个 trial 随机选择 {args.microstate_segments_per_trial} 个 segments")
        else:
            print("微状态模板: 使用所有 segments")
        print("-" * 60)

    all_results = []

    for h5_path in h5_files:
        if not Path(h5_path).exists():
            print(f"警告: 文件不存在，跳过: {h5_path}")
            continue

        # 配置：每个文件创建独立的 config/extractor，避免跨文件遗留状态
        eeg_config = Config()
        eeg_config.use_gpu = not args.no_gpu

        extractor = MergedSegmentExtractor(
            config=eeg_config,
            selection_config=selection_config,
            merge_count=args.merge_count,
            cross_trial=args.cross_trial,
            n_jobs=args.n_jobs,
            microstate_segments_per_trial=args.microstate_segments_per_trial
        )

        # 每个输入文件单独创建子输出目录
        file_out_dir = Path(args.output) / Path(h5_path).stem

        if not args.quiet:
            print(f"\n处理: {h5_path}")

        df_result = extractor.process_h5_file(
            h5_path,
            str(file_out_dir),
            verbose=not args.quiet,
            use_parallel=use_parallel
        )

        if len(df_result) > 0:
            df_result['source_file'] = Path(h5_path).name
            all_results.append(df_result)

    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_path = Path(args.output) / "all_merged_features.csv"
        summary.to_csv(summary_path, index=False, encoding='utf-8')
        if not args.quiet:
            print(f"\n汇总文件保存至: {summary_path}")

    if not args.quiet:
        print("\n" + "=" * 60)
        print("特征提取完成！")
        print("=" * 60)


if __name__ == '__main__':
    main()