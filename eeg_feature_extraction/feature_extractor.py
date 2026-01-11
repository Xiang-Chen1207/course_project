"""
特征提取器：整合所有特征计算模块

支持多CPU并行处理以加速特征提取
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Type, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import warnings
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .config import Config, FrequencyBands, ChannelGroups
from .data_loader import EEGDataLoader, SegmentData
from .psd_computer import PSDComputer, PSDResult
from .features.base import BaseFeature, FeatureRegistry

# 导入所有特征模块以触发注册
from .features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    ComplexityFeatures,
    ConnectivityFeatures,
    NetworkFeatures,
    CompositeFeatures,
    MicrostateFeatures,
    MicrostateAnalyzer,
)


def _get_optimal_n_jobs() -> int:
    """获取最优的并行进程数"""
    cpu_count = mp.cpu_count()
    # 使用所有可用CPU，但至少保留1个给系统
    return max(1, cpu_count - 10)


def _process_segment_task(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    处理单个 segment 的任务函数（用于并行处理）

    Args:
        args: (segment_data, config_dict, feature_groups, output_path, microstate_centroids)

    Returns:
        (output_path, features_dict, error_text)
        - 成功: features_dict 非 None，error_text 为 None
        - 失败: features_dict 为 None，error_text 为异常信息（含 traceback）
    """
    # 支持新旧两种格式的 args
    if len(args) == 5:
        segment_dict, config_dict, feature_groups, output_path, microstate_centroids = args
    else:
        segment_dict, config_dict, feature_groups, output_path = args
        microstate_centroids = None

    try:
        # 重建 Config 对象
        config = Config()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # 还原嵌套的 dataclass 配置，避免被还原为普通 dict 后丢失方法
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

        # 重建 segment 数据
        # 注意: SegmentData 在 data_loader.py 中包含 time_length 字段
        # 并行任务必须显式传递，否则会导致所有任务构造失败。
        time_length = segment_dict.get('time_length')
        if time_length is None:
            # 兼容旧的任务字典（没有 time_length）
            try:
                time_length = float(segment_dict['end_time']) - float(segment_dict['start_time'])
            except Exception:
                time_length = 0.0

        segment = SegmentData(
            eeg_data=segment_dict['eeg_data'],
            trial_id=segment_dict['trial_id'],
            segment_id=segment_dict['segment_id'],
            session_id=segment_dict['session_id'],
            label=segment_dict['label'],
            start_time=segment_dict['start_time'],
            end_time=segment_dict['end_time'],
            time_length=time_length,
        )

        # 创建临时的特征提取器
        extractor = FeatureExtractor(config, n_jobs=1)

        # 如果有预计算的微状态模板，创建 MicrostateAnalyzer
        microstate_analyzer = None
        if microstate_centroids is not None:
            microstate_analyzer = MicrostateAnalyzer(n_states=4)
            microstate_analyzer.centroids = microstate_centroids

        # 提取特征
        features = extractor.extract_features(
            segment.eeg_data, feature_groups,
            microstate_analyzer=microstate_analyzer
        )

        # 添加元信息
        features['trial_id'] = segment.trial_id
        features['segment_id'] = segment.segment_id
        features['session_id'] = segment.session_id
        features['label'] = segment.label
        features['start_time'] = segment.start_time
        features['end_time'] = segment.end_time

        # 转换为 DataFrame 并保存
        df = pd.DataFrame([features])
        meta_cols = ['trial_id', 'segment_id', 'session_id', 'label',
                     'start_time', 'end_time']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols]
        df.to_csv(output_path, index=False, encoding='utf-8')

        return (output_path, features, None)
    except Exception as e:
        return (output_path, None, traceback.format_exc())


class FeatureExtractor:
    """EEG 特征提取器，支持多CPU并行处理"""

    def __init__(self, config: Optional[Config] = None, n_jobs: Optional[int] = None):
        """
        初始化特征提取器

        Args:
            config: 配置对象，如果为 None 则使用默认配置
            n_jobs: 并行进程数，None 表示自动选择（使用所有可用CPU）
        """
        self.config = config or Config()
        self.n_jobs = n_jobs if n_jobs is not None else _get_optimal_n_jobs()

        self._rebuild_computers()

    def _rebuild_computers(self) -> None:
        """根据当前 config 重建 PSDComputer 与各特征计算器。

        说明：部分特征模块会在 __init__ 中缓存 channel_names / PSDComputer。
        当处理不同数据集（通道数、采样率不同）时必须重建，否则可能索引越界或采样率错误。
        """
        self.psd_computer = PSDComputer(
            sampling_rate=self.config.sampling_rate,
            use_gpu=self.config.use_gpu,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft,
        )

        self.feature_computers = {
            name: feature_cls(self.config)
            for name, feature_cls in FeatureRegistry.get_all_feature_classes().items()
        }

        self._feature_names = FeatureRegistry.get_all_feature_names()

    def _sync_config_with_subject_info(self, subject_info: Any) -> None:
        """
        将 H5 内的 subject 信息同步到 config（采样率/通道数/通道名/通道分组）。

        重要改进：
        - 从.h5文件读取电极名称后，自动根据10-20/10-10命名规则确定每个电极的脑区和半球
        - 动态更新 channel_groups，无需预设固定的电极列表
        """
        try:
            # 采样率：以文件内为准（如果文件缺失则保持 config 默认/CLI 传入）
            sr = getattr(subject_info, 'sampling_rate', None)
            if sr is not None and float(sr) > 0:
                self.config.sampling_rate = float(sr)
        except Exception:
            pass

        # 获取通道名
        ch_names = None
        n_channels = None
        try:
            ch_names = getattr(subject_info, 'channel_names', None)
            n_channels = getattr(subject_info, 'n_channels', None)

            if ch_names and len(ch_names) > 0:
                # 有有效的通道名，使用新的动态分组方法
                self.config.update_from_electrode_names(list(ch_names))
            elif n_channels is not None and int(n_channels) > 0:
                # 没有通道名，但有通道数，生成占位名
                placeholder_names = [f"CH{i+1}" for i in range(int(n_channels))]
                self.config.channel_names = placeholder_names
                self.config.n_channels = int(n_channels)
                # 占位名无法确定脑区/半球，保持默认的 channel_groups
        except Exception:
            # 降级处理：尝试仅更新通道数
            try:
                nc = getattr(subject_info, 'n_channels', None)
                if nc is not None and int(nc) > 0:
                    self.config.n_channels = int(nc)
                    self.config.channel_names = [f"CH{i+1}" for i in range(int(nc))]
            except Exception:
                pass

    @property
    def feature_names(self) -> List[str]:
        """获取所有特征名称"""
        return self._feature_names.copy()

    def extract_features(self, eeg_data: np.ndarray,
                         feature_groups: Optional[List[str]] = None,
                         microstate_analyzer: Optional['MicrostateAnalyzer'] = None) -> Dict[str, float]:
        """
        提取单个 segment 的所有特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            feature_groups: 要计算的特征组列表，None 表示计算所有特征
            microstate_analyzer: 预计算的微状态分析器（用于 microstate 特征）

        Returns:
            特征名称到值的字典
        """
        if feature_groups is None:
            feature_groups = list(self.feature_computers.keys())

        # 首先计算 PSD（供多个特征组使用）
        psd_result = self.psd_computer.compute_psd(
            eeg_data,
            bands=self.config.freq_bands.get_all_bands()
        )

        # 可选缓存：相干性矩阵在 connectivity/network 之间共享
        coherence_matrix = None
        if any(g in feature_groups for g in ['connectivity', 'network']):
            try:
                coherence_matrix = self.psd_computer.compute_coherence(
                    eeg_data, band=(8.0, 13.0)
                )
            except Exception as e:
                warnings.warn(f"计算相干性矩阵失败: {e}")
                coherence_matrix = None

        # 计算所有特征
        all_features = {}
        for group_name in feature_groups:
            if group_name not in self.feature_computers:
                warnings.warn(f"未知的特征组: {group_name}")
                continue

            computer = self.feature_computers[group_name]
            try:
                if group_name in ['connectivity', 'network']:
                    features = computer.compute(
                        eeg_data, psd_result=psd_result, coherence_matrix=coherence_matrix
                    )
                elif group_name == 'microstate':
                    features = computer.compute(
                        eeg_data, psd_result=psd_result,
                        microstate_analyzer=microstate_analyzer
                    )
                else:
                    features = computer.compute(eeg_data, psd_result=psd_result)
                all_features.update(features)
            except Exception as e:
                warnings.warn(f"计算特征组 '{group_name}' 时出错: {e}")
                # 填充 NaN
                for name in computer.feature_names:
                    all_features[name] = np.nan

        return all_features

    def extract_segment_to_csv(self, segment: SegmentData, output_path: Path,
                               feature_groups: Optional[List[str]] = None):
        """
        提取单个 segment 的特征并保存到 CSV

        Args:
            segment: SegmentData 对象
            output_path: 输出文件路径
            feature_groups: 要计算的特征组列表
        """
        features = self.extract_features(segment.eeg_data, feature_groups)

        # 添加元信息
        features['trial_id'] = segment.trial_id
        features['segment_id'] = segment.segment_id
        features['session_id'] = segment.session_id
        features['label'] = segment.label
        features['start_time'] = segment.start_time
        features['end_time'] = segment.end_time

        # 转换为 DataFrame 并保存
        df = pd.DataFrame([features])

        # 重新排列列顺序：元信息在前
        meta_cols = ['trial_id', 'segment_id', 'session_id', 'label',
                     'start_time', 'end_time']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols]

        df.to_csv(output_path, index=False, encoding='utf-8')

    def _get_config_dict(self) -> Dict:
        """将配置对象转换为可序列化的字典"""
        config_dict = {}
        for attr in dir(self.config):
            if not attr.startswith('_'):
                value = getattr(self.config, attr)
                if not callable(value):
                    # 处理特殊类型
                    if hasattr(value, '__dict__'):
                        config_dict[attr] = value.__dict__
                    else:
                        try:
                            # 尝试序列化
                            import pickle
                            pickle.dumps(value)
                            config_dict[attr] = value
                        except Exception:
                            pass
        return config_dict

    def process_h5_file(self, h5_path: str, output_dir: str,
                        feature_groups: Optional[List[str]] = None,
                        verbose: bool = True,
                        use_parallel: bool = True,
                        microstate_segments_per_trial: int = 10,
                        microstate_use_gpu: bool = False):
        """
        处理整个 HDF5 文件，为每个 segment 生成特征 CSV

        Args:
            h5_path: HDF5 文件路径
            output_dir: 输出目录
            feature_groups: 要计算的特征组列表
            verbose: 是否显示进度条
            use_parallel: 是否使用多进程并行处理
            microstate_segments_per_trial: 每个 trial 用于微状态模板生成的 segment 数量，
                                           设置为 -1 表示使用所有 segment
            microstate_use_gpu: 是否使用 GPU 加速微状态 K-Means 聚类
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 加载数据
        loader = EEGDataLoader(h5_path)
        subject_info = loader.get_subject_info()

        # 同步采样率/通道信息，并重建计算器，确保不同数据集可复用同一个 extractor 实例
        self._sync_config_with_subject_info(subject_info)
        self._rebuild_computers()

        if verbose:
            print(f"处理被试 {subject_info.subject_id}")
            print(f"采样率: {subject_info.sampling_rate} Hz")
            print(f"通道数: {subject_info.n_channels}")

        # 获取所有 trials 和 segments
        trial_names = loader.get_trial_names()
        total_segments = sum(len(loader.get_segment_names(t)) for t in trial_names)

        if verbose:
            print(f"共 {len(trial_names)} 个 trials, {total_segments} 个 segments")
            if use_parallel:
                print(f"使用 {self.n_jobs} 个CPU进程并行处理")

        # 如果需要计算 microstate 特征，先生成 subject 级别的模板
        microstate_analyzer = None
        if feature_groups is None or 'microstate' in feature_groups:
            microstate_analyzer = self._compute_microstate_template(
                loader, verbose,
                segments_per_trial=microstate_segments_per_trial,
                use_gpu=microstate_use_gpu
            )

        if use_parallel and self.n_jobs > 1:
            self._process_h5_file_parallel(
                loader, subject_info, output_path, feature_groups, verbose, total_segments,
                microstate_analyzer=microstate_analyzer
            )
        else:
            self._process_h5_file_sequential(
                loader, subject_info, output_path, feature_groups, verbose, total_segments,
                microstate_analyzer=microstate_analyzer
            )

        if verbose:
            print(f"特征提取完成，结果保存在: {output_path}")

    def _compute_microstate_template(self, loader: EEGDataLoader,
                                      verbose: bool = True,
                                      segments_per_trial: int = 10,
                                      use_gpu: bool = False,
                                      random_state: Optional[int] = 42) -> MicrostateAnalyzer:
        """
        从 subject 的 trial 中随机抽样 segment 计算微状态模板

        Args:
            loader: EEG 数据加载器
            verbose: 是否显示进度信息
            segments_per_trial: 每个 trial 随机抽取的 segment 数量，
                               设置为 -1 表示使用所有 segment
            use_gpu: 是否使用 GPU 加速 K-Means 聚类
            random_state: 随机种子，用于可重复的随机抽样

        Returns:
            训练好的 MicrostateAnalyzer 对象
        """
        if verbose:
            print("正在生成 subject 级别的微状态模板...")
            if segments_per_trial > 0:
                print(f"  每个 trial 随机抽取 {segments_per_trial} 个 segments")

        # 为降低内存占用：不保留所有 segment 全量数据，仅收集各 segment 的 GFP 峰值地形图
        analyzer = MicrostateAnalyzer(n_states=4, use_gpu=use_gpu, random_state=random_state)

        rng = np.random.RandomState(random_state)
        all_peak_maps: List[np.ndarray] = []
        n_segments_used = 0
        n_peak_maps = 0
        n_trials = 0

        # 按 trial 组织数据，从每个 trial 随机抽样
        for trial_name in loader.get_trial_names():
            segment_names = loader.get_segment_names(trial_name)
            n_trials += 1

            # 随机抽样 segment
            if segments_per_trial > 0 and len(segment_names) > segments_per_trial:
                selected_segments = rng.choice(segment_names, segments_per_trial, replace=False)
            else:
                selected_segments = segment_names

            for segment_name in selected_segments:
                segment = loader.get_segment(trial_name, segment_name)
                data = segment.eeg_data
                # 计算 GFP 并取峰值地形图
                gfp = analyzer.compute_gfp(data)
                peak_indices = analyzer.find_gfp_peaks(gfp)
                peak_maps = data[:, peak_indices].T  # (n_peaks, n_channels)
                if peak_maps.size > 0:
                    all_peak_maps.append(peak_maps)
                    n_peak_maps += peak_maps.shape[0]
                n_segments_used += 1

        if n_segments_used == 0 or n_peak_maps == 0:
            raise ValueError("没有有效的 segment 峰值地形图用于微状态模板生成")

        if verbose:
            print(f"  从 {n_trials} 个 trials 中抽取了 {n_segments_used} 个 segments，"
                  f"共 {n_peak_maps} 个 GFP 峰值地形图用于模板生成")
            if use_gpu:
                print("  使用 GPU 加速 K-Means 聚类")

        combined_maps = np.vstack(all_peak_maps)
        analyzer.centroids = analyzer._polarity_invariant_kmeans(combined_maps)

        if verbose:
            print("  微状态模板生成完成")

        return analyzer

    def _process_h5_file_sequential(self, loader, subject_info, output_path: Path,
                                     feature_groups: Optional[List[str]],
                                     verbose: bool, total_segments: int,
                                     microstate_analyzer: Optional[MicrostateAnalyzer] = None):
        """顺序处理 segments"""
        if verbose:
            pbar = tqdm(total=total_segments, desc="提取特征")

        for trial_name, segment_name, segment in loader.iter_segments():
            trial_dir = output_path / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{segment_name}.csv"
            csv_path = trial_dir / filename
            self._extract_segment_to_csv_with_microstate(
                segment, csv_path, feature_groups, microstate_analyzer
            )

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

    def _extract_segment_to_csv_with_microstate(self, segment: SegmentData, output_path: Path,
                                                  feature_groups: Optional[List[str]],
                                                  microstate_analyzer: Optional[MicrostateAnalyzer] = None):
        """
        提取单个 segment 的特征并保存到 CSV（支持微状态分析器）

        Args:
            segment: SegmentData 对象
            output_path: 输出文件路径
            feature_groups: 要计算的特征组列表
            microstate_analyzer: 预计算的微状态分析器
        """
        features = self.extract_features(
            segment.eeg_data, feature_groups,
            microstate_analyzer=microstate_analyzer
        )

        # 添加元信息
        features['trial_id'] = segment.trial_id
        features['segment_id'] = segment.segment_id
        features['session_id'] = segment.session_id
        features['label'] = segment.label
        features['start_time'] = segment.start_time
        features['end_time'] = segment.end_time

        # 转换为 DataFrame 并保存
        df = pd.DataFrame([features])

        # 重新排列列顺序：元信息在前
        meta_cols = ['trial_id', 'segment_id', 'session_id', 'label',
                     'start_time', 'end_time']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df = df[meta_cols + feature_cols]

        df.to_csv(output_path, index=False, encoding='utf-8')

    def _process_h5_file_parallel(self, loader, subject_info, output_path: Path,
                                   feature_groups: Optional[List[str]],
                                   verbose: bool, total_segments: int,
                                   microstate_analyzer: Optional[MicrostateAnalyzer] = None):
        """并行处理 segments"""
        # 收集所有任务
        tasks = []
        config_dict = self._get_config_dict()

        # 多进程并行时默认禁用 GPU：多数 GPU 后端不支持多进程同时初始化/竞争，
        # 会导致任务在子进程中异常退出（此前异常被吞掉，表现为"处理失败"但无原因）。
        if config_dict.get('use_gpu', False):
            if verbose:
                warnings.warn(
                    "检测到 use_gpu=True 且启用多进程并行；将对子进程强制 use_gpu=False 以保证稳定性。"
                )
            config_dict['use_gpu'] = False

        # 提取微状态模板的 centroids（可序列化的 numpy 数组）
        microstate_centroids = None
        if microstate_analyzer is not None and microstate_analyzer.centroids is not None:
            microstate_centroids = microstate_analyzer.centroids

        for trial_name, segment_name, segment in loader.iter_segments():
            trial_dir = output_path / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{segment_name}.csv"
            csv_path = str(trial_dir / filename)

            # 将 segment 转换为可序列化的字典
            segment_dict = {
                'eeg_data': segment.eeg_data,
                'trial_id': segment.trial_id,
                'segment_id': segment.segment_id,
                'session_id': segment.session_id,
                'label': segment.label,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'time_length': getattr(segment, 'time_length', None),
            }

            tasks.append((segment_dict, config_dict, feature_groups, csv_path, microstate_centroids))

        # 使用进程池并行处理
        if verbose:
            pbar = tqdm(total=len(tasks), desc=f"并行提取特征 ({self.n_jobs} CPUs)")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_process_segment_task, task) for task in tasks]

            failures: List[Tuple[str, str]] = []

            for future in as_completed(futures):
                try:
                    result = future.result()
                    out_csv = result[0] if len(result) > 0 else "<unknown>"
                    features = result[1] if len(result) > 1 else None
                    error_text = result[2] if len(result) > 2 else None

                    if features is None:
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

        # 将失败原因写入日志，便于定位/重跑
        if failures:
            fail_log = output_path / "failures.log"
            try:
                with open(fail_log, 'w', encoding='utf-8') as f:
                    for out_csv, err in failures:
                        f.write(out_csv)
                        f.write("\n")
                        if err:
                            f.write(err.rstrip())
                            f.write("\n")
                        f.write("-" * 80)
                        f.write("\n")
                if verbose:
                    warnings.warn(f"共有 {len(failures)} 个 segment 失败，详见: {fail_log}")
            except Exception:
                warnings.warn(f"写入失败日志出错: {traceback.format_exc()}")

    def process_multiple_files(self, h5_paths: List[str], output_base_dir: str,
                                feature_groups: Optional[List[str]] = None,
                                verbose: bool = True,
                                parallel_files: bool = True,
                                microstate_segments_per_trial: int = 10,
                                microstate_use_gpu: bool = False):
        """
        处理多个 HDF5 文件

        Args:
            h5_paths: HDF5 文件路径列表
            output_base_dir: 输出基础目录
            feature_groups: 要计算的特征组列表
            verbose: 是否显示进度
            parallel_files: 是否并行处理多个文件
            microstate_segments_per_trial: 每个 trial 用于微状态模板生成的 segment 数量
            microstate_use_gpu: 是否使用 GPU 加速微状态 K-Means 聚类
        """
        if parallel_files and len(h5_paths) > 1 and self.n_jobs > 1:
            self._process_multiple_files_parallel(
                h5_paths, output_base_dir, feature_groups, verbose,
                microstate_segments_per_trial, microstate_use_gpu
            )
        else:
            for h5_path in h5_paths:
                h5_name = Path(h5_path).stem
                output_dir = Path(output_base_dir) / h5_name
                self.process_h5_file(
                    h5_path, str(output_dir), feature_groups, verbose,
                    microstate_segments_per_trial=microstate_segments_per_trial,
                    microstate_use_gpu=microstate_use_gpu
                )

    def _process_multiple_files_parallel(self, h5_paths: List[str], output_base_dir: str,
                                          feature_groups: Optional[List[str]],
                                          verbose: bool,
                                          microstate_segments_per_trial: int = 10,
                                          microstate_use_gpu: bool = False):
        """并行处理多个 HDF5 文件"""
        if verbose:
            print(f"并行处理 {len(h5_paths)} 个文件，使用 {self.n_jobs} 个CPU进程")

        # 准备任务
        tasks = []
        for h5_path in h5_paths:
            h5_name = Path(h5_path).stem
            output_dir = str(Path(output_base_dir) / h5_name)
            tasks.append((h5_path, output_dir, feature_groups,
                         microstate_segments_per_trial, microstate_use_gpu))

        def process_single_file(args):
            h5_path, output_dir, fgroups, ms_seg_per_trial, ms_use_gpu = args
            try:
                # 每个进程创建自己的提取器（串行处理该文件内的 segments）
                extractor = FeatureExtractor(self.config, n_jobs=1)
                extractor.process_h5_file(
                    h5_path, output_dir, fgroups, verbose=False, use_parallel=False,
                    microstate_segments_per_trial=ms_seg_per_trial,
                    microstate_use_gpu=ms_use_gpu
                )
                return (h5_path, True)
            except Exception as e:
                return (h5_path, False, str(e))

        if verbose:
            pbar = tqdm(total=len(tasks), desc=f"处理文件 ({self.n_jobs} CPUs)")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(process_single_file, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if not result[1]:
                        warnings.warn(f"处理文件失败: {result[0]} - {result[2] if len(result) > 2 else 'Unknown error'}")
                except Exception as e:
                    warnings.warn(f"任务执行出错: {e}")

                if verbose:
                    pbar.update(1)

        if verbose:
            pbar.close()
            print(f"所有文件处理完成，结果保存在: {output_base_dir}")

    def get_feature_info(self) -> pd.DataFrame:
        """
        获取所有特征的信息

        Returns:
            DataFrame 包含特征名称和所属组
        """
        info = []
        for group_name, computer in self.feature_computers.items():
            for feature_name in computer.feature_names:
                info.append({
                    '特征名称': feature_name,
                    '特征组': group_name
                })
        return pd.DataFrame(info)


def add_custom_feature(name: str, feature_cls: Type[BaseFeature]):
    """
    添加自定义特征计算类

    用于扩展新特征

    Args:
        name: 特征组名称
        feature_cls: 特征计算类（必须继承 BaseFeature）

    Example:
        @FeatureRegistry.register('my_features')
        class MyFeatures(BaseFeature):
            feature_names = ['特征1', '特征2']

            def compute(self, eeg_data, psd_result=None, **kwargs):
                return {'特征1': 1.0, '特征2': 2.0}

        # 或者手动注册
        add_custom_feature('my_features', MyFeatures)
    """
    FeatureRegistry.register(name)(feature_cls)
