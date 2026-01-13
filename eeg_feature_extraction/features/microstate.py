"""
微状态特征计算

基于论文 "Fine-Tuning Large Language Models Using EEG Microstate Features for Mental Workload Assessment"
实现 EEG 微状态分析，提取 4 个微状态类别（A, B, C, D）的 5 个特征。

算法步骤：
1. 计算全局场功率 (GFP)
2. 在 GFP 峰值处使用极性不变的 K-Means 聚类生成模板
3. Backfitting：将每个时间点分配到一个微状态
4. 提取 5 个特征：mean_duration, occurrence, time_coverage, mean_corr, gev
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from typing import Dict, Optional, List, Tuple
import warnings

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult
from ..config import Config

# 尝试导入 CuPy 用于 GPU 加速
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class MicrostateAnalyzer:
    """
    微状态分析器

    实现 GFP 计算、极性不变 K-Means 聚类、Backfitting 和特征提取
    """

    def __init__(self, n_states: int = 4, max_iter: int = 100,
                 random_state: Optional[int] = 42, min_peak_distance: int = 2,
                 use_gpu: bool = False):
        """
        初始化微状态分析器

        Args:
            n_states: 微状态数量 (默认 4，对应 A, B, C, D)
            max_iter: K-Means 最大迭代次数
            random_state: 随机种子
            min_peak_distance: GFP 峰值检测的最小距离
            use_gpu: 是否使用 GPU 加速 K-Means 聚类
        """
        self.n_states = n_states
        self.max_iter = max_iter
        self.random_state = random_state
        self.min_peak_distance = min_peak_distance
        self.use_gpu = use_gpu and HAS_CUPY
        self.centroids: Optional[np.ndarray] = None  # shape: (n_states, n_channels)

        if use_gpu and not HAS_CUPY:
            warnings.warn("CuPy 未安装，将使用 CPU 进行 K-Means 聚类。"
                         "安装 CuPy 以启用 GPU 加速: pip install cupy-cuda12x")

    def compute_gfp(self, data: np.ndarray) -> np.ndarray:
        """
        计算全局场功率 (Global Field Power)

        GFP(t) = sqrt(sum((V_i(t) - mean(V(t)))^2) / N)

        Args:
            data: EEG 数据, shape: (n_channels, n_samples)

        Returns:
            GFP 时间序列, shape: (n_samples,)
        """
        # 对于每个时间点，计算所有通道电压的标准差
        mean_voltage = np.mean(data, axis=0)  # (n_samples,)
        gfp = np.sqrt(np.mean((data - mean_voltage) ** 2, axis=0))
        return gfp

    def find_gfp_peaks(self, gfp: np.ndarray) -> np.ndarray:
        """
        找到 GFP 的局部最大值（峰值）

        Args:
            gfp: GFP 时间序列

        Returns:
            峰值索引数组
        """
        # 使用 scipy 的 find_peaks 函数
        peaks, _ = find_peaks(gfp, distance=self.min_peak_distance)

        # 如果峰值太少，降低距离要求
        if len(peaks) < self.n_states * 2:
            peaks, _ = find_peaks(gfp, distance=1)

        # 如果还是太少，使用所有点
        if len(peaks) < self.n_states:
            # 使用 GFP 最大的点
            peaks = np.argsort(gfp)[-max(self.n_states * 2, len(gfp) // 10):]

        return peaks

    def _spatial_correlation(self, map1: np.ndarray, map2: np.ndarray) -> float:
        """
        计算两个地形图之间的空间相关性

        Args:
            map1: 第一个地形图 (n_channels,)
            map2: 第二个地形图 (n_channels,)

        Returns:
            Pearson 相关系数
        """
        # 处理常数数组的情况
        if np.std(map1) < 1e-10 or np.std(map2) < 1e-10:
            return 0.0

        corr, _ = pearsonr(map1, map2)
        return corr if not np.isnan(corr) else 0.0

    def _polarity_invariant_kmeans(self, maps: np.ndarray) -> np.ndarray:
        """
        极性不变的 K-Means 聚类

        距离度量: 1 - |spatial_correlation(X, Y)|

        Args:
            maps: GFP 峰值处的地形图, shape: (n_maps, n_channels)

        Returns:
            聚类中心, shape: (n_states, n_channels)
        """
        if self.use_gpu:
            return self._polarity_invariant_kmeans_gpu(maps)
        else:
            return self._polarity_invariant_kmeans_vectorized(maps)

    def _polarity_invariant_kmeans_vectorized(self, maps: np.ndarray) -> np.ndarray:
        """
        向量化的极性不变 K-Means 聚类 (CPU 版本)

        Args:
            maps: GFP 峰值处的地形图, shape: (n_maps, n_channels)

        Returns:
            聚类中心, shape: (n_states, n_channels)
        """
        n_maps, n_channels = maps.shape

        if n_maps < self.n_states:
            warnings.warn(f"地形图数量 ({n_maps}) 少于微状态数量 ({self.n_states})，"
                          f"将使用所有地形图作为中心")
            centroids = np.zeros((self.n_states, n_channels))
            centroids[:n_maps] = maps
            return centroids

        # 对数据进行中心化和归一化（用于计算相关性）
        maps_centered = maps - maps.mean(axis=1, keepdims=True)
        maps_norm = np.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_norm[maps_norm < 1e-10] = 1.0  # 避免除零
        maps_normalized = maps_centered / maps_norm

        # 初始化：随机选择初始中心
        rng = np.random.RandomState(self.random_state)
        initial_indices = rng.choice(n_maps, self.n_states, replace=False)
        centroids = maps[initial_indices].copy()

        # 归一化中心
        centroids_centered = centroids - centroids.mean(axis=1, keepdims=True)
        centroids_norm = np.linalg.norm(centroids_centered, axis=1, keepdims=True)
        centroids_norm[centroids_norm < 1e-10] = 1.0
        centroids_normalized = centroids_centered / centroids_norm

        labels = np.zeros(n_maps, dtype=np.int32)

        for iteration in range(self.max_iter):
            old_labels = labels.copy()

            # E-step: 向量化计算所有 map 与所有 centroid 的相关性
            # correlations shape: (n_maps, n_states)
            correlations = maps_normalized @ centroids_normalized.T

            # 取绝对值并找到最大相关性的索引
            abs_correlations = np.abs(correlations)
            labels = np.argmax(abs_correlations, axis=1)

            # M-step: 更新中心
            for k in range(self.n_states):
                cluster_mask = labels == k
                if not np.any(cluster_mask):
                    continue

                cluster_maps = maps[cluster_mask]
                cluster_corrs = correlations[cluster_mask, k]

                # 对齐极性：相关性为负的取反
                polarity = np.sign(cluster_corrs).reshape(-1, 1)
                polarity[polarity == 0] = 1
                aligned_maps = cluster_maps * polarity

                new_centroid = np.mean(aligned_maps, axis=0)
                centroids[k] = new_centroid

            # 更新归一化的中心
            centroids_centered = centroids - centroids.mean(axis=1, keepdims=True)
            centroids_norm = np.linalg.norm(centroids_centered, axis=1, keepdims=True)
            centroids_norm[centroids_norm < 1e-10] = 1.0
            centroids_normalized = centroids_centered / centroids_norm

            # 检查收敛
            if np.array_equal(labels, old_labels):
                break

        # 最终归一化中心
        for i in range(self.n_states):
            norm = np.linalg.norm(centroids[i])
            if norm > 1e-10:
                centroids[i] /= norm

        return centroids

    def _polarity_invariant_kmeans_gpu(self, maps: np.ndarray) -> np.ndarray:
        """
        GPU 加速的极性不变 K-Means 聚类

        Args:
            maps: GFP 峰值处的地形图, shape: (n_maps, n_channels)

        Returns:
            聚类中心, shape: (n_states, n_channels)
        """
        n_maps, n_channels = maps.shape

        if n_maps < self.n_states:
            warnings.warn(f"地形图数量 ({n_maps}) 少于微状态数量 ({self.n_states})，"
                          f"将使用所有地形图作为中心")
            centroids = np.zeros((self.n_states, n_channels))
            centroids[:n_maps] = maps
            return centroids

        # 转移到 GPU
        maps_gpu = cp.asarray(maps, dtype=cp.float32)

        # 对数据进行中心化和归一化
        maps_centered = maps_gpu - maps_gpu.mean(axis=1, keepdims=True)
        maps_norm = cp.linalg.norm(maps_centered, axis=1, keepdims=True)
        maps_norm = cp.maximum(maps_norm, 1e-10)
        maps_normalized = maps_centered / maps_norm

        # 初始化：随机选择初始中心
        rng = np.random.RandomState(self.random_state)
        initial_indices = rng.choice(n_maps, self.n_states, replace=False)
        centroids_gpu = maps_gpu[initial_indices].copy()

        # 归一化中心
        centroids_centered = centroids_gpu - centroids_gpu.mean(axis=1, keepdims=True)
        centroids_norm = cp.linalg.norm(centroids_centered, axis=1, keepdims=True)
        centroids_norm = cp.maximum(centroids_norm, 1e-10)
        centroids_normalized = centroids_centered / centroids_norm

        labels = cp.zeros(n_maps, dtype=cp.int32)

        for iteration in range(self.max_iter):
            old_labels = labels.copy()

            # E-step: 向量化计算所有 map 与所有 centroid 的相关性
            correlations = maps_normalized @ centroids_normalized.T

            # 取绝对值并找到最大相关性的索引
            abs_correlations = cp.abs(correlations)
            labels = cp.argmax(abs_correlations, axis=1)

            # M-step: 更新中心
            for k in range(self.n_states):
                cluster_mask = labels == k
                if not cp.any(cluster_mask):
                    continue

                cluster_maps = maps_gpu[cluster_mask]
                cluster_corrs = correlations[cluster_mask, k]

                # 对齐极性
                polarity = cp.sign(cluster_corrs).reshape(-1, 1)
                polarity = cp.where(polarity == 0, 1, polarity)
                aligned_maps = cluster_maps * polarity

                new_centroid = cp.mean(aligned_maps, axis=0)
                centroids_gpu[k] = new_centroid

            # 更新归一化的中心
            centroids_centered = centroids_gpu - centroids_gpu.mean(axis=1, keepdims=True)
            centroids_norm = cp.linalg.norm(centroids_centered, axis=1, keepdims=True)
            centroids_norm = cp.maximum(centroids_norm, 1e-10)
            centroids_normalized = centroids_centered / centroids_norm

            # 检查收敛
            if cp.array_equal(labels, old_labels):
                break

        # 最终归一化并转回 CPU
        centroids = cp.asnumpy(centroids_gpu)
        for i in range(self.n_states):
            norm = np.linalg.norm(centroids[i])
            if norm > 1e-10:
                centroids[i] /= norm

        return centroids

    def fit(self, data: np.ndarray) -> 'MicrostateAnalyzer':
        """
        从数据中学习微状态模板

        Args:
            data: EEG 数据, shape: (n_channels, n_samples)

        Returns:
            self
        """
        # 1. 计算 GFP
        gfp = self.compute_gfp(data)

        # 2. 找到 GFP 峰值
        peak_indices = self.find_gfp_peaks(gfp)

        # 3. 提取峰值处的地形图
        peak_maps = data[:, peak_indices].T  # shape: (n_peaks, n_channels)

        # 4. 执行极性不变的 K-Means 聚类
        self.centroids = self._polarity_invariant_kmeans(peak_maps)

        return self

    def fit_from_segments(self, segments: List[np.ndarray]) -> 'MicrostateAnalyzer':
        """
        从多个 segments 学习微状态模板

        将所有 segments 的 GFP 峰值处地形图合并后进行聚类

        Args:
            segments: EEG 数据列表, 每个元素 shape: (n_channels, n_samples)

        Returns:
            self
        """
        all_peak_maps = []

        for data in segments:
            # 计算 GFP
            gfp = self.compute_gfp(data)

            # 找到 GFP 峰值
            peak_indices = self.find_gfp_peaks(gfp)

            # 提取峰值处的地形图
            peak_maps = data[:, peak_indices].T  # shape: (n_peaks, n_channels)
            all_peak_maps.append(peak_maps)

        # 合并所有峰值地形图
        if len(all_peak_maps) > 0:
            combined_maps = np.vstack(all_peak_maps)
        else:
            raise ValueError("没有有效的 segment 数据")

        # 执行极性不变的 K-Means 聚类
        self.centroids = self._polarity_invariant_kmeans(combined_maps)

        return self

    def backfit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backfitting: 将每个时间点分配到一个微状态（向量化优化版本）

        Args:
            data: EEG 数据, shape: (n_channels, n_samples)

        Returns:
            labels: 每个时间点的微状态标签, shape: (n_samples,)
            correlations: 每个时间点与其分配微状态的相关性, shape: (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("模型尚未拟合，请先调用 fit() 或 fit_from_segments()")

        n_channels, n_samples = data.shape

        # 向量化计算：对数据和中心进行中心化和归一化
        # data shape: (n_channels, n_samples)
        # centroids shape: (n_states, n_channels)

        # 中心化数据（沿通道轴）
        data_centered = data - data.mean(axis=0, keepdims=True)  # (n_channels, n_samples)
        data_norm = np.linalg.norm(data_centered, axis=0, keepdims=True)  # (1, n_samples)
        data_norm = np.maximum(data_norm, 1e-10)  # 避免除零
        data_normalized = data_centered / data_norm  # (n_channels, n_samples)

        # 中心化和归一化中心
        centroids_centered = self.centroids - self.centroids.mean(axis=1, keepdims=True)  # (n_states, n_channels)
        centroids_norm = np.linalg.norm(centroids_centered, axis=1, keepdims=True)  # (n_states, 1)
        centroids_norm = np.maximum(centroids_norm, 1e-10)
        centroids_normalized = centroids_centered / centroids_norm  # (n_states, n_channels)

        # 向量化计算所有相关性
        # correlations shape: (n_states, n_samples)
        all_correlations = centroids_normalized @ data_normalized

        # 取绝对值（极性不变）
        abs_correlations = np.abs(all_correlations)

        # 找到每个时间点的最大相关性及其对应的微状态
        labels = np.argmax(abs_correlations, axis=0)  # (n_samples,)
        correlations = np.max(abs_correlations, axis=0)  # (n_samples,)

        return labels, correlations

    def extract_features(self, data: np.ndarray, sfreq: float) -> Dict[str, float]:
        """
        提取微状态特征

        对每个微状态类别 (0-3) 计算 5 个特征:
        1. meandurs: 平均持续时间 (秒)
        2. occurrence: 每秒出现次数 (Hz)
        3. timecov: 时间覆盖率 (秒)
        4. mean_corr: 平均相关性
        5. gev: 全局解释方差

        Args:
            data: EEG 数据, shape: (n_channels, n_samples)
            sfreq: 采样频率 (Hz)

        Returns:
            特征字典
        """
        if self.centroids is None:
            raise ValueError("模型尚未拟合")

        # Backfitting
        labels, correlations = self.backfit(data)

        # 计算 GFP
        gfp = self.compute_gfp(data)

        n_samples = data.shape[1]
        total_time = n_samples / sfreq

        # 计算 GFP 平方和 (用于 GEV 计算)
        gfp_squared_sum = np.sum(gfp ** 2)

        features = {}

        for k in range(self.n_states):
            state_name = f"Microstate_{k}"

            # 找到属于该状态的时间点
            state_mask = labels == k
            n_state_samples = np.sum(state_mask)

            # 计算连续段落
            segments = self._find_segments(labels, k)
            n_segments = len(segments)

            # 1. Mean Duration (meandurs)
            if n_segments > 0:
                meandurs = (n_state_samples / sfreq) / n_segments
            else:
                meandurs = 0.0
            features[f"{state_name}_meandurs"] = float(meandurs)

            # 2. Occurrence per second
            occurrence = n_segments / total_time if total_time > 0 else 0.0
            features[f"{state_name}_occurrence"] = float(occurrence)

            # 3. Time Coverage (秒)
            timecov = n_state_samples / sfreq
            features[f"{state_name}_timecov"] = float(timecov)

            # 4. Mean Correlation
            if n_state_samples > 0:
                mean_corr = np.mean(correlations[state_mask])
            else:
                mean_corr = 0.0
            features[f"{state_name}_mean_corr"] = float(mean_corr)

            # 5. Global Explained Variance (GEV)
            if gfp_squared_sum > 0 and n_state_samples > 0:
                # GEV_k = sum_{t in k} (GFP(t) * Corr(t))^2 / sum_{all t} GFP(t)^2
                gev_numerator = np.sum((gfp[state_mask] * correlations[state_mask]) ** 2)
                gev = gev_numerator / gfp_squared_sum
            else:
                gev = 0.0
            features[f"{state_name}_gev"] = float(gev)

        return features

    def _find_segments(self, labels: np.ndarray, state: int) -> List[Tuple[int, int]]:
        """
        找到指定状态的连续段落

        Args:
            labels: 标签序列
            state: 目标状态

        Returns:
            段落列表，每个元素是 (start_idx, end_idx)
        """
        segments = []
        in_segment = False
        start_idx = 0

        for i, label in enumerate(labels):
            if label == state:
                if not in_segment:
                    in_segment = True
                    start_idx = i
            else:
                if in_segment:
                    segments.append((start_idx, i))
                    in_segment = False

        # 处理最后一个段落
        if in_segment:
            segments.append((start_idx, len(labels)))

        return segments


@FeatureRegistry.register('microstate')
class MicrostateFeatures(BaseFeature):
    """
    微状态特征计算

    注意：微状态特征需要从整个 subject 的所有 segment 生成模板，
    然后对每个 segment 进行 backfitting。

    支持三种模式：
    1. 有预计算模板：通过 kwargs 传入 'microstate_analyzer' 对象（推荐）
    2. 提供额外 segments：通过 kwargs 传入 'additional_segments' 列表（降级模式增强）
    3. 单 segment：仅从当前 segment 生成模板（降级模式，不推荐）
    """

    feature_names = [
        # Microstate 0 (A)
        'Microstate_0_meandurs',
        'Microstate_0_occurrence',
        'Microstate_0_timecov',
        'Microstate_0_mean_corr',
        'Microstate_0_gev',
        # Microstate 1 (B)
        'Microstate_1_meandurs',
        'Microstate_1_occurrence',
        'Microstate_1_timecov',
        'Microstate_1_mean_corr',
        'Microstate_1_gev',
        # Microstate 2 (C)
        'Microstate_2_meandurs',
        'Microstate_2_occurrence',
        'Microstate_2_timecov',
        'Microstate_2_mean_corr',
        'Microstate_2_gev',
        # Microstate 3 (D)
        'Microstate_3_meandurs',
        'Microstate_3_occurrence',
        'Microstate_3_timecov',
        'Microstate_3_mean_corr',
        'Microstate_3_gev',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.n_states = 4  # A, B, C, D

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算微状态特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            psd_result: PSD 结果（微状态特征不需要）
            **kwargs: 其他参数
                - microstate_analyzer: 预计算的 MicrostateAnalyzer 对象（推荐）
                - additional_segments: 额外的 segments 列表，用于降级模式的模板生成
                  格式: List[np.ndarray]，每个数组 shape: (n_channels, n_timepoints)

        Returns:
            微状态特征字典
        """
        self._validate_input(eeg_data)

        # 检查是否有预计算的分析器
        analyzer = kwargs.get('microstate_analyzer')

        if analyzer is None:
            # 降级模式：尝试从多个 segments 生成模板
            additional_segments = kwargs.get('additional_segments', [])

            if additional_segments:
                # 使用当前 segment 和额外 segments 生成模板
                all_segments = [eeg_data] + list(additional_segments)
                warnings.warn(
                    f"未提供预计算的微状态模板，将从 {len(all_segments)} 个 segments 生成模板。"
                    "建议在 subject 级别预计算模板以获得更稳定的结果。"
                )
                analyzer = MicrostateAnalyzer(n_states=self.n_states)
                analyzer.fit_from_segments(all_segments)
            else:
                # 仅从当前 segment 生成模板（不推荐）
                warnings.warn(
                    "未提供预计算的微状态模板或额外 segments，将仅从当前 segment 生成模板。"
                    "这可能导致不稳定的结果。强烈建议在 subject 级别预计算模板。"
                )
                analyzer = MicrostateAnalyzer(n_states=self.n_states)
                analyzer.fit(eeg_data)

        # 提取特征
        try:
            features = analyzer.extract_features(eeg_data, self.fs)
        except Exception as e:
            warnings.warn(f"微状态特征提取失败: {e}")
            # 返回 None 值
            features = {name: None for name in self.feature_names}

        return features
