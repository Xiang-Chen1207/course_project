"""
连接性特征计算：通道间相关性、相干性、半球间连接

优化：利用 psd_computer 的并行化 coherence 计算
"""
import numpy as np
from typing import Dict, Optional, List

# 尝试导入 GPU 加速库
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult, PSDComputer
from ..config import Config


@FeatureRegistry.register('connectivity')
class ConnectivityFeatures(BaseFeature):
    """连接性特征计算"""

    feature_names = [
        'mean_interchannel_correlation',
        'mean_alpha_coherence',
        'interhemispheric_alpha_coherence',
        'alpha_beta_band_power_correlation',
        'hemispheric_alpha_asymmetry',
        'frontal_occipital_alpha_ratio',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.channel_groups = config.channel_groups
        self.channel_names = config.channel_names
        self.psd_computer = PSDComputer(
            sampling_rate=config.sampling_rate,
            use_gpu=config.use_gpu,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            nfft=config.nfft
        )

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算连接性特征

        Args:
            eeg_data: EEG 数据
            psd_result: PSD 结果

        Returns:
            连接性特征字典
        """
        self._validate_input(eeg_data)

        features = {}

        # 1. 通道间平均相关系数
        mean_corr = self._compute_mean_correlation(eeg_data)
        features['mean_interchannel_correlation'] = float(mean_corr)

        # 2. 全脑平均连接强度（基于 Alpha 频段相干性）
        # 支持从外部传入缓存的相干性矩阵，避免重复计算
        coherence_matrix = kwargs.get('coherence_matrix', None)
        if coherence_matrix is None:
            coherence_matrix = self.psd_computer.compute_coherence(
                eeg_data, band=(8.0, 13.0)
            )
        # 取上三角矩阵的平均值（不包括对角线）
        upper_tri = coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)]
        mean_coherence = np.nanmean(upper_tri) if upper_tri.size else 0.0
        features['mean_alpha_coherence'] = float(mean_coherence)

        # 3. 左右半球间连接强度
        lr_connectivity = self._compute_lr_connectivity(eeg_data, coherence_matrix)
        features['interhemispheric_alpha_coherence'] = float(lr_connectivity)

        # 4. 频带间功率相关性（Alpha 与 Beta）
        if psd_result is not None:
            band_corr = self._compute_band_correlation(psd_result)
            features['alpha_beta_band_power_correlation'] = float(band_corr)
        else:
            features['alpha_beta_band_power_correlation'] = 0.0

        # 5. 左右半球功率不对称性
        if psd_result is not None:
            asymmetry = self._compute_hemisphere_asymmetry(psd_result)
            features['hemispheric_alpha_asymmetry'] = float(asymmetry)
        else:
            features['hemispheric_alpha_asymmetry'] = 0.0

        # 6. 前后脑区功率梯度
        if psd_result is not None:
            gradient = self._compute_ap_gradient(psd_result)
            features['frontal_occipital_alpha_ratio'] = float(gradient)
        else:
            features['frontal_occipital_alpha_ratio'] = 0.0

        return features

    def _get_channel_indices(self, channel_list: List[str]) -> List[int]:
        """获取通道列表对应的索引"""
        indices = []
        for ch in channel_list:
            if ch in self.channel_names:
                indices.append(self.channel_names.index(ch))
        return indices

    def _compute_mean_correlation(self, eeg_data: np.ndarray) -> float:
        """计算通道间平均相关系数"""
        if self.use_gpu:
            return self._compute_mean_correlation_gpu(eeg_data)
        else:
            return self._compute_mean_correlation_cpu(eeg_data)

    def _compute_mean_correlation_cpu(self, eeg_data: np.ndarray) -> float:
        """CPU 计算通道间相关系数"""
        # 方差过滤，避免常量通道导致 NaN
        std = np.std(eeg_data, axis=1)
        valid = std > 1e-10
        if np.sum(valid) < 2:
            return 0.0

        corr_matrix = np.corrcoef(eeg_data[valid])
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        if upper_tri.size == 0:
            return 0.0
        val = np.nanmean(upper_tri)
        return float(val) if np.isfinite(val) else 0.0

    def _compute_mean_correlation_gpu(self, eeg_data: np.ndarray) -> float:
        """GPU 计算通道间相关系数"""
        # GPU 侧同样做方差过滤（在 CPU 上求 std，避免额外 GPU kernel）
        std = np.std(eeg_data, axis=1)
        valid = std > 1e-10
        if np.sum(valid) < 2:
            return 0.0

        eeg_gpu = cp.asarray(eeg_data[valid])
        corr_matrix = cp.corrcoef(eeg_gpu)
        upper_tri = corr_matrix[cp.triu_indices_from(corr_matrix, k=1)]
        if upper_tri.size == 0:
            return 0.0
        val = float(cp.asnumpy(cp.nanmean(upper_tri)))
        return val if np.isfinite(val) else 0.0

    def _compute_lr_connectivity(self, eeg_data: np.ndarray,
                                  coherence_matrix: np.ndarray) -> float:
        """计算左右半球间连接强度"""
        left_indices = self._get_channel_indices(self.channel_groups.left_hemisphere)
        right_indices = self._get_channel_indices(self.channel_groups.right_hemisphere)

        if not left_indices or not right_indices:
            return 0.0

        # 获取左右半球通道间的相干性
        lr_coherence = []
        for l_idx in left_indices:
            for r_idx in right_indices:
                lr_coherence.append(coherence_matrix[l_idx, r_idx])

        return np.mean(lr_coherence) if lr_coherence else 0.0

    def _compute_band_correlation(self, psd_result: PSDResult) -> float:
        """计算 Alpha 与 Beta 频段功率的跨通道相关性"""
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))
        beta_power = psd_result.band_power.get('beta', np.zeros(self.config.n_channels))

        if len(alpha_power) < 2 or len(beta_power) < 2:
            return 0.0

        # 计算 Pearson 相关系数
        corr = np.corrcoef(alpha_power, beta_power)[0, 1]
        return corr if np.isfinite(corr) else 0.0

    def _compute_hemisphere_asymmetry(self, psd_result: PSDResult) -> float:
        """
        计算左右半球 Alpha 功率不对称性

        公式: (Right - Left) / (Right + Left)
        """
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))

        left_indices = self._get_channel_indices(self.channel_groups.left_hemisphere)
        right_indices = self._get_channel_indices(self.channel_groups.right_hemisphere)

        if not left_indices or not right_indices:
            return 0.0

        left_power = np.mean(alpha_power[left_indices])
        right_power = np.mean(alpha_power[right_indices])

        total = left_power + right_power
        if total > 1e-10:
            return (right_power - left_power) / total
        return 0.0

    def _compute_ap_gradient(self, psd_result: PSDResult) -> float:
        """
        计算前后脑区功率梯度

        前额叶 Alpha 功率 / 枕叶 Alpha 功率
        """
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))

        frontal_indices = self._get_channel_indices(self.channel_groups.frontal)
        occipital_indices = self._get_channel_indices(self.channel_groups.occipital)

        if not frontal_indices or not occipital_indices:
            return 0.0

        frontal_power = np.mean(alpha_power[frontal_indices])
        occipital_power = np.mean(alpha_power[occipital_indices])

        if occipital_power > 1e-10:
            return frontal_power / occipital_power
        return 0.0
