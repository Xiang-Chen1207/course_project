"""
综合特征计算：认知负荷、清醒度、放松/紧张状态等
"""
import numpy as np
from typing import Dict, Optional, List

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult
from ..config import Config


@FeatureRegistry.register('composite')
class CompositeFeatures(BaseFeature):
    """综合特征计算"""

    feature_names = [
        'cognitive_load_estimate',
        'alertness_estimate',
        'relaxation_index',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.channel_groups = config.channel_groups
        self.channel_names = config.channel_names

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算综合特征

        Args:
            eeg_data: EEG 数据
            psd_result: PSD 结果（必须提供）

        Returns:
            综合特征字典
        """
        self._validate_input(eeg_data)

        if psd_result is None:
            raise ValueError("综合特征计算需要预先提供 PSD 结果")

        features = {}

        # 1. 认知负荷水平估计
        cognitive_load = self._compute_cognitive_load(psd_result)
        features['cognitive_load_estimate'] = float(cognitive_load)

        # 2. 清醒度水平估计
        alertness = self._compute_alertness(psd_result)
        features['alertness_estimate'] = float(alertness)

        # 3. 放松 vs 紧张状态判别
        relaxation = self._compute_relaxation_index(psd_result)
        features['relaxation_index'] = float(relaxation)

        return features

    def _get_channel_indices(self, channel_list: List[str]) -> List[int]:
        """获取通道列表对应的索引"""
        indices = []
        for ch in channel_list:
            if ch in self.channel_names:
                indices.append(self.channel_names.index(ch))
        return indices

    def _compute_cognitive_load(self, psd_result: PSDResult) -> float:
        """
        计算认知负荷水平

        基于 Theta/Alpha 比率和前额 Beta 活动

        公式: cognitive_load = sigmoid(w1 * theta_alpha_ratio + w2 * frontal_beta)
        归一化到 0-1 范围
        """
        theta_power = psd_result.band_power.get('theta', np.zeros(self.config.n_channels))
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))
        beta_power = psd_result.band_power.get('beta', np.zeros(self.config.n_channels))

        # 计算全脑 Theta/Alpha 比率
        total_alpha = np.sum(alpha_power)
        total_theta = np.sum(theta_power)
        theta_alpha_ratio = total_theta / total_alpha if total_alpha > 1e-10 else 0

        # 计算前额 Beta 功率
        frontal_indices = self._get_channel_indices(self.channel_groups.frontal)
        if frontal_indices:
            frontal_beta = np.mean(beta_power[frontal_indices])
            # 归一化前额 Beta
            total_beta = np.mean(beta_power)
            frontal_beta_norm = frontal_beta / total_beta if total_beta > 1e-10 else 0
        else:
            frontal_beta_norm = 0

        # 综合计算（简化模型）
        # 高 Theta/Alpha 和高前额 Beta 表示高认知负荷
        raw_score = 0.6 * theta_alpha_ratio + 0.4 * frontal_beta_norm

        # Sigmoid 归一化到 0-1
        # 调整参数使得典型值落在 0-1 范围内
        cognitive_load = 1 / (1 + np.exp(-2 * (raw_score - 1)))

        return np.clip(cognitive_load, 0, 1)

    def _compute_alertness(self, psd_result: PSDResult) -> float:
        """
        计算清醒度水平

        基于 Alpha/Delta 比率

        高 Alpha 和低 Delta 表示高清醒度
        """
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))
        delta_power = psd_result.band_power.get('delta', np.zeros(self.config.n_channels))

        total_alpha = np.sum(alpha_power)
        total_delta = np.sum(delta_power)

        # Alpha/Delta 比率
        alpha_delta_ratio = total_alpha / total_delta if total_delta > 1e-10 else 1

        # Sigmoid 归一化
        # 典型清醒状态的 Alpha/Delta 比率约为 0.5-2
        alertness = 1 / (1 + np.exp(-2 * (alpha_delta_ratio - 0.5)))

        return np.clip(alertness, 0, 1)

    def _compute_relaxation_index(self, psd_result: PSDResult) -> float:
        """
        计算放松 vs 紧张状态

        高 Alpha 功率表示放松
        高 Beta 功率表示紧张/警觉

        公式: relaxation = Alpha / (Alpha + Beta)
        """
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))
        beta_power = psd_result.band_power.get('beta', np.zeros(self.config.n_channels))

        total_alpha = np.sum(alpha_power)
        total_beta = np.sum(beta_power)

        total = total_alpha + total_beta
        if total > 1e-10:
            relaxation = total_alpha / total
        else:
            relaxation = 0.5  # 默认中性状态

        return np.clip(relaxation, 0, 1)
