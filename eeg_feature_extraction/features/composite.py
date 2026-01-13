"""
综合特征计算：认知负荷、清醒度、放松/紧张状态等

特征说明：
- theta_alpha_ratio: 全脑 θ/α 比率，反映认知负荷和注意力状态
  高值表示高认知负荷（更多theta活动相对于alpha）
- frontal_beta_ratio: 前额beta与全脑beta的比值
  高值表示前额区域高度活跃，与认知加工和执行功能相关
- cognitive_load_estimate: 综合认知负荷估计（结合上述两个指标）
- alertness_estimate: 清醒度估计（基于Alpha/Delta比率）
- relaxation_index: 放松指数（Alpha与Beta的相对比例）
"""
import numpy as np
from typing import Dict, Optional, List

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult
from ..config import Config


@FeatureRegistry.register('composite')
class CompositeFeatures(BaseFeature):
    """综合特征计算

    认知负荷相关特征已拆分为独立的指标：
    - theta_alpha_ratio: 全脑θ/α比率
    - frontal_beta_ratio: 前额beta与全脑beta的比值
    - cognitive_load_estimate: 综合认知负荷估计
    """

    feature_names = [
        # 认知负荷拆分特征
        'theta_alpha_ratio',
        'frontal_beta_ratio',
        # 综合指标
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

        # 1. 计算认知负荷相关特征（拆分为独立指标）
        theta_alpha_ratio, frontal_beta_ratio = self._compute_cognitive_load_components(psd_result)
        # 即使为None也要写入，确保CSV列一致
        features['theta_alpha_ratio'] = theta_alpha_ratio
        features['frontal_beta_ratio'] = frontal_beta_ratio

        # 2. 综合认知负荷估计（使用拆分后的特征计算）
        cognitive_load = self._compute_cognitive_load(theta_alpha_ratio, frontal_beta_ratio)
        # 即使为None也要写入，确保CSV列一致
        features['cognitive_load_estimate'] = cognitive_load

        # 3. 清醒度水平估计
        alertness = self._compute_alertness(psd_result)
        # 即使为None也要写入，确保CSV列一致
        features['alertness_estimate'] = alertness

        # 4. 放松 vs 紧张状态判别
        relaxation = self._compute_relaxation_index(psd_result)
        # 即使为None也要写入，确保CSV列一致
        features['relaxation_index'] = relaxation

        return features

    def _get_channel_indices(self, channel_list: List[str]) -> List[int]:
        """获取通道列表对应的索引"""
        indices = []
        for ch in channel_list:
            if ch in self.channel_names:
                indices.append(self.channel_names.index(ch))
        return indices

    def _compute_cognitive_load_components(self, psd_result: PSDResult) -> tuple:
        """
        计算认知负荷的两个组成成分

        Returns:
            tuple: (theta_alpha_ratio, frontal_beta_ratio)
            - theta_alpha_ratio: 全脑 θ/α 比率
            - frontal_beta_ratio: 前额beta与全脑beta的比值
        """
        theta_power = psd_result.band_power.get('theta', np.zeros(self.config.n_channels))
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))
        beta_power = psd_result.band_power.get('beta', np.zeros(self.config.n_channels))

        # 1. 计算全脑 Theta/Alpha 比率
        total_alpha = np.sum(alpha_power)
        total_theta = np.sum(theta_power)
        theta_alpha_ratio = self._safe_ratio(total_theta, total_alpha)

        # 2. 计算前额 Beta 与全脑 Beta 的比值
        frontal_indices = self._get_channel_indices(self.channel_groups.frontal)
        if frontal_indices:
            frontal_beta = np.mean(beta_power[frontal_indices])
            total_beta = np.mean(beta_power)
            frontal_beta_ratio = self._safe_ratio(frontal_beta, total_beta)
        else:
            frontal_beta_ratio = None

        return theta_alpha_ratio, frontal_beta_ratio

    def _compute_cognitive_load(self, theta_alpha_ratio: float,
                                 frontal_beta_ratio: float) -> float:
        """
        计算综合认知负荷水平

        基于 Theta/Alpha 比率和前额 Beta 活动的综合评估

        Args:
            theta_alpha_ratio: 全脑θ/α比率
            frontal_beta_ratio: 前额beta与全脑beta的比值

        Returns:
            认知负荷估计值 (0-1范围)

        公式: cognitive_load = sigmoid(w1 * theta_alpha_ratio + w2 * frontal_beta_ratio)
        """
        # 综合计算（简化模型）
        # 高 Theta/Alpha 和高前额 Beta 比值表示高认知负荷
        if theta_alpha_ratio is None or frontal_beta_ratio is None:
            return None
        raw_score = 0.6 * theta_alpha_ratio + 0.4 * frontal_beta_ratio

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
        alpha_delta_ratio = self._safe_ratio(total_alpha, total_delta)
        if alpha_delta_ratio is None:
            return None

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
        if total <= 0:
            return None

        relaxation = total_alpha / total
        if not np.isfinite(relaxation):
            return None
        # relaxation 理论上在 [0, 1] 范围内，裁剪以确保有效
        return float(np.clip(relaxation, 0, 1))

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
        """返回比值，裁剪到[0.01, 100]范围内，分母无效时返回None"""
        if denominator <= 0:
            return None
        val = numerator / denominator
        if not np.isfinite(val):
            return None
        # 裁剪到有效范围 [0.01, 100]
        return float(np.clip(val, 0.01, 100))
