"""
微分熵 (Differential Entropy, DE) 相关特征计算

包含特征：
- 各频段的微分熵 (DE)
- 差分不对称性 (DASM): DE(Left) - DE(Right)
- 有理不对称性 (RASM): DE(Left) / DE(Right)
- 差分尾部性 (DCAU): DE(Frontal) - DE(Posterior)
- 额叶Alpha不对称性 (FAA): ln(P_α,Right) - ln(P_α,Left)

参考文献：
- Shi, L. C., et al. (2013). Differential entropy feature for EEG-based emotion classification.
- Zheng, W. L., & Lu, B. L. (2015). Investigating critical frequency bands and channels
  for EEG-based emotion recognition with deep neural networks.
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.signal import welch
from scipy.integrate import trapezoid

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult
from ..config import Config


# 14对对称电极配对（用于DASM和RASM计算）
SYMMETRIC_PAIRS = [
    ('FP1', 'FP2'),
    ('F7', 'F8'),
    ('F3', 'F4'),
    ('T7', 'T8'),
    ('P7', 'P8'),
    ('C3', 'C4'),
    ('P3', 'P4'),
    ('O1', 'O2'),
    ('AF3', 'AF4'),
    ('FC5', 'FC6'),
    ('FC1', 'FC2'),
    ('CP5', 'CP6'),
    ('CP1', 'CP2'),
    ('PO3', 'PO4'),
]

# 11对前后电极配对（用于DCAU计算）
FRONTAL_POSTERIOR_PAIRS = [
    ('FC5', 'CP5'),
    ('FC1', 'CP1'),
    ('FC2', 'CP2'),
    ('FC6', 'CP6'),
    ('F7', 'P7'),
    ('F3', 'P3'),
    ('FZ', 'PZ'),
    ('F4', 'P4'),
    ('F8', 'P8'),
    ('FP1', 'O1'),
    ('FP2', 'O2'),
]

# FAA电极配对
FAA_PAIRS = [
    ('F3', 'F4'),    # 额叶中部
    ('F7', 'F8'),    # 额叶外侧
    ('FP1', 'FP2'),  # 前额极
    ('AF3', 'AF4'),  # 前额叶（扩展系统）
]


def compute_de(variance: float) -> float:
    """
    计算微分熵

    假设 EEG 信号服从高斯分布 N(μ, σ²)，DE 计算公式为：
    h(X) = 0.5 * log(2 * π * e * σ²)

    Args:
        variance: 信号方差（σ²）

    Returns:
        微分熵值
    """
    if variance <= 0:
        return 0.0
    return 0.5 * np.log(2 * np.pi * np.e * variance)


def compute_band_variance(signal: np.ndarray, fs: float,
                          band: Tuple[float, float]) -> float:
    """
    计算特定频段的信号方差（使用Welch方法）

    Args:
        signal: 1D信号数组
        fs: 采样率
        band: 频段范围 (low, high)

    Returns:
        该频段的功率（方差近似）
    """
    nperseg = min(len(signal), int(2 * fs))  # 2秒窗口或信号长度
    noverlap = nperseg // 2

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # 提取频段
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0

    # 计算频段功率（积分PSD）
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    band_power = trapezoid(psd[band_mask], dx=freq_resolution)

    return max(band_power, 1e-15)


@FeatureRegistry.register('de_features')
class DEFeatures(BaseFeature):
    """微分熵相关特征计算"""

    feature_names = [
        # 各频段微分熵（全脑平均）
        'de_delta',
        'de_theta',
        'de_alpha',
        'de_beta',
        'de_gamma',
        'de_low_gamma',
        'de_high_gamma',

        # 差分不对称性 (DASM) - 各频段
        'dasm_delta',
        'dasm_theta',
        'dasm_alpha',
        'dasm_beta',
        'dasm_gamma',

        # 有理不对称性 (RASM) - 各频段
        'rasm_delta',
        'rasm_theta',
        'rasm_alpha',
        'rasm_beta',
        'rasm_gamma',

        # 差分尾部性 (DCAU) - 各频段
        'dcau_delta',
        'dcau_theta',
        'dcau_alpha',
        'dcau_beta',
        'dcau_gamma',

        # 额叶Alpha不对称性 (FAA)
        'faa_f3f4',
        'faa_f7f8',
        'faa_fp1fp2',
        'faa_mean',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.channel_names = config.channel_names
        self.freq_bands = config.freq_bands

        # 构建通道名到索引的映射
        self._channel_map = {ch.upper(): idx for idx, ch in enumerate(self.channel_names)}

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算所有DE相关特征

        Args:
            eeg_data: EEG数据, shape: (n_channels, n_timepoints)
            psd_result: PSD结果（可选，用于FAA计算）

        Returns:
            特征字典
        """
        self._validate_input(eeg_data)

        features = {}

        # 定义频段
        bands = {
            'delta': self.freq_bands.delta,
            'theta': self.freq_bands.theta,
            'alpha': self.freq_bands.alpha,
            'beta': self.freq_bands.beta,
            'gamma': self.freq_bands.gamma,
            'low_gamma': self.freq_bands.low_gamma,
            'high_gamma': self.freq_bands.high_gamma,
        }

        # 1. 计算每个通道每个频段的DE
        de_matrix = self._compute_de_matrix(eeg_data, bands)

        # 2. 全脑平均DE
        for band_name in bands.keys():
            if band_name in de_matrix:
                features[f'de_{band_name}'] = float(np.mean(de_matrix[band_name]))

        # 3. DASM (差分不对称性)
        primary_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        dasm_features = self._compute_dasm(de_matrix, primary_bands)
        features.update(dasm_features)

        # 4. RASM (有理不对称性)
        rasm_features = self._compute_rasm(de_matrix, primary_bands)
        features.update(rasm_features)

        # 5. DCAU (差分尾部性)
        dcau_features = self._compute_dcau(de_matrix, primary_bands)
        features.update(dcau_features)

        # 6. FAA (额叶Alpha不对称性)
        faa_features = self._compute_faa(eeg_data, psd_result)
        features.update(faa_features)

        return features

    def _get_channel_idx(self, channel_name: str) -> Optional[int]:
        """获取通道索引"""
        return self._channel_map.get(channel_name.upper())

    def _compute_de_matrix(self, eeg_data: np.ndarray,
                           bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        计算每个通道每个频段的DE

        Args:
            eeg_data: EEG数据
            bands: 频段字典

        Returns:
            {band_name: de_array} 其中 de_array.shape = (n_channels,)
        """
        n_channels = eeg_data.shape[0]
        de_matrix = {}

        for band_name, band_range in bands.items():
            de_values = []
            for ch in range(n_channels):
                # 计算该频段的方差/功率
                variance = compute_band_variance(eeg_data[ch], self.fs, band_range)
                # 计算DE
                de = compute_de(variance)
                de_values.append(de)
            de_matrix[band_name] = np.array(de_values)

        return de_matrix

    def _compute_dasm(self, de_matrix: Dict[str, np.ndarray],
                      bands: List[str]) -> Dict[str, float]:
        """
        计算差分不对称性 (DASM)

        DASM = DE(Left) - DE(Right)
        仅使用数据中存在的对称电极对
        """
        features = {}

        for band in bands:
            if band not in de_matrix:
                features[f'dasm_{band}'] = None
                continue

            de_values = de_matrix[band]
            dasm_values = []

            for left_ch, right_ch in SYMMETRIC_PAIRS:
                left_idx = self._get_channel_idx(left_ch)
                right_idx = self._get_channel_idx(right_ch)

                # 只使用数据中存在的电极对
                if left_idx is not None and right_idx is not None:
                    dasm = de_values[left_idx] - de_values[right_idx]
                    if np.isfinite(dasm):
                        dasm_values.append(dasm)

            if dasm_values:
                features[f'dasm_{band}'] = float(np.mean(dasm_values))
            else:
                features[f'dasm_{band}'] = None

        return features

    def _compute_rasm(self, de_matrix: Dict[str, np.ndarray],
                      bands: List[str]) -> Dict[str, float]:
        """
        计算有理不对称性 (RASM)

        RASM = DE(Left) / DE(Right)
        仅使用数据中存在的对称电极对，当 DE 为负值或比值超出范围时返回 None
        """
        features = {}

        for band in bands:
            if band not in de_matrix:
                features[f'rasm_{band}'] = None
                continue

            de_values = de_matrix[band]
            rasm_values = []

            for left_ch, right_ch in SYMMETRIC_PAIRS:
                left_idx = self._get_channel_idx(left_ch)
                right_idx = self._get_channel_idx(right_ch)

                # 只使用数据中存在的电极对
                if left_idx is not None and right_idx is not None:
                    de_left = de_values[left_idx]
                    de_right = de_values[right_idx]

                    # RASM 要求两个 DE 值都为正（DE 可能为负）
                    if de_left > 0 and de_right > 0:
                        rasm = self._safe_ratio(de_left, de_right)
                        if rasm is not None:
                            rasm_values.append(rasm)

            if rasm_values:
                features[f'rasm_{band}'] = float(np.mean(rasm_values))
            else:
                features[f'rasm_{band}'] = None

        return features

    def _compute_dcau(self, de_matrix: Dict[str, np.ndarray],
                      bands: List[str]) -> Dict[str, float]:
        """
        计算差分尾部性 (DCAU)

        DCAU = DE(Frontal) - DE(Posterior)
        仅使用数据中存在的前后电极对
        """
        features = {}

        for band in bands:
            if band not in de_matrix:
                features[f'dcau_{band}'] = None
                continue

            de_values = de_matrix[band]
            dcau_values = []

            for frontal_ch, posterior_ch in FRONTAL_POSTERIOR_PAIRS:
                frontal_idx = self._get_channel_idx(frontal_ch)
                posterior_idx = self._get_channel_idx(posterior_ch)

                # 只使用数据中存在的电极对
                if frontal_idx is not None and posterior_idx is not None:
                    dcau = de_values[frontal_idx] - de_values[posterior_idx]
                    if np.isfinite(dcau):
                        dcau_values.append(dcau)

            if dcau_values:
                features[f'dcau_{band}'] = float(np.mean(dcau_values))
            else:
                features[f'dcau_{band}'] = None

        return features

    def _compute_faa(self, eeg_data: np.ndarray,
                     psd_result: Optional[PSDResult] = None) -> Dict[str, float]:
        """
        计算额叶Alpha不对称性 (FAA)

        FAA = ln(P_α,Right) - ln(P_α,Left)

        正值表示左半球相对激活更强（趋近/积极情绪）
        负值表示右半球相对激活更强（回避/消极情绪）
        """
        features = {}
        alpha_band = self.freq_bands.alpha

        faa_values = []
        faa_names = ['f3f4', 'f7f8', 'fp1fp2', 'af3af4']

        # 初始化所有 FAA 特征为 None，确保 CSV 列一致
        for i in range(3):
            features[f'faa_{faa_names[i]}'] = None

        for i, (left_ch, right_ch) in enumerate(FAA_PAIRS):
            left_idx = self._get_channel_idx(left_ch)
            right_idx = self._get_channel_idx(right_ch)

            if left_idx is not None and right_idx is not None:
                # 计算alpha功率
                if psd_result is not None:
                    # 使用预计算的PSD结果
                    alpha_power = psd_result.band_power.get('alpha', None)
                    if alpha_power is not None:
                        p_left = alpha_power[left_idx]
                        p_right = alpha_power[right_idx]
                    else:
                        p_left = compute_band_variance(eeg_data[left_idx], self.fs, alpha_band)
                        p_right = compute_band_variance(eeg_data[right_idx], self.fs, alpha_band)
                else:
                    # 直接从信号计算
                    p_left = compute_band_variance(eeg_data[left_idx], self.fs, alpha_band)
                    p_right = compute_band_variance(eeg_data[right_idx], self.fs, alpha_band)

                # 避免log(0)
                p_left = max(p_left, 1e-15)
                p_right = max(p_right, 1e-15)

                # FAA = ln(Right) - ln(Left)
                faa = np.log(p_right) - np.log(p_left)
                faa_values.append(faa)

                # 只记录前三对（标准配对）
                if i < 3:
                    features[f'faa_{faa_names[i]}'] = float(faa)

        # 平均FAA - 即使为None也要写入
        features['faa_mean'] = float(np.mean(faa_values)) if faa_values else None

        return features

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
