"""
连接性特征计算：通道间相关性、相干性、半球间连接、相位锁定值(PLV)

优化：利用 psd_computer 的并行化 coherence 计算

新增特征：
- 相位锁定值 (PLV): 量化两个信号之间的相位同步程度
  PLV = 1 表示完全相位同步
  PLV = 0 表示完全随机相位关系
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.signal import butter, filtfilt, hilbert

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


def _compute_plv_pair(phase1: np.ndarray, phase2: np.ndarray) -> float:
    """
    计算两个信号的相位锁定值

    PLV = |mean(exp(i * (phase1 - phase2)))|

    Args:
        phase1: 信号1的瞬时相位
        phase2: 信号2的瞬时相位

    Returns:
        PLV值，范围[0, 1]
    """
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return float(plv)


def _bandpass_filter(signal: np.ndarray, fs: float,
                     band: Tuple[float, float], order: int = 4) -> Optional[np.ndarray]:
    """
    带通滤波

    Args:
        signal: 输入信号
        fs: 采样率
        band: 频段范围 (low, high)
        order: 滤波器阶数

    Returns:
        滤波后的信号，失败时返回 None
    """
    nyq = fs / 2
    low = band[0] / nyq
    high = band[1] / nyq

    # 确保频率在有效范围内
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        return None

    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal, axis=-1, padlen=min(len(signal) - 1, 3 * max(len(a), len(b))))
        return filtered
    except Exception:
        return None


def _compute_instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    计算瞬时相位（通过Hilbert变换）

    Args:
        signal: 输入信号

    Returns:
        瞬时相位数组
    """
    analytic = hilbert(signal)
    return np.angle(analytic)


def compute_plv_matrix(eeg_data: np.ndarray, fs: float,
                       freq_band: Tuple[float, float],
                       filter_order: int = 4) -> Optional[np.ndarray]:
    """
    计算多通道EEG的PLV连接矩阵（向量化优化版本）

    Args:
        eeg_data: EEG数据, shape: (n_channels, n_timepoints)
        fs: 采样率
        freq_band: 频段范围
        filter_order: 滤波器阶数

    Returns:
        PLV矩阵, shape: (n_channels, n_channels)，对称矩阵；滤波失败返回 None
    """
    n_channels = eeg_data.shape[0]

    # 对所有通道进行带通滤波（向量化）
    filtered_data = []
    valid_channels = []
    for ch in range(n_channels):
        filtered = _bandpass_filter(eeg_data[ch], fs, freq_band, filter_order)
        if filtered is not None:
            filtered_data.append(filtered)
            valid_channels.append(ch)

    # 如果滤波全部失败，返回 None
    if len(filtered_data) < 2:
        return None

    filtered_data = np.array(filtered_data)

    # 向量化计算所有通道的瞬时相位（Hilbert 变换支持批量处理）
    analytic = hilbert(filtered_data, axis=1)
    phases = np.angle(analytic)  # shape: (n_valid_channels, n_timepoints)

    n_valid = len(valid_channels)

    # 向量化计算 PLV 矩阵
    # PLV_ij = |mean(exp(i * (phase_i - phase_j)))|
    # 使用广播和向量化计算

    # 计算相位差矩阵：phases[:, None, :] - phases[None, :, :]
    # shape: (n_valid, n_valid, n_timepoints)
    phase_diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]

    # 计算复指数的平均值
    # exp(i * phase_diff) 并沿时间轴取平均
    plv_valid = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))

    # 构建完整的 PLV 矩阵（包含所有原始通道）
    plv_matrix = np.eye(n_channels)  # 对角线为 1
    for i_idx, i_ch in enumerate(valid_channels):
        for j_idx, j_ch in enumerate(valid_channels):
            if i_ch != j_ch:
                plv_matrix[i_ch, j_ch] = plv_valid[i_idx, j_idx]

    return plv_matrix


@FeatureRegistry.register('connectivity')
class ConnectivityFeatures(BaseFeature):
    """连接性特征计算

    包含：
    - 通道间相关性
    - Alpha频段相干性
    - 半球间连接强度
    - 频带间功率相关性
    - 半球Alpha不对称性
    - 前后脑区功率梯度
    - 相位锁定值 (PLV) - 各频段
    """

    feature_names = [
        'mean_interchannel_correlation',
        'mean_alpha_coherence',
        'interhemispheric_alpha_coherence',
        'alpha_beta_band_power_correlation',
        'hemispheric_alpha_asymmetry',
        'frontal_occipital_alpha_ratio',
        # PLV特征
        'plv_theta_mean',
        'plv_alpha_mean',
        'plv_beta_mean',
        'plv_gamma_mean',
        'plv_theta_interhemispheric',
        'plv_alpha_interhemispheric',
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
            # 即使为None也要写入，确保CSV列一致
            features['frontal_occipital_alpha_ratio'] = gradient
        else:
            features['frontal_occipital_alpha_ratio'] = None

        # 7-12. PLV特征
        plv_features = self._compute_plv_features(eeg_data)
        features.update(plv_features)

        return features

    def _compute_plv_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        计算相位锁定值(PLV)特征

        Args:
            eeg_data: EEG数据

        Returns:
            PLV特征字典
        """
        features = {}
        freq_bands = self.config.freq_bands

        # 定义要计算PLV的频段
        bands = {
            'theta': freq_bands.theta,
            'alpha': freq_bands.alpha,
            'beta': freq_bands.beta,
            'gamma': freq_bands.gamma,
        }

        # 缓存计算过的 PLV 矩阵（避免重复计算）
        plv_cache = {}

        # 计算各频段的全脑平均PLV
        for band_name, band_range in bands.items():
            try:
                plv_matrix = compute_plv_matrix(eeg_data, self.fs, band_range)
                if plv_matrix is None:
                    features[f'plv_{band_name}_mean'] = None
                    continue
                plv_cache[band_name] = plv_matrix
                # 提取上三角（不包括对角线）
                upper_tri = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
                mean_plv = np.nanmean(upper_tri) if upper_tri.size else None
                features[f'plv_{band_name}_mean'] = float(mean_plv) if mean_plv is not None else None
            except Exception:
                features[f'plv_{band_name}_mean'] = None

        # 计算半球间PLV（theta和alpha）
        for band_name in ['theta', 'alpha']:
            try:
                # 使用缓存的 PLV 矩阵
                if band_name in plv_cache:
                    plv_matrix = plv_cache[band_name]
                else:
                    plv_matrix = compute_plv_matrix(eeg_data, self.fs, bands[band_name])

                if plv_matrix is None:
                    features[f'plv_{band_name}_interhemispheric'] = None
                    continue

                inter_plv = self._compute_interhemispheric_plv(plv_matrix)
                features[f'plv_{band_name}_interhemispheric'] = float(inter_plv) if inter_plv else None
            except Exception:
                features[f'plv_{band_name}_interhemispheric'] = None

        return features

    def _compute_interhemispheric_plv(self, plv_matrix: np.ndarray) -> float:
        """
        计算半球间平均PLV

        Args:
            plv_matrix: PLV矩阵

        Returns:
            半球间平均PLV
        """
        left_indices = self._get_channel_indices(self.channel_groups.left_hemisphere)
        right_indices = self._get_channel_indices(self.channel_groups.right_hemisphere)

        if not left_indices or not right_indices:
            return 0.0

        plv_values = []
        for l_idx in left_indices:
            for r_idx in right_indices:
                if l_idx < plv_matrix.shape[0] and r_idx < plv_matrix.shape[1]:
                    plv_values.append(plv_matrix[l_idx, r_idx])

        return float(np.mean(plv_values)) if plv_values else 0.0

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

    def _compute_ap_gradient(self, psd_result: PSDResult) -> Optional[float]:
        """
        计算前后脑区功率梯度

        前额叶 Alpha 功率 / 枕叶 Alpha 功率
        """
        alpha_power = psd_result.band_power.get('alpha', np.zeros(self.config.n_channels))

        frontal_indices = self._get_channel_indices(self.channel_groups.frontal)
        occipital_indices = self._get_channel_indices(self.channel_groups.occipital)

        if not frontal_indices or not occipital_indices:
            return None

        frontal_power = np.mean(alpha_power[frontal_indices])
        occipital_power = np.mean(alpha_power[occipital_indices])

        return self._safe_ratio(frontal_power, occipital_power)

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
