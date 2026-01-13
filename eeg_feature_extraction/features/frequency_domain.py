"""
频域特征计算
"""
import numpy as np
from typing import Dict, Optional
import warnings
from scipy.stats import entropy
from scipy.integrate import trapezoid

# 尝试导入 GPU 加速库
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult
from ..config import Config


# 可选：FOOOF 用于拟合 aperiodic (1/f) 成分
try:
    from fooof import FOOOFGroup
except Exception:  # pragma: no cover
    FOOOFGroup = None


@FeatureRegistry.register('frequency_domain')
class FrequencyDomainFeatures(BaseFeature):
    """频域特征计算

    频段范围:
    - delta: 0.5-4 Hz
    - theta: 4-8 Hz
    - alpha: 8-12 Hz
    - beta: 12-30 Hz
    - low_gamma: 30-50 Hz
    - high_gamma: 50-80 Hz
    - gamma (完整): 30-80 Hz
    """

    feature_names = [
        'delta_power',
        'theta_power',
        'alpha_power',
        'beta_power',
        'gamma_power',
        'low_gamma_power',
        'high_gamma_power',
        'delta_relative_power',
        'theta_relative_power',
        'alpha_relative_power',
        'beta_relative_power',
        'gamma_relative_power',
        'low_gamma_relative_power',
        'high_gamma_relative_power',
        'peak_frequency',
        'spectral_entropy',
        'spectral_centroid',
        'individual_alpha_frequency',
        'theta_beta_ratio',
        'delta_theta_ratio',
        'low_high_power_ratio',
        'aperiodic_exponent',
        'mean_total_power',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.bands = config.freq_bands.get_all_bands()
        self.normalize_spectral_entropy = bool(getattr(config, 'spectral_entropy_normalize', False))

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算频域特征

        Args:
            eeg_data: EEG 数据
            psd_result: 预先计算的 PSD 结果（必须提供）

        Returns:
            频域特征字典
        """
        self._validate_input(eeg_data)

        if psd_result is None:
            raise ValueError("频域特征计算需要预先提供 PSD 结果")

        features = {}

        # 获取各频段功率（每通道）
        band_power = psd_result.band_power
        total_power = psd_result.total_power

        # 1-7. 各频段绝对功率（全通道平均）
        all_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'low_gamma', 'high_gamma']
        for band_name in all_bands:
            power = band_power.get(band_name, np.zeros(self.config.n_channels))
            features[f'{band_name}_power'] = float(np.mean(power))

        # 8-14. 各频段相对功率
        for band_name in all_bands:
            power = band_power.get(band_name, np.zeros(self.config.n_channels))
            rel_power = self._safe_ratio_array(power, total_power)
            # 即使为None也要写入，确保CSV列一致
            features[f'{band_name}_relative_power'] = rel_power

        # 11. 主频率峰值
        peak_freq = self._compute_peak_frequency(psd_result.freqs, psd_result.psd)
        features['peak_frequency'] = float(peak_freq)

        # 12. 频谱熵
        spectral_entropy = self._compute_spectral_entropy(psd_result.psd)
        features['spectral_entropy'] = float(spectral_entropy)

        # 13. 频谱质心
        spectral_centroid = self._compute_spectral_centroid(
            psd_result.freqs, psd_result.psd
        )
        features['spectral_centroid'] = float(spectral_centroid)

        # 14. 个体Alpha频率 (IAF)
        iaf = self._compute_iaf(psd_result.freqs, psd_result.psd)
        features['individual_alpha_frequency'] = float(iaf)

        # 15. Theta-Beta比率
        theta_power = np.mean(band_power.get('theta', np.zeros(1)))
        beta_power = np.mean(band_power.get('beta', np.zeros(1)))
        tbr = self._safe_ratio(theta_power, beta_power)
        # 即使为None也要写入，确保CSV列一致
        features['theta_beta_ratio'] = tbr

        # 16. Delta-Theta比率
        delta_power = np.mean(band_power.get('delta', np.zeros(1)))
        dtr = self._safe_ratio(delta_power, theta_power)
        # 即使为None也要写入，确保CSV列一致
        features['delta_theta_ratio'] = dtr

        # 17. 低频vs高频能量比
        low_high_ratio = self._compute_low_high_ratio(psd_result)
        # 即使为None也要写入，确保CSV列一致
        features['low_high_power_ratio'] = low_high_ratio

        # 18. 非周期性指数（1/f 斜率）
        aperiodic_exp = self._compute_aperiodic_exponent(
            psd_result.freqs, psd_result.psd
        )
        features['aperiodic_exponent'] = float(aperiodic_exp)

        # 19. 总平均功率
        features['mean_total_power'] = float(np.mean(total_power))

        return features

    def _compute_peak_frequency(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """计算主频率峰值"""
        # 在指定频段内找峰值，避免 DC/边界噪声影响
        band_mask = (freqs >= 0.5) & (freqs <= 100.0)
        if not np.any(band_mask):
            peak_indices = np.argmax(psd, axis=1)
            return float(np.mean(freqs[peak_indices]))

        freqs_band = freqs[band_mask]
        psd_band = psd[:, band_mask]
        peak_indices = np.argmax(psd_band, axis=1)
        peak_freqs = freqs_band[peak_indices]
        return float(np.mean(peak_freqs))

    def _compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """计算频谱熵"""
        # 对每个通道计算归一化PSD的熵
        entropies = []
        for ch_psd in psd:
            # 归一化为概率分布
            psd_norm = ch_psd / (np.sum(ch_psd) + 1e-10)
            psd_norm = psd_norm[psd_norm > 0]  # 避免 log(0)
            if len(psd_norm) > 0:
                ent = float(entropy(psd_norm))
                if self.normalize_spectral_entropy and len(psd_norm) > 1:
                    # entropy 使用自然对数，最大值为 ln(N)
                    ent = ent / float(np.log(len(psd_norm)))
                entropies.append(ent)

        return float(np.mean(entropies)) if entropies else 0.0

    def _compute_spectral_centroid(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """计算频谱质心"""
        centroids = []
        for ch_psd in psd:
            total_power = np.sum(ch_psd)
            if total_power > 1e-10:
                centroid = np.sum(freqs * ch_psd) / total_power
                centroids.append(centroid)

        return np.mean(centroids) if centroids else 0.0

    def _compute_iaf(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """计算个体Alpha频率"""
        # 在 Alpha 频段 (8-13 Hz) 内找峰值
        alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
        alpha_freqs = freqs[alpha_mask]
        alpha_psd = psd[:, alpha_mask]

        if len(alpha_freqs) == 0:
            return 10.0  # 默认值

        # 对每个通道找峰值
        peak_indices = np.argmax(alpha_psd, axis=1)
        peak_freqs = alpha_freqs[peak_indices]
        return np.mean(peak_freqs)

    def _compute_low_high_ratio(self, psd_result: PSDResult) -> float:
        """计算低频/高频能量比"""
        freqs = psd_result.freqs
        psd = psd_result.psd

        # 低频: 1-8 Hz
        low_mask = (freqs >= 1.0) & (freqs < 8.0)
        # 高频: 13-40 Hz
        high_mask = (freqs >= 13.0) & (freqs < 40.0)

        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        if np.any(low_mask) and np.any(high_mask):
            low_power = trapezoid(psd[:, low_mask], dx=freq_resolution, axis=1)
            high_power = trapezoid(psd[:, high_mask], dx=freq_resolution, axis=1)
            ratio = self._safe_ratio_array(low_power, high_power)
            return ratio

        return None

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

    def _safe_ratio_array(self, numerator: np.ndarray, denominator: np.ndarray) -> Optional[float]:
        """对向量比值做有效性检查，裁剪到[0.01, 100]范围，返回均值或None"""
        valid_mask = denominator > 0
        if not np.any(valid_mask):
            return None
        ratios = numerator[valid_mask] / denominator[valid_mask]
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size == 0:
            return None
        # 裁剪到有效范围 [0.01, 100]
        ratios = np.clip(ratios, 0.01, 100)
        return float(np.mean(ratios))

    def _compute_aperiodic_exponent(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """
        计算非周期性指数（1/f 斜率）

        优先使用 FOOOF 拟合 aperiodic 成分（推荐），回退到对数线性拟合。
        """
        freq_range = (2.0, 40.0)
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_fit = freqs[freq_mask]
        psd_fit = psd[:, freq_mask]

        if len(freqs_fit) < 3:
            return 1.0

        # 1) FOOOF 拟合（批量）
        if FOOOFGroup is not None:
            # FOOOF 要求功率谱为正值
            psd_fit_pos = np.maximum(psd_fit, 1e-16)
            try:
                fg = FOOOFGroup(
                    aperiodic_mode='fixed',
                    max_n_peaks=3,
                    peak_width_limits=(1, 12),
                    peak_threshold=2.0,
                    verbose=False,
                )
                fg.fit(freqs=freqs_fit, power_spectra=psd_fit_pos, freq_range=freq_range)

                exponents = []
                for idx in range(psd_fit_pos.shape[0]):
                    try:
                        fm = fg.get_fooof(ind=idx, regenerate=True)
                        exp = fm.get_params('aperiodic_params', 'exponent')
                        if np.isfinite(exp):
                            exponents.append(float(exp))
                    except Exception:
                        continue

                if exponents:
                    return float(np.mean(exponents))
            except Exception as e:
                warnings.warn(f"FOOOF 拟合失败，回退到线性拟合: {e}")

        # 2) 回退：对数线性拟合 log10(PSD) = -exponent * log10(freq) + offset
        exponents = []
        for ch_psd in psd_fit:
            valid_mask = ch_psd > 1e-10
            if np.sum(valid_mask) < 3:
                continue

            log_freq = np.log10(freqs_fit[valid_mask])
            log_psd = np.log10(ch_psd[valid_mask])

            try:
                coeffs = np.polyfit(log_freq, log_psd, 1)
                exponents.append(float(-coeffs[0]))
            except np.linalg.LinAlgError:
                continue

        return float(np.mean(exponents)) if exponents else 1.0
