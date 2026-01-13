"""
时域特征计算
"""
import numpy as np
from typing import Dict, Optional

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


@FeatureRegistry.register('time_domain')
class TimeDomainFeatures(BaseFeature):
    """时域特征计算"""

    feature_names = [
        'mean_abs_amplitude',
        'mean_channel_std',
        'mean_peak_to_peak',
        'mean_rms',
        'mean_zero_crossing_rate',
        'hjorth_activity',
        'hjorth_mobility',
        'hjorth_complexity',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_gpu = config.use_gpu and GPU_AVAILABLE

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算时域特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            psd_result: PSD 结果（时域特征不需要）

        Returns:
            时域特征字典
        """
        self._validate_input(eeg_data)

        if self.use_gpu:
            return self._compute_gpu(eeg_data)
        else:
            return self._compute_cpu(eeg_data)

    def _compute_cpu(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """CPU 计算"""
        features = {}

        # 1. 全通道平均幅值
        mean_amplitude = np.mean(np.abs(eeg_data))
        features['mean_abs_amplitude'] = float(mean_amplitude)

        # 2. 全通道标准差
        channel_stds = np.std(eeg_data, axis=1)
        mean_std = np.mean(channel_stds)
        features['mean_channel_std'] = float(mean_std)

        # 3. 全通道峰峰值
        peak_to_peak = np.max(eeg_data, axis=1) - np.min(eeg_data, axis=1)
        mean_ptp = np.mean(peak_to_peak)
        features['mean_peak_to_peak'] = float(mean_ptp)

        # 4. 全通道RMS能量
        rms = np.sqrt(np.mean(eeg_data ** 2, axis=1))
        mean_rms = np.mean(rms)
        features['mean_rms'] = float(mean_rms)

        # 5. 全通道零交叉率
        zero_crossings = self._compute_zero_crossing_rate_cpu(eeg_data)
        features['mean_zero_crossing_rate'] = float(zero_crossings)

        # 6-8. Hjorth 参数
        activity, mobility, complexity = self._compute_hjorth_params_cpu(eeg_data)
        features['hjorth_activity'] = float(activity)
        features['hjorth_mobility'] = float(mobility)
        features['hjorth_complexity'] = float(complexity)

        return features

    def _compute_gpu(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """GPU 加速计算"""
        features = {}

        # 转换到 GPU
        eeg_gpu = cp.asarray(eeg_data)

        # 1. 全通道平均幅值
        mean_amplitude = cp.mean(cp.abs(eeg_gpu))
        features['mean_abs_amplitude'] = float(cp.asnumpy(mean_amplitude))

        # 2. 全通道标准差
        channel_stds = cp.std(eeg_gpu, axis=1)
        mean_std = cp.mean(channel_stds)
        features['mean_channel_std'] = float(cp.asnumpy(mean_std))

        # 3. 全通道峰峰值
        peak_to_peak = cp.max(eeg_gpu, axis=1) - cp.min(eeg_gpu, axis=1)
        mean_ptp = cp.mean(peak_to_peak)
        features['mean_peak_to_peak'] = float(cp.asnumpy(mean_ptp))

        # 4. 全通道RMS能量
        rms = cp.sqrt(cp.mean(eeg_gpu ** 2, axis=1))
        mean_rms = cp.mean(rms)
        features['mean_rms'] = float(cp.asnumpy(mean_rms))

        # 5. 全通道零交叉率
        zero_crossings = self._compute_zero_crossing_rate_gpu(eeg_gpu)
        features['mean_zero_crossing_rate'] = float(cp.asnumpy(zero_crossings))

        # 6-8. Hjorth 参数
        activity, mobility, complexity = self._compute_hjorth_params_gpu(eeg_gpu)
        features['hjorth_activity'] = float(cp.asnumpy(activity))
        features['hjorth_mobility'] = float(cp.asnumpy(mobility))
        features['hjorth_complexity'] = float(cp.asnumpy(complexity))

        return features

    def _compute_zero_crossing_rate_cpu(self, eeg_data: np.ndarray) -> float:
        """CPU 计算零交叉率"""
        n_samples = eeg_data.shape[1]
        duration = n_samples / self.fs

        # 更稳健的零交叉率：
        # 1) 先把 0 用相邻非零符号填充，避免 0 造成重复计数
        # 2) 统计严格的符号变化（+1 -> -1 或 -1 -> +1）
        signs = np.sign(eeg_data)

        # 前向填充 0
        for ch in range(signs.shape[0]):
            s = signs[ch]
            for i in range(1, s.shape[0]):
                if s[i] == 0:
                    s[i] = s[i - 1]
            # 若开头仍是 0，则后向填充
            if s[0] == 0:
                for i in range(1, s.shape[0]):
                    if s[i] != 0:
                        s[:i] = s[i]
                        break

        sign_changes = np.sum((signs[:, :-1] * signs[:, 1:]) < 0, axis=1)
        zcr_per_sec = sign_changes / duration if duration > 0 else sign_changes
        return float(np.mean(zcr_per_sec))

    def _compute_zero_crossing_rate_gpu(self, eeg_gpu) -> float:
        """GPU 计算零交叉率 - 与 CPU 版本一致的实现"""
        n_samples = eeg_gpu.shape[1]
        duration = n_samples / self.fs

        # 转换到 CPU 以使用与 CPU 版本完全一致的零值填充逻辑
        eeg_data = cp.asnumpy(eeg_gpu)
        signs = np.sign(eeg_data)

        # 前向填充 0（与 CPU 版本一致）
        for ch in range(signs.shape[0]):
            s = signs[ch]
            for i in range(1, s.shape[0]):
                if s[i] == 0:
                    s[i] = s[i - 1]
            # 若开头仍是 0，则后向填充
            if s[0] == 0:
                for i in range(1, s.shape[0]):
                    if s[i] != 0:
                        s[:i] = s[i]
                        break

        sign_changes = np.sum((signs[:, :-1] * signs[:, 1:]) < 0, axis=1)
        zcr_per_sec = sign_changes / duration if duration > 0 else sign_changes
        return float(np.mean(zcr_per_sec))

    def _compute_hjorth_params_cpu(self, eeg_data: np.ndarray) -> tuple:
        """CPU 计算 Hjorth 参数"""
        # 一阶导数
        d1 = np.diff(eeg_data, axis=1)
        # 二阶导数
        d2 = np.diff(d1, axis=1)

        # 方差
        var_x = np.var(eeg_data, axis=1)
        var_d1 = np.var(d1, axis=1)
        var_d2 = np.var(d2, axis=1)

        # Activity: 信号方差的平均值
        activity = np.mean(var_x)

        # Mobility: 一阶导数标准差 / 信号标准差
        # 避免除以零
        std_x = np.sqrt(var_x)
        std_d1 = np.sqrt(var_d1)
        std_d2 = np.sqrt(var_d2)

        with np.errstate(divide='ignore', invalid='ignore'):
            mobility = np.where(std_x > 1e-10, std_d1 / std_x, 0)
            mobility_d1 = np.where(std_d1 > 1e-10, std_d2 / std_d1, 0)

        mean_mobility = np.mean(mobility)

        # Complexity: 二阶导数移动性 / 一阶导数移动性
        with np.errstate(divide='ignore', invalid='ignore'):
            complexity = np.where(mobility > 1e-10, mobility_d1 / mobility, 0)

        mean_complexity = np.mean(complexity)

        return activity, mean_mobility, mean_complexity

    def _compute_hjorth_params_gpu(self, eeg_gpu) -> tuple:
        """GPU 计算 Hjorth 参数"""
        # 一阶导数
        d1 = cp.diff(eeg_gpu, axis=1)
        # 二阶导数
        d2 = cp.diff(d1, axis=1)

        # 方差
        var_x = cp.var(eeg_gpu, axis=1)
        var_d1 = cp.var(d1, axis=1)
        var_d2 = cp.var(d2, axis=1)

        # Activity
        activity = cp.mean(var_x)

        # Mobility
        std_x = cp.sqrt(var_x)
        std_d1 = cp.sqrt(var_d1)
        std_d2 = cp.sqrt(var_d2)

        mobility = cp.where(std_x > 1e-10, std_d1 / std_x, 0)
        mobility_d1 = cp.where(std_d1 > 1e-10, std_d2 / std_d1, 0)

        mean_mobility = cp.mean(mobility)

        # Complexity
        complexity = cp.where(mobility > 1e-10, mobility_d1 / mobility, 0)
        mean_complexity = cp.mean(complexity)

        return activity, mean_mobility, mean_complexity
