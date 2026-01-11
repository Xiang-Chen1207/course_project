"""
复杂度特征计算：样本熵、近似熵、Hurst指数、小波能量熵

优化：使用多进程并行化通道计算
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.stats import entropy
import pywt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

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


def _compute_single_channel_entropy(args: Tuple) -> Dict[str, float]:
    """
    计算单个通道的熵值（用于并行处理）

    Args:
        args: (ch_data, m, r_ratio, wavelet, wavelet_level)

    Returns:
        包含各种熵值的字典
    """
    ch_data, m, r_ratio, wavelet, wavelet_level = args
    results = {
        'wavelet_entropy': None,
        'sample_entropy': None,
        'approx_entropy': None,
        'hurst': None
    }

    # 小波能量熵
    try:
        coeffs = pywt.wavedec(ch_data, wavelet, level=wavelet_level)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        total_energy = np.sum(energies)
        if total_energy > 1e-10:
            probs = energies / total_energy
            probs = probs[probs > 0]
            results['wavelet_entropy'] = entropy(probs)
    except Exception:
        pass

    std = np.std(ch_data)
    if std > 1e-10:
        r = r_ratio * std

        # 样本熵
        try:
            results['sample_entropy'] = _sample_entropy_single_optimized(ch_data, m, r)
        except Exception:
            pass

        # 近似熵
        try:
            results['approx_entropy'] = _approx_entropy_single_optimized(ch_data, m, r)
        except Exception:
            pass

    # Hurst 指数
    try:
        results['hurst'] = _hurst_rs_optimized(ch_data)
    except Exception:
        pass

    return results


def _sample_entropy_single_optimized(signal: np.ndarray, m: int, r: float) -> float:
    """优化的单通道样本熵计算"""
    N = len(signal)
    if N < m + 2:
        return 0.0

    # 正确的样本熵：
    # SampEn = -ln(A/B)
    # 其中 B 是 m 维模板的匹配对数（i<j），A 是 m+1 维模板的匹配对数（i<j）。
    # 这里使用简单但正确的实现，避免自匹配/重复计数导致偏差。

    def embed(x: np.ndarray, dim: int) -> np.ndarray:
        n = len(x) - dim + 1
        if n <= 0:
            return np.empty((0, dim), dtype=x.dtype)
        return np.stack([x[i:i + dim] for i in range(n)], axis=0)

    def count_pairs(templates: np.ndarray, tol: float) -> int:
        n_t = templates.shape[0]
        if n_t < 2:
            return 0
        cnt = 0
        for i in range(n_t - 1):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            cnt += int(np.sum(diffs < tol))
        return cnt

    templates_m = embed(signal, m)
    templates_m1 = embed(signal, m + 1)

    B = count_pairs(templates_m, r)
    A = count_pairs(templates_m1, r)

    if B <= 0 or A <= 0:
        return 0.0

    return float(-np.log(A / B))


def _approx_entropy_single_optimized(signal: np.ndarray, m: int, r: float) -> float:
    """优化的单通道近似熵计算"""
    N = len(signal)
    if N < m + 2:
        return 0.0

    def phi(dim):
        n = N - dim + 1
        templates = np.array([signal[i:i + dim] for i in range(n)])

        # 向量化计算每个模板的匹配数
        counts = np.zeros(n)
        for i in range(n):
            distances = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(distances < r)

        counts = counts / n
        counts = counts[counts > 0]
        return np.mean(np.log(counts)) if len(counts) > 0 else 0.0

    phi_m = phi(m)
    phi_m1 = phi(m + 1)

    return phi_m - phi_m1


def _hurst_rs_optimized(signal: np.ndarray) -> float:
    """优化的 Hurst 指数计算"""
    N = len(signal)
    if N < 20:
        return 0.5

    max_k = int(np.log2(N)) - 1
    n_values = [int(2 ** k) for k in range(4, max_k + 1)]
    n_values = [n for n in n_values if n >= 8 and N // n >= 2]

    if len(n_values) < 2:
        return 0.5

    rs_values = []

    for n in n_values:
        num_parts = N // n
        rs_list = []

        for part in range(num_parts):
            segment = signal[part * n:(part + 1) * n]
            mean = np.mean(segment)
            y = np.cumsum(segment - mean)
            R = np.max(y) - np.min(y)
            S = np.std(segment, ddof=1)

            if S > 1e-10:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append((n, np.mean(rs_list)))

    if len(rs_values) < 2:
        return 0.5

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        return coeffs[0]
    except np.linalg.LinAlgError:
        return 0.5


@FeatureRegistry.register('complexity')
class ComplexityFeatures(BaseFeature):
    """复杂度特征计算"""

    feature_names = [
        'wavelet_energy_entropy',
        'sample_entropy',
        'approx_entropy',
        'hurst_exponent',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_gpu = config.use_gpu and GPU_AVAILABLE
        self.m = config.sample_entropy_m
        self.r_ratio = config.sample_entropy_r_ratio
        self.wavelet = config.wavelet
        self.wavelet_level = config.wavelet_level
        # 为避免与 segment 级多进程叠加导致过度并行，默认关闭内部并行
        self.n_workers = int(getattr(config, 'complexity_n_workers', 1))
        if self.n_workers < 1:
            self.n_workers = 1

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算复杂度特征（多进程并行优化版本）

        Args:
            eeg_data: EEG 数据
            psd_result: PSD 结果（复杂度特征不需要）

        Returns:
            复杂度特征字典
        """
        self._validate_input(eeg_data)

        n_channels = eeg_data.shape[0]

        # 准备并行任务参数
        tasks = [
            (eeg_data[ch], self.m, self.r_ratio, self.wavelet, self.wavelet_level)
            for ch in range(n_channels)
        ]

        # 使用线程池并行计算（因为进程池在子进程中可能有问题）
        # 对于 CPU 密集型任务，线程池由于 GIL 限制效率较低
        # 但在已经是并行处理 segments 的情况下，这里使用简单的串行处理
        # 主要的并行化在 segment 级别

        wavelet_entropies = []
        sample_entropies = []
        approx_entropies = []
        hurst_values = []

        # 可选线程池：默认 n_workers=1（串行），避免与外层多进程叠加
        if self.n_workers == 1:
            results = [_compute_single_channel_entropy(t) for t in tasks]
        else:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(_compute_single_channel_entropy, tasks))

        for result in results:
            if result['wavelet_entropy'] is not None:
                wavelet_entropies.append(result['wavelet_entropy'])
            if result['sample_entropy'] is not None and np.isfinite(result['sample_entropy']):
                sample_entropies.append(result['sample_entropy'])
            if result['approx_entropy'] is not None and np.isfinite(result['approx_entropy']):
                approx_entropies.append(result['approx_entropy'])
            if result['hurst'] is not None and np.isfinite(result['hurst']) and 0 < result['hurst'] < 1:
                hurst_values.append(result['hurst'])

        features = {
            'wavelet_energy_entropy': float(np.mean(wavelet_entropies)) if wavelet_entropies else 0.0,
            'sample_entropy': float(np.mean(sample_entropies)) if sample_entropies else 0.0,
            'approx_entropy': float(np.mean(approx_entropies)) if approx_entropies else 0.0,
            'hurst_exponent': float(np.mean(hurst_values)) if hurst_values else 0.5,
        }

        return features

    def _compute_wavelet_entropy(self, eeg_data: np.ndarray) -> float:
        """
        计算小波能量熵

        使用 db4 小波进行 5 层分解，计算各层能量的熵
        """
        entropies = []

        for ch_data in eeg_data:
            try:
                # 小波分解
                coeffs = pywt.wavedec(ch_data, self.wavelet, level=self.wavelet_level)

                # 计算各层能量
                energies = []
                for c in coeffs:
                    energy = np.sum(c ** 2)
                    energies.append(energy)

                energies = np.array(energies)
                total_energy = np.sum(energies)

                if total_energy > 1e-10:
                    # 归一化为概率分布
                    probs = energies / total_energy
                    probs = probs[probs > 0]
                    ent = entropy(probs)
                    entropies.append(ent)
            except Exception:
                continue

        return np.mean(entropies) if entropies else 0.0

    def _compute_sample_entropy(self, eeg_data: np.ndarray) -> float:
        """
        计算样本熵

        参数：m=2, r=0.2*std
        """
        entropies = []

        for ch_data in eeg_data:
            std = np.std(ch_data)
            if std < 1e-10:
                continue

            r = self.r_ratio * std
            try:
                ent = self._sample_entropy_single(ch_data, self.m, r)
                if np.isfinite(ent):
                    entropies.append(ent)
            except Exception:
                continue

        return np.mean(entropies) if entropies else 0.0

    def _sample_entropy_single(self, signal: np.ndarray, m: int, r: float) -> float:
        """
        计算单通道的样本熵

        Args:
            signal: 信号数据
            m: 嵌入维度
            r: 容差阈值

        Returns:
            样本熵值
        """
        N = len(signal)

        # 构建嵌入向量
        def embed(x, dim):
            n = len(x) - dim + 1
            return np.array([x[i:i + dim] for i in range(n)])

        # 计算模板匹配数
        def count_matches(templates, r):
            N_t = len(templates)
            count = 0
            for i in range(N_t):
                for j in range(i + 1, N_t):
                    if np.max(np.abs(templates[i] - templates[j])) < r:
                        count += 2  # 对称性
            return count

        # m 维嵌入
        templates_m = embed(signal, m)
        B = count_matches(templates_m, r)

        # m+1 维嵌入
        templates_m1 = embed(signal, m + 1)
        A = count_matches(templates_m1, r)

        # 计算样本熵
        if B == 0 or A == 0:
            return 0.0

        return -np.log(A / B)

    def _compute_approx_entropy(self, eeg_data: np.ndarray) -> float:
        """
        计算近似熵

        参数：m=2, r=0.2*std
        """
        entropies = []

        for ch_data in eeg_data:
            std = np.std(ch_data)
            if std < 1e-10:
                continue

            r = self.r_ratio * std
            try:
                ent = self._approx_entropy_single(ch_data, self.m, r)
                if np.isfinite(ent):
                    entropies.append(ent)
            except Exception:
                continue

        return np.mean(entropies) if entropies else 0.0

    def _approx_entropy_single(self, signal: np.ndarray, m: int, r: float) -> float:
        """
        计算单通道的近似熵

        Args:
            signal: 信号数据
            m: 嵌入维度
            r: 容差阈值

        Returns:
            近似熵值
        """
        N = len(signal)

        def phi(m):
            # 构建嵌入向量
            n = N - m + 1
            templates = np.array([signal[i:i + m] for i in range(n)])

            # 计算每个模板的匹配比例
            counts = np.zeros(n)
            for i in range(n):
                # 使用向量化计算距离
                distances = np.max(np.abs(templates - templates[i]), axis=1)
                counts[i] = np.sum(distances < r)

            # 计算 phi
            counts = counts / n
            counts = counts[counts > 0]
            return np.mean(np.log(counts))

        phi_m = phi(m)
        phi_m1 = phi(m + 1)

        return phi_m - phi_m1

    def _compute_hurst_exponent(self, eeg_data: np.ndarray) -> float:
        """
        计算 Hurst 指数

        使用 R/S 分析方法
        """
        hurst_values = []

        for ch_data in eeg_data:
            try:
                h = self._hurst_rs(ch_data)
                if np.isfinite(h) and 0 < h < 1:
                    hurst_values.append(h)
            except Exception:
                continue

        return np.mean(hurst_values) if hurst_values else 0.5

    def _hurst_rs(self, signal: np.ndarray) -> float:
        """
        R/S 分析计算 Hurst 指数

        Args:
            signal: 信号数据

        Returns:
            Hurst 指数
        """
        N = len(signal)
        if N < 20:
            return 0.5

        # 定义不同的分割尺度
        max_k = int(np.log2(N)) - 1
        n_values = [int(2 ** k) for k in range(4, max_k + 1)]
        n_values = [n for n in n_values if n >= 8 and N // n >= 2]

        if len(n_values) < 2:
            return 0.5

        rs_values = []

        for n in n_values:
            num_parts = N // n
            rs_list = []

            for part in range(num_parts):
                segment = signal[part * n:(part + 1) * n]

                # 计算累积离差
                mean = np.mean(segment)
                y = np.cumsum(segment - mean)

                # 计算范围 R
                R = np.max(y) - np.min(y)

                # 计算标准差 S
                S = np.std(segment, ddof=1)

                if S > 1e-10:
                    rs_list.append(R / S)

            if rs_list:
                rs_values.append((n, np.mean(rs_list)))

        if len(rs_values) < 2:
            return 0.5

        # 对数线性回归
        log_n = np.log([v[0] for v in rs_values])
        log_rs = np.log([v[1] for v in rs_values])

        try:
            coeffs = np.polyfit(log_n, log_rs, 1)
            return coeffs[0]
        except np.linalg.LinAlgError:
            return 0.5
