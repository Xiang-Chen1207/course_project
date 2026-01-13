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


def _hurst_rs_optimized(signal: np.ndarray) -> Optional[float]:
    """优化的 Hurst 指数计算

    Returns:
        Hurst 指数（可以 > 1），失败时返回 None
    """
    N = len(signal)
    if N < 20:
        return None

    max_k = int(np.log2(N)) - 1
    n_values = [int(2 ** k) for k in range(4, max_k + 1)]
    n_values = [n for n in n_values if n >= 8 and N // n >= 2]

    if len(n_values) < 2:
        return None

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
        return None

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    try:
        coeffs = np.polyfit(log_n, log_rs, 1)
        h = coeffs[0]
        # 接受 H > 0 的所有值（包括 H > 1）
        if np.isfinite(h) and h > 0:
            return float(h)
        return None
    except np.linalg.LinAlgError:
        return None


def _higuchi_fd_single(signal: np.ndarray, kmax: int = 8) -> Optional[float]:
    """
    计算单通道的Higuchi分形维数

    Higuchi算法量化EEG信号的自相似性和复杂度：
    - 高FD：信号复杂、不规则（正常清醒状态、认知负荷）
    - 低FD：信号规则、周期性（癫痫发作、深度睡眠、昏迷）

    Args:
        signal: 1D信号
        kmax: 最大时间间隔（推荐8-20）

    Returns:
        Higuchi分形维数，失败时返回 None
    """
    N = len(signal)
    if N < kmax * 4:
        return None

    L = []
    x = np.arange(1, kmax + 1)

    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            # 构建子序列索引
            indices = np.arange(m - 1, N, k)
            if len(indices) < 2:
                continue

            Lmk = 0
            for i in range(1, len(indices)):
                Lmk += abs(signal[indices[i]] - signal[indices[i - 1]])

            num_segments = len(indices) - 1
            if num_segments > 0:
                # 归一化因子
                norm_factor = (N - 1) / (num_segments * k)
                Lmk = (Lmk / k) * norm_factor
                Lk.append(Lmk)

        if Lk:
            L.append(np.mean(Lk))

    if len(L) < 2:
        return None

    # 线性回归: log(L(k)) vs log(k)
    log_k = np.log(x[:len(L)])
    log_L = np.log(np.array(L) + 1e-15)

    try:
        coeffs = np.polyfit(log_k, log_L, 1)
        fd = -coeffs[0]
        if np.isfinite(fd) and 1.0 <= fd <= 2.0:
            return float(fd)
        return None
    except (np.linalg.LinAlgError, ValueError):
        return None


def _katz_fd_single(signal: np.ndarray) -> Optional[float]:
    """
    计算单通道的Katz分形维数

    Katz算法基于曲线长度和最大距离的比值

    Args:
        signal: 1D信号

    Returns:
        Katz分形维数，失败时返回 None
    """
    N = len(signal)
    if N < 3:
        return None

    # 计算曲线总长度L（假设采样间隔归一化为1）
    diffs = np.abs(np.diff(signal))
    L = np.sum(np.sqrt(1 + diffs ** 2))

    # 计算最大距离d（从第一个点到最远点的欧氏距离）
    indices = np.arange(N)
    distances = np.sqrt(indices ** 2 + (signal - signal[0]) ** 2)
    d = np.max(distances)

    if L < 1e-10 or d < 1e-10:
        return None

    # Katz公式
    fd = np.log10(N - 1) / (np.log10(N - 1) + np.log10(d / L))

    if np.isfinite(fd) and 0.5 <= fd <= 2.0:
        return float(fd)
    return None


def _petrosian_fd_single(signal: np.ndarray) -> Optional[float]:
    """
    计算单通道的Petrosian分形维数

    基于信号符号变化次数，计算效率高

    Args:
        signal: 1D信号

    Returns:
        Petrosian分形维数，失败时返回 None
    """
    N = len(signal)
    if N < 3:
        return None

    # 计算一阶差分
    diff = np.diff(signal)

    # 计算符号变化次数（过零点）
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    N_delta = sign_changes

    if N_delta == 0:
        return None

    # Petrosian公式
    fd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))

    if np.isfinite(fd):
        return float(fd)
    return None


@FeatureRegistry.register('complexity')
class ComplexityFeatures(BaseFeature):
    """复杂度特征计算

    包含：
    - 小波能量熵 (wavelet_energy_entropy)
    - 样本熵 (sample_entropy)
    - 近似熵 (approx_entropy)
    - Hurst指数 (hurst_exponent)
    - Higuchi分形维数 (higuchi_fd)
    - Katz分形维数 (katz_fd)
    - Petrosian分形维数 (petrosian_fd)
    """

    feature_names = [
        'wavelet_energy_entropy',
        'sample_entropy',
        'approx_entropy',
        'hurst_exponent',
        # 分形维数
        'higuchi_fd',
        'katz_fd',
        'petrosian_fd',
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
            # Hurst 指数接受 H > 0（包括 H > 1）
            if result['hurst'] is not None and np.isfinite(result['hurst']) and result['hurst'] > 0:
                hurst_values.append(result['hurst'])

        features = {
            'wavelet_energy_entropy': float(np.mean(wavelet_entropies)) if wavelet_entropies else None,
            'sample_entropy': float(np.mean(sample_entropies)) if sample_entropies else None,
            'approx_entropy': float(np.mean(approx_entropies)) if approx_entropies else None,
            'hurst_exponent': float(np.mean(hurst_values)) if hurst_values else None,
        }

        # 计算分形维数特征（对所有通道取平均）
        fd_features = self._compute_fractal_dimensions(eeg_data)
        features.update(fd_features)

        return features

    def _compute_fractal_dimensions(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        计算三种分形维数特征

        Args:
            eeg_data: EEG数据, shape: (n_channels, n_timepoints)

        Returns:
            分形维数特征字典
        """
        n_channels = eeg_data.shape[0]

        higuchi_values = []
        katz_values = []
        petrosian_values = []

        for ch in range(n_channels):
            ch_data = eeg_data[ch]

            # Higuchi FD
            try:
                hfd = _higuchi_fd_single(ch_data, kmax=8)
                if np.isfinite(hfd):
                    higuchi_values.append(hfd)
            except Exception:
                pass

            # Katz FD
            try:
                kfd = _katz_fd_single(ch_data)
                if np.isfinite(kfd):
                    katz_values.append(kfd)
            except Exception:
                pass

            # Petrosian FD
            try:
                pfd = _petrosian_fd_single(ch_data)
                if np.isfinite(pfd):
                    petrosian_values.append(pfd)
            except Exception:
                pass

        return {
            'higuchi_fd': float(np.mean(higuchi_values)) if higuchi_values else None,
            'katz_fd': float(np.mean(katz_values)) if katz_values else None,
            'petrosian_fd': float(np.mean(petrosian_values)) if petrosian_values else None,
        }

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
