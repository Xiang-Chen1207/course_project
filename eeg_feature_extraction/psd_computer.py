"""
PSD 计算模块：支持 GPU 加速的功率谱密度计算

优化：使用多进程并行化 coherence 计算
"""
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.integrate import trapezoid
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# 尝试导入 GPU 加速库
try:
    import cupy as cp
    from cupyx.scipy.signal import welch as cp_welch
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from scipy.signal import welch as scipy_welch
from scipy.signal import coherence as scipy_coherence


def _compute_coherence_pair(args: Tuple) -> Tuple[int, int, float]:
    """
    计算单对通道的相干性（用于并行处理）

    Args:
        args: (i, j, data_i, data_j, fs, nperseg, noverlap, band)

    Returns:
        (i, j, coherence_value)
    """
    i, j, data_i, data_j, fs, nperseg, noverlap, band = args

    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            freqs, coh = scipy_coherence(
                data_i, data_j,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap
            )
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(band_mask):
            band_vals = coh[band_mask]
            finite_vals = band_vals[np.isfinite(band_vals)]
            if finite_vals.size > 0:
                mean_coh = float(np.mean(finite_vals))
            else:
                mean_coh = 0.0
        else:
            mean_coh = 0.0
        return (i, j, float(mean_coh))
    except Exception:
        return (i, j, 0.0)


@dataclass
class PSDResult:
    """PSD 计算结果"""
    freqs: np.ndarray  # 频率数组
    psd: np.ndarray  # PSD 值, shape: (n_channels, n_freqs)
    band_power: Dict[str, np.ndarray]  # 各频段功率
    total_power: np.ndarray  # 总功率 (per channel)


class PSDComputer:
    """功率谱密度计算器"""

    def __init__(self, sampling_rate: float = 200.0, use_gpu: bool = True,
                 nperseg: int = 256, noverlap: Optional[int] = None, nfft: int = 512):
        """
        初始化 PSD 计算器

        Args:
            sampling_rate: 采样率
            use_gpu: 是否使用 GPU
            nperseg: Welch 方法的窗口长度
            noverlap: 窗口重叠长度
            nfft: FFT 点数
        """
        self.fs = sampling_rate
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.nfft = nfft

        # 默认频段定义
        self.default_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0),
        }

        if self.use_gpu:
            print("PSD 计算将使用 GPU 加速 (CuPy)")
        else:
            if not GPU_AVAILABLE:
                print("CuPy 不可用，使用 CPU 计算")
            else:
                print("PSD 计算将使用 CPU")

    def compute_psd(self, eeg_data: np.ndarray,
                    bands: Optional[Dict[str, Tuple[float, float]]] = None) -> PSDResult:
        """
        计算 PSD

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            bands: 频段定义字典，如 {'alpha': (8, 13)}

        Returns:
            PSDResult 对象
        """
        if bands is None:
            bands = self.default_bands

        if self.use_gpu:
            freqs, psd = self._compute_psd_gpu(eeg_data)
        else:
            freqs, psd = self._compute_psd_cpu(eeg_data)

        # 计算各频段功率
        band_power = self._compute_band_power(freqs, psd, bands)

        # 计算总功率（0.5-100Hz范围，使用半开区间避免边界重复计入）
        total_mask = (freqs >= 0.5) & (freqs < 100.0)
        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_power = trapezoid(psd[:, total_mask], dx=freq_resolution, axis=1)

        return PSDResult(
            freqs=freqs,
            psd=psd,
            band_power=band_power,
            total_power=total_power
        )

    def _compute_psd_gpu(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU 加速的 PSD 计算"""
        # 转换到 GPU
        eeg_gpu = cp.asarray(eeg_data)

        # 使用 CuPy 的 Welch 方法
        freqs, psd = cp_welch(
            eeg_gpu,
            fs=self.fs,
            nperseg=min(self.nperseg, eeg_data.shape[1]),
            noverlap=min(self.noverlap, eeg_data.shape[1] // 2),
            nfft=self.nfft,
            axis=1
        )

        # 转回 CPU
        return cp.asnumpy(freqs), cp.asnumpy(psd)

    def _compute_psd_cpu(self, eeg_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU 的 PSD 计算"""
        freqs, psd = scipy_welch(
            eeg_data,
            fs=self.fs,
            nperseg=min(self.nperseg, eeg_data.shape[1]),
            noverlap=min(self.noverlap, eeg_data.shape[1] // 2),
            nfft=self.nfft,
            axis=1
        )
        return freqs, psd

    def _compute_band_power(self, freqs: np.ndarray, psd: np.ndarray,
                            bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        计算各频段的功率

        Args:
            freqs: 频率数组
            psd: PSD 值
            bands: 频段定义

        Returns:
            各频段功率字典
        """
        band_power = {}
        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        for band_name, (low, high) in bands.items():
            # 找到频段对应的索引
            # 使用半开区间 [low, high) 避免相邻频段在边界频点重复计入
            band_mask = (freqs >= low) & (freqs < high)
            if np.any(band_mask):
                # 使用梯形积分计算功率
                band_power[band_name] = trapezoid(psd[:, band_mask], dx=freq_resolution, axis=1)
            else:
                band_power[band_name] = np.zeros(psd.shape[0])

        return band_power

    def compute_coherence(self, eeg_data: np.ndarray,
                          band: Tuple[float, float] = (8.0, 13.0),
                          use_parallel: bool = True) -> np.ndarray:
        """
        计算通道间相干性矩阵（多线程并行优化版本）

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            band: 频段范围
            use_parallel: 是否使用并行计算

        Returns:
            相干性矩阵, shape: (n_channels, n_channels)
        """
        n_channels = eeg_data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))

        # 设置对角线为1
        np.fill_diagonal(coherence_matrix, 1.0)

        # 获取有效的参数
        nperseg = min(self.nperseg, eeg_data.shape[1])
        noverlap = min(self.noverlap, eeg_data.shape[1] // 2)

        # 生成所有需要计算的通道对
        pairs = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                pairs.append((i, j, eeg_data[i], eeg_data[j], self.fs, nperseg, noverlap, band))

        if use_parallel and len(pairs) > 10:
            # 使用线程池并行计算（1891对需要并行加速）
            n_workers = max(1, mp.cpu_count() - 1)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_compute_coherence_pair, pairs))

            for i, j, coh in results:
                coherence_matrix[i, j] = coh
                coherence_matrix[j, i] = coh
        else:
            # 串行计算（少量对时效率更高）
            for i, j, data_i, data_j, fs, np_seg, nov, bd in pairs:
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        freqs, coh = scipy_coherence(
                            data_i, data_j,
                            fs=fs,
                            nperseg=np_seg,
                            noverlap=nov
                        )
                    band_mask = (freqs >= bd[0]) & (freqs <= bd[1])
                    if np.any(band_mask):
                        band_vals = coh[band_mask]
                        finite_vals = band_vals[np.isfinite(band_vals)]
                        if finite_vals.size > 0:
                            mean_coh = float(np.mean(finite_vals))
                        else:
                            mean_coh = 0.0
                    else:
                        mean_coh = 0.0
                    coherence_matrix[i, j] = mean_coh
                    coherence_matrix[j, i] = mean_coh
                except Exception:
                    pass

        return coherence_matrix

    def compute_coherence_gpu(self, eeg_data: np.ndarray,
                              band: Tuple[float, float] = (8.0, 13.0)) -> np.ndarray:
        """
        GPU 加速的相干性计算（通过 FFT）

        Args:
            eeg_data: EEG 数据
            band: 频段范围

        Returns:
            相干性矩阵
        """
        if not self.use_gpu:
            return self.compute_coherence(eeg_data, band)

        n_channels = eeg_data.shape[0]
        n_samples = eeg_data.shape[1]

        # 转到 GPU
        eeg_gpu = cp.asarray(eeg_data)

        # 计算 FFT
        fft_data = cp.fft.rfft(eeg_gpu, n=self.nfft, axis=1)
        freqs = cp.fft.rfftfreq(self.nfft, 1.0 / self.fs)

        # 获取频段掩码
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        fft_band = fft_data[:, band_mask]

        # 计算互谱
        coherence_matrix = cp.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    # 互谱
                    cross_spec = cp.mean(fft_band[i] * cp.conj(fft_band[j]))
                    auto_i = cp.mean(cp.abs(fft_band[i]) ** 2)
                    auto_j = cp.mean(cp.abs(fft_band[j]) ** 2)
                    if auto_i > 0 and auto_j > 0:
                        coh = cp.abs(cross_spec) ** 2 / (auto_i * auto_j)
                    else:
                        coh = 0.0
                    coherence_matrix[i, j] = coh
                    coherence_matrix[j, i] = coh

        return cp.asnumpy(coherence_matrix)

    def get_peak_frequency(self, freqs: np.ndarray, psd: np.ndarray,
                           band: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        获取峰值频率

        Args:
            freqs: 频率数组
            psd: PSD 值
            band: 可选的频段范围限制

        Returns:
            每个通道的峰值频率
        """
        if band is not None:
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            freqs_band = freqs[band_mask]
            psd_band = psd[:, band_mask]
        else:
            freqs_band = freqs
            psd_band = psd

        peak_indices = np.argmax(psd_band, axis=1)
        return freqs_band[peak_indices]
