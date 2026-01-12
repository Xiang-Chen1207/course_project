"""
配置文件：定义全局参数和频段设置

支持动态电极分组：根据每个.h5文件中的电极名称，自动确定电极所属的脑区和半球。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================================
# 标准 10-20/10-10 电极信息
# ============================================================================

@dataclass
class ElectrodeInfo:
    """单个电极的信息"""
    name: str
    hemisphere: str  # 'left', 'right', 'midline'
    region: str      # 'frontal', 'temporal', 'central', 'parietal', 'occipital', 'reference'


def _determine_hemisphere(name: str) -> str:
    """
    根据电极名称确定其所属半球

    规则（10-20/10-10系统）：
    - 奇数编号（1,3,5,7,9）：左半球
    - 偶数编号（2,4,6,8,10）：右半球
    - Z结尾或纯字母：中线
    - A1, T1, T9：左半球参考/颞区
    - A2, T2, T10：右半球参考/颞区
    """
    name_upper = name.upper()

    # 特殊情况
    if name_upper in ('A1', 'T1', 'T9'):
        return 'left'
    if name_upper in ('A2', 'T2', 'T10'):
        return 'right'

    # Z结尾为中线
    if name_upper.endswith('Z'):
        return 'midline'

    # 提取末尾的数字
    digits = ''
    for char in reversed(name_upper):
        if char.isdigit():
            digits = char + digits
        else:
            break

    if digits:
        num = int(digits)
        if num % 2 == 1:  # 奇数
            return 'left'
        else:  # 偶数
            return 'right'

    # 无数字的情况，默认中线
    return 'midline'


def _determine_region(name: str) -> str:
    """
    根据电极名称确定其所属脑区

    规则（10-20/10-10系统）：
    - FP (Frontopolar), AF (Antero-Frontal), F (Frontal): 额区 frontal
    - FT (Fronto-Temporal), T (Temporal, 不包含TP): 颞区 temporal
    - FC (Fronto-Central), C (Central, 不包含CP): 中央区 central
    - TP (Temporo-Parietal), CP (Centro-Parietal), P (Parietal, 不包含PO): 顶区 parietal
    - PO (Parieto-Occipital), O (Occipital), I (Inion): 枕区 occipital
    - A1, A2: 参考电极 reference
    - CB1, CB2 (Cerebellum): 枕区 occipital
    """
    name_upper = name.upper()

    # 特殊电极
    if name_upper in ('A1', 'A2'):
        return 'reference'
    if name_upper.startswith('CB'):
        return 'occipital'

    # 按前缀优先级匹配（长前缀优先）
    prefix_to_region = [
        ('FP', 'frontal'),      # Frontopolar
        ('AF', 'frontal'),      # Antero-Frontal
        ('FT', 'temporal'),     # Fronto-Temporal
        ('FC', 'central'),      # Fronto-Central (也可算额区，这里归中央区)
        ('F', 'frontal'),       # Frontal
        ('TP', 'parietal'),     # Temporo-Parietal
        ('T', 'temporal'),      # Temporal
        ('CP', 'parietal'),     # Centro-Parietal
        ('C', 'central'),       # Central
        ('PO', 'occipital'),    # Parieto-Occipital
        ('P', 'parietal'),      # Parietal
        ('O', 'occipital'),     # Occipital
        ('I', 'occipital'),     # Inion
    ]

    for prefix, region in prefix_to_region:
        if name_upper.startswith(prefix):
            return region

    # 默认归类为中央区
    return 'central'


class ElectrodeRegistry:
    """
    电极注册表：存储所有已知电极的信息

    支持根据电极名称查询其半球和脑区归属。
    """

    # 标准 90 电极系统（用户提供）
    STANDARD_90_ELECTRODES = [
        'FP1', 'FPZ', 'FP2',
        'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
        'T1', 'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', 'T2',
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
        'A1', 'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', 'A2',
        'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
        'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
        'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
        'O1', 'OZ', 'O2',
        'I1', 'IZ', 'I2',
    ]

    # SEED 62 电极系统（兼容旧代码）
    SEED_62_ELECTRODES = [
        'FP1', 'FPZ', 'FP2',
        'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]

    @classmethod
    def get_electrode_info(cls, name: str) -> ElectrodeInfo:
        """
        获取指定电极的信息

        Args:
            name: 电极名称

        Returns:
            ElectrodeInfo 对象
        """
        return ElectrodeInfo(
            name=name.upper(),
            hemisphere=_determine_hemisphere(name),
            region=_determine_region(name)
        )

    @classmethod
    def get_all_info(cls, electrode_names: List[str]) -> Dict[str, ElectrodeInfo]:
        """
        获取一组电极的信息

        Args:
            electrode_names: 电极名称列表

        Returns:
            {电极名: ElectrodeInfo} 字典
        """
        return {name: cls.get_electrode_info(name) for name in electrode_names}

    @classmethod
    def group_by_hemisphere(cls, electrode_names: List[str]) -> Dict[str, List[str]]:
        """
        按半球分组

        Args:
            electrode_names: 电极名称列表

        Returns:
            {'left': [...], 'right': [...], 'midline': [...]}
        """
        groups = {'left': [], 'right': [], 'midline': []}
        for name in electrode_names:
            info = cls.get_electrode_info(name)
            groups[info.hemisphere].append(name.upper())
        return groups

    @classmethod
    def group_by_region(cls, electrode_names: List[str]) -> Dict[str, List[str]]:
        """
        按脑区分组

        Args:
            electrode_names: 电极名称列表

        Returns:
            {'frontal': [...], 'temporal': [...], 'central': [...],
             'parietal': [...], 'occipital': [...], 'reference': [...]}
        """
        groups = {
            'frontal': [], 'temporal': [], 'central': [],
            'parietal': [], 'occipital': [], 'reference': []
        }
        for name in electrode_names:
            info = cls.get_electrode_info(name)
            groups[info.region].append(name.upper())
        return groups


@dataclass
class FrequencyBands:
    """频段定义

    标准频段范围:
    - delta: 0.5-4 Hz (慢波，与深度睡眠、疲劳相关)
    - theta: 4-8 Hz (与记忆、注意力、情绪相关)
    - alpha: 8-12 Hz (与放松、觉醒状态相关)
    - beta: 12-30 Hz (与警觉、焦虑、认知活动相关)
    - low_gamma: 30-50 Hz (与感知、认知处理相关)
    - high_gamma: 50-80 Hz (与高级认知功能相关)
    - gamma (完整): 30-80 Hz (用于计算二级特征)
    """
    delta: Tuple[float, float] = (0.5, 4.0)
    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 12.0)
    beta: Tuple[float, float] = (12.0, 30.0)
    low_gamma: Tuple[float, float] = (30.0, 50.0)
    high_gamma: Tuple[float, float] = (50.0, 80.0)
    gamma: Tuple[float, float] = (30.0, 80.0)  # 完整gamma频段，用于二级特征计算

    # 扩展频段
    low_freq: Tuple[float, float] = (1.0, 8.0)
    high_freq: Tuple[float, float] = (12.0, 40.0)

    def get_all_bands(self) -> Dict[str, Tuple[float, float]]:
        """获取所有主要频段（用于PSD功率计算）"""
        return {
            'delta': self.delta,
            'theta': self.theta,
            'alpha': self.alpha,
            'beta': self.beta,
            'low_gamma': self.low_gamma,
            'high_gamma': self.high_gamma,
            'gamma': self.gamma,
        }

    def get_primary_bands(self) -> Dict[str, Tuple[float, float]]:
        """获取五个主要频段（不包括细分的gamma）"""
        return {
            'delta': self.delta,
            'theta': self.theta,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
        }

    def get_band_range(self, band_name: str) -> Tuple[float, float]:
        """获取指定频段范围"""
        return getattr(self, band_name)


@dataclass
class ChannelGroups:
    """
    通道分组定义

    支持两种使用方式：
    1. 默认使用 SEED 62通道的预设分组（兼容旧代码）
    2. 使用 from_electrode_names() 从电极名称列表动态生成分组
    """
    # 左半球通道
    left_hemisphere: List[str] = field(default_factory=lambda: [
        'FP1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1',
        'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1',
        'P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1'
    ])

    # 右半球通道
    right_hemisphere: List[str] = field(default_factory=lambda: [
        'FP2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2',
        'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2',
        'P8', 'P6', 'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2'
    ])

    # 中线通道
    midline: List[str] = field(default_factory=lambda: [
        'FPZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ'
    ])

    # 额区通道
    frontal: List[str] = field(default_factory=lambda: [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'
    ])

    # 颞区通道
    temporal: List[str] = field(default_factory=lambda: [
        'FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'
    ])

    # 中央区通道
    central: List[str] = field(default_factory=lambda: [
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6'
    ])

    # 顶区通道
    parietal: List[str] = field(default_factory=lambda: [
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'
    ])

    # 枕区通道
    occipital: List[str] = field(default_factory=lambda: [
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ])

    @classmethod
    def from_electrode_names(cls, electrode_names: List[str]) -> 'ChannelGroups':
        """
        根据电极名称列表动态生成通道分组

        Args:
            electrode_names: 电极名称列表（从.h5文件读取）

        Returns:
            ChannelGroups 对象，包含根据电极名称自动确定的分组
        """
        # 按半球分组
        hemisphere_groups = ElectrodeRegistry.group_by_hemisphere(electrode_names)

        # 按脑区分组
        region_groups = ElectrodeRegistry.group_by_region(electrode_names)

        return cls(
            left_hemisphere=hemisphere_groups['left'],
            right_hemisphere=hemisphere_groups['right'],
            midline=hemisphere_groups['midline'],
            frontal=region_groups['frontal'],
            temporal=region_groups['temporal'],
            central=region_groups['central'],
            parietal=region_groups['parietal'],
            occipital=region_groups['occipital'],
        )

    def get_all_regions(self) -> Dict[str, List[str]]:
        """获取所有脑区分组"""
        return {
            'frontal': self.frontal,
            'temporal': self.temporal,
            'central': self.central,
            'parietal': self.parietal,
            'occipital': self.occipital,
        }

    def get_hemispheres(self) -> Dict[str, List[str]]:
        """获取所有半球分组"""
        return {
            'left': self.left_hemisphere,
            'right': self.right_hemisphere,
            'midline': self.midline,
        }


@dataclass
class Config:
    """全局配置"""
    # 采样参数
    sampling_rate: float = 200.0
    n_channels: int = 62
    segment_length: float = 2.0  # 秒
    n_timepoints: int = 400  # 2.0 * 200

    # 频段配置
    freq_bands: FrequencyBands = field(default_factory=FrequencyBands)

    # 通道分组
    channel_groups: ChannelGroups = field(default_factory=ChannelGroups)

    # PSD 计算参数
    psd_method: str = 'welch'  # 'welch' or 'fft'
    nperseg: int = 256  # Welch 方法的窗口长度
    noverlap: Optional[int] = None  # 默认 nperseg // 2
    nfft: int = 512  # FFT 点数

    # 熵计算参数
    sample_entropy_m: int = 2  # 嵌入维度
    sample_entropy_r_ratio: float = 0.2  # r = ratio * std

    # 小波参数
    wavelet: str = 'db4'
    wavelet_level: int = 5

    # 网络参数
    network_threshold: float = 0.3  # 保留前30%最强连接

    # GPU 设置
    use_gpu: bool = True
    gpu_device: int = 0

    # 频谱熵是否做归一化（除以 log(Nfreq) 使其落在 [0,1] 左右）
    spectral_entropy_normalize: bool = False

    # 复杂度特征的并行度（为避免与 segment 级多进程叠加导致过度并行，默认关闭）
    complexity_n_workers: int = 1

    # 通道名称（默认使用 SEED 62通道，实际使用时会从.h5文件动态读取）
    channel_names: List[str] = field(default_factory=lambda: ElectrodeRegistry.SEED_62_ELECTRODES.copy())

    def get_channel_indices(self, channel_list: List[str]) -> List[int]:
        """获取通道列表对应的索引"""
        # 统一大写比较，避免大小写不一致的问题
        channel_names_upper = [ch.upper() for ch in self.channel_names]
        return [channel_names_upper.index(ch.upper())
                for ch in channel_list if ch.upper() in channel_names_upper]

    def update_from_electrode_names(self, electrode_names: List[str]) -> None:
        """
        根据电极名称列表更新配置

        从.h5文件读取电极名称后调用此方法，自动更新：
        - channel_names
        - n_channels
        - channel_groups

        Args:
            electrode_names: 电极名称列表
        """
        # 标准化电极名称（大写）
        standardized_names = [name.upper() for name in electrode_names]

        self.channel_names = standardized_names
        self.n_channels = len(standardized_names)
        self.channel_groups = ChannelGroups.from_electrode_names(standardized_names)

    def get_electrode_info(self, electrode_name: str) -> ElectrodeInfo:
        """
        获取指定电极的信息（半球、脑区）

        Args:
            electrode_name: 电极名称

        Returns:
            ElectrodeInfo 对象
        """
        return ElectrodeRegistry.get_electrode_info(electrode_name)
