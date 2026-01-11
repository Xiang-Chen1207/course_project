"""
特征计算基类和注册系统
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
import numpy as np
from ..psd_computer import PSDResult
from ..config import Config


class BaseFeature(ABC):
    """特征计算基类"""

    # 类变量：定义该类计算的特征名称列表
    feature_names: List[str] = []

    def __init__(self, config: Config):
        """
        初始化特征计算器

        Args:
            config: 配置对象
        """
        self.config = config
        self.fs = config.sampling_rate

    @abstractmethod
    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            psd_result: 预先计算的 PSD 结果（可选）
            **kwargs: 其他参数

        Returns:
            特征名称到值的字典
        """
        pass

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """获取特征名称列表"""
        return cls.feature_names

    def _validate_input(self, eeg_data: np.ndarray):
        """验证输入数据"""
        if eeg_data.ndim != 2:
            raise ValueError(f"EEG 数据应为二维数组，实际维度: {eeg_data.ndim}")
        if eeg_data.shape[0] != self.config.n_channels:
            raise ValueError(
                f"通道数不匹配，期望 {self.config.n_channels}，实际 {eeg_data.shape[0]}"
            )


class FeatureRegistry:
    """特征注册表：管理所有特征计算类"""

    _registry: Dict[str, Type[BaseFeature]] = {}

    @classmethod
    def register(cls, name: str):
        """
        装饰器：注册特征计算类

        Args:
            name: 特征组名称
        """
        def decorator(feature_cls: Type[BaseFeature]):
            cls._registry[name] = feature_cls
            return feature_cls
        return decorator

    @classmethod
    def get_feature_class(cls, name: str) -> Type[BaseFeature]:
        """获取特征计算类"""
        if name not in cls._registry:
            raise KeyError(f"未注册的特征组: {name}")
        return cls._registry[name]

    @classmethod
    def get_all_feature_classes(cls) -> Dict[str, Type[BaseFeature]]:
        """获取所有注册的特征计算类"""
        return cls._registry.copy()

    @classmethod
    def get_all_feature_names(cls) -> List[str]:
        """获取所有特征名称"""
        names = []
        for feature_cls in cls._registry.values():
            names.extend(feature_cls.feature_names)
        return names

    @classmethod
    def list_registered(cls) -> List[str]:
        """列出所有已注册的特征组"""
        return list(cls._registry.keys())
