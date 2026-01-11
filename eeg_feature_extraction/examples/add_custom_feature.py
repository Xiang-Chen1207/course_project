#!/usr/bin/env python3
"""
示例：如何添加自定义特征

本示例展示如何扩展 EEG 特征提取框架，添加新的特征计算模块。
"""
import numpy as np
from typing import Dict, Optional

# 导入基类和注册器
from eeg_feature_extraction.features.base import BaseFeature, FeatureRegistry
from eeg_feature_extraction.psd_computer import PSDResult
from eeg_feature_extraction.config import Config
from eeg_feature_extraction.feature_extractor import FeatureExtractor


# 方法 1: 使用装饰器注册
@FeatureRegistry.register('my_custom_features')
class MyCustomFeatures(BaseFeature):
    """自定义特征计算示例"""

    # 定义该类计算的所有特征名称
    feature_names = [
        '自定义特征1_信号能量比',
        '自定义特征2_高频占比',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        # 可以添加自定义初始化参数

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算自定义特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)
            psd_result: 预先计算的 PSD 结果

        Returns:
            特征名称到值的字典
        """
        self._validate_input(eeg_data)

        features = {}

        # 特征 1: 信号能量比（高幅值样本占比）
        threshold = np.std(eeg_data) * 2
        high_amplitude_ratio = np.mean(np.abs(eeg_data) > threshold)
        features['自定义特征1_信号能量比'] = float(high_amplitude_ratio)

        # 特征 2: 高频占比（需要 PSD 结果）
        if psd_result is not None:
            high_freq_power = psd_result.band_power.get('gamma', np.zeros(1))
            total_power = psd_result.total_power
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.mean(high_freq_power) / (np.mean(total_power) + 1e-10)
            features['自定义特征2_高频占比'] = float(ratio)
        else:
            features['自定义特征2_高频占比'] = 0.0

        return features


# 方法 2: 手动注册（不使用装饰器）
class AnotherCustomFeature(BaseFeature):
    """另一个自定义特征示例"""

    feature_names = [
        '信号偏度',
        '信号峰度',
    ]

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        from scipy.stats import skew, kurtosis

        self._validate_input(eeg_data)

        features = {}

        # 计算偏度
        channel_skew = skew(eeg_data, axis=1)
        features['信号偏度'] = float(np.mean(channel_skew))

        # 计算峰度
        channel_kurtosis = kurtosis(eeg_data, axis=1)
        features['信号峰度'] = float(np.mean(channel_kurtosis))

        return features


def example_usage():
    """使用示例"""
    # 手动注册第二个特征类
    FeatureRegistry.register('statistical_features')(AnotherCustomFeature)

    # 创建配置
    config = Config(use_gpu=False)

    # 创建特征提取器（会自动包含所有注册的特征）
    extractor = FeatureExtractor(config)

    # 查看所有可用特征
    print("所有可用特征:")
    for name in extractor.feature_names:
        print(f"  - {name}")

    # 创建模拟数据
    eeg_data = np.random.randn(62, 400).astype(np.float32)

    # 提取特征
    features = extractor.extract_features(eeg_data)

    print("\n提取的特征值:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")


if __name__ == '__main__':
    example_usage()
