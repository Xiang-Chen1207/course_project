"""
EEG Feature Extraction Package
可扩展的 EEG 特征提取框架，支持 GPU 加速
"""

from .feature_extractor import FeatureExtractor
from .data_loader import EEGDataLoader
from .config import Config

__version__ = "1.0.0"
__all__ = ["FeatureExtractor", "EEGDataLoader", "Config"]
