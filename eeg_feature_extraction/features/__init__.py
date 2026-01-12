"""
特征计算模块

包含以下特征类别：
- 时域特征 (TimeDomainFeatures)
- 频域特征 (FrequencyDomainFeatures)
- 复杂度特征 (ComplexityFeatures) - 包含分形维数
- 连接性特征 (ConnectivityFeatures) - 包含PLV
- 网络特征 (NetworkFeatures)
- 综合特征 (CompositeFeatures) - 认知负荷拆分为theta_alpha_ratio和frontal_beta_ratio
- 微分熵特征 (DEFeatures) - DE, DASM, RASM, DCAU, FAA
- 微状态特征 (MicrostateFeatures)
"""
from .base import BaseFeature, FeatureRegistry
from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from .complexity import ComplexityFeatures
from .connectivity import ConnectivityFeatures, compute_plv_matrix
from .network import NetworkFeatures
from .composite import CompositeFeatures
from .de_features import DEFeatures
from .microstate import MicrostateFeatures, MicrostateAnalyzer

__all__ = [
    'BaseFeature',
    'FeatureRegistry',
    'TimeDomainFeatures',
    'FrequencyDomainFeatures',
    'ComplexityFeatures',
    'ConnectivityFeatures',
    'NetworkFeatures',
    'CompositeFeatures',
    'DEFeatures',
    'MicrostateFeatures',
    'MicrostateAnalyzer',
    # 工具函数
    'compute_plv_matrix',
]
