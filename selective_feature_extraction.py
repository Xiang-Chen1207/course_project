#!/usr/bin/env python3
"""
可选特征提取脚本

功能：
- 选择性地计算 EEG 特征（按组或按单个特征）
- 预设配置：快速模式、标准模式、完整模式
- 支持配置文件和命令行参数
- 输出：output/<trial_name>/<segment_name>.csv（每个 trial 一个文件夹，每个 segment 一个 CSV）

使用方法：
    # 显示所有可用特征
    python selective_feature_extraction.py --list

    # 使用预设模式
    python selective_feature_extraction.py -i /mnt/dataset2/hdf5_datasets/SEED/sub_2.h5 -o ./output_sub2 --preset standard

    python selective_feature_extraction.py -i /mnt/dataset2/hdf5_datasets/SEED/sub_2.h5 -o ./output_sub2 --preset standard
    # 选择特征组
    python selective_feature_extraction.py -i data.h5 -o ./output --groups time_domain frequency_domain

    # 选择单个特征
    python selective_feature_extraction.py -i data.h5 -o ./output --features alpha_power theta_beta_ratio

    # 排除特定特征
    python selective_feature_extraction.py -i data.h5 -o ./output --preset standard --exclude sample_entropy approx_entropy

    # 使用配置文件
    python selective_feature_extraction.py -i data.h5 -o ./output --config feature_config.yaml
"""

import sys
import os
import argparse
import json
import yaml
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.psd_computer import PSDComputer, PSDResult
from eeg_feature_extraction.data_loader import EEGDataLoader, SegmentData
from eeg_feature_extraction.features.base import BaseFeature, FeatureRegistry

# 导入所有特征模块
from eeg_feature_extraction.features import (
    TimeDomainFeatures,
    FrequencyDomainFeatures,
    ComplexityFeatures,
    ConnectivityFeatures,
    NetworkFeatures,
    CompositeFeatures,
    DEFeatures,
)


# =============================================================================
# 特征配置
# =============================================================================

# 按组定义所有特征
FEATURE_GROUPS = {
    'time_domain': [
        'mean_abs_amplitude',
        'mean_channel_std',
        'mean_peak_to_peak',
        'mean_rms',
        'mean_zero_crossing_rate',
        'hjorth_activity',
        'hjorth_mobility',
        'hjorth_complexity',
    ],
    'frequency_domain': [
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
    ],
    'complexity': [
        'wavelet_energy_entropy',
        'sample_entropy',
        'approx_entropy',
        'hurst_exponent',
        # 分形维数
        'higuchi_fd',
        'katz_fd',
        'petrosian_fd',
    ],
    'connectivity': [
        'mean_interchannel_correlation',
        'mean_alpha_coherence',
        'interhemispheric_alpha_coherence',
        'alpha_beta_band_power_correlation',
        'hemispheric_alpha_asymmetry',
        'frontal_occipital_alpha_ratio',
        # PLV特征
        'plv_theta_mean',
        'plv_alpha_mean',
        'plv_beta_mean',
        'plv_gamma_mean',
        'plv_theta_interhemispheric',
        'plv_alpha_interhemispheric',
    ],
    'network': [
        'network_clustering_coefficient',
        'network_characteristic_path_length',
        'network_global_efficiency',
        'network_small_world_index',
    ],
    'composite': [
        # 认知负荷拆分特征
        'theta_alpha_ratio',
        'frontal_beta_ratio',
        'cognitive_load_estimate',
        'alertness_estimate',
        'relaxation_index',
    ],
    # 新增: 微分熵相关特征
    'de_features': [
        # 各频段微分熵
        'de_delta',
        'de_theta',
        'de_alpha',
        'de_beta',
        'de_gamma',
        'de_low_gamma',
        'de_high_gamma',
        # 差分不对称性 (DASM)
        'dasm_delta',
        'dasm_theta',
        'dasm_alpha',
        'dasm_beta',
        'dasm_gamma',
        # 有理不对称性 (RASM)
        'rasm_delta',
        'rasm_theta',
        'rasm_alpha',
        'rasm_beta',
        'rasm_gamma',
        # 差分尾部性 (DCAU)
        'dcau_delta',
        'dcau_theta',
        'dcau_alpha',
        'dcau_beta',
        'dcau_gamma',
        # 额叶Alpha不对称性 (FAA)
        'faa_f3f4',
        'faa_f7f8',
        'faa_fp1fp2',
        'faa_mean',
    ],
}

# 反向映射：特征名称 -> 组名称
FEATURE_TO_GROUP = {}
for group, features in FEATURE_GROUPS.items():
    for feature in features:
        FEATURE_TO_GROUP[feature] = group

# 预设配置
PRESETS = {
    'fast': {
        'description': '快速模式 - 仅计算高效特征（排除熵、网络、PLV特征）',
        'groups': ['time_domain', 'frequency_domain', 'composite', 'microstate'],
        'exclude_features': [],
    },
    'standard': {
        'description': '标准模式 - 排除最耗时的特征（sample_entropy、approx_entropy、PLV、network）',
        'groups': ['time_domain', 'frequency_domain', 'complexity', 'connectivity', 'composite', 'de_features'],
        'exclude_features': ['sample_entropy', 'approx_entropy',
                             'plv_theta_mean', 'plv_alpha_mean', 'plv_beta_mean', 'plv_gamma_mean',
                             'plv_theta_interhemispheric', 'plv_alpha_interhemispheric'],
    },
    'full': {
        'description': '完整模式 - 计算所有特征（包括DE、PLV、分形维数等）',
        'groups': ['time_domain', 'frequency_domain', 'complexity', 'connectivity', 'network', 'composite', 'de_features'],
        'exclude_features': [],
    },
    'basic': {
        'description': '基础模式 - 仅计算时域和频率功率特征',
        'groups': ['time_domain'],
        'include_features': [
            'delta_power', 'theta_power', 'alpha_power',
            'beta_power', 'gamma_power', 'low_gamma_power', 'high_gamma_power',
            'delta_relative_power', 'theta_relative_power', 'alpha_relative_power',
            'beta_relative_power', 'gamma_relative_power',
            'theta_beta_ratio', 'mean_total_power',
        ],
    },
    'emotion': {
        'description': '情绪分析模式 - 常用于情绪识别的特征（包括DE和FAA）',
        'groups': [],
        'include_features': [
            # 时域
            'mean_abs_amplitude', 'mean_channel_std', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
            # 频域
            'delta_power', 'theta_power', 'alpha_power',
            'beta_power', 'gamma_power',
            'alpha_relative_power', 'beta_relative_power',
            'individual_alpha_frequency', 'theta_beta_ratio',
            # 连接性
            'hemispheric_alpha_asymmetry', 'frontal_occipital_alpha_ratio',
            # 综合
            'alertness_estimate', 'relaxation_index',
            # DE特征
            'de_alpha', 'de_beta', 'de_gamma',
            'dasm_alpha', 'dasm_beta',
            # FAA
            'faa_f3f4', 'faa_f7f8', 'faa_mean',
        ],
    },
    'cognitive': {
        'description': '认知负荷模式 - 用于认知负荷研究的特征（包括拆分后的认知负荷指标）',
        'groups': [],
        'include_features': [
            # 时域
            'mean_rms', 'hjorth_activity', 'hjorth_complexity',
            # 频域
            'theta_power', 'alpha_power', 'beta_power',
            'theta_relative_power', 'alpha_relative_power', 'beta_relative_power',
            'theta_beta_ratio', 'spectral_entropy',
            # 复杂度
            'wavelet_energy_entropy', 'hurst_exponent',
            'higuchi_fd', 'katz_fd',
            # 连接性
            'mean_alpha_coherence',
            # 综合（拆分后的认知负荷指标）
            'theta_alpha_ratio', 'frontal_beta_ratio', 'cognitive_load_estimate',
        ],
    },
    'de_only': {
        'description': '仅DE特征模式 - 仅计算微分熵相关特征',
        'groups': ['de_features'],
        'exclude_features': [],
    },
}


# =============================================================================
# 选择性特征提取器
# =============================================================================

@dataclass
class FeatureSelectionConfig:
    """特征选择配置"""
    selected_features: Set[str] = field(default_factory=set)
    selected_groups: Set[str] = field(default_factory=set)
    excluded_features: Set[str] = field(default_factory=set)

    def get_final_features(self) -> Set[str]:
        """获取最终要计算的特征列表"""
        features = set()

        # 添加选定的组
        for group in self.selected_groups:
            if group in FEATURE_GROUPS:
                features.update(FEATURE_GROUPS[group])

        # 添加单独选定的特征
        features.update(self.selected_features)

        # 移除排除的特征
        features -= self.excluded_features

        return features

    def get_required_groups(self) -> Set[str]:
        """获取需要计算的特征组"""
        final_features = self.get_final_features()
        groups = set()
        for feature in final_features:
            if feature in FEATURE_TO_GROUP:
                groups.add(FEATURE_TO_GROUP[feature])
        return groups


class SelectiveFeatureExtractor:
    """选择性特征提取器"""

    def __init__(self, config: Optional[Config] = None,
                 selection_config: Optional[FeatureSelectionConfig] = None):
        """
        初始化

        Args:
            config: EEG 配置
            selection_config: 特征选择配置
        """
        self.config = config or Config()
        self.selection_config = selection_config or FeatureSelectionConfig()

        # PSD 计算器
        self.psd_computer = PSDComputer(
            sampling_rate=self.config.sampling_rate,
            use_gpu=self.config.use_gpu,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft
        )

        # 初始化需要的特征计算器
        self.feature_computers: Dict[str, BaseFeature] = {}
        self._initialize_computers()

    def _initialize_computers(self):
        """初始化需要的特征计算器"""
        required_groups = self.selection_config.get_required_groups()
        all_feature_classes = FeatureRegistry.get_all_feature_classes()

        for group_name in required_groups:
            if group_name in all_feature_classes:
                self.feature_computers[group_name] = all_feature_classes[group_name](self.config)

    def extract_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        提取选定的特征

        Args:
            eeg_data: EEG 数据, shape: (n_channels, n_timepoints)

        Returns:
            特征字典
        """
        final_features = self.selection_config.get_final_features()

        if not final_features:
            return {}

        # 计算 PSD
        psd_result = self.psd_computer.compute_psd(
            eeg_data,
            bands=self.config.freq_bands.get_all_bands()
        )

        # 提取特征
        all_features = {}
        for group_name, computer in self.feature_computers.items():
            try:
                group_features = computer.compute(eeg_data, psd_result=psd_result)
                # 只保留选定的特征
                for name, value in group_features.items():
                    if name in final_features:
                        all_features[name] = value
            except Exception as e:
                warnings.warn(f"计算特征组 '{group_name}' 时出错: {e}")

        return all_features

    def get_selected_features(self) -> List[str]:
        """获取选定的特征名称列表"""
        return sorted(list(self.selection_config.get_final_features()))

    def process_segment(self, segment: SegmentData) -> Dict:
        """处理单个 segment"""
        features = self.extract_features(segment.eeg_data)

        # 添加元信息
        features['trial_id'] = segment.trial_id
        features['segment_id'] = segment.segment_id
        features['session_id'] = segment.session_id
        features['label'] = segment.label
        features['start_time'] = segment.start_time
        features['end_time'] = segment.end_time

        return features

    def process_h5_file(self, h5_path: str, output_dir: str, verbose: bool = True):
        """
        处理 HDF5 文件

        Args:
            h5_path: HDF5 文件路径
            output_dir: 输出目录
            verbose: 是否显示进度
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        loader = EEGDataLoader(h5_path)
        subject_info = loader.get_subject_info()

        if verbose:
            print(f"\n处理被试 {subject_info.subject_id}")
            print(f"选定特征数: {len(self.get_selected_features())}")

        trial_names = loader.get_trial_names()
        total_segments = sum(len(loader.get_segment_names(t)) for t in trial_names)

        iterator = loader.iter_segments()
        if verbose:
            iterator = tqdm(iterator, total=total_segments, desc="提取特征")

        meta_cols = ['trial_id', 'segment_id', 'session_id', 'label', 'start_time', 'end_time']
        selected_features = self.get_selected_features()

        all_results = []
        for trial_name, segment_name, segment in iterator:
            features = self.process_segment(segment)
            all_results.append(features)

            # 每个 segment 保存为单个 CSV（一个 segment 一个文件）
            df_seg = pd.DataFrame([features])
            feature_cols = [c for c in selected_features if c in df_seg.columns]
            other_cols = [c for c in df_seg.columns if c not in meta_cols and c not in feature_cols]
            df_seg = df_seg[meta_cols + feature_cols + other_cols]

            # 按 trial 建子目录：output/<trial_name>/<segment_name>.csv
            trial_dir = output_path / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            csv_path = trial_dir / f"{segment_name}.csv"
            df_seg.to_csv(csv_path, index=False, encoding='utf-8')

        if verbose:
            print(f"结果保存至目录: {output_path}")
            print(f"共生成 {len(all_results)} 个 segment CSV")

        # 仍返回汇总 DataFrame（方便调用方后续分析）
        df_all = pd.DataFrame(all_results)
        feature_cols_all = [c for c in selected_features if c in df_all.columns]
        other_cols_all = [c for c in df_all.columns if c not in meta_cols and c not in feature_cols_all]
        df_all = df_all[meta_cols + feature_cols_all + other_cols_all]
        return df_all


# =============================================================================
# 配置文件支持
# =============================================================================

def load_config_from_file(config_path: str) -> FeatureSelectionConfig:
    """
    从配置文件加载特征选择配置

    支持 JSON 和 YAML 格式
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    config = FeatureSelectionConfig()

    if 'groups' in data:
        config.selected_groups = set(data['groups'])

    if 'features' in data:
        config.selected_features = set(data['features'])

    if 'exclude' in data:
        config.excluded_features = set(data['exclude'])

    return config


def save_example_config(output_path: str):
    """保存示例配置文件"""
    example_config = {
        'groups': ['time_domain', 'frequency_domain'],
        'features': ['wavelet_energy_entropy', 'hurst_exponent'],
        'exclude': ['sample_entropy', 'approx_entropy'],
    }

    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(example_config, f, allow_unicode=True, default_flow_style=False)
        else:
            json.dump(example_config, f, ensure_ascii=False, indent=2)

    print(f"示例配置文件已保存: {output_path}")


# =============================================================================
# 工具函数
# =============================================================================

def list_all_features():
    """列出所有可用特征"""
    print("\n" + "=" * 70)
    print("可用的 EEG 特征列表")
    print("=" * 70)

    total = 0
    for group_name, features in FEATURE_GROUPS.items():
        print(f"\n{group_name.upper()} ({len(features)} 个)")
        print("-" * 50)
        for i, feature in enumerate(features, 1):
            print(f"  {i:2}. {feature}")
        total += len(features)

    print(f"\n总计: {total} 个特征")
    print("=" * 70)


def list_presets():
    """列出所有预设配置"""
    print("\n" + "=" * 70)
    print("可用的预设配置")
    print("=" * 70)

    for name, preset in PRESETS.items():
        print(f"\n[{name}]")
        print(f"  描述: {preset['description']}")

        if 'groups' in preset and preset['groups']:
            print(f"  包含组: {', '.join(preset['groups'])}")

        if 'include_features' in preset:
            print(f"  包含特征: {len(preset['include_features'])} 个")

        if 'exclude_features' in preset and preset['exclude_features']:
            print(f"  排除特征: {', '.join(preset['exclude_features'])}")

        # 计算特征数量
        config = apply_preset(name)
        features = config.get_final_features()
        print(f"  最终特征数: {len(features)} 个")

    print("\n" + "=" * 70)


def apply_preset(preset_name: str) -> FeatureSelectionConfig:
    """应用预设配置"""
    if preset_name not in PRESETS:
        raise ValueError(f"未知的预设: {preset_name}")

    preset = PRESETS[preset_name]
    config = FeatureSelectionConfig()

    if 'groups' in preset:
        config.selected_groups = set(preset['groups'])

    if 'include_features' in preset:
        config.selected_features = set(preset['include_features'])

    if 'exclude_features' in preset:
        config.excluded_features = set(preset['exclude_features'])

    return config


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='可选特征提取脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 显示所有可用特征
  python selective_feature_extraction.py --list

  # 显示预设配置
  python selective_feature_extraction.py --list-presets

  # 使用快速预设（排除耗时特征）
  python selective_feature_extraction.py -i data.h5 -o ./output --preset fast

  # 选择特定特征组
  python selective_feature_extraction.py -i data.h5 -o ./output --groups time_domain frequency_domain

  # 选择单个特征
    python selective_feature_extraction.py -i data.h5 -o ./output --features alpha_power theta_beta_ratio

  # 完整模式但排除某些特征
    python selective_feature_extraction.py -i data.h5 -o ./output --preset full --exclude sample_entropy approx_entropy

  # 生成示例配置文件
  python selective_feature_extraction.py --generate-config feature_config.yaml
        """
    )

    # 输入输出
    parser.add_argument('-i', '--input', type=str, help='输入 HDF5 文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出目录')

    # 特征选择
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='使用预设配置')
    parser.add_argument('--groups', type=str, nargs='+',
                        choices=list(FEATURE_GROUPS.keys()),
                        help='选择特征组')
    parser.add_argument('--features', type=str, nargs='+',
                        help='选择单个特征')
    parser.add_argument('--exclude', type=str, nargs='+',
                        help='排除特定特征')
    parser.add_argument('--config', type=str, help='配置文件路径 (YAML/JSON)')

    # 工具选项
    parser.add_argument('--list', action='store_true', help='列出所有可用特征')
    parser.add_argument('--list-presets', action='store_true', help='列出预设配置')
    parser.add_argument('--generate-config', type=str, help='生成示例配置文件')
    parser.add_argument('--show-selected', action='store_true', help='显示选定的特征')

    # 其他选项
    parser.add_argument('--no-gpu', action='store_true', help='禁用 GPU')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    # 工具命令
    if args.list:
        list_all_features()
        return

    if args.list_presets:
        list_presets()
        return

    if args.generate_config:
        save_example_config(args.generate_config)
        return

    # 构建特征选择配置
    selection_config = FeatureSelectionConfig()

    # 1. 从配置文件加载
    if args.config:
        selection_config = load_config_from_file(args.config)

    # 2. 应用预设
    if args.preset:
        selection_config = apply_preset(args.preset)

    # 3. 添加命令行指定的组
    if args.groups:
        selection_config.selected_groups.update(args.groups)

    # 4. 添加命令行指定的特征
    if args.features:
        selection_config.selected_features.update(args.features)

    # 5. 应用排除
    if args.exclude:
        selection_config.excluded_features.update(args.exclude)

    # 如果没有选择任何特征，使用默认（标准模式）
    if (not selection_config.selected_groups and
        not selection_config.selected_features):
        print("未选择任何特征，使用 'standard' 预设")
        selection_config = apply_preset('standard')

    # 显示选定的特征
    if args.show_selected or not args.input:
        final_features = selection_config.get_final_features()
        print(f"\n选定的特征 ({len(final_features)} 个):")
        print("-" * 50)
        for feature in sorted(final_features):
            group = FEATURE_TO_GROUP.get(feature, 'unknown')
            print(f"  [{group}] {feature}")

        if not args.input:
            print("\n提示: 使用 -i 参数指定输入文件以开始提取")
            return

    # 执行特征提取
    if args.input and args.output:
        # 配置
        eeg_config = Config()
        eeg_config.use_gpu = not args.no_gpu

        # 创建提取器
        extractor = SelectiveFeatureExtractor(
            config=eeg_config,
            selection_config=selection_config
        )

        if not args.quiet:
            print("\n" + "=" * 50)
            print("开始特征提取")
            print("=" * 50)
            print(f"输入文件: {args.input}")
            print(f"输出目录: {args.output}")
            print(f"选定特征数: {len(extractor.get_selected_features())}")
            print(f"GPU: {'启用' if eeg_config.use_gpu else '禁用'}")

        # 处理文件
        extractor.process_h5_file(
            args.input,
            args.output,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n特征提取完成！")

    elif args.input:
        print("错误: 请使用 -o 参数指定输出目录")
        sys.exit(1)


if __name__ == '__main__':
    main()
