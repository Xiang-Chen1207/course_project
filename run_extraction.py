#!/usr/bin/env python3
"""
便捷运行脚本：处理 SEED 数据集的 EEG 特征提取

可以直接运行此脚本来处理 sub_1.h5 文件
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.feature_extractor import FeatureExtractor
from eeg_feature_extraction.data_loader import EEGDataLoader


def main():
    """主函数"""
    # 配置
    h5_path = "/mnt/dataset2/hdf5_datasets/SEED/sub_1.h5"
    output_dir = "/mnt/dataset4/cx/code/EEG_LLM_text/output/sub1"

    # 检查输入文件
    if not Path(h5_path).exists():
        print(f"错误: 文件不存在: {h5_path}")
        return 1

    # 创建配置（使用 GPU 加速）
    config = Config(
        sampling_rate=200.0,
        use_gpu=True,  # 启用 GPU
    )

    print("=" * 60)
    print("EEG 特征提取")
    print("=" * 60)

    # 显示数据集信息
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()
    stats = loader.get_statistics()

    print(f"\n数据集信息:")
    print(f"  被试 ID: {subject_info.subject_id}")
    print(f"  数据集: {subject_info.dataset_name}")
    print(f"  采样率: {subject_info.sampling_rate} Hz")
    print(f"  通道数: {subject_info.n_channels}")
    print(f"  总 Trials: {stats['n_trials']}")
    print(f"  总 Segments: {stats['n_segments']}")
    print(f"  标签分布:")
    for label, count in stats['labels'].items():
        label_name = {0: 'sad', 1: 'neutral', 2: 'happy'}.get(label, str(label))
        print(f"    {label_name}: {count} segments")

    # 创建特征提取器
    extractor = FeatureExtractor(config)

    print(f"\n将提取的特征:")
    for i, name in enumerate(extractor.feature_names, 1):
        print(f"  {i:2d}. {name}")

    print(f"\n输出目录: {output_dir}")
    print("-" * 60)

    # 执行特征提取
    extractor.process_h5_file(
        h5_path,
        output_dir,
        feature_groups=None,  # 计算所有特征
        verbose=True
    )

    print("\n" + "=" * 60)
    print("特征提取完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
