#!/usr/bin/env python3
"""
EEG 特征提取主程序

使用方法:
    python -m eeg_feature_extraction.main --input /path/to/sub_1.h5 --output ./output

    # 处理多个文件
    python -m eeg_feature_extraction.main --input /mnt/dataset2/hdf5_datasets/Workload_MATB --output /mnt/dataset4/cx/code/EEG_LLM_text/Workload_output

    # 只计算特定特征组
    python -m eeg_feature_extraction.main --input /mnt/dataset2/hdf5_datasets/SleepEDF --output /mnt/dataset4/cx/code/EEG_LLM_text/SleepEDF_output \
        --feature-groups time_domain frequency_domain composite

    # 禁用 GPU
    python -m eeg_feature_extraction.main --input /path/to/sub_1.h5 --output ./output --no-gpu
"""
import argparse
import sys
from pathlib import Path
from glob import glob

from .config import Config
from .feature_extractor import FeatureExtractor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='EEG 特征提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入 HDF5 文件路径（支持通配符，如 /path/to/*.h5）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--feature-groups', '-f',
        type=str,
        nargs='+',
        default=None,
        help='要计算的特征组（默认全部）。可选: time_domain, frequency_domain, '
             'complexity, connectivity, network, composite, microstate'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='禁用 GPU 加速'
    )

    parser.add_argument(
        '--sampling-rate', '-s',
        type=float,
        default=200.0,
        help='采样率（默认: 200.0 Hz）'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不显示进度'
    )

    parser.add_argument(
        '--list-features',
        action='store_true',
        help='列出所有可用特征并退出'
    )

    return parser.parse_args()


def list_features():
    """列出所有可用特征"""
    config = Config()
    extractor = FeatureExtractor(config)
    info = extractor.get_feature_info()

    print("\n可用特征列表:")
    print("-" * 60)

    for group_name in info['特征组'].unique():
        group_features = info[info['特征组'] == group_name]['特征名称'].tolist()
        print(f"\n[{group_name}]")
        for feature in group_features:
            print(f"  - {feature}")

    print("-" * 60)
    print(f"共 {len(info)} 个特征")


def main():
    """主函数"""
    args = parse_args()

    # 列出特征
    if args.list_features:
        list_features()
        return 0

    # 创建配置
    config = Config(
        sampling_rate=args.sampling_rate,
        use_gpu=not args.no_gpu
    )

    # 创建特征提取器
    extractor = FeatureExtractor(config)

    # 获取输入文件列表
    input_path = Path(args.input)
    if input_path.is_dir():
        # 输入是目录时，自动查找其中的 .h5 文件
        h5_files = sorted(str(p) for p in input_path.glob('*.h5'))
    elif '*' in args.input or '?' in args.input:
        # 通配符模式
        h5_files = sorted(glob(args.input))
    else:
        # 单个文件
        h5_files = [args.input]

    if not h5_files:
        print(f"错误: 未找到匹配的文件: {input_path}")
        return 1

    if not args.quiet:
        print(f"找到 {len(h5_files)} 个文件待处理")

    # 处理文件
    for h5_path in h5_files:
        if not Path(h5_path).exists():
            print(f"警告: 文件不存在: {h5_path}")
            continue

        h5_name = Path(h5_path).stem
        output_dir = Path(args.output) / h5_name

        if not args.quiet:
            print(f"\n处理: {h5_path}")

        try:
            extractor.process_h5_file(
                h5_path,
                str(output_dir),
                feature_groups=args.feature_groups,
                verbose=not args.quiet
            )
        except Exception as e:
            print(f"错误处理 {h5_path}: {e}")
            continue

    if not args.quiet:
        print("\n全部处理完成!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
