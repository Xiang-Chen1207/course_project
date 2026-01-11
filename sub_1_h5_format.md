# SEED HDF5 文件格式说明（sub_1.h5）

本文档说明 SEED 数据集经 benchmark_dataloader 转换后的单被试文件 sub_1.h5 的层级结构、字段、形状与含义，便于检查与下游加载。

- 生成逻辑：`benchmark_dataloader/datasets/seed.py`（构建），`benchmark_dataloader/hdf5_io.py`（读写）
- 数据来源：SEED 三分类情绪数据（sad/neutral/happy）
- 典型采样率：`200 Hz`
- 典型通道数：`62`
- 窗口参数（默认）：窗口 2.0 s，步长 2.0 s（无重叠），故每段时间点数 `T = 2.0 * 200 = 400`

---

## 顶层（Root）
HDF5 文件根节点仅包含属性（attributes），保存被试级元信息，不直接包含数据集。

根属性（由 `HDF5Writer._write_subject_attrs` 写入）：
- `subject_id`：int，被试编号（SEED 中为 1–15）。
- `dataset_name`：字符串，如 `"SEED_3class"`。
- `task_type`：字符串，`"emotion"`。
- `downstream_task_type`：字符串，`"classification"`。
- `rsFreq`：float，采样率（典型 200.0）。
- `chn_name`：字符串列表，通道名顺序（典型 62 个 SEED 通道）。
- `chn_pos`：通道位置（若无则为字符串 `"None"`）。
- `chn_ori`：通道朝向（若无则为字符串 `"None"`）。
- `chn_type`：字符串，`"EEG"`。
- `montage`：字符串，例如 `"10_10"`。

> 说明：`chn_pos` / `chn_ori` 如未提供，会以字符串 `"None"` 存储，而不是数组。

---

## 第一层分组：Trial（试次）
- 分组名：`trial{trial_id}`（如 `trial0`、`trial1` ...）。
- 属性：
  - `trial_id`：int，试次编号（跨 session 递增）。
  - `session_id`：int，会话编号（1–3）。

每个 trial 分组下包含若干个 `segment{segment_id}` 分组，每个 segment 是按窗口切片得到的片段。

---

## 第二层分组：Segment（片段）
分组名：`segment{segment_id}`，内部包含一个数据集 `eeg` 和若干属性。

- 数据集：`eeg`
  - 形状：`(n_channels, n_timepoints)`，SEED 默认 `(62, 400)`
  - dtype：`float32`
  - 数据单位：与预处理保持一致（以 `mne.io.RawArray` 创建，预处理后直接存浮点值）
      channels=[
        'FP1', 'FPZ', 'FP2',
        'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
        'CB1', 'O1', 'OZ', 'O2', 'CB2',  # 62 channels total
    ] 奇数代表在左半球，偶数代表在右半球

- `eeg` 的属性（由 `HDF5Writer.add_segment` 写入）：
  - `segment_id`：int，本 trial 内的片段序号（从 0 开始）。
  - `start_time`：float，片段在会话时间轴上的起始秒数。
  - `end_time`：float，片段在会话时间轴上的结束秒数。
  - `time_length`：float，片段时长（秒），默认 2.0。
  - `label`：ndarray（长度 1），该片段所属 trial 的类别标签。

---

## 标签与时间
- 标签映射（SEED 三分类）：`0 = sad`，`1 = neutral`，`2 = happy`。
- 每个 trial 的标签固定，segment 继承其 trial 标签，存为长度 1 的数组，如 `[2]`。
- trial 起止时间使用 `SEED_TIME_META`（单位：秒），对所有被试/会话一致；segment 的 `start_time/end_time` 来源为：
  - `trial_start_sec + [start_sample,end_sample] / rsFreq`
- 切片参数（默认）：
  - `window_sec = 2.0`，`stride_sec = 2.0`（无重叠）
  - `rsFreq = 200.0` → 每段 `400` 个采样点

---

## 典型目录结构示意
```text
/
  (attrs)
    subject_id: 1
    dataset_name: "SEED_3class"
    task_type: "emotion"
    downstream_task_type: "classification"
    rsFreq: 200.0
    chn_name: ["FP1", "FPZ", "FP2", ..., "CB2"]  # 62 通道
    chn_pos: "None"
    chn_ori: "None"
    chn_type: "EEG"
    montage: "10_10"

  /trial0
    (attrs)
      trial_id: 0
      session_id: 1
    /segment0
      /eeg  (float32, shape=(62, 400))
        (attrs)
          segment_id: 0
          start_time: <float sec>
          end_time: <float sec>
          time_length: 2.0
          label: [0|1|2]
    /segment1
      /eeg  (float32, shape=(62, 400))
      ...

  /trial1
    (attrs)
      trial_id: 1
      session_id: 1
    /segment0
      /eeg  (float32, shape=(62, 400))
      ...

  ...
```

> 试次数量：理论上每被试每会话 15 个 trial，共 45 个 trial；若原始会话缺失则会减少。

---

## 读取示例（推荐方式：项目内 Reader）
```python
from benchmark_dataloader.hdf5_io import HDF5Reader

path = "/mnt/dataset2/hdf5_datasets/SEED/sub_1.h5"
with HDF5Reader(path) as reader:
    # 根属性
    subj = reader.subject_attrs
    print(subj.subject_id, subj.dataset_name, subj.rsFreq, len(subj.chn_name))

    # 遍历 trial 与 segment
    for trial_name in reader.get_trial_names():
        tattrs = reader.get_trial_attrs(trial_name)
        seg_names = reader.get_segment_names(trial_name)
        print(trial_name, tattrs.session_id, len(seg_names))

        # 取一个 segment 查看形状与标签
        seg = reader.get_segment(trial_name, seg_names[0])
        print(seg.data.shape, seg.segment.label, seg.segment.start_time, seg.segment.end_time)
```

## 读取示例（h5py 原生）
```python
import h5py

path = "/mnt/dataset2/hdf5_datasets/SEED/sub_1.h5"
with h5py.File(path, "r") as f:
    print(dict(f.attrs))  # 根属性

    for trial_name in f.keys():
        if not trial_name.startswith("trial"):
            continue
        tgrp = f[trial_name]
        print(trial_name, dict(tgrp.attrs))

        for seg_name in tgrp.keys():
            if not seg_name.startswith("segment"):
                continue
            dset = tgrp[seg_name]["eeg"]
            print(seg_name, dset.shape, dset.dtype, dict(dset.attrs))
```

---

## 预处理要点（影响数据形态）
- 陷波（notch）：50 Hz
- 带通（bandpass）：0.1–75 Hz
- 采样率对齐：若非 200 Hz 则重采样到 200 Hz
- 通道命名：若通道数为 62，采用 SEED 约定的 62 通道名；否则回退为 `CH1..CHN`

---

## 小结
- sub_1.h5 采用“被试 → 试次（trial）→ 片段（segment）”的两层分组结构；
- 数据在 `segment{X}/eeg` 数据集中，形状 `(C, T)`，默认 `(62, 400)`，dtype `float32`；
- label 存在 `eeg` 数据集属性中，取值 `[0|1|2]`（sad/neutral/happy）；
- `start_time/end_time` 为会话时间轴秒数，便于时间对齐与可视化分析。
