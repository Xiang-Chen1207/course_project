# EEG 特征提取详细文档

本文档详细介绍 `eeg_feature_extraction/features/` 目录下所有特征的含义、计算方法和应用场景。

## 目录

1. [概述](#概述)
2. [基础知识](#基础知识)
3. [时域特征 (Time Domain Features)](#1-时域特征-time-domain-features)
4. [频域特征 (Frequency Domain Features)](#2-频域特征-frequency-domain-features)
5. [复杂度特征 (Complexity Features)](#3-复杂度特征-complexity-features)
6. [微分熵特征 (Differential Entropy Features)](#4-微分熵特征-differential-entropy-features)
7. [连接性特征 (Connectivity Features)](#5-连接性特征-connectivity-features)
8. [复合特征 (Composite Features)](#6-复合特征-composite-features)
9. [网络拓扑特征 (Network Features)](#7-网络拓扑特征-network-features)
10. [微状态特征 (Microstate Features)](#8-微状态特征-microstate-features)
11. [特征总览](#特征总览)

---

## 概述

本项目实现了 **8 大类共 137 个 EEG 特征**，覆盖了脑电信号分析的多个维度：

| 模块 | 特征数量 | 分析维度 |
|------|----------|----------|
| 时域特征 | 8 | 信号幅度、波动性 |
| 频域特征 | 23 | 功率谱、频率成分 |
| 复杂度特征 | 7 | 信号规律性、分形维度 |
| 微分熵特征 | 44 | 各频带信息量、脑区不对称性 |
| 连接性特征 | 12 | 通道间相关性、相位同步 |
| 复合特征 | 5 | 认知负荷、警觉度 |
| 网络特征 | 4 | 脑网络拓扑结构 |
| 微状态特征 | 20 | 全脑瞬时状态动态 |

---

## 基础知识

### EEG 频带定义

脑电信号按频率可分为以下标准频带：

| 频带 | 频率范围 | 生理意义 |
|------|----------|----------|
| **Delta (δ)** | 0.5-4 Hz | 深度睡眠、无意识状态 |
| **Theta (θ)** | 4-8 Hz | 困倦、深度冥想、记忆巩固 |
| **Alpha (α)** | 8-12 Hz | 放松、闭眼静息、心理空闲 |
| **Beta (β)** | 12-30 Hz | 主动思考、解决问题、警觉 |
| **Low Gamma** | 30-50 Hz | 高级认知处理 |
| **High Gamma** | 50-80 Hz | 活跃认知、感知绑定 |
| **Gamma (γ)** | 30-80 Hz | 注意力、感知整合 |

### 电极位置 (10-20 系统)

```
          前额叶 (Frontal)
    Fp1  Fp2        ← 额极
   F7  F3  Fz  F4  F8  ← 额叶

         中央区 (Central)
   T7  C3  Cz  C4  T8   ← 中央/颞叶

         顶叶 (Parietal)
   P7  P3  Pz  P4  P8   ← 顶叶

         枕叶 (Occipital)
      O1  Oz  O2        ← 枕叶
```

**功能分区：**
- **额叶 (Fp, F)**: 执行功能、决策、运动规划
- **中央 (C)**: 运动/感觉处理
- **颞叶 (T)**: 语言、记忆
- **顶叶 (P)**: 注意力、空间感知
- **枕叶 (O)**: 视觉处理

### 对称电极对 (14对)

用于计算脑区不对称性：

```
左半球 ←→ 右半球
FP1 ←→ FP2    F7 ←→ F8     F3 ←→ F4
AF3 ←→ AF4    FC5 ←→ FC6   FC1 ←→ FC2
T7 ←→ T8      C3 ←→ C4
CP5 ←→ CP6    CP1 ←→ CP2
P7 ←→ P8      P3 ←→ P4     PO3 ←→ PO4
O1 ←→ O2
```

---

## 1. 时域特征 (Time Domain Features)

**文件**: `time_domain.py`
**类名**: `TimeDomainFeatures`

时域特征直接从原始EEG信号的幅度和波形中提取，不需要频率变换。

### 1.1 平均绝对幅度 (mean_abs_amplitude)

**含义**: 信号的平均绝对电压幅度，反映整体信号强度。

**计算方法**:
```
mean_abs_amplitude = mean(|EEG(t)|)
```

**解释**: 对所有通道、所有时间点的EEG电压取绝对值后求均值。

---

### 1.2 平均通道标准差 (mean_channel_std)

**含义**: 各通道信号标准差的均值，反映信号波动程度。

**计算方法**:
```
mean_channel_std = mean(std(EEG_ch))  对每个通道ch
```

**解释**: 先计算每个通道的标准差，再对所有通道求均值。

---

### 1.3 平均峰峰值 (mean_peak_to_peak)

**含义**: 信号的动态范围，即最大值与最小值之差的均值。

**计算方法**:
```
mean_peak_to_peak = mean(max(EEG_ch) - min(EEG_ch))  对每个通道ch
```

**解释**: 每个通道的峰峰值 = 最大电压 - 最小电压，然后对所有通道取均值。

---

### 1.4 均方根值 (mean_rms)

**含义**: 信号的能量指标，反映整体功率水平。

**计算方法**:
```
RMS = √(mean(EEG²))
mean_rms = mean(RMS_ch)  对每个通道ch
```

**解释**: 对每个通道计算均方根值，再求均值。RMS对大幅度波动更敏感。

---

### 1.5 零交叉率 (mean_zero_crossing_rate)

**含义**: 信号穿越零点的频率，粗略反映信号的主要频率成分。

**计算方法**:
```
ZCR = count(sign(EEG(t)) ≠ sign(EEG(t+1))) / 持续时间
mean_zero_crossing_rate = mean(ZCR_ch)
```

**解释**: 统计信号符号变化的次数（从正到负或从负到正），除以信号时长。高ZCR表示信号变化频繁，主频较高。

**注意**: 实现中先对零值进行前向填充处理。

---

### 1.6 Hjorth 活动度 (hjorth_activity)

**含义**: 信号的方差，反映信号的功率或"活动"程度。

**计算方法**:
```
Activity = Var(EEG) = mean((EEG - mean(EEG))²)
hjorth_activity = mean(Activity_ch)
```

**解释**: 信号方差越大，活动度越高，表示信号波动越剧烈。

---

### 1.7 Hjorth 移动度 (hjorth_mobility)

**含义**: 信号一阶导数的标准差与原信号标准差的比值，反映信号的典型频率。

**计算方法**:
```
d¹EEG = diff(EEG)  (一阶差分，近似一阶导数)
Mobility = std(d¹EEG) / std(EEG)
hjorth_mobility = mean(Mobility_ch)
```

**解释**: 移动度值越高，信号中高频成分越多。

---

### 1.8 Hjorth 复杂度 (hjorth_complexity)

**含义**: 二阶导数与一阶导数移动度的比值，反映信号形状的复杂程度。

**计算方法**:
```
d²EEG = diff(d¹EEG)  (二阶差分)
Complexity = Mobility(d¹EEG) / Mobility(EEG)
          = (std(d²EEG)/std(d¹EEG)) / (std(d¹EEG)/std(EEG))
hjorth_complexity = mean(Complexity_ch)
```

**解释**: 复杂度接近1表示信号接近正弦波；偏离1表示信号更复杂、含有更多频率成分。

---

## 2. 频域特征 (Frequency Domain Features)

**文件**: `frequency_domain.py`
**类名**: `FrequencyDomainFeatures`

频域特征通过功率谱密度（PSD）分析信号的频率成分。使用 Welch 方法计算 PSD。

### 2.1 频带绝对功率 (Band Power)

**特征列表**: `delta_power`, `theta_power`, `alpha_power`, `beta_power`, `gamma_power`, `low_gamma_power`, `high_gamma_power`

**含义**: 各频带内的绝对功率，反映该频段神经活动的强度。

**计算方法**:
```
Band_Power = ∫[f_low to f_high] PSD(f) df
           ≈ Σ PSD(f) × Δf  (离散形式)
```

**解释**: 对 PSD 在特定频带范围内积分。

**应用**:
- Delta功率高：深度睡眠
- Alpha功率高：放松状态
- Beta功率高：警觉、认知任务

---

### 2.2 频带相对功率 (Relative Power)

**特征列表**: `delta_relative_power`, `theta_relative_power`, `alpha_relative_power`, `beta_relative_power`, `gamma_relative_power`

**含义**: 各频带功率占总功率的比例，消除了个体差异的影响。

**计算方法**:
```
Relative_Power_band = Band_Power / Total_Power
```

**解释**: 相对功率更适合跨被试比较，因为它不受信号幅度的绝对大小影响。

---

### 2.3 峰值频率 (peak_frequency)

**含义**: PSD 最大值对应的频率，即信号的主导频率。

**计算方法**:
```
peak_frequency = argmax(PSD(f))  在 0.5-100 Hz 范围内
```

**解释**: 找到 PSD 曲线的最高峰对应的频率值。

---

### 2.4 谱熵 (spectral_entropy)

**含义**: 频谱分布的随机性/复杂度，反映信号频率成分的多样性。

**计算方法**:
```
p(f) = PSD(f) / Σ PSD(f)  (归一化PSD为概率分布)
Spectral_Entropy = -Σ p(f) × log(p(f))
```

**解释**:
- 低谱熵：能量集中在少数频率（如纯正弦波）
- 高谱熵：能量分散在多个频率（如白噪声）

---

### 2.5 谱质心 (spectral_centroid)

**含义**: 频谱的"重心"，功率加权的平均频率。

**计算方法**:
```
Spectral_Centroid = Σ(f × PSD(f)) / Σ PSD(f)
```

**解释**: 反映信号的"亮度"或频率中心位置。质心高表示高频成分占主导。

---

### 2.6 个体Alpha频率 (individual_alpha_frequency, IAF)

**含义**: Alpha频带(8-13 Hz)内功率最大的频率，是重要的个体神经特征。

**计算方法**:
```
IAF = argmax(PSD(f))  在 8-13 Hz 范围内
```

**解释**: IAF 因人而异（通常 9-11 Hz），与认知能力和年龄相关。

---

### 2.7 频带功率比值

#### Theta/Beta 比值 (theta_beta_ratio)

**含义**: 认知负荷指标。

**计算方法**:
```
TBR = Theta_Power / Beta_Power
```

**解释**: 高TBR通常与ADHD、低警觉状态相关；低TBR表示高度警觉。

#### Delta/Theta 比值 (delta_theta_ratio)

**含义**: 睡眠/觉醒状态指标。

**计算方法**:
```
DTR = Delta_Power / Theta_Power
```

**解释**: 高DTR表示深度睡眠或意识水平下降。

#### 低频/高频功率比 (low_high_power_ratio)

**含义**: 低频与高频成分的平衡。

**计算方法**:
```
LHPR = Power(1-8 Hz) / Power(13-40 Hz)
```

**解释**: 反映慢波活动与快波活动的相对强度。

---

### 2.8 非周期性指数 (aperiodic_exponent)

**含义**: 功率谱的 1/f 斜率，反映信号的分形特性。

**计算方法**:
```
PSD ∝ 1/f^β  (非周期成分)
aperiodic_exponent = β = -slope(log(PSD) vs log(f))
```

**实现**:
- 优先使用 FOOOF (Fitting Oscillations & One-Over-f) 算法
- 回退方案：对数-对数空间线性拟合

**解释**:
- 较大的指数（更陡的斜率）：与衰老、某些病理状态相关
- 较小的指数：与警觉、年轻状态相关

---

### 2.9 平均总功率 (mean_total_power)

**含义**: 所有频率的平均总功率。

**计算方法**:
```
mean_total_power = mean(Σ PSD(f))  对所有通道
```

---

## 3. 复杂度特征 (Complexity Features)

**文件**: `complexity.py`
**类名**: `ComplexityFeatures`

复杂度特征量化信号的规律性、可预测性和分形特性。

### 3.1 样本熵 (sample_entropy)

**含义**: 信号模式的可预测性，值越低越规则。

**计算方法**:
```
设 m=2 (嵌入维度), r=0.2×std (容忍度)

1. 构造m维向量: X_i = [x(i), x(i+1), ..., x(i+m-1)]
2. 计算B: m维向量中距离<r的配对数
3. 计算A: (m+1)维向量中距离<r的配对数
4. Sample_Entropy = -ln(A/B)
```

**参数**:
- m = 2: 嵌入维度
- r = 0.2 × std(EEG): 容忍度阈值

**解释**: 样本熵度量了当模式长度增加时，模式匹配的概率如何变化。

---

### 3.2 近似熵 (approx_entropy)

**含义**: 类似样本熵，但包含自匹配，计算稍有不同。

**计算方法**:
```
ApEn = φ(m) - φ(m+1)

其中 φ(k) = (1/(N-m+1)) × Σ ln(C_i^m(r))
C_i^m(r) = (符合条件的模式数) / (N-m+1)
```

**解释**: 近似熵是样本熵的早期版本，对短数据更稳定但有一定偏差。

---

### 3.3 小波能量熵 (wavelet_energy_entropy)

**含义**: 小波分解各层能量分布的熵值。

**计算方法**:
```
1. 对信号进行小波分解: EEG → {cA5, cD5, cD4, cD3, cD2, cD1}
2. 计算各层能量: E_i = Σ coeff_i²
3. 计算能量比例: p_i = E_i / Σ E_i
4. Wavelet_Entropy = -Σ p_i × log(p_i)
```

**参数**:
- 小波类型: db4 (Daubechies 4)
- 分解层数: 5

**解释**: 反映信号能量在不同尺度上的分布均匀程度。

---

### 3.4 Hurst 指数 (hurst_exponent)

**含义**: 信号的长程相关性/持续性。

**计算方法**:
```
R/S 分析法:
1. 将信号分成不同长度n的子段
2. 对每个子段计算:
   - 均值调整累积偏差
   - R = max(累积偏差) - min(累积偏差)
   - S = 标准差
3. E[R/S] ∝ n^H
4. H = slope(log(R/S) vs log(n))
```

**解释**:
- H = 0.5: 随机游走（无相关性）
- H > 0.5: 持续性（趋势保持）
- H < 0.5: 反持续性（均值回归）

---

### 3.5 Higuchi 分形维度 (higuchi_fd)

**含义**: 信号的自仿射分形维度（1.0-2.0范围）。

**计算方法**:
```
1. 构造子序列: X_k^m = {x(m), x(m+k), x(m+2k), ...}
2. 计算曲线长度: L_k = (1/k) × Σ|x(m+ik) - x(m+(i-1)k)| × 归一化因子
3. L(k) ∝ k^(-D)
4. Higuchi_FD = D = -slope(log(L) vs log(k))
```

**解释**:
- D ≈ 1.0: 平滑曲线
- D ≈ 2.0: 完全填充平面的复杂曲线

---

### 3.6 Katz 分形维度 (katz_fd)

**含义**: 基于曲线长度和范围的分形维度。

**计算方法**:
```
L = 曲线总长度 = Σ|x(t+1) - x(t)|
d = 最大距离 = max(|x(t) - x(0)|)
N = 数据点数

Katz_FD = log(N-1) / (log(N-1) + log(d/L))
```

**解释**: 计算简单，对噪声敏感度较低。

---

### 3.7 Petrosian 分形维度 (petrosian_fd)

**含义**: 基于差分符号变化的快速分形维度估计。

**计算方法**:
```
N = 数据点数
N_delta = 差分信号的符号变化次数

Petrosian_FD = log(N) / (log(N) + log(N / (N + 0.4 × N_delta)))
```

**解释**: 计算效率最高的分形维度估计方法。

---

## 4. 微分熵特征 (Differential Entropy Features)

**文件**: `de_features.py`
**类名**: `DEFeatures`

微分熵（DE）是连续信号信息量的度量，假设信号服从高斯分布。

### 4.1 频带微分熵 (Band DE)

**特征列表**: `de_delta`, `de_theta`, `de_alpha`, `de_beta`, `de_gamma`, `de_low_gamma`, `de_high_gamma`

**含义**: 各频带信号的信息量/不确定性。

**计算方法**:
```
假设 EEG_band ~ N(μ, σ²)

DE = 0.5 × log(2πe × σ²)
   = 0.5 × log(2πe) + 0.5 × log(σ²)
   = 常数 + log(σ)  (比例关系)
```

**实际计算**: 对频带滤波后的信号计算方差σ²，代入公式。

**解释**: DE值越高，该频带的信号变异性越大，信息量越多。

---

### 4.2 DASM - 微分不对称性 (Differential ASyMetry)

**特征列表**: `dasm_delta`, `dasm_theta`, `dasm_alpha`, `dasm_beta`, `dasm_gamma`

**含义**: 左右脑相同位置电极的DE差值。

**计算方法**:
```
DASM = DE(左半球) - DE(右半球)
     = 0.5 × log(σ²_左) - 0.5 × log(σ²_右)
     = 0.5 × log(σ²_左 / σ²_右)
```

**使用电极对**: (FP1,FP2), (F7,F8), (F3,F4), (T7,T8), (C3,C4), (P7,P8), (P3,P4), (O1,O2) 等14对

**解释**:
- DASM > 0: 左半球活动更强
- DASM < 0: 右半球活动更强

---

### 4.3 RASM - 比率不对称性 (Rational ASyMetry)

**特征列表**: `rasm_delta`, `rasm_theta`, `rasm_alpha`, `rasm_beta`, `rasm_gamma`

**含义**: 左右脑相同位置电极DE的比值。

**计算方法**:
```
RASM = DE(左半球) / DE(右半球)
```

**注意**: 结果限制在 [0.01, 100] 范围内以避免极端值。

---

### 4.4 DCAU - 前后差异 (Differential CaUdal)

**特征列表**: `dcau_delta`, `dcau_theta`, `dcau_alpha`, `dcau_beta`, `dcau_gamma`

**含义**: 额叶与后脑（顶叶/枕叶）的DE差值。

**计算方法**:
```
DCAU = DE(额叶) - DE(后脑)
```

**使用电极对**: (FC5,CP5), (FC1,CP1), (FZ,PZ), (FC2,CP2), (FC6,CP6) 等11对

**解释**:
- DCAU > 0: 额叶活动更强
- DCAU < 0: 后脑活动更强

---

### 4.5 FAA - 额叶Alpha不对称性 (Frontal Alpha Asymmetry)

**特征列表**: `faa_f3f4`, `faa_f7f8`, `faa_fp1fp2`, `faa_mean`

**含义**: 经典的情绪指标，反映趋近-回避动机。

**计算方法**:
```
FAA = ln(P_α,右) - ln(P_α,左)
```

**计算位置**:
- F3/F4: 背外侧前额叶
- F7/F8: 下额叶
- Fp1/Fp2: 额极

**解释** (Davidson模型):
- FAA > 0 (左侧激活高): 趋近动机、积极情绪
- FAA < 0 (右侧激活高): 回避动机、消极情绪

**注意**: Alpha功率与皮层活动负相关（Alpha抑制 = 皮层激活）。

---

## 5. 连接性特征 (Connectivity Features)

**文件**: `connectivity.py`
**类名**: `ConnectivityFeatures`

连接性特征量化不同脑区之间的功能关系。

### 5.1 平均通道间相关 (mean_interchannel_correlation)

**含义**: 所有通道对之间时域信号的平均相关性。

**计算方法**:
```
r_ij = Pearson_Correlation(EEG_i, EEG_j)
mean_interchannel_correlation = mean(r_ij)  对所有 i≠j
```

**范围**: [-1, 1]

**解释**: 高值表示全脑活动高度同步。

---

### 5.2 Alpha相干性 (mean_alpha_coherence)

**含义**: Alpha频带的平均通道间相干性。

**计算方法**:
```
Coherence_ij(f) = |S_ij(f)|² / (S_ii(f) × S_jj(f))

其中 S_ij(f) = 交叉功率谱, S_ii(f) = 自功率谱

mean_alpha_coherence = mean(Coherence_ij)  在8-13Hz, 所有i≠j
```

**范围**: [0, 1]

**解释**: 相干性衡量两个信号在特定频率上的线性相关程度。

---

### 5.3 半球间Alpha相干性 (interhemispheric_alpha_coherence)

**含义**: 左右半球对称电极之间的Alpha相干性。

**计算方法**:
```
使用对称电极对 (如 F3-F4, C3-C4, P3-P4)
interhemispheric_coherence = mean(Coherence(左,右))  在Alpha频带
```

---

### 5.4 Alpha-Beta功率相关 (alpha_beta_band_power_correlation)

**含义**: Alpha和Beta频带功率在各通道间的相关性。

**计算方法**:
```
P_α = [各通道的Alpha功率向量]
P_β = [各通道的Beta功率向量]
correlation = Pearson_Correlation(P_α, P_β)
```

---

### 5.5 半球Alpha不对称性 (hemispheric_alpha_asymmetry)

**含义**: 左右半球Alpha功率的归一化差异。

**计算方法**:
```
asymmetry = (P_α,右 - P_α,左) / (P_α,右 + P_α,左)
```

**范围**: [-1, 1]

---

### 5.6 额-枕Alpha比值 (frontal_occipital_alpha_ratio)

**含义**: 额叶与枕叶Alpha功率的比值。

**计算方法**:
```
ratio = P_α,额叶 / P_α,枕叶
```

**范围**: 限制在 [0.01, 100]

---

### 5.7 相位锁定值 (Phase Locking Value, PLV)

**特征列表**: `plv_theta_mean`, `plv_alpha_mean`, `plv_beta_mean`, `plv_gamma_mean`, `plv_theta_interhemispheric`, `plv_alpha_interhemispheric`

**含义**: 两个信号之间的相位同步程度。

**计算方法**:
```
1. 对信号进行频带滤波
2. 通过Hilbert变换获取瞬时相位: φ(t) = angle(Hilbert(EEG))
3. 计算相位差: Δφ_ij(t) = φ_i(t) - φ_j(t)
4. PLV_ij = |mean(exp(i × Δφ_ij(t)))|
```

**范围**: [0, 1]
- PLV = 0: 相位完全随机
- PLV = 1: 完全相位同步

**解释**: PLV是纯相位指标，不受幅度影响，常用于研究神经振荡同步。

---

## 6. 复合特征 (Composite Features)

**文件**: `composite.py`
**类名**: `CompositeFeatures`

复合特征结合多个基础特征，提供更高层次的认知状态估计。

### 6.1 Theta/Alpha比值 (theta_alpha_ratio)

**含义**: 认知负荷/努力程度指标。

**计算方法**:
```
TAR = Total_Theta_Power / Total_Alpha_Power
```

**解释**:
- 高TAR: 高认知负荷、心理努力
- 低TAR: 放松、低认知需求

---

### 6.2 额叶Beta比值 (frontal_beta_ratio)

**含义**: 额叶在Beta活动中的相对贡献。

**计算方法**:
```
FBR = Beta_Power_额叶 / Beta_Power_全脑
```

**额叶电极**: Fp1, Fp2, F3, F4, F7, F8, Fz

---

### 6.3 认知负荷估计 (cognitive_load_estimate)

**含义**: 综合认知负荷指数 (0-1)。

**计算方法**:
```
raw_score = 0.6 × TAR + 0.4 × FBR
cognitive_load = sigmoid(raw_score) = 1 / (1 + exp(-raw_score))
```

**解释**:
- 接近0: 低认知负荷
- 接近1: 高认知负荷

---

### 6.4 警觉度估计 (alertness_estimate)

**含义**: 觉醒/清醒程度 (0-1)。

**计算方法**:
```
alpha_delta_ratio = Alpha_Power / Delta_Power
alertness = sigmoid(2 × (alpha_delta_ratio - 0.5))
```

**解释**:
- 接近0: 困倦、睡眠
- 接近1: 清醒、警觉

---

### 6.5 放松指数 (relaxation_index)

**含义**: 放松程度 (0-1)。

**计算方法**:
```
relaxation_index = Alpha_Power / (Alpha_Power + Beta_Power)
```

**解释**:
- 接近0: 紧张、焦虑
- 接近1: 放松、平静

---

## 7. 网络拓扑特征 (Network Features)

**文件**: `network.py`
**类名**: `NetworkFeatures`

将脑电通道视为网络节点，使用图论方法分析脑网络结构。

### 7.1 聚类系数 (network_clustering_coefficient)

**含义**: 网络的局部连接密度。

**计算方法**:
```
1. 构建邻接矩阵（基于相干性，保留前30%最强连接）
2. 对每个节点i:
   C_i = 2 × 三角形数 / (k_i × (k_i - 1))
   其中 k_i = 节点i的度
3. C = mean(C_i)
```

**范围**: [0, 1]

**解释**: 高聚类系数表示节点的邻居之间也倾向于相互连接。

---

### 7.2 特征路径长度 (network_characteristic_path_length)

**含义**: 网络中节点对之间的平均最短路径长度。

**计算方法**:
```
d_ij = 节点i到j的最短路径长度
L = mean(d_ij)  对所有连通的 i≠j
```

**解释**: 较短的路径长度表示网络集成度高，信息传递效率高。

---

### 7.3 全局效率 (network_global_efficiency)

**含义**: 网络信息传递效率的度量。

**计算方法**:
```
E = (1 / (N×(N-1))) × Σ (1 / d_ij)  对所有 i≠j
```

**解释**: 全局效率是路径长度的逆指标，对断开的节点也有意义（效率=0）。

---

### 7.4 小世界指数 (network_small_world_index)

**含义**: 网络是否具有小世界特性。

**计算方法**:
```
σ = (C/C_random) / (L/L_random)

其中:
- C_random ≈ k/N (随机网络聚类系数)
- L_random ≈ ln(N)/ln(k) (随机网络路径长度)
- k = 平均度
```

**解释**:
- σ > 1: 小世界网络（高聚类、短路径）
- σ ≈ 1: 随机网络
- σ < 1: 规则格子网络

**神经科学意义**: 健康大脑通常呈现小世界特性，兼顾功能分离（高聚类）和功能整合（短路径）。

---

## 8. 微状态特征 (Microstate Features)

**文件**: `microstate.py`
**类名**: `MicrostateFeatures`, `MicrostateAnalyzer`

脑电微状态是持续约60-120ms的全脑电位地形图，反映大脑的瞬时功能状态。

### 微状态分析流程

```
1. 计算全局场强 GFP(t) = √(Σ(V_i(t) - mean(V))² / N)
2. 找到GFP局部峰值点
3. 对峰值点的地形图进行极性不变的K-Means聚类
4. 将所有时间点回溯分配到最近的微状态模板
5. 提取各微状态的时间统计特征
```

### 四种典型微状态

| 微状态 | 地形特征 | 认知关联 |
|--------|----------|----------|
| A | 左前-右后方向 | 语音处理 |
| B | 右前-左后方向 | 视觉处理 |
| C | 额-枕方向 | 显著性/注意力 |
| D | 额中央 | 注意力/执行功能 |

### 8.1-8.20 每个微状态的5个特征 (×4个状态 = 20个特征)

#### 平均持续时间 (Microstate_X_meandurs)

**含义**: 每次进入该微状态后的平均持续时间。

**计算方法**:
```
meandurs = Σ(每次持续时间) / 出现次数
```

**单位**: 秒

---

#### 出现率 (Microstate_X_occurrence)

**含义**: 单位时间内进入该微状态的次数。

**计算方法**:
```
occurrence = 出现次数 / 总时长
```

**单位**: Hz (次/秒)

---

#### 时间覆盖 (Microstate_X_timecov)

**含义**: 该微状态占总时间的比例。

**计算方法**:
```
timecov = 微状态总持续时间 / 记录总时长
```

**范围**: [0, 1]

---

#### 平均相关性 (Microstate_X_mean_corr)

**含义**: 各时间点与该微状态模板的平均相关程度。

**计算方法**:
```
mean_corr = mean(|correlation(EEG(t), Template_X)|)
对所有分配给X的时间点t
```

**范围**: [0, 1]

---

#### 全局解释方差 (Microstate_X_gev)

**含义**: 该微状态模板对数据方差的解释比例。

**计算方法**:
```
GEV = Σ(GFP(t)² × corr(t)²) / Σ(GFP(t)²)
对所有分配给该微状态的时间点
```

**范围**: [0, 1]

---

## 特征总览

### 按模块统计

| 模块 | 文件 | 特征数 |
|------|------|--------|
| 时域特征 | `time_domain.py` | 8 |
| 频域特征 | `frequency_domain.py` | 23 |
| 复杂度特征 | `complexity.py` | 7 |
| 微分熵特征 | `de_features.py` | 44 |
| 连接性特征 | `connectivity.py` | 12 |
| 复合特征 | `composite.py` | 5 |
| 网络特征 | `network.py` | 4 |
| 微状态特征 | `microstate.py` | 20 |
| **总计** | | **137** |

### 按应用场景分类

#### 情绪识别
- FAA (额叶Alpha不对称性)
- DASM/RASM (脑区不对称性)
- 各频带DE
- 放松指数

#### 认知负荷评估
- Theta/Beta比值
- 认知负荷估计
- 额叶Beta比值
- PLV (相位同步)

#### 睡眠/警觉监测
- Delta功率
- 警觉度估计
- Delta/Theta比值
- Hjorth参数

#### 病理诊断辅助
- 熵特征 (癫痫)
- 分形维度 (复杂性降低)
- 网络特征 (连接异常)
- 微状态参数 (精神分裂症等)

---

## 优化与实现细节

### GPU加速
以下模块支持 CuPy GPU加速:
- `TimeDomainFeatures`: Hjorth参数、零交叉率
- `ConnectivityFeatures`: 相关矩阵计算
- `MicrostateAnalyzer`: K-Means聚类

### 并行处理
- `ComplexityFeatures`: 使用 ThreadPoolExecutor 进行通道并行计算
- 矩阵运算: 向量化操作替代循环

### 安全值处理
- 比值计算限制在 [0.01, 100] 范围
- NaN/Inf 过滤
- 边界情况默认值处理

---

## 参考文献

1. Hjorth, B. (1970). EEG analysis based on time domain properties.
2. Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy.
3. Lachaux, J. P., et al. (1999). Measuring phase synchrony in brain signals.
4. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity.
5. Michel, C. M., & Koenig, T. (2018). EEG microstates as a tool for studying the temporal dynamics of whole-brain neuronal networks.
6. Davidson, R. J. (1998). Anterior electrophysiological asymmetries, emotion, and depression.
