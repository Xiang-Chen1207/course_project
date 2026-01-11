# EEG 特征提取方法详解

本文档详细介绍了 EEG 特征提取模块中实现的 **44 个特征**的计算方法、数学公式、参数设置及实现评估。

---

## 目录

1. [时域特征 (8个)](#1-时域特征-8个)
2. [频域特征 (19个)](#2-频域特征-19个)
3. [复杂度特征 (4个)](#3-复杂度特征-4个)
4. [连接性特征 (6个)](#4-连接性特征-6个)
5. [网络特征 (4个)](#5-网络特征-4个)
6. [综合特征 (3个)](#6-综合特征-3个)
7. [实现评估总结](#7-实现评估总结)
8. [性能瓶颈分析](#8-性能瓶颈分析)

---

## 1. 时域特征 (8个)

文件位置：`eeg_feature_extraction/features/time_domain.py`

### 1.1 全通道平均幅值

**描述**：衡量 EEG 信号的整体强度水平。

**公式**：
```
Mean_Amplitude = (1/NC) × Σ_c Σ_t |x_c(t)| / T
```
其中 NC 为通道数，T 为时间点数。

**实现代码**：
```python
mean_amplitude = np.mean(np.abs(eeg_data))
```

**评估**：✅ 合理 - 标准的信号幅值度量方法。

---

### 1.2 全通道标准差

**描述**：衡量各通道信号的变异程度。

**公式**：
```
Mean_Std = (1/NC) × Σ_c std(x_c)
```

**实现代码**：
```python
channel_stds = np.std(eeg_data, axis=1)
mean_std = np.mean(channel_stds)
```

**评估**：✅ 合理 - 标准统计量。

---

### 1.3 全通道峰峰值

**描述**：信号振幅的动态范围。

**公式**：
```
Mean_PTP = (1/NC) × Σ_c [max(x_c) - min(x_c)]
```

**实现代码**：
```python
peak_to_peak = np.max(eeg_data, axis=1) - np.min(eeg_data, axis=1)
mean_ptp = np.mean(peak_to_peak)
```

**评估**：✅ 合理 - 常用的振幅范围度量。

---

### 1.4 全通道 RMS 能量

**描述**：信号的均方根能量，反映信号强度。

**公式**：
```
RMS = sqrt((1/T) × Σ_t x(t)²)
Mean_RMS = (1/NC) × Σ_c RMS_c
```

**实现代码**：
```python
rms = np.sqrt(np.mean(eeg_data ** 2, axis=1))
mean_rms = np.mean(rms)
```

**评估**：✅ 合理 - 标准的能量度量方法。

---

### 1.5 全通道零交叉率

**描述**：信号穿越零点的频率，反映信号的振荡特性。

**公式**：
```
ZCR = (1/Duration) × Σ_t 1{sign(x(t)) ≠ sign(x(t-1))}
Mean_ZCR = (1/NC) × Σ_c ZCR_c
```

**实现代码**：
```python
sign_changes = np.sum(np.abs(np.diff(np.sign(eeg_data), axis=1)) > 0, axis=1)
zcr_per_sec = sign_changes / duration
```

**评估**：✅ 合理 - 经典的信号特征提取方法。

---

### 1.6 Hjorth 活动性 (Activity)

**描述**：Hjorth 参数之一，代表信号方差，反映信号的功率。

**公式**：
```
Activity = Var(x) = E[(x - μ)²]
Mean_Activity = (1/NC) × Σ_c Var(x_c)
```

**实现代码**：
```python
var_x = np.var(eeg_data, axis=1)
activity = np.mean(var_x)
```

**评估**：✅ 合理 - 标准 Hjorth 参数定义。

---

### 1.7 Hjorth 移动性 (Mobility)

**描述**：信号一阶导数标准差与信号标准差的比值，反映平均频率。

**公式**：
```
Mobility = sqrt(Var(x') / Var(x))
```
其中 x' 是信号的一阶导数（差分近似）。

**实现代码**：
```python
d1 = np.diff(eeg_data, axis=1)
mobility = np.sqrt(var_d1) / np.sqrt(var_x)
```

**评估**：✅ 合理 - 正确实现了 Hjorth 移动性公式。

---

### 1.8 Hjorth 复杂度 (Complexity)

**描述**：一阶导数的移动性与原信号移动性的比值，反映信号的带宽复杂程度。

**公式**：
```
Complexity = Mobility(x') / Mobility(x)
```

**实现代码**：
```python
mobility_d1 = np.sqrt(var_d2) / np.sqrt(var_d1)
complexity = mobility_d1 / mobility
```

**评估**：✅ 合理 - 正确实现了 Hjorth 复杂度公式。

---

## 2. 频域特征 (19个)

文件位置：`eeg_feature_extraction/features/frequency_domain.py`

### 频段定义

| 频段 | 频率范围 (Hz) |
|------|---------------|
| Delta | 0.5 - 4.0 |
| Theta | 4.0 - 8.0 |
| Alpha | 8.0 - 13.0 |
| Beta | 13.0 - 30.0 |
| Gamma | 30.0 - 100.0 |

---

### 2.1-2.5 各频段绝对功率 (5个)

**描述**：特定频段内的信号功率积分。

**公式**：
```
Band_Power = ∫_{f_low}^{f_high} PSD(f) df
Mean_Band_Power = (1/NC) × Σ_c Band_Power_c
```

**实现方法**：使用 Welch 方法计算 PSD，然后对指定频段进行梯形积分。

**Welch 参数**：
- `nperseg = 256`（窗口长度）
- `noverlap = 128`（重叠长度）
- `nfft = 512`（FFT 点数）

**实现代码**：
```python
band_mask = (freqs >= low) & (freqs <= high)
band_power = trapezoid(psd[:, band_mask], dx=freq_resolution, axis=1)
```

**评估**：✅ 合理 - 标准的频段功率计算方法。

---

### 2.6-2.10 各频段相对功率 (5个)

**描述**：各频段功率占总功率的比例。

**公式**：
```
Relative_Power = Band_Power / Total_Power
```

**实现代码**：
```python
rel_power = np.where(total_power > 1e-10, power / total_power, 0)
```

**评估**：✅ 合理 - 标准的归一化方法，含除零保护。

---

### 2.11 主频率峰值

**描述**：PSD 最大值对应的频率。

**公式**：
```
Peak_Freq = argmax_f(PSD(f))
Mean_Peak_Freq = (1/NC) × Σ_c Peak_Freq_c
```

**实现代码**：
```python
peak_indices = np.argmax(psd, axis=1)
peak_freqs = freqs[peak_indices]
```

**评估**：✅ 合理 - 但在低信噪比情况下可能不稳定。

---

### 2.12 频谱熵

**描述**：PSD 的 Shannon 熵，衡量频谱的平坦程度/复杂度。

**公式**：
```
Spectral_Entropy = -Σ_f p(f) × log(p(f))
其中 p(f) = PSD(f) / Σ_f PSD(f)
```

**实现代码**：
```python
psd_norm = ch_psd / (np.sum(ch_psd) + 1e-10)
ent = entropy(psd_norm)  # scipy.stats.entropy
```

**评估**：✅ 合理 - 标准的频谱熵定义。

---

### 2.13 频谱质心

**描述**：频谱的"重心"位置，反映信号的平均频率。

**公式**：
```
Spectral_Centroid = Σ_f (f × PSD(f)) / Σ_f PSD(f)
```

**实现代码**：
```python
centroid = np.sum(freqs * ch_psd) / total_power
```

**评估**：✅ 合理 - 标准的频谱质心定义。

---

### 2.14 个体 Alpha 频率 (IAF)

**描述**：Alpha 频段 (8-13 Hz) 内 PSD 峰值对应的频率。

**公式**：
```
IAF = argmax_{f ∈ [8,13]} PSD(f)
```

**实现代码**：
```python
alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
peak_indices = np.argmax(alpha_psd, axis=1)
```

**评估**：✅ 合理 - IAF 是重要的个体差异指标。缺省值 10.0 Hz 是合理的。

---

### 2.15 Theta-Beta 比率 (TBR)

**描述**：Theta 功率与 Beta 功率的比值，与注意力和 ADHD 相关。

**公式**：
```
TBR = Theta_Power / Beta_Power
```

**实现代码**：
```python
tbr = theta_power / beta_power if beta_power > 1e-10 else 0.0
```

**评估**：✅ 合理 - 临床常用指标。

---

### 2.16 Delta-Theta 比率

**描述**：Delta 与 Theta 功率比，反映慢波活动比例。

**公式**：
```
DTR = Delta_Power / Theta_Power
```

**评估**：✅ 合理 - 用于睡眠/疲劳研究。

---

### 2.17 低频 vs 高频能量比

**描述**：低频段 (1-8 Hz) 与高频段 (13-40 Hz) 功率比（功率积分使用半开区间掩码）。

**公式**：
```
Low_High_Ratio = ∫_{1}^{8} PSD(f) df / ∫_{13}^{40} PSD(f) df   (实现采用 [low, high) 半开区间)
```

**实现代码**：
```python
low_power = trapezoid(psd[:, low_mask], dx=freq_resolution, axis=1)
high_power = trapezoid(psd[:, high_mask], dx=freq_resolution, axis=1)
ratios = low_power / high_power
```

**评估**：✅ 合理 - 用于区分不同认知状态。

---

### 2.18 非周期性指数 (1/f 斜率)

**描述**：PSD 的 1/f 特性斜率，反映神经噪声特性。

**公式**：
```
log(PSD) = -α × log(f) + c
非周期性指数 = α
```

**实现代码（当前实现：优先 FOOOF，失败回退线性拟合）**：
```python
fg = FOOOFGroup(aperiodic_mode='fixed', max_n_peaks=3, peak_width_limits=(1, 12),
                peak_threshold=2.0, verbose=False)
fg.fit(freqs=freqs_fit, power_spectra=psd_fit_pos, freq_range=(2.0, 40.0))

fm = fg.get_fooof(ind=ch_idx, regenerate=True)
exponent = fm.get_params('aperiodic_params', 'exponent')
```

**参数**：
- 拟合频率范围：2-40 Hz
- aperiodic_mode：fixed
- max_n_peaks：3（用于拟合并剥离窄带峰）

**评估**：✅ 合理 - FOOOF 比简化线性拟合更稳健，但结果仍取决于 PSD 质量与峰拟合超参数。

---

### 2.19 总平均功率

**描述**：0.5-100 Hz 范围内的总功率（实现采用半开区间 [0.5, 100)）。

**公式**：
```
Total_Power = ∫_{0.5}^{100} PSD(f) df
```

**评估**：✅ 合理 - 基础的信号能量指标。

---

## 3. 复杂度特征 (4个)

文件位置：`eeg_feature_extraction/features/complexity.py`

### 3.1 小波能量熵

**描述**：小波分解各层能量分布的熵值。

**公式**：
```
E_i = Σ_j |c_i(j)|²   (第 i 层系数能量)
p_i = E_i / Σ_i E_i   (归一化能量)
Wavelet_Entropy = -Σ_i p_i × log(p_i)
```

**参数**：
- 小波基：`db4` (Daubechies-4)
- 分解层数：5 层

**实现代码**：
```python
coeffs = pywt.wavedec(ch_data, 'db4', level=5)
energies = [np.sum(c ** 2) for c in coeffs]
probs = energies / total_energy
ent = entropy(probs)
```

**评估**：✅ 合理 - 标准的小波熵计算方法。

---

### 3.2 样本熵 (Sample Entropy)

**描述**：衡量时间序列复杂度和不可预测性，对自匹配不敏感。

**公式**：
```
SampEn(m, r, N) = -ln(A/B)
其中：
- B = 长度为 m 的模板匹配数
- A = 长度为 m+1 的模板匹配数
- r = 容差阈值 = 0.2 × std(signal)
```

**参数**：
- 嵌入维度 m = 2
- 容差系数 r_ratio = 0.2

**实现代码**：
```python
def count_matches(templates, r):
    for i in range(N_t):
        for j in range(i + 1, N_t):
            if np.max(np.abs(templates[i] - templates[j])) < r:
                count += 2
    return count
```

**评估**：⚠️ 需要优化
- **正确性**：公式实现正确
- **问题**：O(N²) 时间复杂度，是主要性能瓶颈
- **建议**：可以使用 KD-Tree 或向量化方法加速

---

### 3.3 近似熵 (Approximate Entropy)

**描述**：与样本熵类似，但包含自匹配。

**公式**：
```
ApEn(m, r, N) = φ(m) - φ(m+1)
其中：
φ(m) = (1/N-m+1) × Σ_i log(C_i^m / (N-m+1))
C_i^m = 与模板 i 相似的模板数
```

**实现代码**：
```python
def phi(m):
    for i in range(n):
        distances = np.max(np.abs(templates - templates[i]), axis=1)
        counts[i] = np.sum(distances < r)
    return np.mean(np.log(counts / n))
```

**评估**：⚠️ 需要优化
- **正确性**：实现正确
- **问题**：O(N²) 复杂度
- **建议**：使用向量化的距离计算

---

### 3.4 Hurst 指数

**描述**：衡量时间序列的长期记忆性/自相似性。

**公式**：
```
R/S 分析法：
1. 将序列分成长度为 n 的子段
2. 对每个子段计算 R/S = (max(累积离差) - min(累积离差)) / std
3. log(R/S) = H × log(n) + c
4. H = 回归斜率
```

**解释**：
- H = 0.5：随机游走（无记忆）
- H > 0.5：持久性（趋势延续）
- H < 0.5：反持久性（均值回归）

**参数**：
- 最小分割尺度：2^4 = 16
- 最大分割尺度：2^(log2(N)-1)

**实现代码**：
```python
for n in n_values:
    for segment in segments:
        y = np.cumsum(segment - mean)
        R = np.max(y) - np.min(y)
        S = np.std(segment, ddof=1)
        rs_list.append(R / S)
coeffs = np.polyfit(log_n, log_rs, 1)
return coeffs[0]
```

**评估**：✅ 合理 - R/S 分析是经典的 Hurst 指数估计方法。

---

## 4. 连接性特征 (6个)

文件位置：`eeg_feature_extraction/features/connectivity.py`

### 4.1 通道间平均相关系数

**描述**：所有通道对之间的 Pearson 相关系数均值。

**公式**：
```
Mean_Corr = (2 / NC(NC-1)) × Σ_{i<j} corr(x_i, x_j)
```

**实现代码**：
```python
corr_matrix = np.corrcoef(eeg_data)
upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
return np.mean(upper_tri)
```

**评估**：✅ 合理 - 标准的相关性度量。

---

### 4.2 全脑平均连接强度

**描述**：基于 Alpha 频段相干性的全脑连接度量。

**公式**：
```
Coherence(i,j) = |S_xy(f)|² / (S_xx(f) × S_yy(f))
Mean_Coherence = mean(Coh_{i,j}) for all i<j
```

**参数**：
- 频段：Alpha (8-13 Hz)

**实现代码**：
```python
coherence_matrix = psd_computer.compute_coherence(eeg_data, band=(8.0, 13.0))
upper_tri = coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)]
mean_coherence = np.mean(upper_tri)
```

**评估**：⚠️ 性能瓶颈
- **正确性**：实现正确
- **问题**：需要计算 NC×(NC-1)/2 = 1891 对相干性（62通道）
- **建议**：考虑使用 GPU 加速或采样策略

---

### 4.3 左右半球间连接强度

**描述**：左脑通道与右脑通道之间的平均相干性。

**公式**：
```
LR_Connectivity = mean(Coh_{l,r}) for l ∈ Left, r ∈ Right
```

**通道分组**：
- 左半球：27 个通道
- 右半球：27 个通道

**评估**：✅ 合理 - 用于研究半球间交互。

---

### 4.4 频带间功率相关性

**描述**：Alpha 与 Beta 功率在各通道间的 Pearson 相关性。

**公式**：
```
Band_Corr = corr(Alpha_power_vector, Beta_power_vector)
```
其中向量维度为通道数。

**实现代码**：
```python
corr = np.corrcoef(alpha_power, beta_power)[0, 1]
```

**评估**：✅ 合理 - 用于研究跨频段耦合。

---

### 4.5 左右半球功率不对称性

**描述**：左右半球 Alpha 功率的不对称指数。

**公式**：
```
Asymmetry = (Right_Alpha - Left_Alpha) / (Right_Alpha + Left_Alpha)
```

**解释**：
- 正值：右半球活动占优
- 负值：左半球活动占优
- 零：对称

**评估**：✅ 合理 - 经典的脑电不对称指标，与情绪状态相关。

---

### 4.6 前后脑区功率梯度

**描述**：前额叶与枕叶 Alpha 功率比。

**公式**：
```
AP_Gradient = Frontal_Alpha / Occipital_Alpha
```

**通道分组**：
- 前额叶：14 个通道
- 枕叶：12 个通道

**评估**：✅ 合理 - 用于研究前后脑区差异。

---

## 5. 网络特征 (4个)

文件位置：`eeg_feature_extraction/features/network.py`

### 预处理：网络构建

从相干性矩阵构建二值邻接矩阵：
- 保留前 30% 最强连接
- 移除对角线（自环）

```python
threshold_value = np.percentile(upper_tri, 70)  # 保留top 30%
adj[matrix >= threshold_value] = 1
```

---

### 5.1 网络聚类系数

**描述**：网络中三角形的密度，反映局部连通性。

**公式**：
```
C_i = 2 × T_i / (k_i × (k_i - 1))
其中：
- T_i = 节点 i 邻居之间的连接数
- k_i = 节点 i 的度
Mean_CC = (1/N) × Σ_i C_i
```

**实现代码**：
```python
for i in range(n):
    neighbors = np.where(adj_matrix[i] > 0)[0]
    k = len(neighbors)
    for j, l in combinations(neighbors, 2):
        if adj_matrix[neighbors[j], neighbors[l]] > 0:
            neighbor_connections += 1
    cc = neighbor_connections / (k * (k - 1) / 2)
```

**评估**：⚠️ 可优化
- **正确性**：实现正确
- **问题**：O(N × k²) 复杂度
- **建议**：可用矩阵运算加速

---

### 5.2 网络特征路径长度

**描述**：所有节点对之间最短路径的平均长度。

**公式**：
```
L = (1 / N(N-1)) × Σ_{i≠j} d(i,j)
```

**算法**：Floyd-Warshall

**实现代码**：
```python
for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i, k] + dist[k, j] < dist[i, j]:
                dist[i, j] = dist[i, k] + dist[k, j]
```

**评估**：⚠️ 严重性能瓶颈
- **正确性**：实现正确
- **问题**：O(N³) = 238,328 次操作（62节点）
- **建议**：
  - 使用 scipy.sparse.csgraph.shortest_path
  - 或 NetworkX 库优化

---

### 5.3 网络全局效率

**描述**：基于最短路径倒数的网络效率度量。

**公式**：
```
E = (1 / N(N-1)) × Σ_{i≠j} 1/d(i,j)
```

**评估**：⚠️ 代码重复
- **问题**：重复计算了 Floyd-Warshall
- **建议**：合并 `特征路径长度` 和 `全局效率` 的计算

---

### 5.4 网络小世界属性

**描述**：衡量网络是否具有小世界特性（高聚类 + 短路径）。

**公式**：
```
σ = (C/C_random) / (L/L_random)

随机网络参数估计：
- C_random ≈ k / N
- L_random ≈ ln(N) / ln(k)
```

**解释**：
- σ > 1：小世界网络
- σ ≈ 1：随机网络

**评估**：⚠️ 可改进
- 随机网络参数使用了简化估计
- 更精确的方法需要生成多个随机网络取平均

---

## 6. 综合特征 (3个)

文件位置：`eeg_feature_extraction/features/composite.py`

### 6.1 认知负荷水平估计

**描述**：基于 EEG 指标的认知负荷综合评估。

**公式**：
```
Raw_Score = 0.6 × (Theta/Alpha) + 0.4 × (Frontal_Beta / Mean_Beta)
Cognitive_Load = sigmoid(2 × (Raw_Score - 1))
              = 1 / (1 + exp(-2 × (Raw_Score - 1)))
```

**理论依据**：
- 高 Theta/Alpha 比表示工作记忆负荷增加
- 高前额 Beta 活动与注意力/认知控制相关

**输出范围**：0-1（归一化）

**评估**：⚠️ 可商榷
- **合理性**：公式基于文献中的认知负荷指标
- **局限性**：权重 (0.6, 0.4) 是经验值，可能需要针对具体数据集调整
- **建议**：参数可配置化

---

### 6.2 清醒度水平估计

**描述**：基于 Alpha/Delta 比的清醒度评估。

**公式**：
```
Alpha_Delta_Ratio = Total_Alpha / Total_Delta
Alertness = sigmoid(2 × (Alpha_Delta_Ratio - 0.5))
```

**理论依据**：
- 清醒状态：Alpha 活动增强
- 困倦/睡眠：Delta 活动增强

**输出范围**：0-1（1 = 高清醒度）

**评估**：✅ 合理 - 基于经典的睡眠/清醒 EEG 指标。

---

### 6.3 放松 vs 紧张状态判别

**描述**：区分放松与紧张/警觉状态。

**公式**：
```
Relaxation = Alpha / (Alpha + Beta)
```

**理论依据**：
- 放松：Alpha 活动主导
- 紧张/警觉：Beta 活动增强

**输出范围**：0-1（1 = 放松，0 = 紧张）

**评估**：✅ 合理 - 简单有效的状态指标。

---

## 7. 实现评估总结

### 评估等级说明

| 等级 | 含义 |
|------|------|
| ✅ | 实现正确且高效 |
| ⚠️ | 实现正确但有性能或设计问题 |
| ❌ | 实现有误或严重问题 |

### 各类特征评估汇总

| 类别 | 特征数 | ✅ | ⚠️ | ❌ |
|------|--------|----|----|-----|
| 时域 | 8 | 8 | 0 | 0 |
| 频域 | 19 | 19 | 0 | 0 |
| 复杂度 | 4 | 2 | 2 | 0 |
| 连接性 | 6 | 5 | 1 | 0 |
| 网络 | 4 | 0 | 4 | 0 |
| 综合 | 3 | 2 | 1 | 0 |
| **总计** | **44** | **36** | **8** | **0** |

---

## 8. 性能瓶颈分析

### 高耗时特征排序（预估）

| 排名 | 特征 | 复杂度 | 问题 |
|------|------|--------|------|
| 1 | 样本熵 | O(N²×C) | 双重循环匹配 |
| 2 | 近似熵 | O(N²×C) | 双重循环匹配 |
| 3 | 网络特征路径长度 | O(N³) | Floyd-Warshall |
| 4 | 网络全局效率 | O(N³) | Floyd-Warshall（重复）|
| 5 | 全脑平均连接强度 | O(C²×N) | 相干性计算 |
| 6 | 网络聚类系数 | O(C×k²) | 三角形计数 |
| 7 | Hurst 指数 | O(N×logN) | 分段计算 |

其中：N = 时间点数，C = 通道数 (62)

### 优化建议

1. **熵计算**：使用 `antropy` 或 `neurokit2` 库的优化实现
2. **网络分析**：使用 `networkx` 或 `scipy.sparse.csgraph`
3. **相干性**：利用 FFT 批量计算，使用 GPU 加速
4. **缓存**：Floyd-Warshall 结果可在多个特征间共享

---

## 附录：参数配置参考

```python
# 采样参数
sampling_rate = 200.0  # Hz
n_channels = 62
segment_length = 2.0   # 秒

# PSD 参数
nperseg = 256
noverlap = 128
nfft = 512

# 熵参数
sample_entropy_m = 2
sample_entropy_r_ratio = 0.2

# 小波参数
wavelet = 'db4'
wavelet_level = 5

# 网络参数
network_threshold = 0.3
```

---

*文档生成时间：2026-01-09*
*适用版本：EEG Feature Extraction v1.0*
