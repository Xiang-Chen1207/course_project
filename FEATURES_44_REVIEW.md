# EEG 44个特征：计算方法代码审阅与合理性评估

本报告基于目录 `eeg_feature_extraction/` 的实现进行审阅，目标是：
1) 明确 44 个特征各自的**精确计算方法**（以代码为准）；
2) 判断这些计算是否**合理/常见**，以及可能的实现风险与改进建议。

> 结论先行：整体实现属于“经典手工特征 + 少量启发式综合指标”的组合，绝大多数特征定义是常见且可用的；但存在一些值得注意的问题：主频率峰值可能被 0Hz/DC 主导、频谱熵/1/f 指数实现偏“简化”、样本熵遇到无匹配时返回 0 可能产生偏差。

---

## 1. 特征清单与分组（共 44 个）

特征通过注册机制汇总：`FeatureRegistry.get_all_feature_names()`。

- **时域（8）**：`eeg_feature_extraction/features/time_domain.py`
- **频域（19）**：`eeg_feature_extraction/features/frequency_domain.py`
- **复杂度（4）**：`eeg_feature_extraction/features/complexity.py`
- **连接性（6）**：`eeg_feature_extraction/features/connectivity.py`
- **网络（4）**：`eeg_feature_extraction/features/network.py`
- **综合（3）**：`eeg_feature_extraction/features/composite.py`

---

## 2. 共有前置：PSD / 频带功率 / 相干性

### 2.1 PSD 计算（Welch）
代码：`eeg_feature_extraction/psd_computer.py::PSDComputer.compute_psd`

- 默认采样率：`Config.sampling_rate = 200Hz`
- Welch 参数（默认）：
  - `nperseg=256`（但会被 `min(nperseg, n_samples)` 截断；2s@200Hz 时 n_samples=400，所以仍为 256）
  - `noverlap=128`（默认 `nperseg//2`，同样会被 `min(noverlap, n_samples//2)` 截断）
  - `nfft=512`（频率分辨率约 $\Delta f = 200/512 \approx 0.3906\,Hz$）

**合理性**：Welch 是 EEG PSD 的常用做法，参数在 2 秒窗上也基本合理。

**注意点**：未见对原始信号做去趋势/去均值/带通/陷波处理的代码；如果上游数据包含 DC 漂移或工频噪声，会影响频域与连接特征。

### 2.2 频带功率计算（梯形积分）
代码：`PSDComputer._compute_band_power`

对每通道 PSD 在频带掩码内做梯形积分：
$$P_{band}(ch)=\int_{f_{low}}^{f_{high}} PSD_{ch}(f)\,df$$

- 频段（Config 默认）：
  - delta: 0.5–4
  - theta: 4–8
  - alpha: 8–13
  - beta: 13–30
  - gamma: 30–100

**重要细节**：功率积分采用半开区间掩码（例如 [0.5,4)、[4,8)），避免相邻频段在边界频点重复计入。

### 2.3 总功率（用于相对功率等）
代码：`PSDComputer.compute_psd`

总功率按 0.5–100Hz 积分（半开区间）：
$$P_{total}(ch)=\int_{0.5}^{100} PSD_{ch}(f)\,df$$

### 2.4 相干性（coherence）
代码：`PSDComputer.compute_coherence`

对每对通道使用 `scipy.signal.coherence` 计算相干谱 $C_{xy}(f)$，再对指定频带求平均：
$$\overline{C_{xy}}=\frac{1}{|F|}\sum_{f\in F} C_{xy}(f)$$

**合理性**：相干性是常见连接度量。

**注意点**：
- 这里是 CPU 双重循环 $O(N^2)$ 计算（62 通道约 1891 对），会比较慢；虽然有 `compute_coherence_gpu`，但上层未调用。
- 相干性容易受参考方式/体积传导影响；如果未做重参考（CAR 等），网络/连接特征的解释要谨慎。

---

## 3. 各特征详细计算方法与评估

下面按特征组逐一说明。为避免冗长，所有“全通道平均”均表示：先按通道得到每通道统计量，再对通道取均值（除“全通道平均幅值”是直接对全矩阵取均值，但与“先通道后平均”在多数情况下等价）。

### A. 时域特征（8）
代码：`eeg_feature_extraction/features/time_domain.py`

1) **全通道平均幅值**
- 计算：$\mathrm{mean}(|X|)$，其中 $X$ 为所有通道、所有时间点拼接。
- 评估：合理。若希望对通道数不敏感，现实现已平均化。

2) **全通道标准差**
- 计算：对每通道 $\sigma_{ch}=\mathrm{std}(x_{ch})$，再取 $\mathrm{mean}(\sigma_{ch})$。
- 评估：合理；注意对异常尖峰敏感。

3) **全通道峰峰值**
- 计算：$ptp_{ch}=\max(x_{ch})-\min(x_{ch})$，再均值。
- 评估：合理但对伪迹非常敏感（眼动/肌电）。

4) **全通道RMS能量**
- 计算：$rms_{ch}=\sqrt{\mathrm{mean}(x_{ch}^2)}$，再均值。
- 评估：合理。

5) **全通道零交叉率**
- 计算：对每通道统计符号变化次数 $N_{zc}$，再除以时长（秒）得到 $\mathrm{ZCR}=N_{zc}/T$，最后通道均值。
- 评估：常见但实现用 `np.sign`，当样本恰为 0 时可能引入额外交叉计数；一般问题不大。

6) **Hjorth活动性**
- 计算：每通道方差 $\mathrm{var}(x_{ch})$，再均值。
- 评估：合理。

7) **Hjorth移动性**
- 计算：$\sqrt{\mathrm{var}(\Delta x)/\mathrm{var}(x)}$（实现等价写法：$std(\Delta x)/std(x)$），再通道均值。
- 评估：合理且常见；但没有用采样间隔 $dt$ 归一化，因此数值与采样率相关（本项目采样率固定时影响可忽略）。

8) **Hjorth复杂度**
- 计算：$\frac{\sqrt{\mathrm{var}(\Delta^2 x)/\mathrm{var}(\Delta x)}}{\sqrt{\mathrm{var}(\Delta x)/\mathrm{var}(x)}}$，再通道均值。
- 评估：合理；同样依赖采样率。

---

### B. 频域特征（19）
代码：`eeg_feature_extraction/features/frequency_domain.py`

**前置**：使用 `psd_result.band_power`（每通道频带积分）与 `psd_result.total_power`（每通道 0.5–100Hz 积分）。

1–5) **Delta/Theta/Alpha/Beta/Gamma 波段绝对功率**
- 计算：对应频段每通道积分功率的通道均值。
- 评估：合理。

6–10) **Delta/Theta/Alpha/Beta/Gamma 波段相对功率**
- 计算：每通道相对功率 $P_{band}(ch)/P_{total}(ch)$，再通道均值。
- 评估：合理；相对功率是否严格归一还取决于频带覆盖范围与掩码定义。

11) **主频率峰值**
- 计算：每通道在全频率数组上取 `argmax(PSD)` 得到峰值频率，再取均值。
- 评估：概念合理，但实现**可能被 0Hz/DC 或极低频漂移主导**（Welch 输出含 0Hz）。通常建议限制在例如 1–40Hz 或 1–100Hz。

12) **频谱熵**
- 计算：每通道将 PSD 归一化为概率分布 $p(f)=PSD/\sum PSD$，然后用 `scipy.stats.entropy(p)`（自然对数）求熵，再均值。
- 评估：合理，但熵值未按 $\log(N)$ 归一化，数值范围与频点数相关，跨不同 `nfft` 时不直接可比。

13) **频谱质心**
- 计算：$\sum f\cdot PSD(f) / \sum PSD(f)$，每通道算后均值。
- 评估：合理。

14) **个体Alpha频率（IAF）**
- 计算：在 8–13Hz 频带内对每通道取峰值频率，再均值。
- 评估：合理。

15) **Theta-Beta比率**
- 计算：全通道 theta 平均功率 / 全通道 beta 平均功率。
- 评估：常见指标；注意受肌电影响（beta）以及任务/年龄等因素。

16) **Delta-Theta比率**
- 计算：全通道 delta 平均功率 / 全通道 theta 平均功率。
- 评估：可用；对慢漂移与眼动伪迹敏感。

17) **低频vs高频能量比**
- 计算：低频 1–8Hz 与高频 13–40Hz 的 PSD 积分比值（每通道算比再均值）。
- 评估：合理；注意未剔除 alpha（8–13Hz）被排除在两侧之外，这是设计选择。

18) **非周期性指数（1/f 斜率）**
- 计算：优先使用 FOOOF 在 2–40Hz 上拟合 aperiodic 成分，提取 exponent；若 FOOOF 不可用/拟合失败则回退为对数线性拟合。
- 评估：FOOOF 相比简单线性拟合更稳健，但仍会受输入 PSD 质量与峰拟合参数影响。

19) **总平均功率**
- 计算：每通道总功率 $\int_{0.5}^{100} PSD(f)df$，再通道均值。
- 评估：合理。

---

### C. 复杂度特征（4）
代码：`eeg_feature_extraction/features/complexity.py`

1) **小波能量熵**
- 计算：对每通道做 `pywt.wavedec(signal, wavelet='db4', level=5)`，计算每层系数能量 $E_i=\sum c_i^2$，归一化后取 Shannon 熵 $H=-\sum p_i\log p_i$，最后通道均值。
- 评估：合理；若信号长度不足导致小波分解失败，会被 `try/except` 跳过该通道，可能造成“有效通道数”波动。

2) **样本熵（SampEn）**
- 参数：$m=2$，$r=0.2\cdot \mathrm{std}$。
- 计算：按 Chebyshev 距离匹配模板向量，得到 $A,B$ 后返回 $-\log(A/B)$。
- 评估：核心思路正确；但实现中当 $A==0$ 或 $B==0$ 时直接返回 0，而理论上应趋于 $+\infty$（或用平滑/下界）。这会在“非常规则”或“非常短/噪声”通道上引入偏差。

3) **近似熵（ApEn）**
- 参数：$m=2$，$r=0.2\cdot \mathrm{std}$。
- 计算：对每模板计算匹配比例 $C_i^m(r)$（包含自匹配），取 $\phi(m)=\mathrm{mean}(\log C_i^m)$，最后返回 $\phi(m)-\phi(m+1)$。
- 评估：合理。

4) **Hurst指数**
- 计算：R/S 分析，分尺度切片，拟合 $\log(R/S)$ 与 $\log(n)$ 的斜率作为 H。
- 评估：合理；实现对短序列会回退到 0.5。

---

### D. 连接性特征（6）
代码：`eeg_feature_extraction/features/connectivity.py`

1) **通道间平均相关系数**
- 计算：对通道矩阵做 `np.corrcoef(eeg_data)` 得相关矩阵，取上三角（去对角线）均值。
- 评估：合理但强烈依赖参考方式；也容易受共同噪声影响。

2) **全脑平均连接强度**
- 计算：用 alpha(8–13Hz) 相干性矩阵，取上三角均值。
- 评估：合理。

3) **左右半球间连接强度**
- 计算：左半球通道集 × 右半球通道集的相干性均值。
- 评估：合理；通道分组来自 `Config.ChannelGroups`，适用于 SEED 62 通道布局。

4) **频带间功率相关性**
- 计算：取每通道 alpha_power 与 beta_power 两个向量，对通道维做 Pearson 相关。
- 评估：实现正确，但语义是“空间分布上的共变”，不是时间上的 cross-frequency coupling，命名/解释时需避免误解。

5) **左右半球功率不对称性**
- 计算：基于 alpha 功率，$ (R-L)/(R+L) $。
- 评估：合理，是常见不对称指标。

6) **前后脑区功率梯度**
- 计算：前额叶 alpha 均值 / 枕叶 alpha 均值。
- 评估：合理；对分母过小做了保护。

---

### E. 网络特征（4）
代码：`eeg_feature_extraction/features/network.py`

前置：使用 alpha 相干性矩阵构建二值邻接矩阵。

- 阈值：`Config.network_threshold=0.3`，通过上三角百分位得到 `threshold_value`，然后 `matrix >= threshold_value` 置 1。

1) **网络聚类系数**
- 计算：对每节点 i，统计其邻居间实际边数 / 最大可能边数，最后节点均值。
- 评估：合理。

2) **网络特征路径长度**
- 计算：Floyd–Warshall 最短路，取有限距离（>0）的平均值；若全部不连通则返回 N。
- 评估：实现正确；但“返回 N”是一个强行上界，可能使该特征在不连通图上跳变很大。

3) **网络全局效率**
- 计算：$E=\frac{1}{N(N-1)}\sum_{i\ne j}1/d_{ij}$，忽略不可达对。
- 评估：合理。

4) **网络小世界属性**
- 计算：$\sigma=(C/C_{rand})/(L/L_{rand})$，其中 $C_{rand}\approx k/N$，$L_{rand}\approx \ln N/\ln k$，$k$ 近似为 $N\cdot threshold$。
- 评估：公式方向对，但 `k` 的估计非常粗糙（并非由实际图的平均度计算），因此该指标更像“启发式 proxy”，建议在论文级分析中谨慎。

---

### F. 综合特征（3）
代码：`eeg_feature_extraction/features/composite.py`

这些特征本质是“手工规则/启发式评分”，并非从数据学习得到的可验证心理量表。

1) **认知负荷水平估计**
- 计算：
  - 全脑 $theta/alpha$ 比率
  - 前额 beta 与全脑 beta 的比值（frontal_beta_norm）
  - 原始分数：$raw=0.6\cdot (theta/alpha) + 0.4\cdot frontal\_beta\_norm$
  - Sigmoid：$score=\sigma(2\cdot(raw-1))$ 并裁剪到 [0,1]
- 评估：方向与常见经验一致，但参数/阈值（0.6/0.4、平移 1、斜率 2）没有数据驱动依据；更适合作为“可解释的启发式特征”，不应当作严格心理测量。

2) **清醒度水平估计**
- 计算：全脑 $alpha/delta$，再做 $\sigma(2\cdot(ratio-0.5))$ 映射到 [0,1]。
- 评估：方向合理，但同样是启发式映射。

3) **放松vs紧张状态判别**
- 计算：$relax=\frac{Alpha}{Alpha+Beta}$，裁剪到 [0,1]。
- 评估：合理且可解释。

---

## 4. 总体合理性判断（简要）

**总体可用**：
- 时域统计、频带功率、Hjorth、熵类、相关/相干、基本图指标均是 EEG 特征工程中的常见做法。

**主要风险/改进建议**（按重要性）：
1) **主频率峰值**建议限制频段（例如 1–40Hz 或 1–100Hz）以避免 DC 主导。
2) **频带边界重叠**建议改为半开区间或明确边界归属。
3) **非周期性指数**实现偏简化；如要严谨估计 1/f，建议采用更稳健方法（去峰/分离振荡与背景）。
4) **样本熵**在无匹配时返回 0 可能偏差；建议返回 `np.inf` 或使用平滑（例如加 1 计数）并在下游处理。
5) **feature.csv 与代码不一致**：尤其是“总平均功率”的频段范围，建议统一（改代码或改文档）。
6) 上游若未做**基础预处理**（去趋势/陷波/带通/重参考），连接与网络特征解释会受影响。

---

## 5. 参考：默认配置
代码：`eeg_feature_extraction/config.py`

- 采样率：200Hz
- 段长：2s（400 点）
- 频带：delta/theta/alpha/beta/gamma 如上
- Welch：nperseg=256, noverlap=128, nfft=512
- 网络阈值：保留最强 30% 连接
