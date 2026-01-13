# EEG 特征提取代码分析报告

## 目录
1. [代码架构问题分析](#代码架构问题分析)
   - [merged_segment_extraction.py](#merged_segment_extractionpy)
   - [feature_extractor.py](#feature_extractorpy)
2. [特征计算详细说明与问题分析](#特征计算详细说明与问题分析)
   - [时域特征 (Time Domain)](#时域特征-time-domain)
   - [频域特征 (Frequency Domain)](#频域特征-frequency-domain)
   - [复杂度特征 (Complexity)](#复杂度特征-complexity)
   - [连接性特征 (Connectivity)](#连接性特征-connectivity)
   - [网络特征 (Network)](#网络特征-network)
   - [综合特征 (Composite)](#综合特征-composite)
   - [微分熵特征 (DE Features)](#微分熵特征-de-features)
   - [微状态特征 (Microstate)](#微状态特征-microstate)
3. [改进建议汇总](#改进建议汇总)

---

## 代码架构问题分析

### merged_segment_extraction.py

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 代码位置 |
|---------|---------|---------|---------|
| M1 | **高** | `signal.SIGALRM` 只在 Unix/Linux 系统上可用，Windows 上会报错 | 行 260-266 |
| M2 | **中** | 超时机制在多进程子进程中使用 SIGALRM 可能产生竞态条件 | `_run_with_timeout()` |
| M3 | **低** | `_get_config_dict()` 的 pickle 序列化可能对复杂嵌套对象失败 | 行 472-488 |
| M4 | **低** | `FEATURE_TIMEOUT_SEC = 30` 硬编码，不可配置 | 行 216 |

#### 详细分析

**M1: 跨平台兼容性问题**
```python
def _run_with_timeout(self, fn, timeout_sec: int, *args, **kwargs):
    def _handler(signum, frame):
        raise TimeoutError("feature computation timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)  # Windows 不支持
    signal.alarm(timeout_sec)  # Windows 不支持
```
**建议**: 使用 `concurrent.futures.ThreadPoolExecutor` 配合 `timeout` 参数，或使用 `multiprocessing` 的超时机制。

**M2: 多进程中的信号处理**
在 `ProcessPoolExecutor` 的子进程中设置 SIGALRM 可能会与主进程产生冲突，且信号只能被主线程捕获。

---

### feature_extractor.py

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 代码位置 |
|---------|---------|---------|---------|
| F1 | **低** | `n_jobs` 计算逻辑 `cpu_count - 10` 可能导致服务器资源浪费 | 行 35-39 |
| F2 | **低** | 每次 `_rebuild_computers()` 都会重新实例化所有特征计算器 | 行 155-174 |
| F3 | **中** | `time_length` 兼容性处理可能丢失精度 | 行 84-90 |

#### 详细分析

**F1: n_jobs 计算**
```python
def _get_optimal_n_jobs() -> int:
    cpu_count = mp.cpu_count()
    return max(1, cpu_count - 10)  # 小型服务器上可能浪费资源
```
**建议**: 改为 `max(1, int(cpu_count * 0.75))` 或使其可配置。

---

## 特征计算详细说明与问题分析

### 时域特征 (Time Domain)

**文件**: `eeg_feature_extraction/features/time_domain.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `mean_abs_amplitude` | $\bar{\|x\|} = \frac{1}{N \cdot C} \sum_{c=1}^{C} \sum_{i=1}^{N} \|x_{c,i}\|$ | 信号平均绝对幅值，反映整体能量水平 | μV 级别 |
| `mean_channel_std` | $\bar{\sigma} = \frac{1}{C} \sum_{c=1}^{C} \sigma_c$ | 通道标准差均值，反映信号变异性 | μV 级别 |
| `mean_peak_to_peak` | $\bar{PTP} = \frac{1}{C} \sum_{c=1}^{C} (\max(x_c) - \min(x_c))$ | 峰峰值均值，反映信号动态范围 | μV 级别 |
| `mean_rms` | $\bar{RMS} = \frac{1}{C} \sum_{c=1}^{C} \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_{c,i}^2}$ | RMS 均值，反映信号功率 | μV 级别 |
| `mean_zero_crossing_rate` | $\bar{ZCR} = \frac{1}{C} \sum_{c=1}^{C} \frac{\sum_{i=1}^{N-1} \mathbb{1}[x_{c,i} \cdot x_{c,i+1} < 0]}{T}$ | 零交叉率，反映信号频率特性 | Hz |
| `hjorth_activity` | $Activity = \bar{\sigma^2_x} = \frac{1}{C} \sum_{c=1}^{C} Var(x_c)$ | Hjorth 活动度，即方差 | μV² |
| `hjorth_mobility` | $Mobility = \sqrt{\frac{Var(x')}{Var(x)}}$ | Hjorth 移动度，反映信号频率 | 无量纲 |
| `hjorth_complexity` | $Complexity = \frac{Mobility(x')}{Mobility(x)}$ | Hjorth 复杂度，反映波形复杂程度 | 无量纲 |

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| T1 | **中** | CPU 和 GPU 版本的零交叉率计算方式不一致 | 可能导致结果差异 |
| T2 | **低** | 零交叉处理：CPU 版本做了前向填充，GPU 版本没有 | 精度微小差异 |
| T3 | **低** | Hjorth 复杂度计算中，当 `mobility ≈ 0` 时除法不稳定 | 可能产生 0 或 inf |

**T1 详细分析**:
```python
# CPU 版本 (行 131-156): 处理了零值情况
signs = np.sign(eeg_data)
for ch in range(signs.shape[0]):
    s = signs[ch]
    for i in range(1, s.shape[0]):
        if s[i] == 0:
            s[i] = s[i - 1]  # 前向填充

# GPU 版本 (行 158-168): 没有处理零值
signs = cp.sign(eeg_gpu)
sign_changes = cp.sum((signs[:, :-1] * signs[:, 1:]) < 0, axis=1)
```
**影响**: 当信号中存在精确为 0 的点时，两个版本结果会略有差异。

---

### 频域特征 (Frequency Domain)

**文件**: `eeg_feature_extraction/features/frequency_domain.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `delta_power` | $P_\delta = \frac{1}{C} \sum_{c=1}^{C} \int_{0.5}^{4} S_c(f) df$ | δ波功率 (0.5-4 Hz)，与深度睡眠相关 | μV²/Hz |
| `theta_power` | $P_\theta = \frac{1}{C} \sum_{c=1}^{C} \int_{4}^{8} S_c(f) df$ | θ波功率 (4-8 Hz)，与记忆/注意相关 | μV²/Hz |
| `alpha_power` | $P_\alpha = \frac{1}{C} \sum_{c=1}^{C} \int_{8}^{12} S_c(f) df$ | α波功率 (8-12 Hz)，与放松/觉醒相关 | μV²/Hz |
| `beta_power` | $P_\beta = \frac{1}{C} \sum_{c=1}^{C} \int_{12}^{30} S_c(f) df$ | β波功率 (12-30 Hz)，与警觉/认知相关 | μV²/Hz |
| `gamma_power` | $P_\gamma = \frac{1}{C} \sum_{c=1}^{C} \int_{30}^{80} S_c(f) df$ | γ波功率 (30-80 Hz)，与高级认知相关 | μV²/Hz |
| `low_gamma_power` | $P_{l\gamma} = \int_{30}^{50} S(f) df$ | 低γ波功率 (30-50 Hz) | μV²/Hz |
| `high_gamma_power` | $P_{h\gamma} = \int_{50}^{80} S(f) df$ | 高γ波功率 (50-80 Hz) | μV²/Hz |
| `*_relative_power` | $P_{rel,band} = \frac{P_{band}}{P_{total}}$ | 各频段相对功率 | [0, 1] |
| `peak_frequency` | $f_{peak} = \arg\max_f S(f)$ | 主峰频率 | 0.5-100 Hz |
| `spectral_entropy` | $H_s = -\sum_i p_i \log(p_i)$，其中 $p_i = \frac{S(f_i)}{\sum S}$ | 频谱熵，反映频率分布均匀程度 | ≥0 (nats) |
| `spectral_centroid` | $f_c = \frac{\sum f \cdot S(f)}{\sum S(f)}$ | 频谱质心，反映"重心"频率 | Hz |
| `individual_alpha_frequency` | $IAF = \arg\max_{f \in [8,13]} S(f)$ | 个体α频率，神经效率标记 | 8-13 Hz |
| `theta_beta_ratio` | $TBR = \frac{P_\theta}{P_\beta}$ | θ/β比率，与注意力/ADHD相关 | [0.01, 100] |
| `delta_theta_ratio` | $DTR = \frac{P_\delta}{P_\theta}$ | δ/θ比率 | [0.01, 100] |
| `low_high_power_ratio` | $LHPR = \frac{\int_{1}^{8} S df}{\int_{13}^{40} S df}$ | 低频/高频功率比 | [0.01, 100] |
| `aperiodic_exponent` | 1/f 斜率：$\log S(f) \propto -\chi \log f$ | 非周期性指数，反映神经噪声 | ~1-3 |
| `mean_total_power` | $\bar{P}_{total} = \frac{1}{C} \sum_{c=1}^{C} \int S_c(f) df$ | 总功率均值 | μV²/Hz |

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| FQ1 | **中** | `_safe_ratio` 将比值限制在 [0.01, 100]，可能丢失有效数据 | 极端值被过滤 |
| FQ2 | **低** | IAF 默认值 10.0 硬编码 | 当 α 频段无数据时返回固定值 |
| FQ3 | **低** | `aperiodic_exponent` 回退方法的精度较低 | FOOOF 失败时结果可能不准确 |
| FQ4 | **低** | 频谱熵未归一化时与频率分辨率相关 | 不同数据长度的结果不可比 |

**FQ1 详细分析**:
```python
@staticmethod
def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """返回位于[0.01, 100]的比值，否则返回None"""
    if denominator <= 0:
        return None
    val = numerator / denominator
    if np.isfinite(val) and 0.01 <= val <= 100:
        return float(val)
    return None  # 问题：有效但极端的比值被丢弃
```
**建议**: 考虑使用对数变换或更宽的范围，并在结果中标记极端值而非丢弃。

**FQ4 详细分析**:
```python
def _compute_spectral_entropy(self, psd: np.ndarray) -> float:
    # ...
    ent = float(entropy(psd_norm))
    if self.normalize_spectral_entropy and len(psd_norm) > 1:
        ent = ent / float(np.log(len(psd_norm)))  # 归一化
```
**问题**: 默认 `spectral_entropy_normalize = False`，导致熵值与频率分辨率相关，不同采样率/窗长的数据不可直接比较。

---

### 复杂度特征 (Complexity)

**文件**: `eeg_feature_extraction/features/complexity.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `wavelet_energy_entropy` | $H_w = -\sum_{j=1}^{J} p_j \log(p_j)$，其中 $p_j = \frac{E_j}{\sum E}$ | 小波能量熵，反映信号能量分布 | ≥0 (nats) |
| `sample_entropy` | $SampEn = -\ln\frac{A}{B}$ | 样本熵，量化信号规律性/可预测性 | ≥0，通常 0.1-2 |
| `approx_entropy` | $ApEn = \phi^m(r) - \phi^{m+1}(r)$ | 近似熵，与样本熵类似但包含自匹配 | ≥0，通常 0.1-2 |
| `hurst_exponent` | R/S 分析: $E[R(n)/S(n)] = C \cdot n^H$ | Hurst 指数，反映长程相关性 | (0, 1)，0.5=随机 |
| `higuchi_fd` | $D = -\frac{\log(L(k))}{\log(k)}$ | Higuchi 分形维数，量化信号复杂度 | 1.0-2.0，EEG 约 1.4-1.7 |
| `katz_fd` | $D = \frac{\log_{10}(N-1)}{\log_{10}(N-1) + \log_{10}(d/L)}$ | Katz 分形维数 | 1.0-2.0 |
| `petrosian_fd` | $D = \frac{\log_{10}(N)}{\log_{10}(N) + \log_{10}(\frac{N}{N + 0.4 N_\Delta})}$ | Petrosian 分形维数，基于符号变化 | ~1.0 |

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| C1 | **高** | 样本熵当 A=0 或 B=0 时返回 0.0，应为 inf | 结果数学上不正确 |
| C2 | **中** | Hurst 指数只接受 (0, 1) 范围，超出范围的有效值被丢弃 | 理论上 H 可以 >1 |
| C3 | **中** | 分形维数计算失败时返回硬编码默认值 | 隐藏计算问题 |
| C4 | **低** | 样本熵/近似熵的 `m=2, r=0.2*std` 参数固定 | 可能不适合所有数据 |

**C1 详细分析**:
```python
def _sample_entropy_single_optimized(signal: np.ndarray, m: int, r: float) -> float:
    # ...
    B = count_pairs(templates_m, r)
    A = count_pairs(templates_m1, r)

    if B <= 0 or A <= 0:
        return 0.0  # 问题：数学上应该返回 inf 或 NaN
    return float(-np.log(A / B))
```
**影响**: 当信号高度规则（A=0）或高度随机（B=0）时，返回 0.0 是错误的。正确的样本熵定义中，A=0 意味着 SampEn = ∞。

**C3 详细分析**:
```python
def _higuchi_fd_single(signal: np.ndarray, kmax: int = 8) -> float:
    # ...
    if len(L) < 2:
        return 1.5  # 硬编码默认值
```
**建议**: 返回 NaN 并在上层处理，或记录警告。

---

### 连接性特征 (Connectivity)

**文件**: `eeg_feature_extraction/features/connectivity.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `mean_interchannel_correlation` | $\bar{r} = \frac{2}{C(C-1)} \sum_{i<j} r_{ij}$ | 通道间平均相关性 | [-1, 1] |
| `mean_alpha_coherence` | $\bar{Coh}_\alpha = \frac{2}{C(C-1)} \sum_{i<j} Coh_{ij}(\alpha)$ | α频段平均相干性 | [0, 1] |
| `interhemispheric_alpha_coherence` | $Coh_{LR} = \frac{1}{\|L\|\|R\|} \sum_{l \in L, r \in R} Coh_{lr}$ | 半球间α相干性 | [0, 1] |
| `alpha_beta_band_power_correlation` | $r_{\alpha\beta} = corr(P_\alpha, P_\beta)$ | α与β功率的跨通道相关 | [-1, 1] |
| `hemispheric_alpha_asymmetry` | $Asym = \frac{P_{\alpha,R} - P_{\alpha,L}}{P_{\alpha,R} + P_{\alpha,L}}$ | α功率半球不对称性 | [-1, 1] |
| `frontal_occipital_alpha_ratio` | $Ratio = \frac{P_{\alpha,Frontal}}{P_{\alpha,Occipital}}$ | 前后α功率比 | [0.01, 100] |
| `plv_theta_mean` | $\overline{PLV}_\theta = \frac{2}{C(C-1)} \sum_{i<j} PLV_{ij}(\theta)$ | θ频段全脑平均 PLV | [0, 1] |
| `plv_alpha_mean` | $\overline{PLV}_\alpha$ | α频段全脑平均 PLV | [0, 1] |
| `plv_beta_mean` | $\overline{PLV}_\beta$ | β频段全脑平均 PLV | [0, 1] |
| `plv_gamma_mean` | $\overline{PLV}_\gamma$ | γ频段全脑平均 PLV | [0, 1] |
| `plv_theta_interhemispheric` | 半球间θ频段 PLV | 半球间θ相位同步 | [0, 1] |
| `plv_alpha_interhemispheric` | 半球间α频段 PLV | 半球间α相位同步 | [0, 1] |

**PLV 计算公式**:
$$PLV = \left| \frac{1}{N} \sum_{t=1}^{N} e^{i(\phi_1(t) - \phi_2(t))} \right|$$

其中 $\phi$ 是通过 Hilbert 变换获得的瞬时相位。

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| CN1 | **中** | 带通滤波失败时返回原始信号 | PLV 结果可能无意义 |
| CN2 | **低** | `_compute_lr_connectivity` 未检查索引边界 | 可能的 IndexError |
| CN3 | **低** | PLV 计算效率较低（逐对计算） | 大量通道时速度慢 |

**CN1 详细分析**:
```python
def _bandpass_filter(signal: np.ndarray, fs: float,
                     band: Tuple[float, float], order: int = 4) -> np.ndarray:
    # ...
    try:
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, signal, ...)
        return filtered
    except Exception:
        return signal  # 问题：返回未滤波的信号，PLV 结果无效
```
**建议**: 应该抛出异常或返回 None，让调用者处理。

---

### 网络特征 (Network)

**文件**: `eeg_feature_extraction/features/network.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `network_clustering_coefficient` | $C = \frac{1}{N} \sum_i \frac{2 t_i}{k_i(k_i-1)}$ | 网络聚类系数，量化局部连接密度 | [0, 1] |
| `network_characteristic_path_length` | $L = \frac{1}{N(N-1)} \sum_{i \neq j} d_{ij}$ | 特征路径长度，信息传递效率 | ≥1 |
| `network_global_efficiency` | $E = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$ | 全局效率，网络整合程度 | [0, 1] |
| `network_small_world_index` | $\sigma = \frac{C/C_{rand}}{L/L_{rand}}$ | 小世界指数，σ>1 表示小世界属性 | >0，>1 为小世界 |

其中：
- $t_i$: 节点 i 的三角形数量
- $k_i$: 节点 i 的度数
- $d_{ij}$: 节点 i 到 j 的最短路径长度

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| N1 | **中** | 小世界指数使用近似公式估算随机网络参数 | 结果可能不准确 |
| N2 | **低** | 当网络不连通时，路径长度取节点数作为默认值 | 可能误导解释 |
| N3 | **低** | 阈值化（保留前 30% 连接）是硬编码的 | 不同数据集可能需要不同阈值 |

**N1 详细分析**:
```python
def _compute_small_world_index(self, clustering: float, path_length: float,
                                n_nodes: int) -> float:
    # 估算随机网络的参数（近似公式）
    avg_degree = max(1, n_nodes * self.threshold)
    c_random = avg_degree / n_nodes  # 简化估计
    l_random = np.log(n_nodes) / np.log(avg_degree)  # 简化估计
```
**问题**: 真正的随机网络参数应该通过生成多个 Erdős-Rényi 随机图取平均来估算。近似公式在稀疏网络或小网络上可能有较大误差。

---

### 综合特征 (Composite)

**文件**: `eeg_feature_extraction/features/composite.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `theta_alpha_ratio` | $TAR = \frac{\sum P_\theta}{\sum P_\alpha}$ | 全脑θ/α比率，反映认知负荷 | [0.01, 100] |
| `frontal_beta_ratio` | $FBR = \frac{\bar{P}_{\beta,Frontal}}{\bar{P}_{\beta,All}}$ | 前额β与全脑β比值 | [0.01, 100] |
| `cognitive_load_estimate` | $CL = \sigma(0.6 \cdot TAR + 0.4 \cdot FBR - 1)$ | 综合认知负荷估计 | [0, 1] |
| `alertness_estimate` | $Alert = \sigma(2 \cdot \frac{P_\alpha}{P_\delta} - 1)$ | 清醒度估计 | [0, 1] |
| `relaxation_index` | $RI = \frac{P_\alpha}{P_\alpha + P_\beta}$ | 放松指数 | [0, 1] |

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数。

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| CP1 | **中** | Sigmoid 参数（如 `-2 * (raw_score - 1)`）是硬编码的 | 可能不适合所有数据集 |
| CP2 | **低** | 当 `theta_alpha_ratio` 或 `frontal_beta_ratio` 为 None 时，`cognitive_load_estimate` 返回 None | 特征缺失 |

**CP1 详细分析**:
```python
def _compute_cognitive_load(self, theta_alpha_ratio: float,
                             frontal_beta_ratio: float) -> float:
    raw_score = 0.6 * theta_alpha_ratio + 0.4 * frontal_beta_ratio
    cognitive_load = 1 / (1 + np.exp(-2 * (raw_score - 1)))  # 硬编码参数
```
**问题**: 权重 (0.6, 0.4) 和 sigmoid 参数 (2, 1) 没有理论或数据支持，可能需要根据具体任务和数据集调整。

---

### 微分熵特征 (DE Features)

**文件**: `eeg_feature_extraction/features/de_features.py`

#### 特征列表

| 特征名 | 计算公式 | 物理意义 | 典型范围 |
|-------|---------|---------|---------|
| `de_delta/theta/alpha/beta/gamma` | $DE = \frac{1}{2}\ln(2\pi e \sigma^2)$ | 各频段微分熵（假设高斯分布） | 实数，常 >0 |
| `de_low_gamma` | 同上 | 低γ频段 DE | 实数 |
| `de_high_gamma` | 同上 | 高γ频段 DE | 实数 |
| `dasm_*` | $DASM = DE_{Left} - DE_{Right}$ | 差分不对称性（14对对称电极） | 实数 |
| `rasm_*` | $RASM = DE_{Left} / DE_{Right}$ | 有理不对称性 | [0.01, 100] |
| `dcau_*` | $DCAU = DE_{Frontal} - DE_{Posterior}$ | 差分尾部性（11对前后电极） | 实数 |
| `faa_f3f4` | $FAA = \ln(P_{\alpha,F4}) - \ln(P_{\alpha,F3})$ | F3/F4 额叶α不对称性 | 实数 |
| `faa_f7f8` | 同上 | F7/F8 额叶α不对称性 | 实数 |
| `faa_fp1fp2` | 同上 | FP1/FP2 额叶α不对称性 | 实数 |
| `faa_mean` | FAA 各配对的平均值 | 平均额叶α不对称性 | 实数 |

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| DE1 | **中** | DE 可以为负值（当 σ² < 1/(2πe) 时），但 RASM 要求正值 | RASM 计算可能失败 |
| DE2 | **中** | 缺少电极时返回 0.0 而非 NaN | 结果难以区分"零值"和"缺失值" |
| DE3 | **低** | 电极配对是硬编码的，不支持非标准电极布局 | 自定义电极布局无法使用 |

**DE1 详细分析**:
```python
def compute_de(variance: float) -> float:
    if variance <= 0:
        return 0.0
    return 0.5 * np.log(2 * np.pi * np.e * variance)
    # 当 variance < 1/(2*pi*e) ≈ 0.058 时，DE < 0
```
**问题**: 微分熵可以是负值，这在 RASM 比值计算中可能导致问题：
```python
rasm = self._safe_ratio(de_left, de_right)  # 如果 de_right < 0，比值无意义
```

---

### 微状态特征 (Microstate)

**文件**: `eeg_feature_extraction/features/microstate.py`

#### 特征列表

每个微状态类别 (0-3，对应 A, B, C, D) 有 5 个特征：

| 特征名模式 | 计算公式 | 物理意义 | 典型范围 |
|-----------|---------|---------|---------|
| `Microstate_k_meandurs` | $\bar{D}_k = \frac{\sum_{seg \in k} duration_{seg}}{N_k}$ | 平均持续时间 | 50-150 ms |
| `Microstate_k_occurrence` | $Occ_k = \frac{N_k}{T_{total}}$ | 每秒出现次数 | 2-10 Hz |
| `Microstate_k_timecov` | $TC_k = \frac{samples_k}{f_s}$ | 时间覆盖（秒） | 取决于信号长度 |
| `Microstate_k_mean_corr` | $\bar{r}_k = \frac{1}{\|S_k\|} \sum_{t \in S_k} corr(x_t, \mu_k)$ | 与模板的平均相关性 | [0, 1] |
| `Microstate_k_gev` | $GEV_k = \frac{\sum_{t \in S_k} (GFP_t \cdot r_t)^2}{\sum_t GFP_t^2}$ | 全局解释方差 | [0, 1] |

其中：
- GFP (Global Field Power): $GFP_t = \sqrt{\frac{1}{C}\sum_c (V_{c,t} - \bar{V}_t)^2}$
- 微状态分配使用极性不变的空间相关性

#### 发现的问题

| 问题编号 | 严重程度 | 问题描述 | 影响 |
|---------|---------|---------|------|
| MS1 | **高** | Backfitting 逐时间点计算，非常慢 | 大数据集计算时间长 |
| MS2 | **中** | 降级模式（无预计算模板）会从单个 segment 生成模板 | 结果不稳定，与 subject 级模板不一致 |
| MS3 | **中** | GFP 峰值不足时的回退策略可能导致不稳定的聚类 | 短 segment 上结果不可靠 |
| MS4 | **低** | K-Means 初始化是随机的 | 不同运行可能得到不同的微状态顺序 |

**MS1 详细分析**:
```python
def backfit(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 逐时间点计算 - O(n_samples * n_states)
    for t in range(n_samples):  # 非向量化！
        current_map = data[:, t]
        for k in range(self.n_states):
            corr = abs(self._spatial_correlation(current_map, self.centroids[k]))
```
**建议**: 向量化计算：
```python
# 建议的向量化实现
data_centered = data - data.mean(axis=0, keepdims=True)
data_norm = np.linalg.norm(data_centered, axis=0, keepdims=True)
data_normalized = data_centered / (data_norm + 1e-10)

centroids_centered = centroids - centroids.mean(axis=1, keepdims=True)
centroids_norm = np.linalg.norm(centroids_centered, axis=1, keepdims=True)
centroids_normalized = centroids_centered / (centroids_norm + 1e-10)

correlations = np.abs(centroids_normalized @ data_normalized)  # (n_states, n_samples)
labels = np.argmax(correlations, axis=0)
```

---

## 改进建议汇总

### 高优先级

1. **跨平台兼容性** (M1): 替换 `signal.SIGALRM` 为跨平台的超时机制
2. **样本熵计算** (C1): 当 A=0 或 B=0 时返回 inf 或 NaN 而非 0.0
3. **Backfitting 向量化** (MS1): 将逐时间点循环改为矩阵运算

### 中优先级

4. **安全比值** (FQ1, DE1): 扩大有效范围或使用对数变换
5. **Hurst 指数范围** (C2): 允许理论上有效的超出 (0,1) 范围的值
6. **滤波失败处理** (CN1): 滤波失败时抛出异常而非返回原始信号
7. **Sigmoid 参数** (CP1): 使参数可配置或基于数据自适应

### 低优先级

8. **默认值处理** (C3, FQ2): 返回 NaN 而非硬编码默认值
9. **配置序列化** (M3): 使用更健壮的序列化方法
10. **小世界指数** (N1): 使用真实随机网络而非近似公式

---

## 附录：频段定义

| 频段 | 频率范围 | 相关功能状态 |
|-----|---------|------------|
| Delta (δ) | 0.5-4 Hz | 深度睡眠、无意识 |
| Theta (θ) | 4-8 Hz | 记忆编码、冥想、注意 |
| Alpha (α) | 8-12 Hz | 放松觉醒、闭眼休息 |
| Beta (β) | 12-30 Hz | 警觉、焦虑、活跃思考 |
| Low Gamma | 30-50 Hz | 感知绑定、认知处理 |
| High Gamma | 50-80 Hz | 高级认知、语言处理 |

---

*文档生成时间：2024年1月*
*版本：1.0*
