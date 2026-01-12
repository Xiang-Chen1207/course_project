
## 1. 额叶Alpha不对称性 (Frontal Alpha Asymmetry, FAA)

### 1.1 神经科学背景

额叶Alpha不对称性反映左右半球前额叶皮层的相对激活水平。由于Alpha功率与皮层激活呈**负相关**（Alpha抑制假说），因此：
- **正值FAA**（右侧Alpha > 左侧Alpha）→ 左侧皮层更活跃 → 趋近动机/积极情绪
- **负值FAA**（左侧Alpha > 右侧Alpha）→ 右侧皮层更活跃 → 回避动机/消极情绪

### 1.2 数学公式

#### 基本定义
$$
FAA = \ln(P_{\alpha,Right}) - \ln(P_{\alpha,Left})
$$

其中：
- $P_{\alpha,Right}$：右侧额叶电极的Alpha频段功率
- $P_{\alpha,Left}$：对应左侧额叶电极的Alpha频段功率
- Alpha频段：**8-13 Hz**（有时细分为8-10 Hz低Alpha和10-13 Hz高Alpha）

#### 功率谱密度计算
使用Welch方法计算功率谱密度(PSD)：
$$
P_{\alpha} = \int_{f_1}^{f_2} S(f) \, df
$$

其中 $S(f)$ 是功率谱密度，$f_1=8$ Hz，$f_2=13$ Hz。

### 1.3 电极配对方案

| 配对名称 | 左侧电极 | 右侧电极 | 10-20系统位置 |
|---------|---------|---------|--------------|
| F3-F4   | F3      | F4      | 额叶中部 |
| F7-F8   | F7      | F8      | 额叶外侧 |
| Fp1-Fp2 | Fp1     | Fp2     | 前额极 |
| AF3-AF4 | AF3     | AF4     | 前额叶（扩展系统） |

**推荐**：F3-F4配对最常用，也可计算多对取平均。

### 1.4 算法步骤

```
输入：
    - eeg_left: 左侧电极EEG信号，shape = (n_samples,) 或 (n_epochs, n_samples)
    - eeg_right: 右侧电极EEG信号，shape同上
    - fs: 采样率 (Hz)
    - alpha_band: Alpha频段范围，默认 (8, 13)
    - window_sec: Welch方法窗口长度（秒），默认 2.0
    - overlap: 窗口重叠比例，默认 0.5

输出：
    - faa: 额叶Alpha不对称性指数，标量或shape = (n_epochs,)

算法：
1. 预处理检查
   - 验证左右信号长度一致
   - 检查采样率是否满足Nyquist准则（fs > 2 * alpha_band[1]）

2. 计算功率谱密度
   FOR each epoch (如果有多个epoch):
       a. 使用Welch方法计算PSD
          - nperseg = int(window_sec * fs)
          - noverlap = int(nperseg * overlap)
          - freqs, psd_left = welch(eeg_left, fs, nperseg=nperseg, noverlap=noverlap)
          - freqs, psd_right = welch(eeg_right, fs, nperseg=nperseg, noverlap=noverlap)
       
       b. 提取Alpha频段
          - alpha_idx = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
       
       c. 计算Alpha频段功率（积分或平均）
          - 方法1（积分）: P_alpha = trapz(psd[alpha_idx], freqs[alpha_idx])
          - 方法2（平均）: P_alpha = mean(psd[alpha_idx])
          
       d. 计算FAA
          - faa = log(P_alpha_right) - log(P_alpha_left)

3. 返回结果
```

### 1.5 代码实现要点

```python
# 函数签名
def compute_frontal_alpha_asymmetry(
    eeg_left: np.ndarray,          # 左侧电极数据
    eeg_right: np.ndarray,         # 右侧电极数据  
    fs: float,                      # 采样率
    alpha_band: tuple = (8, 13),   # Alpha频段
    window_sec: float = 2.0,       # Welch窗口长度
    overlap: float = 0.5,          # 重叠比例
    method: str = 'welch'          # PSD方法：'welch', 'multitaper', 'fft'
) -> float | np.ndarray:
    """
    计算额叶Alpha不对称性指数
    
    Returns:
        faa: FAA值，正值表示左半球相对激活更强（趋近/积极）
    """
    pass
```

### 1.6 注意事项与边界情况

1. **功率值为零或负数**：在取对数前检查，添加小常数（如1e-10）避免log(0)
2. **伪迹影响**：眼动、肌电伪迹会严重影响额叶记录，建议先进行伪迹去除
3. **参考电极**：使用平均参考或linked mastoids参考，避免单侧参考引入系统偏差
4. **个体基线**：FAA存在显著个体差异，建议计算相对于静息态基线的变化
5. **频段细分**：可分别计算低Alpha(8-10Hz)和高Alpha(10-13Hz)的FAA

---

## 2. 分形维数 (Fractal Dimension, FD)

### 2.1 神经科学背景

分形维数量化EEG信号的**自相似性和复杂度**：
- **高FD**：信号复杂、不规则（正常清醒状态、认知负荷）
- **低FD**：信号规则、周期性（癫痫发作、深度睡眠、昏迷）

常用方法：**Higuchi算法**（最推荐）、Katz算法、Petrosian算法

### 2.2 Higuchi算法

#### 数学定义

给定时间序列 $X = \{x(1), x(2), ..., x(N)\}$，Higuchi算法步骤：

**Step 1**: 构建新序列 $X_k^m$

对于时间间隔 $k = 1, 2, ..., k_{max}$ 和起始点 $m = 1, 2, ..., k$：
$$
X_k^m = \{x(m), x(m+k), x(m+2k), ..., x(m+\lfloor\frac{N-m}{k}\rfloor \cdot k)\}
$$

**Step 2**: 计算每个序列的长度 $L_m(k)$
$$
L_m(k) = \frac{1}{k} \left[ \left( \sum_{i=1}^{\lfloor(N-m)/k\rfloor} |x(m+ik) - x(m+(i-1)k)| \right) \cdot \frac{N-1}{\lfloor(N-m)/k\rfloor \cdot k} \right]
$$

其中 $\frac{N-1}{\lfloor(N-m)/k\rfloor \cdot k}$ 是归一化因子。

**Step 3**: 对所有起始点取平均
$$
L(k) = \frac{1}{k} \sum_{m=1}^{k} L_m(k)
$$

**Step 4**: 线性回归求分形维数

在 $\log(k)$ vs $\log(L(k))$ 图上，斜率的负值即为Higuchi分形维数：
$$
FD_{Higuchi} = -\frac{d(\log L(k))}{d(\log k)}
$$

#### 参数选择
- **$k_{max}$**：最大时间间隔，通常取 $k_{max} = \lfloor N/4 \rfloor$ 或固定值如8-64
- 推荐：$k_{max} = 8$ 至 $k_{max} = 20$（根据信号长度调整）

### 2.3 Katz算法

#### 数学定义
$$
FD_{Katz} = \frac{\log_{10}(L/a)}{\log_{10}(d/a)}
$$

其中：
- $L$：曲线总长度 = $\sum_{i=1}^{N-1} \sqrt{1 + (x(i+1)-x(i))^2}$（假设采样间隔归一化为1）
- $a$：平均步长 = $L / (N-1)$
- $d$：最大距离 = $\max_i \sqrt{(i-1)^2 + (x(i)-x(1))^2}$

简化公式（常用形式）：
$$
FD_{Katz} = \frac{\log_{10}(N-1)}{\log_{10}(N-1) + \log_{10}(d/L)}
$$

### 2.4 Petrosian算法

#### 数学定义

基于信号符号变化次数：
$$
FD_{Petrosian} = \frac{\log_{10}(N)}{\log_{10}(N) + \log_{10}\left(\frac{N}{N + 0.4 \cdot N_{\Delta}}\right)}
$$

其中：
- $N$：信号长度
- $N_{\Delta}$：符号变化次数，即二值化序列 $\text{sign}(x(i+1)-x(i))$ 中符号改变的次数

### 2.5 算法步骤（Higuchi为例）

```
输入：
    - signal: EEG信号，shape = (n_samples,)
    - kmax: 最大时间间隔，默认 8
    
输出：
    - fd: Higuchi分形维数，标量，范围约 [1, 2]

算法：
1. 获取信号长度 N

2. 对每个 k = 1, 2, ..., kmax:
   a. 初始化 Lk_sum = 0
   
   b. 对每个起始点 m = 1, 2, ..., k:
      i. 构建子序列索引：indices = [m-1, m-1+k, m-1+2k, ...]（Python 0-indexed）
      ii. 提取子序列值：X_k_m = signal[indices]
      iii. 计算相邻差分绝对值之和：diff_sum = sum(|X_k_m[i+1] - X_k_m[i]|)
      iv. 计算序列长度并归一化：
          num_segments = len(X_k_m) - 1
          norm_factor = (N - 1) / (num_segments * k)
          L_m_k = (diff_sum / k) * norm_factor
      v. 累加：Lk_sum += L_m_k
   
   c. 计算平均长度：L[k] = Lk_sum / k

3. 线性回归
   - x = log(1/k) for k = 1, ..., kmax  # 或 x = log(k)
   - y = log(L[k])
   - 使用最小二乘法拟合 y = slope * x + intercept
   - fd = slope  # 如果用log(1/k)；若用log(k)则 fd = -slope

4. 返回 fd
```

### 2.6 代码实现要点

```python
def compute_higuchi_fd(
    signal: np.ndarray,
    kmax: int = 8
) -> float:
    """
    计算Higuchi分形维数
    
    Args:
        signal: 1D EEG信号
        kmax: 最大时间间隔（推荐8-20）
    
    Returns:
        fd: 分形维数，正常EEG约1.4-1.7，癫痫发作时可降至1.2以下
    """
    pass

def compute_katz_fd(signal: np.ndarray) -> float:
    """
    计算Katz分形维数
    """
    pass

def compute_petrosian_fd(signal: np.ndarray) -> float:
    """
    计算Petrosian分形维数
    """
    pass
```

### 2.7 注意事项与边界情况

1. **信号长度要求**：Higuchi需要 $N > k_{max} \times k_{max}$，建议至少256点
2. **kmax选择**：太小会欠拟合，太大会引入噪声；建议通过检查$\log$-$\log$图的线性度来选择
3. **数值稳定性**：长度接近零时log会出问题，添加小常数保护
4. **信号预处理**：建议先去趋势（detrend）和滤波
5. **多通道处理**：可对每个通道分别计算，或对特定脑区平均后计算

---

## 3. 相位锁定值 (Phase Locking Value, PLV)

### 3.1 神经科学背景

相位锁定值量化两个神经信号之间的**相位同步程度**，反映功能连接：
- **PLV = 1**：完全相位同步
- **PLV = 0**：完全随机相位关系
- 不同频段的PLV反映不同的功能网络（如gamma-PLV与情绪、theta-PLV与记忆）

### 3.2 数学公式

#### 解析信号与瞬时相位

首先通过**Hilbert变换**获取解析信号：
$$
z(t) = x(t) + i \cdot \mathcal{H}[x(t)]
$$

瞬时相位：
$$
\phi(t) = \arctan\left(\frac{\mathcal{H}[x(t)]}{x(t)}\right) = \text{angle}(z(t))
$$

#### PLV定义

给定两个信号 $x_1(t)$ 和 $x_2(t)$ 的瞬时相位 $\phi_1(t)$ 和 $\phi_2(t)$：

$$
PLV = \left| \frac{1}{N} \sum_{t=1}^{N} e^{i(\phi_1(t) - \phi_2(t))} \right|
$$

等价形式：
$$
PLV = \sqrt{\left(\frac{1}{N}\sum_{t=1}^{N}\cos(\Delta\phi(t))\right)^2 + \left(\frac{1}{N}\sum_{t=1}^{N}\sin(\Delta\phi(t))\right)^2}
$$

其中 $\Delta\phi(t) = \phi_1(t) - \phi_2(t)$。

#### 跨试次PLV（Event-Related PLV）

对于多试次数据（n_trials × n_samples）：
$$
PLV(t) = \left| \frac{1}{N_{trials}} \sum_{k=1}^{N_{trials}} e^{i(\phi_1^k(t) - \phi_2^k(t))} \right|
$$

### 3.3 带通滤波要求

**关键**：PLV必须在特定频段内计算，直接对宽带信号计算无意义！

```
常用频段：
- Delta:  0.5 - 4 Hz
- Theta:  4 - 8 Hz
- Alpha:  8 - 13 Hz
- Beta:   13 - 30 Hz
- Gamma:  30 - 100 Hz

滤波器设计：
- 类型：零相位带通滤波（filtfilt）
- 推荐：Butterworth 4-6阶，或FIR滤波器
- 过渡带宽度：建议1-2 Hz
```

### 3.4 算法步骤

```
输入：
    - signal1: 第一个通道EEG信号，shape = (n_samples,) 或 (n_epochs, n_samples)
    - signal2: 第二个通道EEG信号，shape同上
    - fs: 采样率 (Hz)
    - freq_band: 感兴趣频段，如 (8, 13) 表示Alpha
    - filter_order: 滤波器阶数，默认 4

输出：
    - plv: 相位锁定值
        - 单epoch：标量
        - 多epoch取平均：标量
        - 时间分辨：shape = (n_samples,)

算法：
1. 参数验证
   - 检查信号长度一致
   - 检查freq_band[1] < fs/2（Nyquist准则）

2. 带通滤波
   - 设计Butterworth带通滤波器
     nyq = fs / 2
     low = freq_band[0] / nyq
     high = freq_band[1] / nyq
     b, a = butter(filter_order, [low, high], btype='band')
   
   - 零相位滤波
     signal1_filt = filtfilt(b, a, signal1, axis=-1)
     signal2_filt = filtfilt(b, a, signal2, axis=-1)

3. 计算解析信号（Hilbert变换）
   - analytic1 = hilbert(signal1_filt, axis=-1)
   - analytic2 = hilbert(signal2_filt, axis=-1)

4. 提取瞬时相位
   - phase1 = np.angle(analytic1)
   - phase2 = np.angle(analytic2)

5. 计算相位差
   - phase_diff = phase1 - phase2

6. 计算PLV
   方法A（单epoch，时间平均）：
       plv = np.abs(np.mean(np.exp(1j * phase_diff)))
   
   方法B（多epoch，跨试次平均）：
       plv_per_time = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))  # 每个时间点
       plv = np.mean(plv_per_time)  # 或保留时间分辨率
   
   方法C（滑动窗口PLV）：
       FOR each window:
           plv_window[i] = np.abs(np.mean(np.exp(1j * phase_diff[window_start:window_end])))

7. 返回PLV值
```

### 3.5 连接矩阵计算

对于多通道EEG（n_channels × n_samples），计算全脑PLV连接矩阵：

```
输入：
    - eeg_data: shape = (n_channels, n_samples) 或 (n_epochs, n_channels, n_samples)
    - fs, freq_band: 同上
    - channel_names: 可选，通道名列表

输出：
    - plv_matrix: shape = (n_channels, n_channels)，对称矩阵，对角线为1

算法：
1. 对所有通道进行带通滤波和Hilbert变换
2. 初始化 plv_matrix = np.zeros((n_channels, n_channels))
3. FOR i = 0 to n_channels-1:
       FOR j = i+1 to n_channels-1:
           plv_matrix[i, j] = compute_plv(phase[i], phase[j])
           plv_matrix[j, i] = plv_matrix[i, j]  # 对称
4. np.fill_diagonal(plv_matrix, 1.0)
5. 返回 plv_matrix
```

### 3.6 代码实现要点

```python
def compute_plv(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float,
    freq_band: tuple,
    filter_order: int = 4,
    method: str = 'mean'  # 'mean', 'time_resolved', 'sliding_window'
) -> float | np.ndarray:
    """
    计算两个信号之间的相位锁定值
    
    Args:
        signal1, signal2: EEG信号
        fs: 采样率
        freq_band: 频段范围，如(8, 13)
        filter_order: Butterworth滤波器阶数
        method: 
            'mean' - 返回时间平均PLV（标量）
            'time_resolved' - 返回每个时间点的PLV（需要多epoch）
            'sliding_window' - 返回滑动窗口PLV
    
    Returns:
        plv: 相位锁定值，范围[0, 1]
    """
    pass

def compute_plv_matrix(
    eeg_data: np.ndarray,
    fs: float,
    freq_band: tuple,
    filter_order: int = 4
) -> np.ndarray:
    """
    计算多通道EEG的PLV连接矩阵
    
    Args:
        eeg_data: shape = (n_channels, n_samples) 或 (n_epochs, n_channels, n_samples)
        fs: 采样率
        freq_band: 频段范围
    
    Returns:
        plv_matrix: shape = (n_channels, n_channels)，对称矩阵
    """
    pass
```

### 3.7 注意事项与边界情况

1. **必须带通滤波**：宽带信号的PLV无意义，因为瞬时相位定义依赖于窄带假设
2. **滤波器边缘效应**：信号首尾可能有伪迹，建议丢弃首尾各几个周期的数据
3. **信号长度**：至少包含几个完整的低频周期（如8Hz需要至少250ms，建议1-2秒）
4. **统计显著性**：可通过surrogate方法（相位随机化）建立null分布，检验PLV显著性
5. **体积传导**：相邻电极可能因体积传导而表现出伪同步，可考虑使用PLI（Phase Lag Index）作为替代
6. **对角线处理**：自身与自身的PLV恒为1，通常设为1或NaN

---

## 4. 完整特征提取类设计

### 4.1 推荐类结构

```python
class EEGFeatureExtractor:
    """
    EEG特征提取器
    
    支持特征：
    - Frontal Alpha Asymmetry (FAA)
    - Fractal Dimension (Higuchi, Katz, Petrosian)
    - Phase Locking Value (PLV)
    """
    
    def __init__(self, fs: float, montage: str = '10-20'):
        """
        Args:
            fs: 采样率
            montage: 电极配置，用于确定电极对应关系
        """
        self.fs = fs
        self.montage = montage
        
    def compute_faa(
        self, 
        eeg_data: np.ndarray,
        left_channels: list,
        right_channels: list,
        alpha_band: tuple = (8, 13)
    ) -> np.ndarray:
        """计算额叶Alpha不对称性"""
        pass
    
    def compute_fractal_dimension(
        self,
        signal: np.ndarray,
        method: str = 'higuchi',
        kmax: int = 8
    ) -> float:
        """计算分形维数"""
        pass
    
    def compute_plv(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        freq_band: tuple
    ) -> float:
        """计算相位锁定值"""
        pass
    
    def compute_plv_matrix(
        self,
        eeg_data: np.ndarray,
        freq_band: tuple
    ) -> np.ndarray:
        """计算全脑PLV连接矩阵"""
        pass
    
    def extract_all_features(
        self,
        eeg_data: np.ndarray,
        channel_names: list
    ) -> dict:
        """
        提取所有特征
        
        Returns:
            features: {
                'faa_f3f4': float,
                'faa_f7f8': float,
                'fd_higuchi': np.ndarray (per channel),
                'fd_katz': np.ndarray (per channel),
                'plv_theta': np.ndarray (connectivity matrix),
                'plv_alpha': np.ndarray (connectivity matrix),
                'plv_gamma': np.ndarray (connectivity matrix),
            }
        """
        pass
```

### 4.2 依赖库

```python
# 必需
import numpy as np
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import linregress

# 可选（用于加速或额外功能）
from numba import jit  # 加速Higuchi算法
import mne  # EEG数据处理
from sklearn.preprocessing import StandardScaler  # 特征标准化
```

### 4.3 单元测试建议

```python
def test_faa():
    """测试FAA计算"""
    # 生成测试信号：左侧alpha更强应产生负FAA
    fs = 256
    t = np.arange(0, 10, 1/fs)
    left = 2 * np.sin(2*np.pi*10*t) + np.random.randn(len(t)) * 0.5
    right = 1 * np.sin(2*np.pi*10*t) + np.random.randn(len(t)) * 0.5
    faa = compute_faa(left, right, fs)
    assert faa < 0, "左侧alpha更强时FAA应为负"

def test_higuchi_fd():
    """测试Higuchi分形维数"""
    # 白噪声FD应接近2，正弦波FD应接近1
    white_noise = np.random.randn(1024)
    sine_wave = np.sin(np.linspace(0, 20*np.pi, 1024))
    
    fd_noise = compute_higuchi_fd(white_noise, kmax=8)
    fd_sine = compute_higuchi_fd(sine_wave, kmax=8)
    
    assert 1.8 < fd_noise < 2.1, f"白噪声FD应接近2，实际{fd_noise}"
    assert 1.0 < fd_sine < 1.3, f"正弦波FD应接近1，实际{fd_sine}"

def test_plv():
    """测试PLV计算"""
    fs = 256
    t = np.arange(0, 5, 1/fs)
    # 完全同步信号
    sig1 = np.sin(2*np.pi*10*t)
    sig2 = np.sin(2*np.pi*10*t)
    plv_sync = compute_plv(sig1, sig2, fs, freq_band=(8, 13))
    assert plv_sync > 0.95, f"同步信号PLV应接近1，实际{plv_sync}"
    
    # 随机相位信号
    sig3 = np.sin(2*np.pi*10*t + np.random.rand()*2*np.pi)
    # 多次随机化取平均验证
```