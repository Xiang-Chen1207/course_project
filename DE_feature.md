

### 2. 特征 1：微分熵 (Differential Entropy, DE)
**数学定义：**
假设 EEG 信号服从高斯分布 $N(\mu, \sigma^2)$，DE 的计算公式为：
$$ h(X) = \frac{1}{2} \log(2 \pi e \sigma^2) $$
*   其中 $\pi$ 是圆周率，$e$ 是欧拉常数 (np.e)。
*   $\sigma^2$ 是该频段的信号方差（即平均功率能量）。
*   **代码要求：** 编写一个函数 `compute_de(variance)`，输入方差，返回 DE 值。

### 3. 特征 2 & 3：差分不对称性 (DASM) 与 有理不对称性 (RASM)
这两个特征基于大脑左右半球的**14对**对称电极计算。

**电极配对列表 (Left - Right)：**
1.  Fp1 - Fp2
2.  F7 - F8
3.  F3 - F4
4.  T7 - T8
5.  P7 - P8
6.  C3 - C4
7.  P3 - P4
8.  O1 - O2
9.  AF3 - AF4
10. FC5 - FC6
11. FC1 - FC2
12. CP5 - CP6
13. CP1 - CP2
14. PO3 - PO4

**计算公式：**
*   **DASM:** $DE(Left) - DE(Right)$
*   **RASM:** $DE(Left) / DE(Right)$

**代码要求：**
编写函数 `compute_asymmetry(de_features, channel_map)`。
*   `de_features`: 包含所有通道 DE 值的数组。
*   `channel_map`: 一个字典，将电极名称映射到数组索引（例如 `{'Fp1': 0, 'Fp2': 1, ...}`）。
*   函数应返回 DASM 和 RASM 特征向量。

### 4. 特征 4：差分尾部性 (Differential Caudality, DCAU)
该特征基于大脑**额叶（Frontal）与后部（Posterior）**区域的**11对**电极计算。

**电极配对列表 (Frontal - Posterior)：**
1.  FC5 - CP5
2.  FC1 - CP1
3.  FC2 - CP2
4.  FC6 - CP6
5.  F7 - P7
6.  F3 - P3
7.  Fz - Pz
8.  F4 - P4
9.  F8 - P8
10. Fp1 - O1
11. Fp2 - O2

**计算公式：**
*   **DCAU:** $DE(Frontal) - DE(Posterior)$
