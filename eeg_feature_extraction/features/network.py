"""
网络特征计算：聚类系数、特征路径长度、全局效率、小世界属性

优化：使用 scipy.sparse.csgraph 替代 Floyd-Warshall 算法
     使用向量化计算替代循环
"""
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

from .base import BaseFeature, FeatureRegistry
from ..psd_computer import PSDResult, PSDComputer
from ..config import Config


@FeatureRegistry.register('network')
class NetworkFeatures(BaseFeature):
    """网络特征计算"""

    feature_names = [
        'network_clustering_coefficient',
        'network_characteristic_path_length',
        'network_global_efficiency',
        'network_small_world_index',
    ]

    def __init__(self, config: Config):
        super().__init__(config)
        self.threshold = config.network_threshold
        self.psd_computer = PSDComputer(
            sampling_rate=config.sampling_rate,
            use_gpu=config.use_gpu,
            nperseg=config.nperseg,
            noverlap=config.noverlap,
            nfft=config.nfft
        )

    def compute(self, eeg_data: np.ndarray, psd_result: Optional[PSDResult] = None,
                **kwargs) -> Dict[str, float]:
        """
        计算网络特征

        Args:
            eeg_data: EEG 数据
            psd_result: PSD 结果

        Returns:
            网络特征字典
        """
        self._validate_input(eeg_data)

        features = {}

        # 计算相干性矩阵作为连接矩阵
        # 支持从外部传入缓存的相干性矩阵，避免与 connectivity 重复计算
        coherence_matrix = kwargs.get('coherence_matrix', None)
        if coherence_matrix is None:
            coherence_matrix = self.psd_computer.compute_coherence(
                eeg_data, band=(8.0, 13.0)
            )

        # 应用阈值，保留前 30% 最强连接
        adj_matrix = self._threshold_matrix(coherence_matrix)

        # 1. 网络聚类系数
        clustering = self._compute_clustering_coefficient(adj_matrix)
        features['network_clustering_coefficient'] = float(clustering)

        # 2. 网络特征路径长度
        path_length = self._compute_characteristic_path_length(adj_matrix)
        features['network_characteristic_path_length'] = float(path_length)

        # 3. 网络全局效率
        global_efficiency = self._compute_global_efficiency(adj_matrix)
        features['network_global_efficiency'] = float(global_efficiency)

        # 4. 网络小世界属性
        small_world = self._compute_small_world_index(
            clustering, path_length, adj_matrix.shape[0]
        )
        features['network_small_world_index'] = float(small_world)

        return features

    def _threshold_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        应用阈值，保留前 threshold% 最强连接

        Args:
            matrix: 连接矩阵

        Returns:
            二值化的邻接矩阵
        """
        # 获取上三角元素（不包括对角线）
        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]

        if len(upper_tri) == 0:
            return np.zeros_like(matrix)

        # 计算阈值（保留前 threshold% 最强连接）
        threshold_value = np.percentile(upper_tri, (1 - self.threshold) * 100)

        # 创建二值化邻接矩阵
        adj = np.zeros_like(matrix)
        adj[matrix >= threshold_value] = 1
        np.fill_diagonal(adj, 0)  # 移除自环

        return adj

    def _compute_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """
        计算平均聚类系数（向量化优化版本）

        聚类系数 = 实际三角形数 / 可能的三角形数

        使用矩阵乘法快速计算三角形数量
        """
        n = adj_matrix.shape[0]

        # 确保是二值矩阵
        A = (adj_matrix > 0).astype(np.float64)

        # 计算每个节点的度
        degrees = A.sum(axis=1)

        # 使用矩阵乘法计算三角形数量
        # A^3 的对角线元素 = 经过节点 i 的闭合三角形数量 * 2
        A2 = A @ A
        triangles = np.diag(A2 @ A) / 2

        # 计算最大可能的三角形数量
        max_triangles = degrees * (degrees - 1) / 2

        # 计算聚类系数
        with np.errstate(divide='ignore', invalid='ignore'):
            cc = np.where(max_triangles > 0, triangles / max_triangles, 0.0)

        return float(np.mean(cc))

    def _compute_characteristic_path_length(self, adj_matrix: np.ndarray) -> float:
        """
        计算特征路径长度

        使用 scipy.sparse.csgraph.shortest_path 优化（比 Floyd-Warshall 快 10-50x）
        """
        n = adj_matrix.shape[0]

        # 创建稀疏距离矩阵（有边的距离为1）
        dist_matrix = np.where(adj_matrix > 0, 1.0, 0.0)
        sparse_matrix = csr_matrix(dist_matrix)

        # 使用 scipy 的最短路径算法（自动选择最优算法）
        dist = shortest_path(sparse_matrix, directed=False, unweighted=True)

        # 计算平均路径长度（忽略无穷大和自身）
        finite_dist = dist[np.isfinite(dist) & (dist > 0)]
        if len(finite_dist) > 0:
            return float(np.mean(finite_dist))
        return float(n)  # 如果图不连通，返回节点数作为最大路径长度

    def _compute_global_efficiency(self, adj_matrix: np.ndarray) -> float:
        """
        计算全局效率（向量化优化版本）

        E = (1 / N(N-1)) * Σ(1 / d_ij)

        使用 scipy.sparse.csgraph.shortest_path 优化
        """
        n = adj_matrix.shape[0]

        if n < 2:
            return 0.0

        # 创建稀疏距离矩阵
        dist_matrix = np.where(adj_matrix > 0, 1.0, 0.0)
        sparse_matrix = csr_matrix(dist_matrix)

        # 使用 scipy 的最短路径算法
        dist = shortest_path(sparse_matrix, directed=False, unweighted=True)

        # 向量化计算效率
        # 创建掩码排除对角线和无穷大
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        mask &= np.isfinite(dist) & (dist > 0)

        # 计算效率
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist = np.where(mask, 1.0 / dist, 0.0)

        efficiency = np.sum(inv_dist)

        return float(efficiency / (n * (n - 1)))

    def _compute_small_world_index(self, clustering: float, path_length: float,
                                    n_nodes: int) -> float:
        """
        计算小世界属性

        σ = (C / C_random) / (L / L_random)

        对于随机网络:
        C_random ≈ k / N
        L_random ≈ ln(N) / ln(k)

        其中 k 是平均度
        """
        if n_nodes < 2 or path_length <= 0:
            return 1.0

        # 估算随机网络的参数
        # 假设平均度约为节点数的 threshold 比例
        avg_degree = max(1, n_nodes * self.threshold)

        # 随机网络的聚类系数
        c_random = avg_degree / n_nodes

        # 随机网络的路径长度
        if avg_degree > 1:
            l_random = np.log(n_nodes) / np.log(avg_degree)
        else:
            l_random = n_nodes

        # 避免除以零
        if c_random < 1e-10 or l_random < 1e-10:
            return 1.0

        # 小世界系数
        gamma = clustering / c_random if c_random > 0 else 1.0
        lambd = path_length / l_random if l_random > 0 else 1.0

        if lambd < 1e-10:
            return 1.0

        return gamma / lambd
