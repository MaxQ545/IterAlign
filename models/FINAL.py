# 原始代码来自 GAlign，以下为 FINAL 类的实现
import numpy as np
from numpy import inf
from sklearn.preprocessing import normalize
from time import time

class FINAL:
    """
    FINAL: 与 MMNC 类似的网络对齐模型，对 PyTorch Geometric 的 Data 输入进行对齐。
    """

    def __init__(self, config):
        self.alpha = config['alpha']
        self.maxiter = config['max_iter']
        self.tol = config['tol']
        self.train_ratio = config['train_ratio']
        self.no_edge_feat = True
        self.H = None

    def prepare_data(self, G):
        """
        将 PyTorch Geometric 的 Data 转换为邻接矩阵和特征矩阵。

        参数：
            G (torch_geometric.data.Data): 输入图

        返回：
            A (np.ndarray): 邻接矩阵 (n x n)
            N (np.ndarray 或 None): 节点特征矩阵 (n x k)，若无特征则为 None
        """
        edge_index = G.edge_index
        n = G.num_nodes
        A = np.zeros((n, n))
        edges = edge_index.cpu().numpy().T
        for (u, v) in edges:
            A[u, v] = 1
            A[v, u] = 1

        N = None
        if hasattr(G, 'x') and G.x is not None:
            N = G.x.cpu().numpy()

        return A, N

    def run(self, G1, G2, alignment):
        """
        执行 FINAL 对齐，与 MMNC 的 run 类似。

        参数：
            G1, G2: torch_geometric.data.Data类型的图数据（源图和目标图）
            _ : 忽略，仅为接口一致性

        返回:
            align_links (list): [G1中的节点列表, 对应的G2中对齐节点列表]
            align_rank (list): 对G1中每个节点在G2中所有节点的相似度排序列表
        """

        # 将输入的 PyG Data 转换为邻接矩阵与节点特征矩阵
        self.A1, self.N1 = self.prepare_data(G1)
        self.A2, self.N2 = self.prepare_data(G2)

        start_time = time()

        # 若无节点特征，则使用全 1 特征
        if self.N1 is None and self.N2 is None:
            self.N1 = np.ones((self.A1.shape[0], 1))
            self.N2 = np.ones((self.A2.shape[0], 1))

        # 无边特征时，使用邻接矩阵作为唯一的边特征输入
        self.E1 = np.zeros((1, self.A1.shape[0], self.A1.shape[1]))
        self.E2 = np.zeros((1, self.A2.shape[0], self.A2.shape[1]))
        self.E1[0] = self.A1
        self.E2[0] = self.A2
        self.no_edge_feat = True

        # 对节点特征归一化处理
        self.N1 = normalize(self.N1)
        self.N2 = normalize(self.N2)

        L = self.E1.shape[0]  # 边特征的维度数（这里为1）
        K = self.N1.shape[1]  # 节点特征维度
        n1 = self.A1.shape[0]  # G1的节点数
        n2 = self.A2.shape[0]  # G2的节点数

        # 计算节点特征的 Kronecker 积累和，以构建 N 向量
        # N向量存储节点特征之间的两两对应相似度（通过 Kronecker 积对特征进行组合）
        N = np.zeros(n1 * n2)
        for k in range(K):
            # np.kron(self.N1[:, k], self.N2[:, k]) 将G1第k维特征与G2第k维特征进行Kronecker积
            # 将两个特征列的内积展开为所有节点间的对应特征相似度
            N += np.kron(self.N1[:, k], self.N2[:, k])

        # 计算 d 向量，用于反映结构特征(基于度或其他结构性统计)对齐
        # 通过 E1[i]*A1 和 E2[i]*A2 对节点特征进行加权，从而得到结构-特征混合的表示
        d = np.zeros_like(N)
        for i in range(L):
            for k in range(K):
                # np.dot(self.E1[i]*self.A1, self.N1[:, k]) 表示在源图上使用边特征E1[i]和邻接A1加权节点特征N1[:, k]后的结构特征值
                # np.dot(self.E2[i]*self.A2, self.N2[:, k]) 同理，在目标图上的结构特征值
                # 再通过 Kronecker 积，将两个图的结构特征合并到一起
                d += np.kron(np.dot(self.E1[i] * self.A1, self.N1[:, k]),
                             np.dot(self.E2[i] * self.A2, self.N2[:, k]))
                # print(np.argsort(np.dot(self.E1[i] * self.A1, self.N1[:, k]))[:1000])
                # print(np.argsort(np.dot(self.E2[i] * self.A2, self.N2[:, k]))[1000:])

        # D = N*d 将节点特征相似度N与结构特征d组合
        D = N * d
        # DD = 1./np.sqrt(D) 对D中每个元素开方取倒数，用于后续归一化处理
        DD = 1. / np.sqrt(D)
        # 对无穷值进行处理
        DD[DD == inf] = 0
        # q = DD*N 用DD来加权N，得到修正后的特征分布q
        q = DD * N

        # 初始化先验相似度矩阵H，这里使用均匀分布作为初始相似度
        self.H = np.ones((n2, n1)) * (1 / n1)
        # 从alignment中取出20%的锚连接
        if alignment and len(alignment) > 0:
            anchor_count = int(self.train_ratio * len(alignment))
            anchors = list(alignment.items())[:anchor_count]

            # 对选中的锚连接进行强制先验约束
            for g1_node, g2_node in anchors:
                # 将H的对应行（g2_node行）设置为0，并在对应g1_node列设为1
                # H的维度：(n2, n1), 行对应G2节点, 列对应G1节点
                self.H[g2_node, :] = 0
                self.H[g2_node, int(g1_node)] = 1.0

        # 将H展平为向量h
        h = self.H.flatten('F')
        s = h

        # 迭代固定点对齐过程
        for i in range(self.maxiter):
            prev = s
            # M = (q*s).reshape((n2, n1), order='F') 将对齐向量s与q元素相乘后，变回矩阵形式(n2 x n1)
            # M的元素表示在当前对齐状态下每个目标节点与源节点的潜在匹配度
            M = (q * s).reshape((n2, n1), order='F')

            # S 用于存储在结构约束下的对齐评分更新
            S = np.zeros((n2, n1))
            for l in range(L):
                # 通过 E2[l]*A2 和 E1[l]*A1 对 M 进行双侧加权，模拟在两图结构下的关联
                # S += E2[l]*A2 * M * E1[l]*A1 将对齐情况M嵌入到结构约束下重新计算
                S += np.dot(np.dot(self.E2[l] * self.A2, M), self.E1[l] * self.A1)

            # s = (1 - self.alpha)*h + self.alpha*q*S.flatten('F')
            # 使用 (1 - alpha)*h 将先验相似度和 alpha*q*S 的更新值融合
            # s 向量更新后再在下次迭代使用
            s = (1 - self.alpha) * h + self.alpha * q * S.flatten('F')

            # 计算变化量 diff 来判断是否收敛
            diff = np.sqrt(np.sum((s - prev) ** 2))
            if diff < self.tol:
                break

        # 将最终的对齐向量 s reshape 为对齐矩阵 alignment_matrix (n1 x n2)
        # alignment_matrix[i, j] 表示源图中节点i与目标图中节点j的对齐得分
        alignment_matrix = s.reshape(n1, n2)

        # 从 alignment_matrix 中选出最佳匹配
        # best_matches[i] 找到第 i 个源节点对应得分最高的目标节点
        best_matches = np.argmax(alignment_matrix, axis=1)
        align_links = [list(range(n1)), best_matches.tolist()]

        # align_rank 为每个源节点在目标图中节点相似度的排序（从高到低）
        align_rank = []
        for i in range(n1):
            row = alignment_matrix[i]
            # np.argsort(-row) 对 row 的值进行降序排序，返回目标节点索引
            ranked = np.argsort(-row)
            align_rank.append(ranked.tolist())

        for anchor in anchors:
            for idx in range(len(align_links[0])):
                if int(anchor[0]) == align_links[0][idx]:
                    align_links[0].pop(idx)
                    align_links[1].pop(idx)
                    align_rank.pop(idx)
                    break
        end_time = time()

        return align_links, align_rank, end_time - start_time
