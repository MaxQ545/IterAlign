import numpy as np
import heapq
import torch
from sklearn.neighbors import KDTree
import networkx as nx
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from tqdm import tqdm
from time import time


class iMMNC:
    def __init__(self, config):
        self.k_de = config['k_de']
        self.k_nei = config['k_nei']
        self.train_ratio = config['train_ratio']
        self.fast = config['fast_select']
        self.degree_thresold = config['degree_thresold']
        self.rate = config['rate']
        self.niter = config['niter']

    def CenaExtractNodeFeature(self, g, layers):
        g_degree_dict = cal_degree_dict(list(g.nodes()), g, layers)
        g_nodes = [i for i in range(len(g))]
        N1 = len(g_nodes)
        feature_mat = []
        for layer in range(layers + 1):
            L_max = [np.log( np.max(g_degree_dict[layer][x]) + 1) for x in g_nodes]
            L_med= [np.log(np.median(g_degree_dict[layer][x]) + 1) for x in g_nodes]
            L_min=  [np.log( np.min(g_degree_dict[layer][x]) + 1) for x in g_nodes]
            L_75 = [np.log(np.percentile(g_degree_dict[layer][x], 75) + 1) for x in g_nodes]
            L_25 = [np.log( np.percentile(g_degree_dict[layer][x], 25) + 1) for x in g_nodes]
            feature_mat.append(L_max)
            feature_mat.append(L_min)
            feature_mat.append(L_med)
            feature_mat.append(L_75)
            feature_mat.append(L_25)
        feature_mat = np.array(feature_mat).reshape((-1,N1))
        return feature_mat.transpose()

    def select_train_nodes(self, e1, e2, train_ratio=0.01, distance_metric="euclidean", num_top=1):
        n_nodes = e1.shape[0]

        kd_tree = KDTree(e2, metric=distance_metric)

        dist, ind = kd_tree.query(e1, k=num_top)
        dist_list = -dist[:,0]
        ind_list = ind[:,0]

        index_l = heapq.nlargest(int(train_ratio*n_nodes), range(len(dist_list)), dist_list.__getitem__)
        train_data_dict = {i: ind_list[i] for i in index_l}

        return train_data_dict

    def fast_select_train_nodes(self, g1, g2, e1, e2, train_ratio=0.01, distance_metric="euclidean", num_top=1, degree_threshold=6):
        n = min(len(g1),len(g2))
        select_nodes1 = [node for node in g1.nodes() if g1.degree[node] >= degree_threshold]
        select_nodes2 = [node for node in g2.nodes() if g2.degree[node] >= degree_threshold]

        index_dict1 = dict(zip(list(range(len(select_nodes1))),select_nodes1))
        index_dict2 = dict(zip(list(range(len(select_nodes2))),select_nodes2))

        new_e1 = e1[select_nodes1]
        new_e2 = e2[select_nodes2]

        kd_tree = KDTree(new_e2, metric=distance_metric)

        dist, ind = kd_tree.query(new_e1, k=num_top)
        dist_list = -dist[:, 0]
        ind_list = ind[:, 0]
        if int(train_ratio * n)>min(len(select_nodes1),len(select_nodes2)):
            num = min(len(select_nodes1),len(select_nodes2))
        else:
            num = int(train_ratio * n)

        index_l = heapq.nlargest(num, range(len(dist_list)), dist_list.__getitem__)
        train_data_dict = {index_dict1[i]: index_dict2[ind_list[i]] for i in index_l}
        return train_data_dict

    def align_embedding(self, g1, g2 ,nodes1 ,nodes2, K_nei, e1=None, e2=None):
        adj1 = nx.to_numpy_array(g1, nodelist=list(range(len(g1))))
        adj2 = nx.to_numpy_array(g2, nodelist=list(range(len(g2))))
        D1 = np.sum(adj1,axis=0)
        D2 = np.sum(adj2,axis=0)

        e1 = netmf(sps.csr_matrix(adj1),dim=128)
        e2 = netmf(sps.csr_matrix(adj2),dim=128)

        obj = e1[nodes1].T @  e2[nodes2]
        e1_star = e1
        e2_star = e2

        combined_e1 = [e1]
        combined_e2 = [e2]

        tmp1 = sps.csr_matrix(np.diag(1 / D1))@sps.csr_matrix(adj1)
        tmp2 = sps.csr_matrix(np.diag(1 / D2))@sps.csr_matrix(adj2)

        for i in range(K_nei):
            e1_star = tmp1 @ e1_star
            e2_star = tmp2 @ e2_star
            combined_e1.append(e1_star)
            combined_e2.append(e2_star)
            obj += e1_star[nodes1].T @  e2_star[nodes2]

        obj = obj / K_nei
        u, _, v = np.linalg.svd(obj)
        R = u @ v
        trans_e1 = e1 @ R

        trans_combined_e1 = np.concatenate([item@ R for item in combined_e1],axis=-1)
        combined_e2 = np.concatenate(combined_e2, axis=-1)

        return trans_e1, e2, trans_combined_e1,combined_e2

    def run(self, G1, G2, _):
        g1 = pyg_to_nx(G1, directed=False)
        g2 = pyg_to_nx(G2, directed=False)

        start_time = time()

        e1 = self.CenaExtractNodeFeature(g1, self.k_de)
        e2 = self.CenaExtractNodeFeature(g2, self.k_de)

        train_dict = self.fast_select_train_nodes(g1, g2, e1, e2,
                                                  train_ratio=self.train_ratio,
                                                  degree_threshold=self.degree_thresold)
        nodes1 = list(train_dict.keys())
        nodes2 = list([train_dict[i] for i in nodes1])
        # Step 2: Align Embedding Spaces
        for i in tqdm(range(self.niter)):
            # Step 2: Align Embedding Spaces
            if i == 0:
                aligned_embed1, embed2, trans_combined_e1, combined_e2 = self.align_embedding(g1, g2, nodes1,
                                                                                         nodes2, self.k_nei)

            else:
                aligned_embed1, embed2, trans_combined_e1, combined_e2 = self.align_embedding(g1, g2, nodes1,
                                                                                         nodes2, self.k_nei,
                                                                                         aligned_embed1, embed2)

            # Step 3: Match Nodes with Similar Embeddings
            if self.fast:
                train_dict = self.fast_select_train_nodes(g1, g2, trans_combined_e1, combined_e2,
                                                     train_ratio=max(self.train_ratio + self.rate * (i + 1), 1.0),
                                                     degree_threshold=self.degree_thresold)
            else:
                train_dict = self.select_train_nodes(trans_combined_e1, combined_e2,
                                                train_ratio=max(self.train_ratio + self.rate * (i + 1), 0.5))

            nodes1 = list(train_dict.keys())
            nodes2 = list([train_dict[i] for i in nodes1])

        # Step 3: Match Nodes with Similar Embeddings
        align_links = KDTreeAlignmentHit1(aligned_embed1, embed2)
        align_rank = KDTreeAlignmentHitK(aligned_embed1, embed2)

        end_time = time()

        return align_links, align_rank, end_time - start_time


def cal_degree_dict(G_list, G, layer):
    G_degree = G.degree()
    degree_dict = {}
    degree_dict[0] = {}
    for node in G_list:
        degree_dict[0][node] = {node}
    for i in range(1, layer + 1):
        degree_dict[i] = {}
        for node in G_list:
            neighbor_set = []
            for neighbor in degree_dict[i - 1][node]:
                neighbor_set += nx.neighbors(G, neighbor)
            neighbor_set = set(neighbor_set)
            for j in range(i - 1, -1, -1):
                neighbor_set -= degree_dict[j][node]
            degree_dict[i][node] = neighbor_set
    for i in range(layer + 1):
        for node in G_list:
            if len(degree_dict[i][node]) == 0:
                degree_dict[i][node] = [0]
            else:
                degree_dict[i][node] = node_to_degree(G_degree, degree_dict[i][node])
    return degree_dict

def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET

#Full NMF matrix (which NMF factorizes with SVD)
#Taken from MILE code
def netmf_mat_full(A, window = 10, b=1.0):
    if not sps.issparse(A):
        A = sps.csr_matrix(A)
    #print "A shape", A.shape
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = sps.csgraph.laplacian(A, normed=True, return_diag=True)
    X = sps.identity(n) - L
    S = np.zeros_like(X)
    X_power = sps.identity(n)
    for i in range(window):
        #print "Compute matrix %d-th power" % (i + 1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sps.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    result = np.log(np.maximum(M.todense(),1))
    return sps.csr_matrix(result)


#Used in NetMF, AROPE
def svd_embed(prox_sim, dim):

    # 无语，sparse.linalg.svds居然有误差，同一个矩阵分解出来的差很多，SOS
    u, s, v = sps.linalg.svds(prox_sim, dim, return_singular_vectors="u")

    # u,s,v = np.linalg.svd(prox_sim.todense(),dim)
    return sps.diags(np.sqrt(s)).dot(u.T).T


def netmf(A, dim = 128, window=5, b=1.0, normalize = True):
    prox_sim = netmf_mat_full(A, window, b)
    embed = svd_embed(prox_sim, dim)
    if normalize:
        norms = np.linalg.norm(embed, axis = 1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    return embed


def pyg_to_nx(data, directed=False):
    """
    将 torch_geometric 的 Data 对象转换为 NetworkX 的图。

    参数：
    - data (torch_geometric.data.Data): 包含 edge_index 和 num_nodes 的 Data 对象。
    - directed (bool): 是否创建有向图。默认为 False（无向图）。

    返回：
    - G (networkx.Graph 或 networkx.DiGraph): 转换后的 NetworkX 图。
    """
    # 确定创建有向图还是无向图
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # 获取节点数量
    num_nodes = data.num_nodes
    # 添加节点（节点编号从 0 到 num_nodes-1）
    G.add_nodes_from(range(num_nodes))

    # 获取 edge_index，并确保它在 CPU 上
    edge_index = data.edge_index
    if edge_index.is_cuda:
        edge_index = edge_index.cpu()

    # 将 edge_index 转换为 NumPy 数组，并转置为边列表
    edge_list = edge_index.numpy().T.tolist()

    # 如果是无向图，确保每条边只添加一次
    if not directed:
        # 对边进行去重处理（因为无向图中 (u, v) 和 (v, u) 是相同的）
        edge_set = set()
        for edge in edge_list:
            sorted_edge = tuple(sorted(edge))
            edge_set.add(sorted_edge)
        unique_edge_list = list(edge_set)
        G.add_edges_from(unique_edge_list)
    else:
        G.add_edges_from(edge_list)

    return G


def KDTreeAlignmentHit1(emb1, emb2, distance_metric="euclidean"):
    ## 稀疏矩阵对齐
    kd_tree = KDTree(emb2, metric=distance_metric)

    kd_tree.reset_n_calls()

    dist, ind = kd_tree.query(emb1, k=1)
    ind_list = ind[:, 0]

    align_links = [[], []]
    align_links[0] = list(range(emb1.shape[0]))
    align_links[1] = ind_list.tolist()

    return align_links


def KDTreeAlignmentHitK(emb1, emb2, distance_metric="euclidean"):
    kd_tree = KDTree(emb2, metric=distance_metric)
    dist, ind = kd_tree.query(emb1, k=emb2.shape[0])
    # ind is of shape (emb1.shape[0], emb2.shape[0])
    # For each node in emb1, ind[i] contains indices of emb2 nodes, ordered from closest to farthest
    ind = ind.tolist()
    return ind


def get_counterpart(alignment_matrix, true_alignments, K):
    n_nodes = alignment_matrix.shape[0]

    correct_nodes_hits =[]

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None: #if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix.todense()[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]

        if target_alignment in node_sorted_indices[-K:]:
            correct_nodes_hits.append(node_index)

    return correct_nodes_hits
