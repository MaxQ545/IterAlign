import math
import json
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor

from time import time


class oIterAlign:
    """
    oIterAlign model for graph node alignment using damping propagation and diffusion alignment.

    This class provides methods for damping propagation and diffusion alignment
    to process graph structures and align their nodes based on their features.
    """

    def __init__(self, config):
        self.device = config["device"]
        self.dp_min_degree = config["dp_min_degree"]
        self.num_dp_select = config["num_dp_select"]
        self.num_diffusion_select = config["num_diffusion_select"]
        self.num_step = config["diffusion_step"]

        print("\nmodels Settings:")
        print(f"{'=' * 50}")
        print("DP minimum feature mean:", self.dp_min_degree)
        print("Number of nodes selected by CENAExtractor:", self.num_dp_select)
        print("Number of nodes selected by Diffusion:", self.num_diffusion_select)
        print("Number of diffusion steps:", self.num_step)

    def random_walk_from_node(self, start_node, edges, num_steps, num_nodes):
        visited = torch.zeros(num_nodes, dtype=torch.float32)
        current_node = start_node

        for _ in range(num_steps):
            visited[current_node] += 1
            neighbors = edges[current_node]
            if neighbors:
                current_node = random.choice(neighbors)
            else:
                current_node = start_node  # 回到起始节点

        if visited.sum() > 0:
            visited /= visited.sum()  # 归一化为概率分布
        return visited

    def build_adjacency_list(self, edge_index, num_nodes):
        edges = {i: [] for i in range(num_nodes)}
        for i, j in edge_index.t().tolist():
            edges[i].append(j)
        return edges

    def damping_propagation(self, edge_index, output_dim):
        num_nodes = edge_index.max().item() + 1
        num_steps = 1000  # 随机游走的步数
        node_features = torch.zeros(num_nodes, output_dim, dtype=torch.float32, device=self.device)

        edges = self.build_adjacency_list(edge_index, num_nodes)

        with ProcessPoolExecutor() as executor:
            futures = []
            for node in range(num_nodes):
                futures.append(executor.submit(self.random_walk_from_node, node, edges, num_steps, num_nodes))

            for i, future in enumerate(futures):
                walk_result = future.result()
                node_features[i] = torch.sort(walk_result, descending=True)[0][:output_dim]

        return node_features

    def random_walk_distribution(self, start_node, edges, num_steps, num_nodes):
        visited = [0] * num_nodes
        current_node = start_node

        for _ in range(num_steps):
            visited[current_node] += 1
            neighbors = edges[current_node]
            if neighbors:
                current_node = random.choice(neighbors)
            else:
                current_node = start_node

        visited = torch.tensor(visited, dtype=torch.float32)
        if visited.sum() > 0:
            visited /= visited.sum()
        return visited

    def diffusion_align(self, x1_deg, x2_deg, e1, e2, num_epoch):
        x1_deg = x1_deg.to(self.device, dtype=torch.float32)
        x2_deg = x2_deg.to(self.device, dtype=torch.float32)

        num_nodes1 = x1_deg.size(0)
        num_nodes2 = x2_deg.size(0)

        x1 = torch.empty((num_nodes1, 0), dtype=torch.float32, device=self.device)
        x2 = torch.empty((num_nodes2, 0), dtype=torch.float32, device=self.device)

        edges1 = self.build_adjacency_list(e1, num_nodes1)
        edges2 = self.build_adjacency_list(e2, num_nodes2)

        align_links = [[], []]
        align_ranks = []

        for epoch in tqdm(range(num_epoch)):
            if epoch == 0:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1_deg, x2_deg, e1, e2, self.num_dp_select, self.dp_min_degree, None
                )
            else:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1, x2, e1, e2, self.num_diffusion_select, None, align_links
                )

            hot_matrix1 = torch.zeros((num_nodes1, len(hot_nodes1)), dtype=torch.float32, device=self.device)
            hot_matrix2 = torch.zeros((num_nodes2, len(hot_nodes2)), dtype=torch.float32, device=self.device)

            with ProcessPoolExecutor() as executor:
                futures1 = []
                for hot_index, hot_node in enumerate(hot_nodes1):
                    futures1.append(executor.submit(self.random_walk_distribution, hot_node, edges1, 1000, num_nodes1))
                for hot_index, f in enumerate(futures1):
                    walk_result = f.result()
                    hot_matrix1[:, hot_index] = walk_result.to(self.device)

                futures2 = []
                for hot_index, hot_node in enumerate(hot_nodes2):
                    futures2.append(executor.submit(self.random_walk_distribution, hot_node, edges2, 1000, num_nodes2))
                for hot_index, f in enumerate(futures2):
                    walk_result = f.result()
                    hot_matrix2[:, hot_index] = walk_result.to(self.device)

            x1 = torch.cat([x1, hot_matrix1], dim=1)
            x2 = torch.cat([x2, hot_matrix2], dim=1)

            align_links[0].extend(hot_nodes1)
            align_links[1].extend(hot_nodes2)
            align_ranks.extend(hot_nodes1_rank)

        return align_links, align_ranks

    def run(self, graph1, graph2, alignment):
        print("\nRunning......")
        print(f"{'=' * 50}")

        start_time = time()
        output_dim = min(graph1.num_nodes, graph2.num_nodes)

        # Step 1: Extract node features
        print("Extracting node features on plain graph...")
        x1 = self.damping_propagation(graph1.edge_index, output_dim)
        x2 = self.damping_propagation(graph2.edge_index, output_dim)

        # Step 2: Perform diffusion alignment
        print("Diffusion aligning...")
        num_epoch = math.ceil((len(alignment) - self.num_dp_select) / self.num_diffusion_select + 1)
        align_links, align_ranks = self.diffusion_align(
            x1, x2,
            graph1.edge_index, graph2.edge_index,
            num_epoch=num_epoch
        )
        end_time = time()

        return align_links, align_ranks, end_time - start_time

    def _select_nodes(self, x1, x2, adj1, adj2, num_select, min_degree, align_links, K=50):
        """
        选择节点——基于最小度和 L2 距离，通过匈牙利算法找到最优匹配对。
        """
        if align_links is None:
            deg1 = adj1.sum(dim=1)
            deg2 = adj2.sum(dim=1)
            mask1 = deg1 >= min_degree
            mask2 = deg2 >= min_degree
        else:
            mask1 = torch.ones(x1.size(0), dtype=torch.bool, device=self.device)
            mask2 = torch.ones(x2.size(0), dtype=torch.bool, device=self.device)
            mask1[align_links[0]] = False
            mask2[align_links[1]] = False

        select_nodes1 = torch.nonzero(mask1, as_tuple=True)[0]
        select_nodes2 = torch.nonzero(mask2, as_tuple=True)[0]

        x1_sub = x1[select_nodes1]
        x2_sub = x2[select_nodes2]

        dist, ind = l2_nestest(x1_sub, x2_sub, device=self.device)

        # 只取每个行节点最近的K个候选列节点
        K = min(K, x2_sub.size(0))
        topK_ind = ind[:, :K]  # shape: [N1_sub, K]

        # 构建一个稀疏问题：只计算 topK_ind 对应的距离
        N1_sub = x1_sub.size(0)
        N2_sub = x2_sub.size(0)
        large_val = 1e9
        dist_matrix = torch.full((N1_sub, N2_sub), large_val, device=self.device, dtype=torch.float32)

        # 只为 topK_ind 赋予真实距离值
        topK_x2_indices = topK_ind
        for i in range(N1_sub):
            cols = topK_x2_indices[i]
            dist_matrix[i, cols] = torch.cdist(x1_sub[i].unsqueeze(0), x2_sub[cols], p=2).flatten()

        dist_matrix_np = dist_matrix.cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(dist_matrix_np)

        matched_dist = dist_matrix_np[row_idx, col_idx]
        matched_dist_t = torch.tensor(matched_dist, device=self.device, dtype=torch.float32)
        sorted_indices = torch.argsort(matched_dist_t)
        selected_indices = sorted_indices[:num_select]

        row_idx_t = torch.tensor(row_idx, device=self.device, dtype=torch.long)
        col_idx_t = torch.tensor(col_idx, device=self.device, dtype=torch.long)

        final_row = row_idx_t[selected_indices]
        final_col = col_idx_t[selected_indices]

        node1_indices = select_nodes1[final_row].cpu().tolist()
        node2_indices = select_nodes2[final_col].cpu().tolist()

        # 使用 ind 恢复 node1_rank
        node1_rank = select_nodes2[ind[final_row]].tolist()

        return node1_indices, node2_indices, node1_rank


def edge2adj(edge_index):
    """
    Convert edge_index to adjacency matrix

    Parameters:
    edge_index: a 2D tensor with shape [2, E]

    Return: Adjacency matrix
    """
    n = edge_index.max().item() + 1
    # 这里统一使用 float32
    adj = torch.zeros(n, n, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    return adj


def l2_nestest(x1, x2, batch_size=5000, device='cuda'):
    """
    Compute L2 distances between two sets of vectors in batches using PyTorch,
    returning sorted results.

    Parameters:
    - x1: PyTorch tensor of shape (n1, d), the query vector set
    - x2: PyTorch tensor of shape (n2, d), the target vector set
    - batch_size: int, the batch size for processing x1 in chunks
    - device: str, 'cuda' or 'cpu', specifying the computation device

    Returns:
    - sorted_distances: PyTorch tensor of shape (n1, n2), sorted distance matrix
    - sorted_indices: PyTorch tensor of shape (n1, n2), sorted target vector indices
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    # 这里统一使用 float32
    sorted_distances = torch.empty((n1, n2), dtype=torch.float32, device=device)
    sorted_indices = torch.empty((n1, n2), dtype=torch.long, device=device)

    for start_idx in range(0, n1, batch_size):
        end_idx = min(start_idx + batch_size, n1)
        x1_batch = x1[start_idx:end_idx]
        with torch.no_grad():
            dist_matrix = torch.cdist(x1_batch, x2, p=2)  # [batch_size, n2], float32
        dist_batch, ind_batch = torch.sort(dist_matrix, dim=1)
        sorted_distances[start_idx:end_idx] = dist_batch
        sorted_indices[start_idx:end_idx] = ind_batch

    return sorted_distances, sorted_indices


def draw_hot_graph_with_intensity(adj_matrix, hot_matrix, step, graph_name="Graph"):
    """
    Draw the graph with node color intensity based on hot_matrix[:, dim].

    Parameters:
    adj_matrix: PyTorch tensor of the adjacency matrix.
    hot_matrix: PyTorch tensor where the node intensities are stored.
    step: The current diffusion step (used for visualization).
    graph_name: Name of the graph (used in the title of the plot).
    """
    adj_np = adj_matrix.cpu().numpy()
    hot_values = hot_matrix.sum(dim=1).cpu().numpy()

    # 这里将 numpy 转回 torch 时也使用 float32
    norm_hot_values = torch.tensor(hot_values, dtype=torch.float32, device='cpu')
    # 使用 sigmoid 进行归一化
    norm_hot_values = norm_hot_values.sigmoid().numpy()

    g = nx.from_numpy_array(adj_np)
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.get_cmap('coolwarm')
    node_collection = nx.draw_networkx_nodes(
        g, pos,
        node_color=norm_hot_values,
        cmap=cmap,
        node_size=5,
        vmin=0, vmax=1
    )
    plt.colorbar(node_collection, label='Node Intensity')
    plt.title(f"{graph_name} at Diffusion Step {step}")
    plt.savefig(f"image/hot_graph_{graph_name}_step_{step}.png")
    plt.close()
