import math
import json
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F


class MMNC_CENA:
    """
    This is MMNC's first step: CENA
    """

    def __init__(self, config):
        """
        Initializes the SPNA model.

        Args:
            device (torch.device): The device to perform computations.
            df (int): Damping factor
            dp_step (int): Number of propagation steps.
            damping_factor (float): Factor controlling feature retention and propagation.
            num_dp_select (int): Number of nodes selected by damping propagation.
            dp_min_degree (float): Minimum degree for node selection.
            num_diffusion_select (int): Number of nodes selected in subsequent epochs.
            num_epoch (int): Number of epochs to run alignment.
            num_step (int): Number of diffusion steps.
        """
        self.device = config["device"]
        self.dp_min_degree = config["dp_min_degree"]
        self.num_dp_select = config["num_dp_select"]
        self.num_diffusion_select = config["num_diffusion_select"]
        self.num_step = config["diffusion_step"]

        # 打印模型设置
        print("\nmodels Settings:")
        print(f"{'=' * 50}")
        print("DP minimum feature mean:", self.dp_min_degree)
        print("Number of nodes selected by CENAExtractor:", self.num_dp_select)
        print("Number of nodes selected by Diffusion:", self.num_diffusion_select)
        print("Number of diffusion steps:", self.num_step)

    def cena(self, edge_index, num_layer, device):
        """
        Generate node features based on the degree of the nodes using efficient PyTorch operations.

        Parameters:
        edge_index: Edge index of the graph with shape [2, E]
        num_layer: Number of layers to be considered
        device: Device to run the method

        Return: Node features with shape [N, 5 * (num_layer + 1)]
        """
        adj = edge2adj(edge_index).to(device)
        deg = adj.sum(dim=1)

        degree_stats = cal_degree_dict_torch(adj, deg, num_layer)

        # Collect features
        feature_list = []
        for layer_stats in degree_stats:
            max_deg, median_deg, min_deg, q75_deg, q25_deg = layer_stats
            feature_list.extend([
                torch.log(max_deg + 1),
                torch.log(median_deg + 1),
                torch.log(min_deg + 1),
                torch.log(q75_deg + 1),
                torch.log(q25_deg + 1)
            ])

        feature_mat = torch.stack(feature_list, dim=1)  # Shape [N, 5 * (num_layer + 1)]
        return feature_mat

    def _select_nodes(self, x1, x2, adj1, adj2, num_select, min_degree, align_links):
        """
        Selects nodes based on their degree and L2 distance between features.

        Args:
            x1 (torch.Tensor): Node features of graph 1 with shape [N1, F1].
            x2 (torch.Tensor): Node features of graph 2 with shape [N2, F2].
            adj1 (torch.Tensor): Adjacency matrix of graph 1 with shape [N1, N1].
            adj2 (torch.Tensor): Adjacency matrix of graph 2 with shape [N2, N2].
            num_select (int): Number of nodes to select.
            min_degree (float): Minimum degree threshold for node selection.
            align_links (list): Existing alignment links to exclude.

        Returns:
            tuple: Three lists:
                - node1_indices (list): Selected node indices in graph 1.
                - node2_indices (list): Corresponding selected node indices in graph 2.
                - node1_rank (list): Rank of node2_indices in the original embeddings.
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

        x1 = x1[select_nodes1]
        x2 = x2[select_nodes2]

        dist, ind = l2_nestest(x1, x2, device=self.device)

        first_dist = dist[:, 0]
        first_ind = ind[:, 0]

        selected_indices = torch.argsort(first_dist)[:num_select]
        best_ind = first_ind[selected_indices]

        node1_indices = [select_nodes1[i].item() for i in selected_indices]
        node2_indices = [select_nodes2[i].item() for i in best_ind]
        node1_rank = select_nodes2[ind[selected_indices]].tolist()

        return node1_indices, node2_indices, node1_rank

    def diffusion_align(self, x1_deg, x2_deg, e1, e2, num_epoch):
        """
        Performs diffusion alignment to align two graphs.

        Args:
            x1_deg (torch.Tensor): Node features of graph 1 with shape [N1, F1].
            x2_deg (torch.Tensor): Node features of graph 2 with shape [N2, F2].
            e1 (torch.Tensor): Edge index of graph 1 with shape [2, E1].
            e2 (torch.Tensor): Edge index of graph 2 with shape [2, E2].
            num_epoch (int): Number of epochs to run alignment.

        Returns:
            tuple: Alignment links and ranks as two lists:
                - align_links (list): Alignment links for the two graphs.
                - align_ranks (list): Rank information for aligned nodes.
        """
        x1_deg = x1_deg.to(self.device)
        x2_deg = x2_deg.to(self.device)

        num_nodes1 = x1_deg.size(0)
        num_nodes2 = x2_deg.size(0)
        x1 = torch.empty((num_nodes1, 0), dtype=torch.float32, device=self.device)
        x2 = torch.empty((num_nodes2, 0), dtype=torch.float32, device=self.device)
        adj1 = edge2adj(e1).to(self.device)
        adj2 = edge2adj(e2).to(self.device)

        deg_inv_sqrt1 = torch.diag(torch.pow(adj1.sum(dim=1), -0.5))
        adj_norm1 = deg_inv_sqrt1 @ adj1 @ deg_inv_sqrt1
        deg_inv_sqrt2 = torch.diag(torch.pow(adj2.sum(dim=1), -0.5))
        adj_norm2 = deg_inv_sqrt2 @ adj2 @ deg_inv_sqrt2

        align_links = [[], []]
        align_ranks = []

        for epoch in range(num_epoch):
            if epoch == 0:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1_deg, x2_deg, adj1, adj2, self.num_dp_select, self.dp_min_degree, None
                )
            else:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1, x2, adj1, adj2, self.num_diffusion_select, None, align_links
                )

            hot_matrix1 = torch.zeros((num_nodes1, len(hot_nodes1)), dtype=torch.float32, device=self.device)
            hot_matrix2 = torch.zeros((num_nodes2, len(hot_nodes2)), dtype=torch.float32, device=self.device)

            hot_matrix1[hot_nodes1, range(len(hot_nodes1))] = 1
            hot_matrix2[hot_nodes2, range(len(hot_nodes2))] = 1

            for step in range(self.num_step):
                hot_matrix1 = adj_norm1 @ hot_matrix1 + hot_matrix1
                hot_matrix2 = adj_norm2 @ hot_matrix2 + hot_matrix2

            x1 = torch.cat([x1, hot_matrix1], dim=1)
            x2 = torch.cat([x2, hot_matrix2], dim=1)

            align_links[0].extend(hot_nodes1)
            align_links[1].extend(hot_nodes2)
            align_ranks.extend(hot_nodes1_rank)

        return align_links, align_ranks

    def run(self, graph1, graph2, alignment):
        print("\nRunning......")
        print(f"{'=' * 50}")

        # Step 1 Extract node features
        print("Extracting node features using CENA...")
        output_dim = min(graph1.num_nodes, graph2.num_nodes)

        x1 = self.cena(graph1.edge_index, 2, device=self.device)
        x2 = self.cena(graph2.edge_index, 2, device=self.device)

        # Step 2 Perform diffusion alignment
        print("Diffusion aligning...")
        num_epoch = math.ceil((len(alignment) - self.num_dp_select) / self.num_diffusion_select + 1)
        align_links, align_ranks = self.diffusion_align(x1, x2, graph1.edge_index, graph2.edge_index, num_epoch=num_epoch)

        return align_links, align_ranks, 0


def edge2adj(edge_index):
    """
    Convert edge_index to adjacency matrix

    Parameters:
    edge_index: a 2D tensor with shape [2, E]

    Return: Adjacency matrix
    """
    n = edge_index.max().item() + 1
    adj = torch.zeros(n, n, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1

    return adj


def l2_nestest(x1, x2, batch_size=5000, device='cuda'):
    """
    Compute L2 distances between two sets of vectors in batches using PyTorch,
    returning sorted results.

    Parameters:
    - x1: NumPy array of shape (n1, d), the query vector set
    - x2: NumPy array of shape (n2, d), the target vector set
    - batch_size: int, the batch size for processing x1 in chunks
    - device: str, 'cuda' or 'cpu', specifying the computation device

    Returns:
    - sorted_distances: NumPy array of shape (n1, n2), sorted distance matrix
    - sorted_indices: NumPy array of shape (n1, n2), sorted target vector indices
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    sorted_distances = torch.empty((n1, n2), dtype=torch.float32, device=device)
    sorted_indices = torch.empty((n1, n2), dtype=torch.int64, device=device)

    for start_idx in range(0, n1, batch_size):
        end_idx = min(start_idx + batch_size, n1)
        x1_batch = x1[start_idx:end_idx]
        with torch.no_grad():
            dist_matrix = torch.cdist(x1_batch, x2, p=2)  # [batch_size, n2]
        dist_batch, ind_batch = torch.sort(dist_matrix, dim=1)
        sorted_distances[start_idx:end_idx] = dist_batch
        sorted_indices[start_idx:end_idx] = ind_batch

    sorted_distances = sorted_distances
    sorted_indices = sorted_indices

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

    # Normalize the hot values
    if hot_values.max() == hot_values.min():
        norm_hot_values = hot_values / hot_values.max()
    else:
        norm_hot_values = (hot_values - hot_values.min()) / (hot_values.max() - hot_values.min())

    # Draw the graph
    g = nx.from_numpy_array(adj_np)
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.get_cmap('coolwarm')
    node_collection = nx.draw_networkx_nodes(g, pos, node_color=norm_hot_values, cmap=cmap,
                                             node_size=5, vmin=0, vmax=1)
    # nx.draw_networkx_edges(g, pos, alpha=0.5)
    # nx.draw_networkx_labels(g, pos, font_size=8)
    plt.colorbar(node_collection, label='Node Intensity')
    plt.title(f"{graph_name} at Diffusion Step {step}")
    plt.savefig(f"image/hot_graph_{graph_name}_step_{step}.png")


def cal_degree_dict_torch(adj, deg, num_layer):
    N = adj.size(0)
    device = adj.device

    neighbor_masks = torch.zeros(num_layer + 1, N, N, dtype=torch.bool, device=device)

    # Layer 0: self nodes
    neighbor_masks[0] = torch.eye(N, dtype=torch.bool, device=device)

    for i in range(1, num_layer + 1):
        prev_neighbors = torch.any(neighbor_masks[:i], dim=0)
        new_neighbors = (neighbor_masks[i - 1].float() @ adj.float()) > 0
        neighbor_masks[i] = new_neighbors & (~prev_neighbors)

    degree_stats = []

    for i in range(num_layer + 1):
        degrees_per_node_per_layer = neighbor_masks[i].float() * deg.view(1, -1)
        valid_mask = degrees_per_node_per_layer != 0

        # Replace zeros with -inf for max, inf for min
        degrees_for_max = degrees_per_node_per_layer.clone()
        degrees_for_min = degrees_per_node_per_layer.clone()

        degrees_for_max[~valid_mask] = float('-inf')
        degrees_for_min[~valid_mask] = float('inf')

        max_deg = degrees_for_max.max(dim=1)[0]
        min_deg = degrees_for_min.min(dim=1)[0]

        # Handle cases where all values are -inf or inf
        max_deg[max_deg == float('-inf')] = 0.0
        min_deg[min_deg == float('inf')] = 0.0

        # For median and quantiles, we need to handle masks
        # This requires more complex operations or loops

        # Collecting degrees into a list for each node
        degrees_list = [degrees_per_node_per_layer[i][valid_mask[i]] for i in range(N)]

        median_deg = torch.zeros(N, device=device)
        q75_deg = torch.zeros(N, device=device)
        q25_deg = torch.zeros(N, device=device)

        for node, degrees_i in enumerate(degrees_list):
            if degrees_i.numel() == 0:
                median_deg[node] = 0.0
                q75_deg[node] = 0.0
                q25_deg[node] = 0.0
            else:
                sorted_degrees_i, _ = degrees_i.sort()
                num_elements = sorted_degrees_i.numel()
                median_deg[node] = sorted_degrees_i[num_elements // 2]
                idx_75 = int(0.75 * (num_elements - 1))
                idx_25 = int(0.25 * (num_elements - 1))
                q75_deg[node] = sorted_degrees_i[idx_75]
                q25_deg[node] = sorted_degrees_i[idx_25]

        degree_stats.append((max_deg, median_deg, min_deg, q75_deg, q25_deg))

    return degree_stats
