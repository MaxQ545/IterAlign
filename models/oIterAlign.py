import math
import json
import random

import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor
from time import time


class oIterAlign:
    """
    oIterAlign model for graph node alignment.
    It contains three main procedures:
        1) Diffusion without anchors
        2) Diffusion with anchors
        3) Node pair selection based on a Linear Assignment approach
    """
    def __init__(self, config):
        """
        Initializes the oIterAlign model with the given configuration.

        Args:
            config (dict): A dictionary containing the following keys:
                - "device" (str or torch.device): The computation device.
                - "dp_min_degree" (float): The minimum degree threshold for node selection of first iteration.
                - "num_dp_select" (int): Number of node pairs to select in the first iteration.
                - "num_diffusion_select" (int): Number of node pairs to select in subsequent iterations.
                - "diffusion_step" (int): Number of diffusion steps to perform.
        """
        self.device = config["device"]
        self.dp_min_degree = config["dp_min_degree"]
        self.num_dp_select = config["num_dp_select"]
        self.num_diffusion_select = config["num_diffusion_select"]
        self.num_step = config["diffusion_step"]

        print("\nmodels Settings:")
        print(f"{'=' * 50}")
        print("DP minimum feature mean:", self.dp_min_degree)
        print("Number of nodes selected by First Iter:", self.num_dp_select)
        print("Number of nodes selected by Later Iter:", self.num_diffusion_select)
        print("Number of diffusion steps:", self.num_step)

    def diffusion_without_anchor(self, edge_index, output_dim):
        """
        Performs diffusion without any anchor constraints.

        The method constructs a one-hot encoding (identity matrix), applies
        a symmetrically normalized adjacency repeatedly for the configured
        number of steps, and then retains the top `output_dim` features for each node.

        Args:
            edge_index (torch.Tensor): Edge index with shape [2, E].
            output_dim (int): The number of top features (columns) to retain
                              after diffusion and sorting.

        Returns:
            torch.Tensor: A feature matrix of shape [N, output_dim],
                          where N is the number of nodes.
        """
        adj = edge2adj(edge_index).to(self.device, dtype=torch.float32)

        # Symmetric normalization of the adjacency matrix
        deg_inv_sqrt1 = torch.diag(torch.pow(adj.sum(dim=1), -0.5))
        adj_norm = deg_inv_sqrt1 @ adj @ deg_inv_sqrt1

        # Initialize a one-hot matrix (identity-like) for diffusion
        hot_matrix = torch.zeros_like(adj, dtype=torch.float32, device=self.device)
        hot_matrix[range(adj.shape[0]), range(adj.shape[0])] = 1.0

        # Perform diffusion for the specified number of steps
        for t in range(self.num_step + 1):
            hot_matrix = adj_norm @ hot_matrix + hot_matrix

        # Sort features per node and retain the top 'output_dim'
        feature = torch.sort(hot_matrix, dim=1)[0]
        feature = feature[:, -output_dim:]

        return feature

    def _select_nodes(self, x1, x2, adj1, adj2, num_select, min_degree, align_links, K=50):
        """
        Selects node pairs using the Linear Assignment Algorithm with partial distance computation.

        This function:
          1) Filters nodes by minimum degree threshold if align_links is None.
             Otherwise, it masks out already aligned nodes in align_links.
          2) Computes distances in batches, but only keeps the top K closest
             candidates per node to form a sparse cost matrix.
          3) Uses the Hungarian (linear_sum_assignment) algorithm to match pairs
             based on this cost matrix.
          4) Returns the best 'num_select' pairs according to the smallest distances.

        Args:
            x1 (torch.Tensor): Node feature matrix for graph 1, shape [N1, F].
            x2 (torch.Tensor): Node feature matrix for graph 2, shape [N2, F].
            adj1 (torch.Tensor): Adjacency matrix for graph 1, shape [N1, N1].
            adj2 (torch.Tensor): Adjacency matrix for graph 2, shape [N2, N2].
            num_select (int): Number of pairs to select after alignment.
            min_degree (float or None): Minimum node degree threshold for initial filtering.
                                        If None, it uses existing alignment to filter.
            align_links (list or None): Already matched pairs in the form [list1, list2]
                                        or None if no matches yet.
            K (int, optional): The number of nearest candidates to keep in the cost matrix
                               for each node (default: 50).

        Returns:
            tuple of:
              - node1_indices (list): Indices of the selected nodes in graph 1.
              - node2_indices (list): Indices of the selected nodes in graph 2.
              - node1_rank (list): For each selected node in graph 1, the ranks of its
                                   nearest candidates in graph 2.
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

        # For each node in x1_sub, only keep the K closest nodes from x2_sub
        K = min(K, x2_sub.size(0))
        topK_ind = ind[:, :K]  # [N1_sub, K]

        N1_sub = x1_sub.size(0)
        N2_sub = x2_sub.size(0)
        large_val = 1e9
        dist_matrix = torch.full((N1_sub, N2_sub), large_val, device=self.device, dtype=torch.float32)

        # Fill in the real distances only for the top K candidates
        for i in range(N1_sub):
            cols = topK_ind[i]
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

        # node1_rank gives, for each selected node in graph 1, a list of potential partners in graph 2
        node1_rank = select_nodes2[ind[final_row]].tolist()

        return node1_indices, node2_indices, node1_rank

    def diffusion_with_anchor(self, x1_deg, x2_deg, e1, e2, num_epoch):
        """
        Performs iterative diffusion using existing anchor (aligned) nodes.

        In each epoch:
          1) A set of node pairs is selected via _select_nodes().
          2) These pairs are encoded as one-hot vectors and diffused across
             the respective graphs.
          3) The resulting diffusion features are concatenated to the existing
             features to refine subsequent alignment.

        Args:
            x1_deg (torch.Tensor): Node features of graph 1, shape [N1, F1].
            x2_deg (torch.Tensor): Node features of graph 2, shape [N2, F2].
            e1 (torch.Tensor): Edge index for graph 1, shape [2, E1].
            e2 (torch.Tensor): Edge index for graph 2, shape [2, E2].
            num_epoch (int): Number of alignment epochs to perform.

        Returns:
            tuple:
                - align_links (list): A list of matched node indices for graph 1 and graph 2.
                                      align_links[0] stores matched indices in graph 1,
                                      align_links[1] stores matched indices in graph 2.
                - align_ranks (list): Rank information for aligned nodes (nearest neighbors).
        """
        x1_deg = x1_deg.to(self.device, dtype=torch.float32)
        x2_deg = x2_deg.to(self.device, dtype=torch.float32)

        num_nodes1 = x1_deg.size(0)
        num_nodes2 = x2_deg.size(0)
        x1 = torch.empty((num_nodes1, 0), dtype=torch.float32, device=self.device)
        x2 = torch.empty((num_nodes2, 0), dtype=torch.float32, device=self.device)

        adj1 = edge2adj(e1).to(self.device, dtype=torch.float32)
        adj2 = edge2adj(e2).to(self.device, dtype=torch.float32)

        deg_inv_sqrt1 = torch.diag(torch.pow(adj1.sum(dim=1), -0.5))
        deg_inv_sqrt2 = torch.diag(torch.pow(adj2.sum(dim=1), -0.5))
        adj_norm1 = deg_inv_sqrt1 @ adj1 @ deg_inv_sqrt1
        adj_norm2 = deg_inv_sqrt2 @ adj2 @ deg_inv_sqrt2

        # Add self-loop
        adj_norm1 = torch.eye(adj_norm1.shape[0], dtype=torch.float32, device=self.device) + adj_norm1
        adj_norm2 = torch.eye(adj_norm2.shape[0], dtype=torch.float32, device=self.device) + adj_norm2

        # Precompute diffusion kernels
        diffusion_kernel1 = torch.eye(x1.shape[0], dtype=torch.float32, device=self.device)
        diffusion_kernel2 = torch.eye(x2.shape[0], dtype=torch.float32, device=self.device)
        for step in range(self.num_step):
            diffusion_kernel1 = adj_norm1 @ diffusion_kernel1
            diffusion_kernel2 = adj_norm2 @ diffusion_kernel2

        align_links = [[], []]
        align_ranks = []

        for epoch in tqdm(range(num_epoch)):
            if epoch == 0:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1_deg, x2_deg, adj1, adj2,
                    self.num_dp_select, self.dp_min_degree,
                    None
                )
            else:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1, x2, adj1, adj2,
                    self.num_diffusion_select, None,
                    align_links
                )

            hot_matrix1 = torch.zeros((num_nodes1, len(hot_nodes1)), dtype=torch.float32, device=self.device)
            hot_matrix2 = torch.zeros((num_nodes2, len(hot_nodes2)), dtype=torch.float32, device=self.device)

            hot_matrix1[hot_nodes1, range(len(hot_nodes1))] = 1.0
            hot_matrix2[hot_nodes2, range(len(hot_nodes2))] = 1.0

            # Diffuse using precomputed kernels
            hot_matrix1 = diffusion_kernel1 @ hot_matrix1
            hot_matrix2 = diffusion_kernel2 @ hot_matrix2

            # Diffusion by steps if not compute diffusion kernel first
            # for step in range(self.num_step):
            #     hot_matrix1 = adj_norm1 @ hot_matrix1
            #     hot_matrix2 = adj_norm2 @ hot_matrix2

            # Concatenate diffusion results with the existing features
            x1 = torch.cat([x1, hot_matrix1], dim=1)
            x2 = torch.cat([x2, hot_matrix2], dim=1)

            align_links[0].extend(hot_nodes1)
            align_links[1].extend(hot_nodes2)
            align_ranks.extend(hot_nodes1_rank)

        return align_links, align_ranks

    def run(self, graph1, graph2, alignment):
        """
        Executes the alignment process on two given graphs.

        The procedure is:
          1) Diffusion without anchors to obtain initial node features.
          2) Calculation of the required number of epochs based on alignment size.
          3) Iterative diffusion with anchors to refine alignments.
          4) Returns the final alignment links, ranks, and the elapsed time.

        Args:
            graph1: A graph structure with 'edge_index' and 'num_nodes'.
            graph2: Another graph structure with 'edge_index' and 'num_nodes'.
            alignment (list): A list of known aligned node pairs (used for epoch calculation).

        Returns:
            tuple:
                - align_links (list): Final alignment links [list1, list2].
                - align_ranks (list): Rank information for aligned nodes.
                - elapsed_time (float): The total execution time in seconds.
        """
        print("\nRunning......")
        print(f"{'=' * 50}")

        start_time = time()
        # Step 1: Obtain node features via diffusion without anchors
        print("Extracting node features on plain graph...")
        output_dim = min(graph1.num_nodes, graph2.num_nodes)

        x1 = self.diffusion_without_anchor(graph1.edge_index, output_dim)
        x2 = self.diffusion_without_anchor(graph2.edge_index, output_dim)

        # Step 2: Perform iterative diffusion alignment
        print("Diffusion aligning...")
        num_epoch = math.ceil((len(alignment) - self.num_dp_select) / self.num_diffusion_select + 1)
        align_links, align_ranks = self.diffusion_with_anchor(
            x1, x2,
            graph1.edge_index, graph2.edge_index,
            num_epoch=num_epoch
        )
        end_time = time()

        return align_links, align_ranks, end_time - start_time


def edge2adj(edge_index):
    """
    Converts an edge index to a PyTorch adjacency matrix (undirected).

    Args:
        edge_index (torch.Tensor): A 2D tensor of shape [2, E], where each column
                                   represents an edge between two nodes.

    Returns:
        torch.Tensor: A [N, N] adjacency matrix (float32), where
                      N = (max node index) + 1.
    """
    n = edge_index.max().item() + 1
    adj = torch.zeros(n, n, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    return adj


def l2_nestest(x1, x2, batch_size=5000, device='cuda'):
    """
    Computes pairwise L2 distances between two sets of vectors, returning
    distances and corresponding indices sorted for each row in ascending order.

    Args:
        x1 (torch.Tensor): Query set of size (n1, d).
        x2 (torch.Tensor): Target set of size (n2, d).
        batch_size (int): Batch size for processing x1 in chunks.
        device (str): 'cuda' or 'cpu', specifying the computation device.

    Returns:
        (torch.Tensor, torch.Tensor):
            - sorted_distances of shape (n1, n2): Distances sorted per row.
            - sorted_indices of shape (n1, n2): Indices of x2 that produce
              the sorted distances.
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    sorted_distances = torch.empty((n1, n2), dtype=torch.float32, device=device)
    sorted_indices = torch.empty((n1, n2), dtype=torch.long, device=device)

    for start_idx in range(0, n1, batch_size):
        end_idx = min(start_idx + batch_size, n1)
        x1_batch = x1[start_idx:end_idx]
        with torch.no_grad():
            dist_matrix = torch.cdist(x1_batch, x2, p=2)  # [batch_size, n2]
        dist_batch, ind_batch = torch.sort(dist_matrix, dim=1)
        sorted_distances[start_idx:end_idx] = dist_batch
        sorted_indices[start_idx:end_idx] = ind_batch

    return sorted_distances, sorted_indices


def draw_hot_graph_with_intensity(adj_matrix, hot_matrix, step, graph_name="Graph"):
    """
    Draws a graph using node color intensity based on the summed values of hot_matrix.

    Specifically, it:
      1) Sums hot_matrix across its columns for each node.
      2) Applies a sigmoid function to normalize values to [0,1].
      3) Uses a spring layout to plot the graph with node colors based on these intensities.

    Args:
        adj_matrix (torch.Tensor): Adjacency matrix of shape [N, N].
        hot_matrix (torch.Tensor): Matrix of node intensities, shape [N, F].
        step (int): Current diffusion step (for labeling in the figure title).
        graph_name (str): Name of the graph (used in the figure title and filename).
    """
    adj_np = adj_matrix.cpu().numpy()
    hot_values = hot_matrix.sum(dim=1).cpu().numpy()

    norm_hot_values = torch.tensor(hot_values, dtype=torch.float32, device='cpu')
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
