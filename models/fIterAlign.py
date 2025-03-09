import math
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from time import time


class fIterAlign:
    """
    fIterAlign model for graph node alignment.
    It contains three main procedures:
        1) Diffusion without anchors
        2) Diffusion with anchors
        3) Fast node pair selection
    """

    def __init__(self, config):
        """
        Initializes the fIterAlign model with the given configuration.

        Args:
            config (dict): A dictionary containing the following keys:
                - "device" (str or torch.device): The device to run computations on.
                - "dp_min_degree" (float): The minimum node degree threshold for the first iteration.
                - "num_dp_select" (int): Number of node pairs to select in the first iteration.
                - "num_diffusion_select" (int): Number of node pairs to select in subsequent iterations.
                - "diffusion_step" (int): Number of diffusion steps to perform in each epoch.
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
        Performs diffusion without anchor constraints.

        This method:
          1) Converts the edge index to an adjacency matrix.
          2) Applies a symmetric normalization to the adjacency matrix.
          3) Initializes a one-hot-like matrix (identity) for the nodes.
          4) Performs several steps of diffusion.
          5) Sorts the resulting feature vectors per node and retains the top 'output_dim' columns.

        Args:
            edge_index (torch.Tensor): Edge index of the graph with shape [2, E].
            output_dim (int): Number of top columns to keep after diffusion.

        Returns:
            torch.Tensor: A node feature matrix of shape [N, output_dim],
                          where N is the number of nodes in the graph.
        """
        adj = edge2adj(edge_index).to(self.device)
        # Symmetric normalization
        deg_inv_sqrt1 = torch.diag(torch.pow(adj.sum(dim=1), -0.5))
        adj_norm = deg_inv_sqrt1 @ adj @ deg_inv_sqrt1

        # Initialize a one-hot matrix
        hot_matrix = torch.zeros_like(adj, device=self.device)
        hot_matrix[range(adj.shape[0]), range(adj.shape[0])] = 1

        # Diffusion steps
        for t in range(self.num_step):
            hot_matrix = adj_norm @ hot_matrix + hot_matrix

        # Sort and keep top 'output_dim' features
        feature = torch.sort(hot_matrix, dim=1)[0]
        feature = feature[:, -output_dim:]

        return feature

    def _select_nodes(self, x1, x2, adj1, adj2, num_select, min_degree, align_links):
        """
        Selects node pairs by finding the nearest neighbor in x2 for each node in x1,
        then choosing the top 'num_select' pairs with the smallest distances.

        Steps:
          1) If align_links is None, filter nodes whose degrees are >= min_degree.
             Otherwise, mask out already aligned nodes.
          2) Compute all pairwise L2 distances between remaining nodes in x1 and x2.
          3) For each node in x1, select the closest node in x2 (distance[:,0]).
          4) Sort these pairwise distances and pick the top 'num_select' smallest ones.
          5) Return the final matched indices for both graphs, along with the rank list.

        Args:
            x1 (torch.Tensor): Feature matrix for graph 1 (shape [N1, F]).
            x2 (torch.Tensor): Feature matrix for graph 2 (shape [N2, F]).
            adj1 (torch.Tensor): Adjacency matrix for graph 1 (shape [N1, N1]).
            adj2 (torch.Tensor): Adjacency matrix for graph 2 (shape [N2, N2]).
            num_select (int): Number of node pairs to select.
            min_degree (float or None): If not None, applies a degree threshold to filter nodes.
            align_links (list or None): If not None, a list of already matched nodes
                                        [matched_in_g1, matched_in_g2].

        Returns:
            (list, list, list):
                - node1_indices: Indices of the selected nodes in graph 1.
                - node2_indices: Indices of the selected nodes in graph 2.
                - node1_rank: For each selected node in graph 1, a list of its
                              candidate matches in graph 2 sorted by ascending distance.
        """
        # Step 1: Filter nodes by min_degree (if first iteration) or mask aligned nodes
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

        # Step 2: Compute L2 distances
        dist, ind = l2_nestest(x1, x2, device=self.device)

        # Step 3: Take the closest (distance[:,0]) for each node in x1
        first_dist = dist[:, 0]
        first_ind = ind[:, 0]

        # Step 4: Sort by these closest distances and pick top 'num_select'
        selected_indices = torch.argsort(first_dist)[:num_select]
        best_ind = first_ind[selected_indices]

        # Step 5: Collect final node indices and ranks
        node1_indices = [select_nodes1[i].item() for i in selected_indices]
        node2_indices = [select_nodes2[i].item() for i in best_ind]
        node1_rank = select_nodes2[ind[selected_indices]].tolist()

        return node1_indices, node2_indices, node1_rank

    def diffusion_with_anchor(self, x1_deg, x2_deg, e1, e2, num_epoch):
        """
        Performs iterative diffusion-based alignment using anchors.

        Procedure per epoch:
          1) Select node pairs using _select_nodes().
          2) Convert selected nodes into one-hot matrices (hot_matrix1, hot_matrix2).
          3) Diffuse each hot_matrix for 'self.num_step' steps using adjacency normalization.
          4) Concatenate these diffusion results with existing features.
          5) Update alignment lists with newly selected pairs.

        Args:
            x1_deg (torch.Tensor): Feature matrix for graph 1, shape [N1, F1].
            x2_deg (torch.Tensor): Feature matrix for graph 2, shape [N2, F2].
            e1 (torch.Tensor): Edge index for graph 1, shape [2, E1].
            e2 (torch.Tensor): Edge index for graph 2, shape [2, E2].
            num_epoch (int): Number of iterations (epochs) to perform.

        Returns:
            (list, list):
                - align_links: Two lists of matched node indices [aligned_in_g1, aligned_in_g2].
                - align_ranks: List of nearest-neighbor ranks for the newly aligned nodes in graph 1.
        """
        x1_deg = x1_deg.to(self.device)
        x2_deg = x2_deg.to(self.device)

        num_nodes1 = x1_deg.size(0)
        num_nodes2 = x2_deg.size(0)

        # Initial empty feature sets for each graph
        x1 = torch.empty((num_nodes1, 0), dtype=torch.float32, device=self.device)
        x2 = torch.empty((num_nodes2, 0), dtype=torch.float32, device=self.device)

        # Build adjacency matrices
        adj1 = edge2adj(e1).to(self.device)
        adj2 = edge2adj(e2).to(self.device)

        # Symmetric normalization for adjacency
        deg_inv_sqrt1 = torch.diag(torch.pow(adj1.sum(dim=1), -0.5))
        adj_norm1 = deg_inv_sqrt1 @ adj1 @ deg_inv_sqrt1

        deg_inv_sqrt2 = torch.diag(torch.pow(adj2.sum(dim=1), -0.5))
        adj_norm2 = deg_inv_sqrt2 @ adj2 @ deg_inv_sqrt2

        align_links = [[], []]
        align_ranks = []

        for epoch in range(num_epoch):
            # Step 1: Select node pairs
            if epoch == 0:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1_deg, x2_deg, adj1, adj2, self.num_dp_select, self.dp_min_degree, None
                )
            else:
                hot_nodes1, hot_nodes2, hot_nodes1_rank = self._select_nodes(
                    x1, x2, adj1, adj2, self.num_diffusion_select, None, align_links
                )

            # Step 2: Build one-hot matrices
            hot_matrix1 = torch.zeros((num_nodes1, len(hot_nodes1)), dtype=torch.float32, device=self.device)
            hot_matrix2 = torch.zeros((num_nodes2, len(hot_nodes2)), dtype=torch.float32, device=self.device)
            hot_matrix1[hot_nodes1, range(len(hot_nodes1))] = 1
            hot_matrix2[hot_nodes2, range(len(hot_nodes2))] = 1

            # Step 3: Diffuse these matrices
            for step in range(self.num_step):
                hot_matrix1 = adj_norm1 @ hot_matrix1 + hot_matrix1
                hot_matrix2 = adj_norm2 @ hot_matrix2 + hot_matrix2

            # Step 4: Concatenate the diffused features
            x1 = torch.cat([x1, hot_matrix1], dim=1)
            x2 = torch.cat([x2, hot_matrix2], dim=1)

            # Step 5: Update alignment
            align_links[0].extend(hot_nodes1)
            align_links[1].extend(hot_nodes2)
            align_ranks.extend(hot_nodes1_rank)

        return align_links, align_ranks

    def run(self, graph1, graph2, alignment):
        """
        Runs the entire alignment pipeline on two graphs.

        Steps:
          1) Extract node features (via diffusion_without_anchor).
          2) Determine the required number of epochs based on the existing alignment.
          3) Perform iterative diffusion with anchors (diffusion_with_anchor).
          4) Return the discovered alignments, ranks, and the elapsed time.

        Args:
            graph1: An object with attributes 'edge_index' and 'num_nodes' for the first graph.
            graph2: An object with attributes 'edge_index' and 'num_nodes' for the second graph.
            alignment (list): A list of known aligned node pairs (used for epoch calculation).

        Returns:
            (list, list, float):
                - align_links: Final lists of matched nodes for graph 1 and graph 2.
                - align_ranks: Rank lists for the matched nodes.
                - elapsed_time: The total execution time in seconds.
        """
        print("\nRunning......")
        print(f"{'=' * 50}")

        start_time = time()
        # Step 1: Diffusion without anchors to get initial features
        print("Extracting node features on plain graph...")
        output_dim = min(graph1.num_nodes, graph2.num_nodes)

        x1 = self.diffusion_without_anchor(graph1.edge_index, output_dim)
        x2 = self.diffusion_without_anchor(graph2.edge_index, output_dim)

        # Step 2: Perform iterative diffusion alignment
        print("Diffusion aligning...")
        num_epoch = math.ceil((len(alignment) - self.num_dp_select) / self.num_diffusion_select + 1)
        align_links, align_ranks = self.diffusion_with_anchor(
            x1, x2, graph1.edge_index, graph2.edge_index, num_epoch=num_epoch
        )
        end_time = time()

        return align_links, align_ranks, end_time - start_time


def edge2adj(edge_index):
    """
    Converts a 2D edge index to an adjacency matrix.

    Args:
        edge_index (torch.Tensor): A 2D tensor with shape [2, E],
                                   where each column represents an edge.

    Returns:
        torch.Tensor: An NxN adjacency matrix (float), where N is (max node index + 1).
    """
    n = edge_index.max().item() + 1
    adj = torch.zeros(n, n, dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1

    return adj


def l2_nestest(x1, x2, batch_size=5000, device='cuda'):
    """
    Computes pairwise L2 distances between two sets of vectors in batches,
    then returns sorted distances and their corresponding indices.

    Args:
        x1 (torch.Tensor): A tensor of shape (n1, d), representing the query set.
        x2 (torch.Tensor): A tensor of shape (n2, d), representing the target set.
        batch_size (int): The batch size for processing x1 in chunks.
        device (str): The computation device ('cuda' or 'cpu').

    Returns:
        (torch.Tensor, torch.Tensor):
            - sorted_distances (n1, n2): L2 distances sorted in ascending order per row.
            - sorted_indices (n1, n2): Indices in x2 that yield the sorted distances for each row.
    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]

    sorted_distances = torch.empty((n1, n2), dtype=torch.float32, device=device)
    sorted_indices = torch.empty((n1, n2), dtype=torch.int64, device=device)

    for start_idx in range(0, n1, batch_size):
        end_idx = min(start_idx + batch_size, n1)
        x1_batch = x1[start_idx:end_idx]
        with torch.no_grad():
            dist_matrix = torch.cdist(x1_batch, x2, p=2)  # shape: [batch_size, n2]
        dist_batch, ind_batch = torch.sort(dist_matrix, dim=1)
        sorted_distances[start_idx:end_idx] = dist_batch
        sorted_indices[start_idx:end_idx] = ind_batch

    return sorted_distances, sorted_indices


def draw_hot_graph_with_intensity(adj_matrix, hot_matrix, step, graph_name="Graph"):
    """
    Draws a graph where nodes are colored based on the sum of intensities in hot_matrix.

    Steps:
      1) Compute hot_values by summing over columns of hot_matrix.
      2) Apply min-max normalization to hot_values.
      3) Use NetworkX to draw the graph with node colors corresponding to normalized intensities.

    Args:
        adj_matrix (torch.Tensor): NxN adjacency matrix of the graph.
        hot_matrix (torch.Tensor): NxF matrix containing diffusion intensities per node.
        step (int): The current diffusion step, used in the figure title.
        graph_name (str): A label for the graph, used in the plot title and filename.
    """
    adj_np = adj_matrix.cpu().numpy()
    hot_values = hot_matrix.sum(dim=1).cpu().numpy()

    # Min-max normalization
    if hot_values.max() == hot_values.min():
        norm_hot_values = hot_values / (hot_values.max() + 1e-9)
    else:
        norm_hot_values = (hot_values - hot_values.min()) / (hot_values.max() - hot_values.min())

    # Draw the graph
    g = nx.from_numpy_array(adj_np)
    pos = nx.spring_layout(g)
    plt.figure(figsize=(8, 8))
    cmap = plt.cm.get_cmap('coolwarm')
    node_collection = nx.draw_networkx_nodes(
        g, pos, node_color=norm_hot_values,
        cmap=cmap, node_size=5, vmin=0, vmax=1
    )

    plt.colorbar(node_collection, label='Node Intensity')
    plt.title(f"{graph_name} at Diffusion Step {step}")
    plt.savefig(f"image/hot_graph_{graph_name}_step_{step}.png")
