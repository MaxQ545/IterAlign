import networkx as nx
import numpy as np
import time
import math
from sklearn.neighbors import KDTree
import sklearn


class REGAL:
    def __init__(self, config):
        self.max_layer = config['max_layer']
        self.alpha = config['alpha']
        self.buckets = config['buckets']
        self.k = config['k']
        self.gammastruc = config['gammastruc']
        self.gammaattr = config['gammaattr']

    def run(self, G1, G2, alignment):
        print("\nRunning......")
        print(f"{'=' * 50}")

        nx_graph1 = pyg_to_nx(G1)
        nx_graph2 = pyg_to_nx(G2)
        offset = max(nx_graph1.nodes) + 1 if len(nx_graph1.nodes) > 0 else 0
        graph2_reindexed = nx.relabel_nodes(nx_graph2, lambda x: x + offset)
        nx_graph = nx.compose(nx_graph1, graph2_reindexed)
        print("read in graph")
        adj = nx.adjacency_matrix(nx_graph, nodelist=range(nx_graph.number_of_nodes()))
        print("got adj matrix")

        graph = Graph(adj)
        max_layer = self.max_layer
        if self.max_layer == 0:
            max_layer = None

        alpha = self.alpha
        num_buckets = self.buckets  # BASE OF LOG FOR LOG SCALE
        if num_buckets == 1:
            num_buckets = None

        start_time = time.time()
        rep_method = RepMethod(max_layer=max_layer,
                               alpha=alpha,
                               k=self.k,
                               num_buckets=num_buckets,
                               normalize=True,
                               gammastruc=self.gammastruc,
                               gammaattr=self.gammaattr)
        if max_layer is None:
            max_layer = 1000
        print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
        representations = get_representations(graph, rep_method)

        emb1, emb2 = get_embeddings(representations, offset)
        alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = None)

        align_links = [[], []]
        align_links[0] = list(range(alignment_matrix.shape[0]))
        align_links[1] = np.argmax(alignment_matrix, axis=1).tolist()

        align_ranks = np.argsort(-alignment_matrix, axis=1).tolist()

        end_time = time.time()

        # ------------------------ 新增的可信锚点排序逻辑 ------------------------ #
        link_sims = []
        for i in range(alignment_matrix.shape[0]):
            matched_j = align_links[1][i]
            sim_ij = alignment_matrix[i, matched_j]
            link_sims.append((i, sim_ij))

        # 按相似度从高到低排序
        link_sims.sort(key=lambda x: x[1], reverse=True)

        # 重新构建 align_links 和 align_ranks
        sorted_align_links = [[], []]
        sorted_align_ranks = []
        for idx, sim_val in link_sims:
            sorted_align_links[0].append(align_links[0][idx])
            sorted_align_links[1].append(align_links[1][idx])
            sorted_align_ranks.append(align_ranks[idx])

        align_links = sorted_align_links
        align_ranks = sorted_align_ranks
        # ------------------------ 新增代码结束 ------------------------ #

        return align_links, align_ranks, end_time - start_time

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



class Graph():
	#Undirected, unweighted
	def __init__(self,
				adj,
				num_buckets=None,
				node_labels = None,
				edge_labels = None,
				graph_label = None,
				node_attributes = None,
				true_alignments = None):
		self.G_adj = adj #adjacency matrix
		self.N = self.G_adj.shape[0] #number of nodes
		self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
		self.max_degree = max(self.node_degrees)
		self.num_buckets = num_buckets #how many buckets to break node features into

		self.node_labels = node_labels
		self.edge_labels = edge_labels
		self.graph_label = graph_label
		self.node_attributes = node_attributes #N x A matrix, where N is # of nodes, and A is # of attributes
		self.kneighbors = None #dict of k-hop neighbors for each node
		self.true_alignments = true_alignments #dict of true alignments, if this graph is a combination of multiple graphs


class RepMethod():
	def __init__(self,
				align_info = None,
				p=None,
				k=10,
				max_layer=None,
				alpha = 0.1,
				num_buckets = None,
				normalize = True,
				gammastruc = 1,
				gammaattr = 1):
		self.p = p #sample p points
		self.k = k #control sample size
		self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
		self.alpha = alpha #discount factor for higher layers
		self.num_buckets = num_buckets #number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
		self.normalize = normalize #whether to normalize node embeddings
		self.gammastruc = gammastruc #parameter weighing structural similarity in node identity
		self.gammaattr = gammaattr #parameter weighing attribute similarity in node identity


# xNetMF pipeline
def get_representations(graph, rep_method, verbose=True):
    # Node identity extraction
    feature_matrix = get_features(graph, rep_method, verbose)

    # Efficient similarity-based representation
    # Get landmark nodes
    if rep_method.p is None:
        rep_method.p = get_feature_dimensionality(graph, rep_method, verbose=verbose)  # k*log(n), where k = 10
    elif rep_method.p > graph.N:
        print("Warning: dimensionality greater than number of nodes. Reducing to n")
        rep_method.p = graph.N
    landmarks = get_sample_nodes(graph, rep_method, verbose=verbose)

    # Explicitly compute similarities of all nodes to these landmarks
    before_computesim = time.time()
    C = np.zeros((graph.N, rep_method.p))
    for node_index in range(graph.N):  # for each of N nodes
        for landmark_index in range(rep_method.p):  # for each of p landmarks
            # select the p-th landmark
            C[node_index, landmark_index] = compute_similarity(graph,
                                                               rep_method,
                                                               feature_matrix[node_index],
                                                               feature_matrix[landmarks[landmark_index]],
                                                               graph.node_attributes,
                                                               (node_index, landmarks[landmark_index]))

    before_computerep = time.time()

    # Compute Nystrom-based node embeddings
    W_pinv = np.linalg.pinv(C[landmarks])
    U, X, V = np.linalg.svd(W_pinv)
    Wfac = np.dot(U, np.diag(np.sqrt(X)))
    reprsn = np.dot(C, Wfac)
    after_computerep = time.time()
    if verbose:
        print("computed representation in time: ", after_computerep - before_computerep)

    # Post-processing step to normalize embeddings (true by default, for use with REGAL)
    if rep_method.normalize:
        reprsn = reprsn / np.linalg.norm(reprsn, axis=1).reshape((reprsn.shape[0], 1))
    return reprsn


def get_khop_neighbors(graph, rep_method):
    if rep_method.max_layer is None:
        rep_method.max_layer = graph.N  # Don't need this line, just sanity prevent infinite loop

    kneighbors_dict = {}

    # only 0-hop neighbor of a node is itself
    # neighbors of a node have nonzero connections to it in adj matrix
    for node in range(graph.N):
        neighbors = np.nonzero(graph.G_adj[:, [node]])[0].tolist()  ###
        # print(graph.G_adj[:, [node]])
        # print(np.nonzero(graph.G_adj[:, [node]]))
        # print(neighbors)
        if len(neighbors) == 0:  # disconnected node
            print("Warning: node %d is disconnected" % node)
            kneighbors_dict[node] = {0: set([node]), 1: set()}
        else:
            if type(neighbors[0]) is list:
                neighbors = neighbors[0]
            kneighbors_dict[node] = {0: set([node]), 1: set(neighbors) - set([node])}

        # For each node, keep track of neighbors we've already seen
    all_neighbors = {}
    for node in range(graph.N):
        all_neighbors[node] = set([node])
        all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])

    # Recursively compute neighbors in k
    # Neighbors of k-1 hop neighbors, unless we've already seen them before
    current_layer = 2  # need to at least consider neighbors
    while True:
        if rep_method.max_layer is not None and current_layer > rep_method.max_layer: break
        reached_max_layer = True  # whether we've reached the graph diameter

        for i in range(graph.N):
            # All neighbors k-1 hops away
            neighbors_prevhop = kneighbors_dict[i][current_layer - 1]

            khop_neighbors = set()
            # Add neighbors of each k-1 hop neighbors
            for n in neighbors_prevhop:
                neighbors_of_n = kneighbors_dict[n][1]
                for neighbor2nd in neighbors_of_n:
                    khop_neighbors.add(neighbor2nd)

            # Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
            khop_neighbors = khop_neighbors - all_neighbors[i]

            # Add neighbors at this hop to set of nodes we've already seen
            num_nodes_seen_before = len(all_neighbors[i])
            all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
            num_nodes_seen_after = len(all_neighbors[i])

            # See if we've added any more neighbors
            # If so, we may not have reached the max layer: we have to see if these nodes have neighbors
            if len(khop_neighbors) > 0:
                reached_max_layer = False

            # add neighbors
            kneighbors_dict[i][current_layer] = khop_neighbors  # k-hop neighbors must be at least k hops away

        if reached_max_layer:
            break  # finished finding neighborhoods (to the depth that we want)
        else:
            current_layer += 1  # move out to next layer

    return kneighbors_dict


# Turn lists of neighbors into a degree sequence
# Input: graph, RepMethod, node's neighbors at a given layer, the node
# Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence(graph, rep_method, kneighbors, current_node):
    if rep_method.num_buckets is not None:
        degree_counts = [0] * int(math.log(graph.max_degree, rep_method.num_buckets) + 1)
    else:
        degree_counts = [0] * (graph.max_degree + 1)

    # For each node in k-hop neighbors, count its degree
    for kn in kneighbors:
        weight = 1  # unweighted graphs supported here
        degree = graph.node_degrees[kn]
        if rep_method.num_buckets is not None:
            try:
                degree_counts[int(math.log(degree, rep_method.num_buckets))] += weight
            except:
                print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
        else:
            degree_counts[degree] += weight
    return degree_counts


# Get structural features for nodes in a graph based on degree sequences of neighbors
# Input: graph, RepMethod
# Output: nxD feature matrix
def get_features(graph, rep_method, verbose=True):
    before_khop = time.time()
    # Get k-hop neighbors of all nodes
    khop_neighbors_nobfs = get_khop_neighbors(graph, rep_method)

    graph.khop_neighbors = khop_neighbors_nobfs

    if verbose:
        print("max degree: ", graph.max_degree)
        after_khop = time.time()
        print("got k hop neighbors in time: ", after_khop - before_khop)

    G_adj = graph.G_adj
    num_nodes = G_adj.shape[0]
    if rep_method.num_buckets is None:  # 1 bin for every possible degree value
        num_features = graph.max_degree + 1  # count from 0 to max degree...could change if bucketizing degree sequences
    else:  # logarithmic binning with num_buckets as the base of logarithm (default: base 2)
        num_features = int(math.log(graph.max_degree, rep_method.num_buckets)) + 1
    feature_matrix = np.zeros((num_nodes, num_features))

    before_degseqs = time.time()
    for n in range(num_nodes):
        for layer in graph.khop_neighbors[n].keys():  # construct feature matrix one layer at a time
            if len(graph.khop_neighbors[n][layer]) > 0:
                # degree sequence of node n at layer "layer"
                deg_seq = get_degree_sequence(graph, rep_method, graph.khop_neighbors[n][layer], n)
                # add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                feature_matrix[n] += [(rep_method.alpha ** layer) * x for x in deg_seq]
    after_degseqs = time.time()

    if verbose:
        print("got degree sequences in time: ", after_degseqs - before_degseqs)

    return feature_matrix


# Input: two vectors of the same length
# Optional: tuple of (same length) vectors of node attributes for corresponding nodes
# Output: number between 0 and 1 representing their similarity
def compute_similarity(graph, rep_method, vec1, vec2, node_attributes=None, node_indices=None):
    dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2)  # compare distances between structural identities
    if graph.node_attributes is not None:
        # distance is number of disagreeing attributes
        attr_dist = np.sum(graph.node_attributes[node_indices[0]] != graph.node_attributes[node_indices[1]])
        dist += rep_method.gammaattr * attr_dist
    return np.exp(-dist)  # convert distances (weighted by coefficients on structure and attributes) to similarities


# Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
# Input: graph (just need graph size here), RepMethod (just need dimensionality here)
# Output: np array of node IDs
def get_sample_nodes(graph, rep_method, verbose=True):
    # Sample uniformly at random
    sample = np.random.RandomState(seed=42).permutation((np.arange(graph.N)))[:rep_method.p]
    return sample


# Get dimensionality of learned representations
# Related to rank of similarity matrix approximations
# Input: graph, RepMethod
# Output: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
def get_feature_dimensionality(graph, rep_method, verbose=True):
    p = int(rep_method.k * math.log(graph.N, 2))  # k*log(n) -- user can set k, default 10
    if verbose:
        print("feature dimensionality is ", min(p, graph.N))
    rep_method.p = min(p, graph.N)  # don't return larger dimensionality than # of nodes
    return rep_method.p


#Split embeddings in half
#Right now asssume graphs are same size (as done in paper's experiments)
#NOTE: to handle graphs of different sizes, pass in an arbitrary split index
#Similarly, to embed >2 graphs, change to pass in a list of splits and return list of embeddings
def get_embeddings(combined_embed, graph_split_idx = None):
	if graph_split_idx is None:
		graph_split_idx = int(combined_embed.shape[0] / 2)
	dim = combined_embed.shape[1]
	embed1 = combined_embed[:graph_split_idx]
	embed2 = combined_embed[graph_split_idx:]

	return embed1, embed2

def get_embedding_similarities(embed, embed2 = None, sim_measure = "euclidean", num_top = None):
	n_nodes, dim = embed.shape
	if embed2 is None:
		embed2 = embed

	if num_top is not None: #KD tree with only top similarities computed
		kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, num_top = num_top)
		return kd_sim

	#All pairwise distance computation
	if sim_measure == "cosine":
		similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
	else:
		similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
		similarity_matrix = np.exp(-similarity_matrix)

	return similarity_matrix


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()
