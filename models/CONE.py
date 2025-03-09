import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import sklearn
from sklearn.neighbors import KDTree
import collections
import io
import ot
from time import time

# This method require the numbers of nodes in two graph to be equal

class CONE:
    def __init__(self, config):
        self.embmethod = config['embmethod']
        self.dim = config['dim']
        self.window = config['window']
        self.negative = config['negative']
        self.embsim = config['embsim']
        self.niter_init = config['niter_init']
        self.reg_init = config['reg_init']
        self.nepoch = config['nepoch']
        self.lr = config['lr']
        self.bsz = config['bsz']
        self.alignmethod = config['alignmethod']
        self.numtop = config['numtop']
        self.embsim = config['embsim']
        self.niter_align = config['niter_align']
        self.reg_align = config['reg_align']
        

    def align_embeddings(self, embed1, embed2, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
        # Step 2: Align Embedding Spaces
        corr = None
        if struc_embed is not None and struc_embed2 is not None:
            if self.embsim == "cosine":
                corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
            else:
                corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
                corr = np.exp(-corr)

            # Take only top correspondences
            matches = np.zeros(corr.shape)
            matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
            corr = matches

        # Convex Initialization
        if adj1 is not None and adj2 is not None:
            if not sps.issparse(adj1): adj1 = sps.csr_matrix(adj1)
            if not sps.issparse(adj2): adj2 = sps.csr_matrix(adj2)
            init_sim, corr_mat = convex_init_sps(embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False,
                                                             niter=self.niter_init, reg=self.reg_init, P=corr)
        else:
            init_sim, corr_mat = convex_init(embed1, embed2, apply_sqrt=False, niter=self.niter_init,
                                                         reg=self.reg_init, P=corr)
        print(corr_mat)
        print(np.max(corr_mat, axis=0))
        print(np.max(corr_mat, axis=1))

        # Stochastic Alternating Optimization
        dim_align_matrix, corr_mat = align(embed1, embed2, init_sim, lr=self.lr, bsz=self.bsz,
                                                       nepoch=self.nepoch, niter=self.niter_align, reg=self.reg_align)
        print(dim_align_matrix.shape, corr_mat.shape)

        # Step 3: Match Nodes with Similar Embeddings
        # Align embedding spaces
        aligned_embed1 = embed1.dot(dim_align_matrix)
        # Greedily match nodes
        if self.alignmethod == 'greedy':  # greedily align each embedding to most similar neighbor
            # KD tree with only top similarities computed
            if self.numtop is not None:
                alignment_matrix = kd_align(aligned_embed1, embed2, distance_metric=self.embsim, num_top=self.numtop)
            # All pairwise distance computation
            else:
                if self.embsim == "cosine":
                    alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(aligned_embed1, embed2)
                else:
                    alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)
                    alignment_matrix = np.exp(-alignment_matrix)

        return alignment_matrix

    def run(self, G1, G2, alignment):

        print("\nRunning......")
        print(f"{'=' * 50}")

        true_align = {int(x):alignment[x] for x in alignment}  # 使用数字作为key值
        
        nx_graphA = pyg_to_nx(G1)
        nx_graphB = pyg_to_nx(G2)

        adjA = nx.adjacency_matrix(nx_graphA)
        adjB = nx.adjacency_matrix(nx_graphB)
        node_num = adjA.shape[0]

        start_time = time()
        # step1: obtain normalized proximity-preserving node embeddings
        if (self.embmethod == "netMF"):
            emb_matrixA = netmf(adjA, dim=self.dim, window=self.window, b=self.negative, normalize=True)
            emb_matrixB = netmf(adjB, dim=self.dim, window=self.window, b=self.negative, normalize=True)
            print("netMF done.")

        # step2 and 3: align embedding spaces and match nodes with similar embeddings
        alignment_matrix = self.align_embeddings(emb_matrixA, emb_matrixB, adj1=csr_matrix(adjA), adj2=csr_matrix(adjB),
                                            struc_embed=None, struc_embed2=None)

        align_list, align_rank = get_counterpart(alignment_matrix, true_align)
        end_time = time()

        # ------------------------ 以下为新增的可信锚点排序逻辑 ------------------------ #
        link_sims = []
        for i in range(alignment_matrix.shape[0]):
            matched_j = align_list[1][i]  # 与第 i 个节点匹配的节点 j
            # alignment_matrix[i, matched_j] 为该匹配对 (i, j) 的相似度
            sim_ij = alignment_matrix[i, matched_j]
            link_sims.append((i, sim_ij))

        # 按相似度从高到低排序
        link_sims.sort(key=lambda x: x[1], reverse=True)

        # 重新构建 align_list 和 align_rank
        sorted_align_list = [[], []]
        sorted_align_rank = []
        print(link_sims)
        for idx, sim_val in link_sims:
            sorted_align_list[0].append(align_list[0][idx])
            sorted_align_list[1].append(align_list[1][idx])
            sorted_align_rank.append(align_rank[idx])

        align_list = sorted_align_list
        align_rank = sorted_align_rank
        # ------------------------ 新增代码结束 ------------------------ #

        return align_list, align_rank, end_time - start_time


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
    M_dense = M.toarray().astype(np.float32)
    Y = np.log(np.maximum(M_dense, 1))

    return sps.csr_matrix(Y)

#Used in NetMF, AROPE
def svd_embed(prox_sim, dim):
    u, s, v = sps.linalg.svds(prox_sim, dim, return_singular_vectors="u")
    return sps.diags(np.sqrt(s)).dot(u.T).T

def netmf(A, dim = 128, window=10, b=1.0, normalize = True):
    prox_sim = netmf_mat_full(A, window, b)
    embed = svd_embed(prox_sim, dim)
    if normalize:
        norms = np.linalg.norm(embed, axis = 1).reshape((embed.shape[0], 1))
        norms[norms == 0] = 1
        embed = embed / norms
    return embed



def get_counterpart(alignment_matrix, true_alignments):
    n_nodes = alignment_matrix.shape[0]

    correct_nodes = []
    align_list = [[], []]
    align_rank = []

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None: #if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]

        rank = node_sorted_indices.tolist()
        rank = rank[-1: 0: -1]
        align_rank.append(rank)

        if target_alignment in node_sorted_indices[-1:]:
            correct_nodes.append(node_index)
        counterpart = node_sorted_indices[-1]

        align_list[0].append(node_index)
        align_list[1].append(counterpart)

    return align_list, align_rank


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    emb1 = np.asarray(emb1)
    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sps_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sps_align_matrix.tocsr()


def objective(X, Y, R, n=5):
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    C = np.array(C)
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n


def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


def align(X, Y, R, lr=1.0, bsz=10, nepoch=5, niter=10,
          nmax=10, reg=0.05, verbose=True, project_every = True):
    for epoch in range(1, nepoch + 1):
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            #print bsz, C.shape
            C = np.array(C)
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            #print P.shape, C.shape
            # compute gradient
            #print "random values from embeddings:", xt, yt
            #print "sinkhorn", np.isnan(P).any(), np.isinf(P).any()
            #Pyt = np.dot(P, yt)
            #print "Pyt", np.isnan(Pyt).any(), np.isinf(Pyt).any()
            G = - np.dot(xt.T, np.dot(P, yt))
            #print "G", np.isnan(G).any(), np.isinf(G).any()
            update = lr / bsz * G
            print(("Update: %.3f (norm G %.3f)" % (np.linalg.norm(update), np.linalg.norm(G))))
            R -= update

            # project on orthogonal matrices
            if project_every:
                U, s, VT = np.linalg.svd(R)
                R = np.dot(U, VT)
        niter //= 4
        if verbose:
            print(("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R))))
    if not project_every:
        U, s, VT = np.linalg.svd(R)
        R = np.dot(U, VT)
    return R, P


def convex_init(X, Y, niter=10, reg=1.0, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    print(obj)
    return procrustes(np.dot(P, X), Y).T, P

def convex_init_sps(X, Y, K_X = None, K_Y = None, niter=10, reg=1.0, apply_sqrt=False, P = None):
    if P is not None: #already given initial correspondence--then just procrustes
        return procrustes(P.dot(X), Y).T, P
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    if K_X is None:
        K_X = np.dot(X, X.T)
    if K_Y is None:
        K_Y = np.dot(Y, Y.T)
    K_Y = K_Y.astype(float)
    K_X = K_X.astype(float)
    K_Y *= sps.linalg.norm(K_X) / sps.linalg.norm(K_Y)
    K2_X, K2_Y = K_X.dot(K_X), K_Y.dot(K_Y)
    #print K_X, K_Y, K2_X, K2_Y
    K_X, K_Y, K2_X, K2_Y = K_X.toarray(), K_Y.toarray(), K2_X.toarray(), K2_Y.toarray()
    P = np.ones([n, n]) / float(n)


    for it in range(1, niter + 1):
        if it % 10 == 0: print(it)
        G = P.dot(K2_X) + K2_Y.dot(P) - 2 * K_Y.dot(P.dot(K_X))
        # G = G.todense() #TODO how to get around this??
        G = np.array(G)  # 将 G 转换为 ndarray
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        q = sps.csr_matrix(q)
        #print q.shape
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm( P.dot(K_X) - K_Y.dot(P) )
    print(obj)
    return procrustes(P.dot(X), Y).T, P


def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print(("Loading vectors from %s" % fname))
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = list(map(int, fin.readline().split()))
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print(("%d word vectors loaded" % (len(words))))
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write("%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(["%.4f" % a for a in x[i, :]]) + "\n")
    fout.close()


def save_matrix(fname, x):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write("%d %d\n" % (n, d))
    for i in range(n):
        fout.write(" ".join(["%.4f" % a for a in x[i, :]]) + "\n")
    fout.close()


def procrustes(X_src, Y_tgt):
    '''
    print "procrustes:", Y_tgt, X_src
    print np.isnan(Y_tgt).any(), np.isinf(Y_tgt).any()
    print np.isnan(X_src).any(), np.isinf(X_src).any()
    print np.min(Y_tgt), np.max(Y_tgt)
    print np.min(X_src), np.max(X_src)
    dot = np.dot(Y_tgt.T, X_src)
    print np.isnan(dot).any(), np.isinf(dot).any()
    print np.min(dot), np.max(dot)
    '''
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = np.zeros([n, d])
    y = np.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y


def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print(("Coverage of source vocab: %.4f" % (coverage)))
    return lexicon, float(len(vocab))


def load_pairs(filename, idx_src, idx_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    pairs = []
    tot = 0
    for line in f:
        a, b = line.rstrip().split(' ')
        tot += 1
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if verbose:
        coverage = (1.0 * len(pairs)) / tot
        print(("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage)))
    return pairs


def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        pred = scores.argmax(axis=0)
        for j in range(i, e):
            if pred[j - i] in lexicon[idx_src[j]]:
                acc += 1.0
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    sc = np.dot(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    similarities -= sc2[np.newaxis, :]

    nn = np.argmax(similarities, axis=1).tolist()
    correct = 0.0
    for k in range(0, len(lexicon)):
        if nn[k] in lexicon[idx_src[k]]:
            correct += 1.0
    return correct / lexicon_size
