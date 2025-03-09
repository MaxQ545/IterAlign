import numpy as np
import networkx as nx
import random
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import deque
from time import time

class CENA:
    def __init__(self, config):
        self.alpha = config['alpha']
        self.layer = config['layer']
        self.q = config['q']
        self.c = config['c']
        self.multi_walk = config['multi_walk']
        self.align_train_prop = config['align_train_prop']

    def run(self, G1, G2, alignment):
        random.seed(22)
        np.random.seed(22)

        print("\nRunning......")
        print(f"{'=' * 50}")

        alignment_dict = {int(x): alignment[x] for x in alignment}
        attribute = []

        G1 = pyg_to_nx(G1)
        G2 = pyg_to_nx(G2)

        start_time = time()
        mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))

        if self.align_train_prop == 0:
            seed_list1 = []
            seed_list2 = []
        else:
            anchor_count = int(self.align_train_prop * len(alignment))
            anchors = list(alignment_dict.keys())[:anchor_count]
            seed_list1 = list(alignment_dict.keys())[:anchor_count]

            # seed_list1 = list(np.random.choice(list(alignment_dict.keys()), int(self.align_train_prop * len(alignment_dict)),
            #                                    replace=False))
            seed_list2 = [alignment_dict[seed_list1[x]] for x in range(len(seed_list1))]
        k = np.inf
        seed_list_num = len(seed_list1)

        index = list(G1.nodes())
        columns = list(G2.nodes())

        index = list(set(index) - set(seed_list1))
        columns = list(set(columns) - set(seed_list2))

        columns = [x + mul + 1 for x in columns]

        if k != 0:
            print('structing...', end='')
            G1_degree_dict = cal_degree_dict(list(G1.nodes()), G1, self.layer)
            G2_degree_dict = cal_degree_dict(list(G2.nodes()), G2, self.layer)
            struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2 = \
                structing(self.layer, G1, G2, G1_degree_dict, G2_degree_dict, attribute, self.alpha, self.c)
            print('finished!')
        print('walking...', end='')
        if self.multi_walk == True:
            multi_simulate_walks(G1, G2, self.q, struc_neighbor1, struc_neighbor2,
                                 struc_neighbor_sim1, struc_neighbor_sim2,
                                 seed_list1, seed_list2,
                                 num_walks=20, walk_length=80, workers=20)
        else:
            single_simulate_walks(G1, G2, self.q, struc_neighbor1, struc_neighbor2,
                                  struc_neighbor_sim1, struc_neighbor_sim2,
                                  seed_list1, seed_list2,
                                  num_walks=20, walk_length=80, workers=20)
        walks = LineSentence('random_walks.txt')
        print('finished!')
        print('embedding...', end='')
        # in the old version, it's:
        # model = Word2Vec(walks, size=64, window=5, min_count=0, hs=1, sg=1, workers=32, iter=5)
        # in the new version, it's:
        model = Word2Vec(walks, vector_size=64, window=5, min_count=0, hs=1, sg=1, negative=10, workers=32,
                         epochs=5)
        print('finished!')
        columns = [x - mul - 1 for x in columns]

        embedding1 = np.array([model.wv[str(x)] for x in index])
        embedding2 = np.array([model.wv[str(x + mul + 1)] for x in columns])

        cos = cosine_similarity(embedding1, embedding2)
        adj_matrix = np.zeros((len(index) * len(columns), 3))
        for i in range(len(index)):
            for j in range(len(columns)):
                adj_matrix[i * len(columns) + j, 0] = index[i]
                adj_matrix[i * len(columns) + j, 1] = columns[j]
                adj_matrix[i * len(columns) + j, 2] = cos[i, j]
        adj_matrix[:, 2] = list(map(clip, adj_matrix[:, 2]))
        if len(seed_list1) != 0:
            adj_matrix2 = caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns)
            adj_matrix[:, 2] *= adj_matrix2[:, 2]

        adj_matrix = adj_matrix[np.argsort(-adj_matrix[:, 2])]

        seed1 = []
        seed2 = []
        len_adj_matrix = len(adj_matrix)
        if len_adj_matrix != 0:
            T = len(alignment_dict)
            while len(adj_matrix) > 0 and T > 0:
                T -= 1
                node1, node2 = int(adj_matrix[0, 0]), int(adj_matrix[0, 1])
                seed1.append(node1)
                seed2.append(node2)
                adj_matrix = adj_matrix[adj_matrix[:, 0] != node1, :]
                adj_matrix = adj_matrix[adj_matrix[:, 1] != node2, :]

        seed_list1 += seed1
        seed_list2 += seed2
        print('Add seed nodes : {}'.format(len(seed1)), end='\t')

        count = 0
        for i in range(len(seed_list1)):
            try:
                if alignment_dict[seed_list1[i]] == seed_list2[i]:
                    count += 1
            except:
                continue

        print('All seed accuracy : %.2f%%' % (100 * count / len(seed_list1)))
        count -= seed_list_num
        precision = 100 * count / (len(seed_list1) - seed_list_num)
        recall = 100 * count / (len(alignment_dict) - seed_list_num)
        print('Precision : %.2f%%\tRecall :  %.2f%%' % (precision, recall))

        seed_list1 = seed_list1[seed_list_num:]
        seed_list2 = seed_list2[seed_list_num:]
        embedding1 = np.array([model.wv[str(x)] for x in seed_list1])
        embedding2 = np.array([model.wv[str(x + mul + 1)] for x in seed_list2])
        S = cosine_similarity(embedding1, embedding2)
        rank_matrix = []

        for j in range(len(seed_list1)):
            sim_vec = S[j, :]
            sorted_idx = np.argsort(-sim_vec)
            sorted_g2_nodes = [seed_list2[idx] for idx in sorted_idx]
            rank_matrix.append(sorted_g2_nodes)

        idx_dict = {seed_list1[x]: x for x in range(len(seed_list1))}
        rank_matrix = [rank_matrix[idx_dict[idx]] for idx in seed_list1]

        end_time = time()

        align_links = [seed_list1, seed_list2]

        ##############################################################################
        # 新增的排序代码：根据种子对 (seed_list1[i], seed_list2[i]) 的最终相似度 S[i,i] 排序
        ##############################################################################
        pairs = []
        for i in range(len(seed_list1)):
            # 对角线 S[i, i] 即第 i 个匹配对 (seed_list1[i], seed_list2[i]) 的相似度
            sim = S[i, i]
            # 将该匹配对及其对应的 rank_matrix[i] 一起存储
            pairs.append((seed_list1[i], seed_list2[i], sim, rank_matrix[i]))

        # 按相似度从大到小排序
        pairs.sort(key=lambda x: -x[2])

        # 排序后重新解包回 seed_list1, seed_list2, rank_matrix
        seed_list1 = [p[0] for p in pairs]
        seed_list2 = [p[1] for p in pairs]
        rank_matrix = [p[3] for p in pairs]

        # 如果需要的话，align_links 也更新成排序后的版本
        align_links = [seed_list1, seed_list2]
        ##############################################################################

        return align_links, rank_matrix, end_time - start_time


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



def seed_link(seed_list1, seed_list2, G1, G2, anchor):
    k = 0
    for i in range(len(seed_list1) - 1):
        for j in range(np.max([anchor + 1, i + 1]), len(seed_list1)):
            if G1.has_edge(seed_list1[i], seed_list1[j]) and not G2.has_edge(seed_list2[i], seed_list2[j]):
                G2.add_edges_from([[seed_list2[i], seed_list2[j]]])
                k += 1
            if not G1.has_edge(seed_list1[i], seed_list1[j]) and G2.has_edge(seed_list2[i], seed_list2[j]):
                G1.add_edges_from([[seed_list1[i], seed_list1[j]]])
                k += 1
    print('Add seed links : {}'.format(k), end = '\t')
    return k


def structing(layers, G1, G2, G1_degree_dict, G2_degree_dict, attribute, alpha, c):
    G1_nodes = list(G1.nodes())
    G2_nodes = list(G2.nodes())

    k1 = k2 = 1
    pp_dist_matrix = {}
    pp_dist_df = pd.DataFrame(np.zeros((G1.number_of_nodes(), G2.number_of_nodes())),
                              index=G1_nodes, columns=G2_nodes)

    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.max(G1_degree_dict[layer][x]) + np.e) for x in G1_nodes]
        L2 = [np.log(k2 * np.max(G2_degree_dict[layer][x]) + np.e) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    for layer in range(layers + 1):
        L1 = [np.log(k1 * np.min(G1_degree_dict[layer][x]) + 1) for x in G1_nodes]
        L2 = [np.log(k2 * np.min(G2_degree_dict[layer][x]) + 1) for x in G2_nodes]
        pp_dist_matrix[layer, 0] = pd.DataFrame(
            np.transpose(np.array(L1 * G2.number_of_nodes()).reshape(-1, G1.number_of_nodes())),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_matrix[layer, 1] = pd.DataFrame(
            np.array(list(L2 * G1.number_of_nodes())).reshape(-1, G2.number_of_nodes()),
            index=G1_nodes, columns=G2_nodes)
        pp_dist_df += abs(pp_dist_matrix[layer, 0] - pp_dist_matrix[layer, 1])
    pp_dist_df /= 2
    pp_dist_df = np.exp(-alpha * pp_dist_df)
    if len(attribute) != 0:
        pp_dist_df = c * pp_dist_df + np.array(attribute) * (1 - c)
    struc_neighbor1 = {}
    struc_neighbor2 = {}
    struc_neighbor_sim1 = {}
    struc_neighbor_sim2 = {}
    for i in range(G1.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor1[G1_nodes[i]] = list(pp.index[:10])
        struc_neighbor_sim1[G1_nodes[i]] = np.array(pp[:10])
        struc_neighbor_sim1[G1_nodes[i]] /= np.sum(struc_neighbor_sim1[G1_nodes[i]])
    pp_dist_df = pp_dist_df.transpose()
    for i in range(G2.number_of_nodes()):
        pp = pp_dist_df.iloc[i, np.argsort(-pp_dist_df.iloc[i, :])]
        struc_neighbor2[G2_nodes[i]] = list(pp.index[:10])
        struc_neighbor_sim2[G2_nodes[i]] = np.array(pp[:10])
        struc_neighbor_sim2[G2_nodes[i]] /= np.sum(struc_neighbor_sim2[G2_nodes[i]])
    return struc_neighbor1, struc_neighbor2, struc_neighbor_sim1, struc_neighbor_sim2


def multi_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2,
                         struc_neighbor_sim1, struc_neighbor_sim2,
                         seed_list1=[], seed_list2=[],
                         num_walks=20, walk_length=80, workers=20):
    walks = deque()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        G1_list = list(G1.nodes())
        G2_list = list(G2.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(G1_list)
            random.shuffle(G2_list)
            job = executor.submit(simulate_walks, G1, G2, q, struc_neighbor1, struc_neighbor2,
                                  struc_neighbor_sim1, struc_neighbor_sim2,
                                  G1_list, G2_list, seed_list1=[], seed_list2=[],
                                  walk_length=80)
            futures[job] = walk_iter
            # part += 1

        for job in as_completed(futures):
            walk = job.result()
            walks.extend(walk)
        del futures
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v) + ' '
            line += '\n'
            file.write(line)


def single_simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2,
                          struc_neighbor_sim1, struc_neighbor_sim2,
                          seed_list1=[], seed_list2=[],
                          num_walks=20, walk_length=80, workers=20):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    walks = []
    for i in tqdm(range(num_walks)):
        for node in list(G1.nodes()):

            walk = []
            walk.append(node)
            curr_graph = 1

            for j in range(walk_length - 1):
                r = np.random.random()
                if r < q:
                    if curr_graph == 1:
                        list_Gneighbors = list(G1.neighbors(node))
                        node = np.random.choice(list_Gneighbors)
                    elif curr_graph == 2:
                        list_Gneighbors = list(G2.neighbors(node - mul - 1))
                        node = np.random.choice(list_Gneighbors) + mul + 1
                else:
                    if curr_graph == 1:
                        try:
                            node = seed_list2[seed_list1.index(node)] + mul + 1
                        except:
                            node = random.choices(population=struc_neighbor1[node], weights=struc_neighbor_sim1[node])[
                                       0] + mul + 1
                        curr_graph = 2
                    else:
                        try:
                            node = seed_list1[seed_list2.index(node - mul - 1)]
                        except:
                            node = random.choices(population=struc_neighbor2[node - mul - 1],
                                                  weights=struc_neighbor_sim2[node - mul - 1])[0]
                        curr_graph = 1
                walk.append(node)
            walks.append(walk)
        for node in list(G2.nodes()):
            node += mul + 1
            walk = []
            walk.append(node)
            curr_graph = 2

            for j in range(walk_length - 1):
                r = np.random.random()
                if r < q:
                    if curr_graph == 1:
                        list_Gneighbors = list(G1.neighbors(node))
                        node = np.random.choice(list_Gneighbors)
                    elif curr_graph == 2:
                        list_Gneighbors = list(G2.neighbors(node - mul - 1))
                        node = np.random.choice(list_Gneighbors) + mul + 1
                else:
                    if curr_graph == 1:
                        try:
                            node = seed_list2[seed_list1.index(node)] + mul + 1
                        except:
                            node = random.choices(population=struc_neighbor1[node], weights=struc_neighbor_sim1[node])[
                                       0] + mul + 1
                        curr_graph = 2
                    else:
                        try:
                            node = seed_list1[seed_list2.index(node - mul - 1)]
                        except:
                            node = random.choices(population=struc_neighbor2[node - mul - 1],
                                                  weights=struc_neighbor_sim2[node - mul - 1])[0]
                        curr_graph = 1
                walk.append(node)
            walks.append(walk)
    with open('random_walks.txt', 'w') as file:
        for walk in walks:
            line = ''
            for v in walk:
                line += str(v) + ' '
            line += '\n'
            file.write(line)

def clip(x):
    if x <= 0:
        return 0
    else:
        return x


def caculate_jaccard_coefficient(G1, G2, seed_list1, seed_list2, index, columns, alignment_dict=None):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    seed1_dict = {}
    seed1_dict_reversed = {}
    seed2_dict = {}
    seed2_dict_reversed = {}
    for i in range(len(seed_list1)):
        seed1_dict[i + 2 * (mul + 1)] = seed_list1[i]
        seed1_dict_reversed[seed_list1[i]] = i + 2 * (mul + 1)
        seed2_dict[i + 2 * (mul + 1)] = seed_list2[i] + mul + 1
        seed2_dict_reversed[seed_list2[i] + mul + 1] = i + 2 * (mul + 1)
    G1_edges = pd.DataFrame(G1.edges())
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 1] = G1_edges.iloc[:, 1].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G1_edges.iloc[:, 0] = G1_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed1_dict_reversed))
    G2_edges = pd.DataFrame(G2.edges())
    G2_edges += mul + 1
    G2_edges.iloc[:, 0] = G2_edges.iloc[:, 0].apply(lambda x: to_seed(x, seed2_dict_reversed))
    G2_edges.iloc[:, 1] = G2_edges.iloc[:, 1].apply(lambda x: to_seed(x, seed2_dict_reversed))
    adj = nx.Graph()
    adj.add_edges_from(np.array(G1_edges))
    adj.add_edges_from(np.array(G2_edges))
    jaccard_dict = {}
    for G1_node in index:
        for G2_node in columns:
            if (G1_node, G2_node) not in jaccard_dict.keys():
                jaccard_dict[G1_node, G2_node] = 0
            jaccard_dict[G1_node, G2_node] += calculate_adj(adj.neighbors(G1_node), adj.neighbors(G2_node + mul + 1))

    jaccard_dict = [[x[0][0], x[0][1], x[1]] for x in jaccard_dict.items()]
    adj_matrix = np.array(jaccard_dict)
    return adj_matrix

def to_seed(x, dictionary):
    try:
        return dictionary[x]
    except:
        return x


def calculate_adj(setA, setB):
    setA = set(setA)
    setB = set(setB)
    ep = 0.5
    inter = len(setA & setB) + ep
    union = len(setA | setB) + ep

    adj = inter / union
    return adj


def simulate_walks(G1, G2, q, struc_neighbor1, struc_neighbor2,
                   struc_neighbor_sim1, struc_neighbor_sim2,
                   G1_list, G2_list, seed_list1=[], seed_list2=[],
                   walk_length=80):
    walks = deque()
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    for node in G1_list:
        walks.append(simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2,
                                   struc_neighbor_sim1, struc_neighbor_sim2,
                                   seed_list1=[], seed_list2=[],
                                   walk_length=80, node=node, curr_graph=1))
    for node in G2_list:
        node += mul + 1
        walks.append(simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2,
                                   struc_neighbor_sim1, struc_neighbor_sim2,
                                   seed_list1=[], seed_list2=[],
                                   walk_length=80, node=node, curr_graph=2))
    return walks

def simulate_walk(G1, G2, q, struc_neighbor1, struc_neighbor2,
                         struc_neighbor_sim1, struc_neighbor_sim2,
                         seed_list1 = [], seed_list2 = [],
                         walk_length = 80, node = 0, curr_graph = 1
                  ):
    mul = int(np.max([np.max(G1.nodes()), np.max(G2.nodes())]))
    walk = deque()
    walk.append(node)
    for i in range(walk_length - 1):
        r = np.random.random()
        if r < q:
            if curr_graph == 1:
                list_Gneighbors = list(G1.neighbors(node))
                node = np.random.choice(list_Gneighbors)
            elif curr_graph == 2:
                list_Gneighbors = list(G2.neighbors(node - mul - 1))
                node = np.random.choice(list_Gneighbors) + mul + 1
        else:
            if curr_graph == 1:
                try:
                    node = seed_list2[seed_list1.index(node)] + mul + 1
                except:
                    node = random.choices(population=struc_neighbor1[node], weights = struc_neighbor_sim1[node])[0] + mul + 1
                curr_graph = 2
            else:
                try:
                    node = seed_list1[seed_list2.index(node - mul - 1)]
                except:
                    node = random.choices(population=struc_neighbor2[node - mul - 1], weights = struc_neighbor_sim2[node - mul - 1])[0]
                curr_graph = 1
        walk.append(node)
    return walk


def edge_sample(G):
    edges = list(G.edges())
    test_edges_false = []
    while len(test_edges_false) < G.number_of_edges():
        node1 = np.random.choice(G.nodes())
        node2 = np.random.choice(G.nodes())
        if node1 == node2:
            continue
        if G.has_edge(node1, node2):
            continue
        test_edges_false.append([min(node1, node2), max(node1, node2)])
    edges = edges + test_edges_false
    return edges


def node_to_degree(G_degree, SET):
    SET = list(SET)
    SET = sorted([G_degree[x] for x in SET])
    return SET