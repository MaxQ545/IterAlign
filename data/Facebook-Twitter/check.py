import torch
import math
from collections import defaultdict
from tqdm import tqdm  # 导入 tqdm 以显示进度
import networkx as nx  # 使用 networkx 进行自同构群判别

def read_edgelist(file_path):
    """
    读取边列表文件，返回边的列表和节点总数。

    参数:
        file_path (str): 边列表文件的路径。

    返回:
        edges (list of tuples): 边的列表，每个边表示为 (u, v)。
        num_nodes (int): 节点的总数。
    """
    edges = []
    max_node = -1
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue  # 跳过空行
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            max_node = max(max_node, u, v)
    return edges, max_node + 1  # 假设节点编号从0开始

def build_adjacency_matrix(edges, num_nodes):
    """
    根据边列表构建邻接矩阵 (PyTorch)。

    参数:
        edges (list of tuples): 边的列表。
        num_nodes (int): 节点的总数。

    返回:
        adj (torch.Tensor): 邻接矩阵 (num_nodes x num_nodes)。
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1  # 无向图对称
    return adj

def graph_to_networkx(adj):
    """
    将邻接矩阵转换为 NetworkX 图对象。

    参数:
        adj (torch.Tensor): 邻接矩阵。

    返回:
        G (networkx.Graph): NetworkX 图对象。
    """
    G = nx.Graph()
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i][j] == 1:
                G.add_edge(i, j)
    return G

def group_equivalent_nodes_automorphism(adj):
    """
    使用自同构群算法来分组等价节点。

    参数:
        adj (torch.Tensor): 邻接矩阵。

    返回:
        groups (defaultdict): {color_value: [节点列表]}，即颜色分组（等价组）。
    """
    # Step 1: Convert adjacency matrix to a networkx graph
    G = graph_to_networkx(adj)

    # Step 2: Find automorphisms of the graph using networkx
    # Detecting graph automorphisms (node equivalence) using networkx's built-in function
    GM = nx.algorithms.isomorphism.GraphMatcher(G, G)
    automorphisms = list(GM.isomorphisms_iter())

    # Step 3: Group nodes based on automorphisms
    groups = defaultdict(list)
    for auto in automorphisms:
        for node, mapped_node in auto.items():
            groups[node].append(mapped_node)

    return groups

def combination(n, k):
    """
    计算组合数 C(n, k)。

    参数:
        n (int): 总数。
        k (int): 选择数。

    返回:
        int: 组合数。
    """
    if n < k:
        return 0
    return math.comb(n, k)  # Python 3.10 及以上支持

def count_equivalent_groups(groups):
    """
    计算每种大小的等价节点组数，仅统计每个等价类本身，排除其子组合。

    参数:
        groups (dict): {color_value: [节点列表]}。

    返回:
        counts (defaultdict): 每种大小 n -> 等价组数。
    """
    counts = defaultdict(int)
    for nodes in groups.values():
        size = len(nodes)
        if size >= 2:
            counts[size] += 1
    return counts

def calculate_matching_accuracy(groups, num_nodes):
    """
    计算节点匹配的最高正确率。

    假设：
    - 等价组包含 n 个节点，则组内每个节点的匹配正确率为 1/n。
    - 单节点组（n=1）的匹配正确率为 1。

    计算方法：
    - 对于每个等价组（大小 >=2），每个节点的匹配正确率为 1/n。
    - 对于单节点组，匹配正确率为 1。
    - 最终的匹配正确率为所有节点匹配正确率的平均值。

    参数:
        groups (dict): {color_value: [节点列表]}。
        num_nodes (int): 节点的总数。

    返回:
        accuracy (float): 匹配的最高正确率（百分比）。
    """
    total_accuracy = 0.0
    for nodes in groups.values():
        size = len(nodes)
        if size >= 2:
            # 每个节点的匹配正确率为 1/size
            total_accuracy += size * (1.0 / size)  # sum(1/size for _ in nodes)
        else:
            # 单节点组，匹配正确率为1
            total_accuracy += 1.0

    accuracy = (total_accuracy / num_nodes) * 100
    return accuracy

def main():
    """
    主函数，执行整个流程并输出结果。
    """
    file_path1 = "G1.edgelist"  # 替换为你的edgelist文件路径
    file_path2 = "G2.edgelist"
    edges1, num_nodes1 = read_edgelist(file_path1)
    edges2, num_nodes2 = read_edgelist(file_path2)

    adj1 = build_adjacency_matrix(edges1, num_nodes1)
    adj2 = build_adjacency_matrix(edges2, num_nodes2)

    # 1. 使用自同构群算法进行节点等价分组
    groups1 = group_equivalent_nodes_automorphism(adj1)
    groups2 = group_equivalent_nodes_automorphism(adj2)

    # 2. 统计每种大小的等价组数量（排除子组合）
    counts1 = count_equivalent_groups(groups1)
    counts2 = count_equivalent_groups(groups2)

    # 3. 输出：n个节点两两等价的有X组
    for n in sorted(counts1.keys()):
        print(f"G1 {n}个节点两两等价的有{counts1[n]}组")
    for n in sorted(counts2.keys()):
        print(f"G2 {n}个节点两两等价的有{counts2[n]}组")

    # 4. 计算并输出匹配正确率
    accuracy = calculate_matching_accuracy(groups1, num_nodes1)
    print(f"\nG1节点匹配的最高正确率为：{accuracy:.2f}%，等价类有{accuracy * num_nodes1 / 100:.2f}")
    accuracy = calculate_matching_accuracy(groups2, num_nodes2)
    print(f"\nG2节点匹配的最高正确率为：{accuracy:.2f}%，等价类有{accuracy * num_nodes2 / 100:.2f}")

if __name__ == "__main__":
    main()
