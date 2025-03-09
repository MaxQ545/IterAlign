import networkx as nx

# Load the edgelist from the uploaded file
file_path = 'G1.edgelist'
G = nx.read_edgelist(file_path, nodetype=int)

# Calculate the number of edges to remove (10% of the total)
rm_rate = 0.20
edges_to_remove = int(rm_rate * G.number_of_edges())

# Ensure the graph remains connected after removing edges
import random

def remove_edges_while_connected(graph, num_edges_to_remove):
    edges = list(graph.edges())
    removed_edges = []
    for _ in range(num_edges_to_remove):
        edge = random.choice(edges)
        graph.remove_edge(*edge)
        if nx.is_connected(graph):
            removed_edges.append(edge)
            edges.remove(edge)
        else:
            graph.add_edge(*edge)  # Restore edge if graph becomes disconnected
    return graph, removed_edges

# Create a copy of the graph to avoid modifying the original
G_copy = G.copy()
G_modified, removed_edges = remove_edges_while_connected(G_copy, edges_to_remove)

# Save the modified graph to a new file
output_file_path = f'G1_rm_{rm_rate:.2f}.edgelist'
nx.write_edgelist(G_modified, output_file_path, data=False)
