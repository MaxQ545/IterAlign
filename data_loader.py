import torch
import json
from torch_geometric.data import Data


class DataLoader:
    """
    Handles dataset loading and conversion to PyTorch Geometric Data objects.
    """
    def __init__(self, dataset_path):
        """
        Initializes the DataLoader with the dataset path.

        Args:
            dataset_path (str): Path to the dataset folder.
        """
        self.dataset_path = dataset_path
        

    def load_data(self, name, rm_rate):
        """
        Loads the data for a specific dataset.

        Args:
            name (str): Name of the dataset to load.

        Returns:
            tuple: 
                - G1 (Data): PyTorch Geometric Data object for graph 1.
                - G2 (Data): PyTorch Geometric Data object for graph 2.
                - alignment (dict): Alignment links between graph 1 and graph 2 nodes.
        """
        # Load graph 1 edge list
        if rm_rate > 0:
            with open(f"{self.dataset_path}/{name}/G1_rm_{rm_rate:.2f}.edgelist", "r") as f:
                edge_g1 = [list(map(int, line.split())) for line in f.readlines()]
                edge_g1 = torch.tensor(edge_g1, dtype=torch.long).t().contiguous()
        else:
            with open(f"{self.dataset_path}/{name}/G1.edgelist", "r") as f:
                edge_g1 = [list(map(int, line.split())) for line in f.readlines()]
                edge_g1 = torch.tensor(edge_g1, dtype=torch.long).t().contiguous()


        # Calculate number of nodes in graph 1
        num_nodes_g1 = edge_g1.max().item() + 1

        # Load graph 2 edge list
        with open(f"{self.dataset_path}/{name}/G2.edgelist", "r") as f:
            edge_g2 = [list(map(int, line.split())) for line in f.readlines()]
            edge_g2 = torch.tensor(edge_g2, dtype=torch.long).t().contiguous()
        
        # Calculate number of nodes in graph 2
        num_nodes_g2 = edge_g2.max().item() + 1

        # Load alignment links
        with open(f"{self.dataset_path}/{name}/Alignment.json", "r") as f:
            alignment = json.load(f)

        # Create PyTorch Geometric Data objects
        G1 = Data(edge_index=edge_g1, num_nodes=num_nodes_g1)
        G2 = Data(edge_index=edge_g2, num_nodes=num_nodes_g2)

        # Dataset abstract
        print("\nDataset:", name)
        print(f"{'=' * 50}")
        print(f"G1: {G1.num_nodes} nodes, {G1.num_edges} edges")
        print(f"G2: {G2.num_nodes} nodes, {G2.num_edges} edges")
        print(f"Number of alignment links: {len(alignment)}")

        return G1, G2, alignment