import argparse
from models import *
from data_loader import DataLoader
from evaluate import Evaluator
from config import configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="fIterAlign")
    parser.add_argument('--dataset', type=str, default="Facebook-Twitter")
    parser.add_argument('--remove-rate', type=float, default=0.00)
    args = parser.parse_args()

    # Step 1: Initialize dataset and model
    model_name = args.model
    dataset_name = args.dataset
    config = configs[model_name][dataset_name]

    data_loader = DataLoader(dataset_path="data")
    G1, G2, alignment = data_loader.load_data(dataset_name, args.remove_rate)
    model = globals()[model_name]
    model = model(config)

    # Step 2: Run the model and get alignment
    align_links, align_ranks, time = model.run(G1, G2, alignment)

    # Step 3: Evaluate model
    evaluator = Evaluator(
        align_links=align_links,
        align_ranks=align_ranks,
        alignment=alignment,
        time=time
    )
    evaluator.evaluate(100)
