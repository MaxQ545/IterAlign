import argparse
from models import *
from data import DataLoader
from evaluate import Evaluator
from config import configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="oIterAlign")
    parser.add_argument('--dataset', type=str, default="Arxiv1-Arxiv2")
    parser.add_argument('--num_dp_select', type=int)
    parser.add_argument('--num_diffusion_select', type=int)
    parser.add_argument('--diffusion_step', type=int)
    parser.add_argument('--device', type=str)
    args = parser.parse_args()

    # Step 1: Initialize dataset and model
    model_name = args.model
    dataset_name = args.dataset
    config = configs[model_name][dataset_name]

    config['num_dp_select'] = args.num_dp_select
    config['num_diffusion_select'] = args.num_diffusion_select
    config['diffusion_step'] = args.diffusion_step
    config['device'] = args.device

    data_loader = DataLoader(dataset_path="data")
    G1, G2, alignment = data_loader.load_data(dataset_name, 0)
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
    evaluator.evaluate(0)
