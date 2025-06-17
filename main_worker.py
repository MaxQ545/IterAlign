import argparse
import json
import sys
import os

# Local imports from your project
from models import *
from data_loader import DataLoader
from evaluate import Evaluator
from config import configs


def run_single_experiment(model_name, dataset_name, remove_rate):
    """
    Runs one experiment and reports the execution time upon completion.
    Resource monitoring is handled by the calling manager script.
    """
    # Step 1: Initialize dataset and model
    try:
        config = configs[model_name][dataset_name]
    except KeyError:
        print(f"Error: No configuration found for model '{model_name}' and dataset '{dataset_name}'.")
        exit(1)

    data_loader = DataLoader(dataset_path="data")
    G1, G2, alignment = data_loader.load_data(dataset_name, remove_rate)

    model_class = globals().get(model_name)
    if not model_class:
        print(f"Error: Model class '{model_name}' not found.")
        exit(1)

    model = model_class(config)

    # Step 2: Run the model and get alignment
    # The model.run() method is expected to return: (align_links, align_ranks, time)
    align_links, align_ranks, time_taken = model.run(G1, G2, alignment)

    # Step 3: (Optional) Evaluate model
    evaluator = Evaluator(
        align_links=align_links,
        align_ranks=align_ranks,
        alignment=alignment,
        time=time_taken
    )
    evaluator.evaluate(100)

    # Step 4: Prepare results for output
    # The only thing we need to report is the final execution time.
    results = {
        "time": time_taken,
    }
    # Print results as a JSON string to stdout for the manager script
    sys.stdout = original_stdout
    print(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--remove-rate', type=float, default=0.00)
    args = parser.parse_args()

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    run_single_experiment(args.model, args.dataset, args.remove_rate)