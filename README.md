# Empowering Iterative Graph Alignment Using Heat Diffusion (IterAlign)

This repository contains the official implementation of the paper: **Empowering Iterative Graph Alignment Using Heat Diffusion**.

Our proposed method, **IterAlign**, is a novel, parameter-free, and efficient approach for Unsupervised Plain Graph Alignment (UPGA). IterAlign leverages heat diffusion to generate stable node representations, mitigating the over-reliance on strict structural consistency. By alternating between representation generation and node alignment, our model iteratively refines the alignment process, leading to superior and robust performance.

## About The Project

Existing Unsupervised Plain Graph Alignment (UPGA) methods often struggle with structural inconsistencies between real-world graphs, leading to biased node representations and suboptimal alignment. To address these limitations, we propose **IterAlign**, which introduces several key innovations:

  * **Heat Diffusion for Stable Representations:** We are the first to apply graph heat diffusion for UPGA, creating stable node representations that effectively reduce the impact of structural noise.
  * **Iterative Refinement:** IterAlign employs an iterative process that alternates between generating node representations and performing node alignment. This allows the model to progressively correct erroneous matches and refine the alignment.
  * **Parameter-Free and Efficient:** Our method is designed to be computationally efficient and free of hyperparameters, making it easy to apply to a wide range of graph alignment tasks.

Extensive experiments on public benchmark datasets demonstrate that IterAlign not only outperforms state-of-the-art UPGA approaches with lower computational overhead but also approaches the theoretical accuracy upper bound for unsupervised plain graph alignment.

## Getting Started

To get a local copy up and running, please follow these simple steps.

### Prerequisites

This project requires Python 3. Make sure you have it installed on your system.

### Installation

1.  Clone the repo:
    ```sh
    git clone https://github.com/MaxQ545/IterAlign.git
    ```
2.  Install the required Python packages using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

You can run the model using the following command structure. The example below demonstrates how to run `fIterAlign` on the `Facebook-Twitter` dataset.

```sh
python main.py --model fIterAlign --dataset Facebook-Twitter --remove-rate 0.00
```

### Arguments

  * `--model`: The model to be used. For our proposed method, use `fIterAlign and oIterAlign`.
  * `--dataset`: The dataset to perform graph alignment on (e.g., `Facebook-Twitter, DBLP1-DBLP2 and Arxiv1-Arxiv2`).
  * `--remove-rate`: The rate at which nodes are removed to test robustness. `0.00` indicates no nodes are removed.

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{wang2025empowering,
  title={Empowering Iterative Graph Alignment Using Heat Diffusion},
  author={Wang, Boyan and Feng, Weijie and Huang, Jinyang and Guo, Dan and Liu, Zhi},
  journal={arXiv preprint arXiv:2506.17640},
  year={2025}
}
```

## Contact

Boyan Wang - [wangboyan@mail.hfut.edu.cn]

Project Link: [https://github.com/MaxQ545/IterAlign](https://www.google.com/search?q=https://github.com/your_username/your_repository_name)