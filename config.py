configs = {
    "oIterAlign": {
        "Facebook-Twitter": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 10,  # 10 for best
            "num_diffusion_select": 20,
            "diffusion_step": 5,
        },
        "DBLP1-DBLP2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 200,  # 200 for original dataset
            "num_diffusion_select": 30,
            "diffusion_step": 7,
        },
        "Arxiv1-Arxiv2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 100,  # 100 for default
            "num_diffusion_select": 100,
            "diffusion_step": 8,
        },
    },

    "fIterAlign": {
        "Facebook-Twitter": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 20,  # 20 for default
            "num_diffusion_select": 20,
            "diffusion_step": 5,
        },
        "DBLP1-DBLP2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 200,  # 200 for major experience
            "num_diffusion_select": 30,
            "diffusion_step": 7,
        },
        "Arxiv1-Arxiv2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 100,  # 100 for default
            "num_diffusion_select": 100,
            "diffusion_step": 8,
        },
    },

    "MMNC_CENA": {
        "Facebook-Twitter": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 100,
            "num_diffusion_select": 20,
            "diffusion_step": 5,
        },
        "DBLP1-DBLP2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 1000,
            "num_diffusion_select": 20,
            "diffusion_step": 7,
        },
        "Arxiv1-Arxiv2": {
            "device": "cuda",
            "seed": 42,
            "dp_min_degree": 6,
            "num_dp_select": 200,
            "num_diffusion_select": 100,
            "diffusion_step": 8,
        },
    },

    "MMNC": {
        "Facebook-Twitter": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": False,
            "degree_thresold": 6,
        },
        "DBLP1-DBLP2": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": False,
            "degree_thresold": 6,
        },
        "Arxiv1-Arxiv2": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": True,
            "degree_thresold": 6,
        },
    },

    "iMMNC": {
        "Facebook-Twitter": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": False,
            "degree_thresold": 6,
            "rate": 0.02,
            "niter": 5,
        },
        "DBLP1-DBLP2": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": False,
            "degree_thresold": 6,
            "rate": 0.02,
            "niter": 5,
        },
        "Arxiv1-Arxiv2": {
            "train_ratio": 0.04,
            "k_de": 3,
            "k_nei": 7,
            "fast_select": True,
            "degree_thresold": 6,
            "rate": 0.02,
            "niter": 5,
        }
    },

    "CENA": {
        "Facebook-Twitter": {
            "alpha": 5,
            'layer': 3,
            'q': 0.5,
            'c': 0.5,
            'multi_walk': True,
            'align_train_prop': 0.1
        },
        "DBLP1-DBLP2": {
            "alpha": 5,
            'layer': 3,
            'q': 0.5,
            'c': 0.5,
            'multi_walk': True,
            'align_train_prop': 0.1
        },
        "Arxiv1-Arxiv2": {
            "alpha": 5,
            'layer': 3,
            'q': 0.5,
            'c': 0.5,
            'multi_walk': True,
            'align_train_prop': 0.1
        }
    },

    "GradAlign": {
        "Facebook-Twitter": {
            "k_hop": 2,
            "hid_dim": 150,
            "train_ratio": 0.1,
            "centrality": 'eigenvector',
            "att_portion": 0,
        },
        "DBLP1-DBLP2": {
            "k_hop": 2,
            "hid_dim": 150,
            "train_ratio": 0.1,
            "centrality": 'eigenvector',
            "att_portion": 0,
        },
        "Arxiv1-Arxiv2": {
            "k_hop": 2,
            "hid_dim": 150,
            "train_ratio": 0.1,
            "centrality": 'eigenvector',
            "att_portion": 0,
        },
    },

    "FINAL": {
        "Facebook-Twitter": {
            "alpha": 0.82,
            "max_iter": 30,
            "tol": 1e-4,
            "H": None,
            'train_ratio': 0.1,
        },
        "DBLP1-DBLP2": {
            "alpha": 0.82,
            "max_iter": 30,
            "tol": 1e-4,
            "H": None,
            'train_ratio': 0.1,
        },
        "Arxiv1-Arxiv2": {
            "alpha": 0.82,
            "max_iter": 30,
            "tol": 1e-4,
            "H": None,
            'train_ratio': 0.1,
        }
    },
    "REGAL": {
        "Facebook-Twitter": {
            'k': 10,
            "max_layer": 2,
            "alpha": 0.01,
            "gammastruc": 1,
            "gammaattr": 1,
            "buckets": 2,
        },
        "DBLP1-DBLP2": {
            'k': 10,
            "max_layer": 2,
            "alpha": 0.01,
            "gammastruc": 1,
            "gammaattr": 1,
            "buckets": 2,
        },
        "Arxiv1-Arxiv2": {
            'k': 10,
            "max_layer": 2,
            "alpha": 0.01,
            "gammastruc": 1,
            "gammaattr": 1,
            "buckets": 2,
        }
    },
    "WLAlign": {
        "Facebook-Twitter": {
            "train_ratio": 10,
        },
        "DBLP1-DBLP2": {
            "train_ratio": 10,
        },
        "Arxiv1-Arxiv2": {
            "train_ratio": 10,
        }
    },
    "CONE": {
        "DBLP1-DBLP2": {
            "embmethod": "netMF",
            "dim": 1024,
            "window": 10,
            "negative": 1.0,
            "niter_init": 10,
            "reg_init": 1.0,
            "nepoch": 5,
            "niter_align": 10,
            "reg_align": 0.05,
            "bsz": 10,
            "lr": 1.0,
            "embsim": "euclidean",
            "alignmethod": "greedy",
            "numtop": 10
        },
        "Arxiv1-Arxiv2": {
            "embmethod": "netMF",
            "dim": 1024,
            "window": 10,
            "negative": 1.0,
            "niter_init": 10,
            "reg_init": 1.0,
            "nepoch": 5,
            "niter_align": 10,
            "reg_align": 0.05,
            "bsz": 10,
            "lr": 1.0,
            "embsim": "euclidean",
            "alignmethod": "greedy",
            "numtop": 10
        },
    }
}