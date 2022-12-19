#from graphormer.data import register_dataset
#from dgl.data import QM9
from customized_dataset.main_radgraphs import RadGraphs
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json 

#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

#torch.multiprocessing.set_start_method('spawn')
#@register_dataset("customized_radgraph_dataset")

def create_customized_dataset():
    print(" ..... WE ARE HERE AT CUSTOMIZED DATA.PY ......")
    file = "/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json"
    with open(file) as f:
        d = json.load(f)
    dataset = RadGraphs(d)
    num_graphs = len(dataset)
    print(num_graphs)
    # customized dataset split
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_graphs), test_size=num_graphs // 5, random_state=0
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx, test_size=num_graphs // 5, random_state=0
    )
    return {
        "dataset": dataset,
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "source": "pyg"
    }
