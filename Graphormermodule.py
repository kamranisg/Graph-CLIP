from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
import json

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
# import dataset classes

from customized_dataset.customized_dataset import create_customized_dataset
#from graphormer.data.dataset import GraphormerDataset
from graphormer.data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
from customized_dataset.MIMIC_CXR.mimic_data import create_loader
from graphormer.data.dataset import BatchedDataDataset
from graphormer.data.collator import collator

class GraphormerDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()


        self.data_dict = create_customized_dataset()
        print("outside customized")
        self.graph_dataset=GraphormerPYGDataset(dataset=self.data_dict["dataset"],train_idx=self.data_dict["train_idx"],valid_idx=self.data_dict["valid_idx"],test_idx=self.data_dict["test_idx"])

        self.graph_dataset = self.graph_dataset
        print("outside graph dataset init")

    def train_dataloader(self):

        train_data = self.graph_dataset.train_data
        batched_train_data = BatchedDataDataset(train_data)
        #print(batched_train_data[12])
        radgraph_train_loader = torch.utils.data.DataLoader(batched_train_data,batch_size=64,num_workers=0,collate_fn=collator,shuffle=False)
      
        return radgraph_train_loader


def main():
    obj = GraphormerDataModule()
    train_loaders = obj.train_dataloader()
    
    #print(train_loaders["Radgraphs"],train_loaders["MIMIC"])
    #for idx,batch in enumerate(train_loaders["Radgraphs"]):
     #   print(idx)
    #iterator=iter(train_loader)
    #print(next(iterator))


if __name__ == '__main__':
    main()