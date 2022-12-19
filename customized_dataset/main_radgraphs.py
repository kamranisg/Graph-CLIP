#!/usr/bin/...python3
from customized_dataset.rad.Graph_reports import buildKG
import pandas as pd
import numpy as np
import torch

import importlib
from torch_geometric.data import Data 
import faulthandler

faulthandler.enable()

class RadGraphs():
    def __init__(self,d):
        super(RadGraphs,self).__init__()
        self.d = d
        

    def __getitem__(self, item):
        
        print("Inside Main_Radgraphs ISNT IT .... We are at Graph Number ",item)
        obj=buildKG(self.d,item)
        src,dst = obj.src_dst()
        #print(src,dst)

        # DGL STUFF
        #g=dgl.graph((src,dst))
        #g.ndata['attr']=torch.tensor(obj.entity_embeds())
        #g.edata['edge_attr']=torch.tensor(obj.relation_embeds())

        # PYG STUFF
        
        #print(obj.entity_embeds(),obj.edge_embeds(),edge_index)
        edge_index = torch.tensor([src,dst],dtype=torch.long)

        g = Data(x=obj.entity_embeds(), edge_attr = obj.edge_embeds(), edge_index=edge_index, y=torch.tensor(item))

        return (g)
    def __len__(self):

        return len(self.d)

def main():
    #print("heo")
    #file = "/u/home/mohammed/Graphormer/examples/customized_dataset/train.json" 
    file = "/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json" 
    #dataset=RadGraphs(file)
    #print(dataset[2])
    #for graph in dataset:
     #   print(graph)
        
if __name__ == '__main__':
    main()