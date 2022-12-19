from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
import torch_geometric
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import argparse

import numpy as np
# the VIC (Very Important Class)
from Graphmodels.graphormer_graph_encoder import GraphormerGraphEncoder
from customized_dataset.MIMIC_CXR.mimic_data import create_loader
from transformers import CLIPProcessor, CLIPModel
import clip
from Graphormermodule import GraphormerDataModule 

class GraphCLIP(pl.LightningModule):
    def __init__(self,
            num_atoms,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            #warmup_updates,
            #tot_updates,
            peak_lr,
            #end_lr,
            weight_decay
            ): 
        super().__init__()
        self.save_hyperparameters()

        print("hey")
        self.num_atoms = num_atoms
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        #self.warmup_updates = warmup_updates
        #self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        #self.end_lr = end_lr
        self.weight_decay = weight_decay


        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=self.num_atoms,
            num_in_degree=self.num_in_degree,
            num_out_degree=self.num_out_degree,
            num_edges=self.num_edges,
            num_spatial=self.num_spatial,
            num_edge_dis=self.num_edge_dis,
            edge_type=self.edge_type,
            multi_hop_max_dist=self.multi_hop_max_dist,)
        
        
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_encoder = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", device="cuda")
        self.model, _ = clip.load("ViT-B/32",device="cuda")
        self.projection_dim = 512
        self.graph_embed_dim = 768
        self.image_embed_dim = 512
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature = 1.0

        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.graph_embed_dim, self.graph_embed_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.graph_embed_dim, self.projection_dim)
        )
    
        self.image_proj_head = nn.Sequential(
          nn.Linear(self.image_embed_dim, self.image_embed_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.image_embed_dim, self.projection_dim)
        )
    
    def cross_entropy(self,preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
    
    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.peak_lr,
            weight_decay=self.weight_decay
            )


        return optimizer

    #def forward(self,features_graph,features_image):
    def training_step(self,batch):
        
        # Encode Graphs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _,graph_feat=self.graph_encoder(batch)
        graph_feat=self.graph_proj_head(graph_feat)

        graph_feat = graph_feat.to("cuda")
        # Load similar images of MIMIC-CXR
        obj=create_loader()
        img_batch=obj.bring_loaders(batch["y"])

        # Encode Images
        image_feat=self.model.encode_image(img_batch.to("cuda"))
        #print(image_feat)
        #image_feat=self.image_proj_head(image_feat)


        #image_feat = image_feat / image_feat.norm(dim=1, keepdim=True)
        #graph_feat = graph_feat / graph_feat.norm(dim=1, keepdim=True)

        # cosine similarity as logits

        graph_feat = graph_feat.to(torch.float16)
        logits = (graph_feat @ image_feat.T) / self.temperature
        image_similarity = image_feat @ image_feat.T
        graph_similarity = graph_feat @ graph_feat.T

        targets = F.softmax((image_similarity + graph_similarity)/2 * self.temperature, dim=-1)
        #print(logits,targets)
        images_loss = self.cross_entropy(logits,targets,reduction='none')
        graphs_loss = self.cross_entropy(logits.T,targets.T,reduction='none')
        loss = (images_loss + graphs_loss)/2.0
        return loss.mean()


def add_model_args(parent_parser):
    group = parent_parser.add_argument_group("GraphCLIP")
    group.add_argument('--num_atoms', type=int, default=512*9, help='number of atom types')
    group.add_argument('--num_in_degree', type=int, default=512, help='number of in degree types')
    group.add_argument('--num_out_degree', type=int, default=512, help='number of out degree types')
    group.add_argument('--num_edges', type=int, default=512*3, help='number of edge types')
    group.add_argument('--num_spatial', type=int, default=512, help='number of spatial types')
    group.add_argument('--num_edge_dis', type=int, default=128, help='number of edge dis types')
    group.add_argument('--edge_type', type=str, default='multi_hop', help='edge type')
    group.add_argument('--multi_hop_max_dist', type=int, default=10, help='multi_hop_max_dist')

    # Optimization
    #group.add_argument('--warmup_updates', type=int, default=60000)
    #group.add_argument('--tot_updates', type=int, default=1000000)
    group.add_argument('--peak_lr', type=float, default=1e-5)
    #group.add_argument('--end_lr', type=float, default=1e-9)
    group.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    #parent_parser.print_help()

    return parent_parser

def main():
    
    #Arguments
    parent_parser=argparse.ArgumentParser()
    opt=add_model_args(parent_parser)
    args = opt.parse_args()
    args_dict=vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #parent_parser.print_help()

    # MAIN ENCODER (COMBINED)
    GraphImageEncoder = GraphCLIP(**args_dict)
    
    optimizer = GraphImageEncoder.configure_optimizers()

    # DATA MODULE
    Graphdata = GraphormerDataModule()
    
    train_loader = Graphdata.train_dataloader()
    #print(len(train_loader))
    #batch=next(iter(train_loader))

    trainer = pl.Trainer()
    trainer.fit(model = GraphImageEncoder,train_dataloaders=train_loader )
    for idx, batch in enumerate(train_loader):
        #print(" size = ",batch)
        loss=GraphImageEncoder.training_step(batch=batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss = ",loss)
    #for idx,batch in enumerate(train_loader):
     #   print(idx,batch)
      #  Encoder.training_step(batch,idx)


if __name__ == '__main__':
    main()
