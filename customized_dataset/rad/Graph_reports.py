
import json
import pandas as pd
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data 
import torch
import itertools
import pickle
from transformers import AutoTokenizer
import time
#torch.multiprocessing.set_start_method('spawn', force=True)

class buildKG():
   def __init__(self,d,i):
      super(buildKG,self).__init__()
      
      self.d = d
      self.index = i
      self.reports=pickle.load(open("/u/home/mohammed/Graphormer/pickles/Frontal_Graph_3000.pkl","rb"))
      self.data = self.d[self.reports[self.index]]

      self.embeds = pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_1_TO_2700.pkl","rb"))
      
      #self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
      
      #self.dict=pickle.load(open("e_word2idx_425.pickle","rb"))


   def get_entity(self,idx):          
      e=self.data["entities"][str(idx)]["tokens"]
      return e

   def create_Graph(self):
      count_entities=(len(self.data["entities"]))
      #print(count_entities)
      
      kg=pd.DataFrame(columns = ['source','target','edge'])
      for idx in range(count_entities):
         idx=idx+1
       
         if str(idx) in self.data["entities"]:  # Problem for report 2, "11" not present 
            n=(self.data["entities"][str(idx)]["tokens"])
            no_of_edges = len(self.data["entities"][str(idx)]["relations"])
         
         #if str(idx) in self.data["relations"]:
            for e in range(no_of_edges):
                idx2=self.data["entities"][str(idx)]["relations"][e][1]
                relation=self.data["entities"][str(idx)]["relations"][e][0]
         #if (!idx2 && !relation):
                kg.loc[len(kg)]=[n,self.get_entity(idx2),relation]
    #else:
     # kg.loc[len(kg)]=[n,"-","-"]
         if no_of_edges==0:
            kg.loc[len(kg)]=[n,"-","-"]
      return kg


   def network_graph(self):
      kg=self.create_Graph()
      G=nx.from_pandas_edgelist(kg, "source", "target", 
                          edge_attr=True, create_using=nx.MultiDiGraph())

      
      return G
   
   def plot(self):
      G=self.network_graph()
      plt.figure(figsize=(12,12))
      pos = nx.spring_layout(G, k = 0.8) # k regulates the distance between nodes
      nx.draw(G, with_labels=True, node_color='red', node_size=1000, edge_cmap=plt.cm.Blues, pos = pos, font_weight='bold')
      plt.show()

   def extract_entity_names_and_edges(self):
      kg = self.create_Graph()

      source=list(kg["source"])
      targets = list(kg["target"])
      edges = list(kg["edge"])

      entities=(list(set(source+targets)))
      entities.remove('-')

      edges=list(filter(lambda a: a != "-", edges)) 

      return entities,edges
   
   

   def extract_triples(self):
      triples=list((nx.to_edgelist(self.network_graph())))
      trilist = []
      for i in range(len(triples)):
         if triples[i][2]['edge'] !="-" or triples[i][1] != "-":
            string = triples[i][0] + " " + triples[i][2]['edge'] + " " +  triples[i][1] 
            trilist.append(string)
      #print(" len triples ",len(trilist))
      return trilist

   def entity_embeds(self):

      entity_embeds = self.embeds[self.index][0].to("cpu")
  
      return entity_embeds #self.embeds[self.index][0]
      
   def edge_embeds(self):
      edge_embeds = self.embeds[self.index][1].to("cpu")
   
      return edge_embeds #self.embeds[self.index][1]

   
   def src_dst(self):
      e,_=self.extract_entity_names_and_edges()
      src = []
      dst = []

      triples=list((nx.to_edgelist(self.network_graph())))
      for i in range(len(triples)):
         if triples[i][2]['edge'] != "-" or triples[i][1] != "-": # Checks if node is not alone and has existing edge
            
            #src.append(self.dict[triples[i][0]])
            #dst.append(self.dict[triples[i][1]])
            src.append(e.index(triples[i][0]))
            dst.append(e.index(triples[i][1]))
            #print(triples[i][0],triples[i][1])
            #src = list(map(float,src))
            #dst = list(map(float,dst))
      return (src,dst)



def main():

  
   file = "/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json"
   with open(file) as f:
      d = json.load(f)

   obj=buildKG(d,1620)
   n,e=obj.extract_entity_names_and_edges()
   
   print(n,e)
   #print(obj.relation_embeds())
   #obj.plot()
   #print((obj.extract_triples()))
   #print(len(obj.extract_triples()))
   #print((obj.entity_embeds()).shape) 
   #print(("hello"))

   # Graph Creation
   #src,dst = obj.src_dst()
   #edge_index = torch.tensor([src,dst],dtype=torch.long)
   #g = Data(x=torch.tensor(obj.entity_embeds()), edge_attr = torch.tensor(obj.relation_embeds()), edge_index=edge_index,y=torch.tensor(0))
   #print(g)
   #print(src,dst)
   #edge_index = torch.tensor([src,dst],dtype=torch.long)
   #g = Data(x=torch.tensor(obj.entity_embeds()), edge_attr = torch.tensor(obj.relation_embeds()), edge_index=edge_index)
   
if __name__ == '__main__':
    main()
