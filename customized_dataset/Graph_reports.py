
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import dgl
import importlib

class buildKG():
   def __init__(self,file,i):
      super(buildKG,self).__init__()
      with open(file) as f:
         self.d = json.load(f)
      print("graph reports .py")
      self.index = i
      self.reports = list(self.d)
      print(self.reports)
      print(len(self.reports))
      self.data = self.d[self.reports[self.index]]

   def get_entity(self,idx):          
      e=self.data["entities"][str(idx)]["tokens"]
      return e

   def create_Graph(self):
      count_entities=(len(self.data["entities"]))
      kg=pd.DataFrame(columns = ['source','target','edge'])
      for idx in range(count_entities):
         idx=idx+1
         n=(self.data["entities"][str(idx)]["tokens"])
         no_of_edges = len(self.data["entities"][str(idx)]["relations"])
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

   def extract_entity_names(self):
      source=list(self.create_Graph()["source"])
      targets = list(self.create_Graph()["target"])
      e=(list(set(source+targets)))
      e.remove('-')
      return e
      

   def extract_triples(self):
      triples=list((nx.to_edgelist(self.network_graph())))
      trilist = []
      for i in range(len(triples)):
         if triples[i][2]['edge'] !="-" or triples[i][1] != "-":
            string = triples[i][0] + " " + triples[i][2]['edge'] + " " +  triples[i][1] 
            trilist.append(string)
      return trilist

   def entity_embeds(self):
      model = SentenceTransformer('all-MiniLM-L6-v2') 
      sentences = self.extract_entity_names()
      print(sentences)
      e_embed = model.encode(sentences)
      return e_embed
      
   def relation_embeds(self):
      model = SentenceTransformer('all-MiniLM-L6-v2') 
      sentences = self.extract_triples()
      print(sentences)
      t_embed = model.encode(sentences)
      return t_embed
   
   def src_dst(self):
      e=self.extract_entity_names()
      #print(e,len(e))
      src = []
      dst = []

      triples=list((nx.to_edgelist(self.network_graph())))
      trilist = []
      for i in range(len(triples)):
         if triples[i][2]['edge'] != "-" or triples[i][1] != "-":
            
            src.append(e.index(triples[i][0]))
            dst.append(e.index(triples[i][1]))
            #print(triples[i][0],triples[i][1])

      return (src,dst)

def main():

   obj=buildKG('train.json',115)

   #print((obj.extract_entity_names()))
   #obj.plot()
   #print((obj.extract_triples()))
   #print(len(obj.extract_triples()))
   #print((obj.entity_embeds()).shape)
   #print((obj.src_dst()))
if __name__ == '__main__':
    main()
