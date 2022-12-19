from customized_dataset.rad.Graph_reports import buildKG
from collections import Counter
import pickle
import pandas as pd

def vocab_425():
    file='/u/home/mohammed/Graphormer/examples/customized_dataset/train.json'
    reports=pickle.load(open("Frontal_Graph_425.pkl","rb"))
    
    e_vocab_nocopy=[]
    for i in range(len(reports)):
      print(i)
      obj=buildKG(file,i)
      e_vocab_nocopy = e_vocab_nocopy + obj.extract_entity_names()
    
    e_vocab_nocopy = list(set(e_vocab_nocopy))
    e_word2idx_425 = { word: ind for ind,word in enumerate(e_vocab_nocopy)}

    pd.to_pickle(e_word2idx_425,"e_word2idx_425.pickle")
    
def vocab():
    file='/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json'
    reports=pickle.load(open("/u/home/mohammed/Graphormer/pickles/Frontal_Graph_3000.pkl","rb"))
    
    e_vocab_nocopy=[]
    for i in range(len(reports)):
      #print(i)
      obj=buildKG(file,i)
      e_vocab_nocopy = e_vocab_nocopy + obj.extract_entity_names()
    
    e_vocab_nocopy = list(set(e_vocab_nocopy))
    e_word2idx_3000 = {word: ind for ind,word in enumerate(e_vocab_nocopy)}
    print(e_word2idx_3000)
    pd.to_pickle(e_word2idx_3000,"e_word2idx_3000.pickle")
    
def t_vocab():
    file='/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json'
    reports=pickle.load(open("Frontal_Graph.pkl","rb"))
    
    t_vocab_nocopy=[]
    for i in range(len(reports)):
      obj=buildKG(file,i)
      t_vocab_nocopy = t_vocab_nocopy + obj.extract_triples()
    
    print(len(t_vocab_nocopy))
    t_vocab_nocopy = list(set(t_vocab_nocopy))
    print(len(t_vocab_nocopy))
    t_triple2idx_full_frontal = { triple: ind for ind,triple in enumerate(t_vocab_nocopy)}

    pd.to_pickle(t_triple2idx_full_frontal,"t_triple2idx_425_full_frontal.pickle")

def id2patient_425():
    idx2patient_425 = {}
    reports=pickle.load(open("Frontal_Graph_425.pkl","rb"))

    for ind,patient in enumerate(reports):
      print(ind)
      x=patient.split("/")
      patient_id = x[1]
      study_id = x[2].split(".")[0]
      idx2patient_425[ind]=[patient_id[1:],study_id[1:]]
      #print(self.idx2patient_425)
      
      # Run for each new data file 
      pd.to_pickle(idx2patient_425,"idx2patient_425.pickle")

def id2patient():
    idx2patient_3000 = {}
    reports=pickle.load(open("Frontal_Graph_3000.pkl","rb"))
    for ind,patient in enumerate(reports):
      x=patient.split("/")
      patient_id = x[1]
      study_id = x[2].split(".")[0]
      idx2patient_3000[ind]=[patient_id[1:],study_id[1:]]
      #print(self.idx2patient_425)
      
      # Run for each new data file 
      pd.to_pickle(idx2patient_3000,"idx2patient_3000.pickle")

def testpkl():
  emd=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_200.pkl","rb"))
  print(len(emd.keys()))
def main():
  #vocab()
  testpkl()

if __name__ == '__main__':
    main()
