import pandas as pd
import pickle
from customized_dataset.rad.Graph_reports import buildKG
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import time
import json


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeds(sentences,tokenizer,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_output = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                               add_special_tokens=True,
                                               padding='longest',
                                               return_tensors='pt')
    tokenizer_output=tokenizer_output.to(device)
    embeddings = model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                 attention_mask=tokenizer_output.attention_mask).to(device)

    return embeddings

def main():
    CXR_BERT={}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports=pickle.load(open("/u/home/mohammed/Graphormer/pickles/Frontal_Graph_3000.pkl","rb"))
    file='/u/home/mohammed/Graphormer/examples/customized_dataset/MIMIC-CXR_graphs.json'
    # dict = { 0 : [node,edge] , 1: [node,edge] }
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True)
    model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True).to(device)   
    torch.cuda.empty_cache()
    with open(file) as f:
        d = json.load(f)

    for idx, report in enumerate(reports):

        idx = idx+2701
        obj=buildKG(d,idx)
        print("GRAPH NUMBER = ",idx)
   
        nodes,edges = obj.extract_entity_names_and_edges()
   
        if edges and nodes:

            
            node_embeds=get_embeds(nodes,tokenizer,model)
           
            edge_embeds=get_embeds(edges,tokenizer,model)
            

            print(node_embeds.shape,edge_embeds.shape)
            CXR_BERT[idx]=[node_embeds,edge_embeds]
        
        #if idx%300 == 0:
         #   print("Saving the embeds ......")
          #  pd.to_pickle(CXR_BERT,"/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_{}.pkl".format(idx))

    pd.to_pickle(CXR_BERT,"/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_{}.pkl".format(idx))

if __name__ == '__main__':
    main()