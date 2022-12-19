
import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def findAllMissingNumbers(a):
   b = sorted(a)
   return list(set(range(b[0], b[-1])) - set(b))

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

    e1=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_300.pkl","rb"))
    e2=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_600.pkl","rb"))
    e3=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_900.pkl","rb"))
    e4=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_1200.pkl","rb"))
    e5=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_1500.pkl","rb"))
    e6=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_1800.pkl","rb"))
    e7=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_2100.pkl","rb"))
    e8=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_2400.pkl","rb"))
    e9=pickle.load(open("/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_2700.pkl","rb"))
    dict = {**e1,**e2,**e3,**e4,**e5,**e6,**e7,**e8,**e9}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True)
    model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",trust_remote_code=True).to(device)   
    s1=['change']
    s2=['significant','change']
    edge=torch.randn(0,128)
    dict[60]=[get_embeds(s1,tokenizer,model),edge]
    dict[1024]=[get_embeds(s2,tokenizer,model),edge]
    dict[1027]=[get_embeds(s1,tokenizer,model),edge]
    dict[1620]=[get_embeds(s1,tokenizer,model),edge]    
    pd.to_pickle(dict,"/u/home/mohammed/Graphormer/Embeddings/CXR_BERT_EMBEDDINGS_1_TO_2700.pkl")
    a=[]
    i=0
    for key,value in dict.items():
        i=i+1
        #print(value[0].shape)
        a.append(key)

    print(i)
    print(findAllMissingNumbers(a))
    #print(dict.keys())


if __name__ == '__main__':
    main()