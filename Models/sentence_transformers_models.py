from transformers import AutoModel, AutoTokenizer
import torch
from utils.__helper__ import mean_pooling
import os

class sentence_transformers_models:
    def __init__(self,model_name:str,device:torch.device,model_version:str=None):
        if model_name=="sbert":
            model_version = "sentence-transformers/all-MiniLM-L6-v2"
        elif model_name=="simcse":
            model_version = "princeton-nlp/sup-simcse-bert-large-uncased"
        else:
            model_version = model_version
            
        self.device = device
        self.model = AutoModel.from_pretrained(model_version).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_version)
        self.model.eval()
    
    
    def encode(self,sentence: str):
        sentence_tokenizer = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True,max_length=1024,return_token_type_ids=False).to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**sentence_tokenizer,use_cache=False)
            
        sentence_embeddings = mean_pooling(outputs.last_hidden_state,sentence_tokenizer['attention_mask']).squeeze()
        return sentence_embeddings
    
   
        
if __name__ == "__main__":
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = sentence_transformers_models(model_name="simcse",device=device)
    sentence = "this is testing"
   
    sentence_embedding= model.encode(sentence)
    
    print("done")
    
    """
    Models: 
    simcse: princeton-nlp/sup-simcse-bert-base-uncased
    use: sentence-transformers/use-cmlm-multilingual
    
    """