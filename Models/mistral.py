
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import torch
import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union, List
import sys
from utils.read_files import mkdir_p,full_path
from utils.__helper__ import mean_pooling
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM

dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")


class Mistral:
    def __init__(self,model_dir:str,hf_token:str,device:torch.device):  
        self.device= device
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.float16,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
    
    def encode(self,sentence: Union[str, List[str]]):
        
        sentence_tokenizer = self.tokenizer(sentence, return_tensors='pt',padding=True, truncation=True,max_length=1024).to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**sentence_tokenizer,output_hidden_states=True,use_cache=False)
        
        sentence_embeddings = mean_pooling(outputs.last_hidden_state, sentence_tokenizer['attention_mask']).squeeze()
        return sentence_embeddings


if __name__ ==  "__main__":
    model_path = str(full_path("/data/shared/mistral-7b-v03/Mistral-7B-v0.3_shard_size_1GB"))
    device = 'cuda:3'
    model = Mistral(model_path, hf, device)
    message = ["Language modeling is gifted to the community "]
    
    sentence_embedding = model.encode(sentence=message)
    print("done")



