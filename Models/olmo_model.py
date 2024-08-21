# from hf_olmo import * # registers the Auto* classes
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import torch
import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union, List
import sys
sys.path.insert(0, "./ContextWordEmbedding/")
from utils.read_files import mkdir_p,full_path
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.__helper__ import mean_pooling

dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")

class OlMo:
    def __init__(self,model_dir:str,hf_token:str,device:torch.device):    
        self.device= device
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,torch_dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model.eval() 
            
    def encode(self,sentence: Union[str, List[str]]):
        sentence_tokenizer = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True,max_length=1024,return_token_type_ids=False).to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**sentence_tokenizer,output_hidden_states=True,use_cache=False)
            
        sentence_embeddings = mean_pooling(outputs.hidden_states[-1], sentence_tokenizer['attention_mask']).squeeze()
        return sentence_embeddings
    
    
if __name__ ==  "__main__":
    # make sure to install : pip install ai2-olmo
    model_path = str(full_path("/data/shared/olmo/OLMo-7B_shard_size_1GB"))
    device = 'cuda:2'
    model= OlMo(model_path, hf, device)
    
    x='this is testing'
    y="this is not a testing"
    z= "this is just a random sentence"
    zz = " this is another sentence"
    zy="This sentence is longer than all other sentence"
    sent = [x,y,z,zz,zy]
    batch = sent * 10
   
    sentence_embedding = model.encode(batch)
                                            
    print("done")



