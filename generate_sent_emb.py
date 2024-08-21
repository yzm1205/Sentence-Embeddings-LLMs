import torch
import json
import sys
from typing import Tuple, Union, List
import sys
import openai

# Load Models
from Models.mistral import Mistral
from Models.llama import LLaMA
from Models.olmo_model import OlMo
from Models.openelm import OpenELM
from Models.sentence_transformers_models import sentence_transformers_models as stm

from utils.read_files import mkdir_p, full_path
import os
import dotenv
dotenv.load_dotenv(os.getenv("./Model/.env"))
hf = os.getenv("huggingface_token")
hf_auth = hf


class ModelLoader:
    def __init__(self, model_name:str , default_gpu="cuda:1") -> None:
        self.device =default_gpu
        self.is_valid_model = True
        cache_dir = mkdir_p("./data/shared/llm_cache")
        
        if model_name in ["sbert","simcse"]:
            self.model = stm(model_name=model_name,device=self.device)
        
        elif model_name in ["llama2" ,"Llama2"]:
            self.model_dir = str(full_path("/data/shared/Llama-2-7b-hf/Llama-2-7b-hf_shard_size_1GB"))
            self.model = LLaMA(self.model_dir,hf_auth,self.device)
        
        elif model_name in ["llama3"]:
            self.model_dir = str(full_path(f"/data/shared/llama3-8b/Meta-Llama-3-8B_shard_size_1GB"))
            self.model = LLaMA(self.model_dir,hf_auth,self.device)
            
        elif model_name in ["mistral"]:
            model_path = str(full_path("/data/shared/mistral-7b-v03/Mistral-7B-v0.3_shard_size_1GB"))
            self.model = Mistral(model_path, hf, self.device)
            
        elif model_name in ["olmo"]:
            model_path = str(full_path("/data/shared/olmo/OLMo-7B_shard_size_1GB"))
            self.model = OlMo(model_path, hf_token = hf, device= self.device)
            
        elif model_name in ["openelm"]:
            self.model = OpenELM(self.device)
        
        else:
            self.is_valid_model = False
            ValueError(f"Unknown model name : {model_name}")
            sys.exit()
        # print(f"Model {model_name} loaded successfully.")
    
            
class EmbeddingGenerator(ModelLoader):
    def __init__(self,model_name:str,default_gpu="cuda",
                 gpt3Model:str= "text-embedding-ada-002") -> None:
        super().__init__(model_name,default_gpu)
        self.device=default_gpu
        self.model_name = model_name
        if self.model_name in ["gpt3", "chatgpt"]:
            self.gpt3model = gpt3Model
                
    def encode(self,anchor_words:List[str],sentences: List[str],batch_size:int=32)->Tuple[torch.Tensor,torch.Tensor]:
        if not self.is_valid_model:
            print(f"Error: Unknown model name '{self.model_name}'. Embedding generation aborted. ")
            return None
        sentence_embeddings = self.model.encode(sentences=sentences,batch_size=batch_size)
        return sentence_embeddings
        
            
        
# Usage
if __name__ == "__main__":
    
    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
    
    x='this is testing'
    y="this is not a testing"
    z= "this is just a random sentence"
    zz = " this is another sentence"
    zy="This sentence is longer than all other sentence"
    sentences = [x,y,z,zz,zy]
    anchor_word = ["testing","testing","random","sentence","longer"]
    # batch = sente * 10
    generator = EmbeddingGenerator("openelm",default_gpu=device)
    sentence_embeddings = generator.encode(sentences=sentences)
  
    print("done")
    
