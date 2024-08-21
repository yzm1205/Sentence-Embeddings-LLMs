import token
from transformers import AutoModelForCausalLM,AutoTokenizer
import dotenv
import os
import torch
from utils.__helper__ import mean_pooling

dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")
# breakpoint()

class OpenELM:
    def __init__(self,device:torch.device):  
        self.device= device
        self.model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-3B",trust_remote_code=True,token=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',token=hf,add_bos_token=True,trust_remote_code=True,add_eos_token=True,add_special_tokens=True,
        return_tensors='pt')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device).eval()
        

    
    def get_sentence_embedding(self,sentence: str):
        sentence_tokenizer = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True,max_length=1024,return_token_type_ids=False).to(device=self.device)
        with torch.no_grad():
            outputs = self.model(**sentence_tokenizer,use_cache=False,output_hidden_states=True)
        
        sentence_embeddings = mean_pooling(outputs.hidden_states[-1],sentence_tokenizer["attention_mask"]).squeeze()
        return sentence_embeddings
    
    


if __name__ == "__main__":
    device = 'cuda:2'
    model= OpenELM(device)
    x='this is testing'
    y="this is not a testing"
    z= "this is just a random sentence"
    zz = " this is another sentence"
    zy="This sentence is longer than all other sentence"
    sent = [x,y,z,zz,zy]
    batch = sent * 10
    sentence_embedding = model.encode(batch)
    
    print("done")