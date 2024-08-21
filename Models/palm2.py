import re
import numpy as np
import dotenv
import os
import torch
import google.generativeai as palm

dotenv.load_dotenv(os.getenv("./models/.env"))


class Palm2:
    def __init__(self,device:torch.device):
        palm.configure(api_key=os.getenv("palm_key"))
        self.model_version = "models/embedding-gecko-001"
        models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
        # print(f"Available Models: {models}")
        # originally used version 001
        self.model = next(filter(lambda m: m.name == 'models/embedding-gecko-001', models))
        
    def sentence_embedding(self, sentence):
        sentence = sentence.strip()
        response = palm.generate_embeddings(text=sentence, model=self.model)
        return np.array(response['embedding'])
    
        
    
if __name__ == "__main__":
    # install the package: pip install google-generativeai
    device = 'cuda:1'
    model = Palm2(device)
    message = "Language modeling is gifted to the community "
    embeddings = model.sentence_embedding(message)
    print("done")