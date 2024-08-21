import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow_text import SentencepieceTokenizer


class USE:
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)
        print("USE model loaded successfully....")
    
    def encode(self,sentence):
       
       sentence_embeddings = self.model(sentence).numpy()[0]
       return sentence_embeddings
         
    
    

    

if __name__ == "__main__":
    model = USE()
    sentence = ["Language modeling is gifted to the community"]
    embeddings = model.sentence_embeddings(sentence)
    