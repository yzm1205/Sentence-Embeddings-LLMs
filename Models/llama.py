import imp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import dotenv
import os
import torch
import time
from pathlib import Path
from typing import Union, List
import sys
from utils.read_files import mkdir_p, full_path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils.__helper__ import mean_pooling
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


dotenv.load_dotenv(os.getenv("./models/.env"))
hf = os.getenv("huggingface_token")


class LLaMA:
    def __init__(self, model_dir: str, hf_token: str, device: torch.device):
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.eval()

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32):

        sentence_tokenizer = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_token_type_ids=False,
        ).to(device=self.device)

        with torch.no_grad():
            outputs = self.model(
                **sentence_tokenizer, output_hidden_states=True, use_cache=False
            )

        # Sentence Embeddings
        sentence_embeddings = mean_pooling(
            outputs.last_hidden_state, sentence_tokenizer["attention_mask"]
        )

        return sentence_embeddings


def run():
    model_path = str(full_path("/data/shared/llama3-8b/Meta-Llama-3-8B_shard_size_1GB"))
    # model_path2= str(full_path("/data/shared/Llama-2-7b-hf/Llama-2-7b-hf_shard_size_1GB"))
    device = "cuda:1"
    model = LLaMA(model_path, hf, device)

    # Batch 1
    sentences = [
        "The actress was deeply adored by her fans for her talent and humility.",
        "The company launched a new and improved version of their software, which significantly enhanced its functionality.",
        "The team of huskies raced across the snowy terrain, their sled cutting through the icy path with ease.",
        "The government implemented various policies aimed at ameliorating the economic situation in the country.",
        "The detective carefully examined the burked crime scene for any clues.",
    ]

    sentence_embeddings = model.encode(sentences, batch_size=32)

    print("Done")


if __name__ == "__main__":
    run()
    print("done")
