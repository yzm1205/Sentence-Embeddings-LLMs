from argparse import ArgumentParser
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(dotenv_path="./ContextWordEmbedding/Model/.env")


class GPTModel:
    
    def __init__(self,model_name='gpt-3.5-turbo-0125'):
    
        self.client = OpenAI(api_key=os.getenv("gpt3_key"))
        self.model_name = model_name
        print(f"\n {self.model_name} \n")
        

    def generate_response(self,system_role,prompt, temperature=0.9, top_p=1.0):
        """
        Generate text using the GPT-3 model.

        Args:
            prompt (str): The prompt to use for text generation.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
            temperature (float): The temperature to use for text generation. Defaults to 0.7.
            top_p (float): The top-p value to use for text generation. Defaults to 1.0.
            n (int): The number of completions to generate. Defaults to 1.

        Returns:
            list: A list of generated text outputs.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", 
                    "content": system_role},
                    {"role":"user",
                    "content":prompt}
                    ],
            
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content


if __name__ == "__main__":
# Example usage
    profile = "Gen-Y"
    system_content= ""

    model = GPTModel(system_role=system_content)
    prompt = "What happened to voyger 1 spacecraft?"
    generated_text = model.generate_response(prompt)
    print(generated_text)