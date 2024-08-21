import anthropic
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="./ContextWordEmbedding/Model/.env")


class ClaudeModel:

    def __init__(self, model="claude-3-sonnet-20240229"):
      
        self.client = anthropic.Anthropic(api_key=os.getenv("claude_key"))
        self.model = model
        
        print(f"\n {model} \n")

    def generate_response(self, system_role, prompt, max_tokens=1000, temperature=1):
        """
        Generate a response using the Claude model.

        Args:
            prompt (str): The prompt to use for generating the response.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 100.
            temperature (float): The temperature to use for generating the response. Defaults to 0.7.
    
        Returns:
            list: A list of generated responses.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_role,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        return response.content[0].text

if __name__ == "__main__":
    profile = "Gen-Y"
    system_content= f"You are a chat bot assiting people with their queries. The responses should be genereated for the user profile as {profile}. Note that, the repsonses should align with the user profile. For instance, example 1: If the user profile has 'age' keyword and its value is 'age' and the people to address are 'kids', then the chatbot should reply in a way that is suitable for kids. -  Similarly, Example 2: if the user profile has'political view' category and if its value is 'left wing', then the responses to the quires should address leftist people only. - Example 3: In the user profile, there could be multiple keywords such as 'age', political_view' and many more and its value could be 'adult', leftist' respectively. The keywords and its values define the user profile. So, generate responses such that it only intereset to that user profile."

    # Example usage
    model = ClaudeModel(system_role=system_content)
    prompt = "what is the purpose of life?"
    response = model.generate_response(prompt)
    print(response)
