from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
load_dotenv()

class DeepseekAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        self.client = self.init_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def init_client(self):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        client = OpenAI(api_key = os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        return client

    def make_prediction(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        # print(f"Prompt: {prompt}")
        response = self.chat_completion(messages)
        # print(response)
        # Uncomment to print token usage
        # print(f"USAGE|PROMPT|{response.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{response.usage.completion_tokens}")
        # print(f"Response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()

    def chat_completion(self, messages: str) -> str:
        # print(messages)
        return self.client.chat.completions.create(
            model=self.model_name,
            # TODO: parameterize the reasoning_effort (including not setting it since it's only supported
            # o1, as of 12/19/2024)
            # Default value for o1 is 'medium'.
            # Uncomment to set a different value.
            # reasoning_effort='high',
            messages=messages,
        )
