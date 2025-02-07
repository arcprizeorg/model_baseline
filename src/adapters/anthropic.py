from .provider import ProviderAdapter
from src.models import ARCTaskOutput
import anthropic
import os
from dotenv import load_dotenv
import json
from typing import List
load_dotenv()

class AnthropicAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        # Initialize VertexAI model
        self.model = self.init_model()
        self.model_name = model_name
        self.max_tokens = max_tokens

    def init_model(self):
        """
        Initialize the Anthropic model
        """
        
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        return client
    
    def make_prediction(self, prompt: str) -> str:
        """
        Make a prediction with the Anthropic model
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)

        return response.content[0].text

    def chat_completion(self, messages, tools=[]) -> str:
        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools
        )
        return response

if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-3-5-sonnet-20240620")
    print(type(adapter.extract_json_from_response("[[1, 2, 3], [4, 5, 6]]")))