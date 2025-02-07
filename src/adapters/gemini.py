from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
import google.generativeai as genai
from typing import List, Optional

load_dotenv()

class GeminiAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        self.model = self.init_model(model_name)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": 0.0
        }

    def init_model(self, model_name: str):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai.GenerativeModel(model_name)

    def make_prediction(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_completion(messages)
        return response.text

    def chat_completion(self, messages: list) -> str:
        # Convert to Gemini's message format
        history = [{"parts": [msg["content"]], "role": msg["role"]} for msg in messages]
        
        response = self.model.generate_content(
            contents=history,
            generation_config=self.generation_config,
            stream=False
        )
        return response