from .provider import ProviderAdapter
import os
from dotenv import load_dotenv
import json
from openai import OpenAI
load_dotenv()


class DeepSeekAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        self.client = self.init_client()
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def init_client(self):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        client = OpenAI()
        return client

    def make_prediction(self, prompt: str) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        # print(f"Prompt: {prompt}")
        response = self.chat_completion(messages)
        
        # Uncomment to print token usage
        # print(f"USAGE|PROMPT|{response.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{response.usage.completion_tokens}")
        # print(f"Response: {response.choices[0].message.content.strip()}")
        return response.choices[0].message.content.strip()

    def chat_completion(self, messages: str) -> str:
        return self.client.chat.completions.create(
            model=self.model_name,
            # TODO: parameterize the reasoning_effort (including not setting it since it's only supported
            # o1, as of 12/19/2024)
            # Default value for o1 is 'medium'.
            # Uncomment to set a different value.
            # reasoning_effort='high',
            messages=messages,
        )

    def extract_json_from_response(self, input_response: str) -> list[list[int]] | None:


        prompt = f"""
You are a helpful assistant. Extract only the JSON of the test output from the following response. 
Do not include any explanation or additional text; only return valid JSON.

Response:
{input_response}

The JSON should be in this format:
{{
"response": [
    [1, 2, 3],
    [4, 5, 6]
]
}}
"""

        # print(f"Input response: {input_response}")
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        # Uncomment to print token usage 
        # print(f"USAGE|PROMPT|{completion.usage.prompt_tokens}")
        # print(f"USAGE|COMPLETION|{completion.usage.completion_tokens}")

        assistant_content = completion.choices[0].message.content.strip()

        # Some oai models like to wrap the response in a code block
        if assistant_content.startswith("```json"):
            assistant_content = "\n".join(assistant_content.split("\n")[1:])
        
        if assistant_content.endswith("```"):
            assistant_content = "\n".join(assistant_content.split("\n")[:-1])

        # Attempt to parse the returned content as JSON
        # print(f"For input response: {input_response}, got extracted content: {assistant_content}")
        try:
            json_entities = json.loads(assistant_content)
            return json_entities.get("response")
        except json.JSONDecodeError:
            return None