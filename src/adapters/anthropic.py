from .provider import ProviderAdapter
from src.models import ARCTaskOutput, AttemptMetadata, Choice, Message, Usage, Cost, CompletionTokensDetails, Attempt
import anthropic
import os
from dotenv import load_dotenv
import json
from typing import List
from datetime import datetime
load_dotenv()

class AnthropicAdapter(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        # Initialize VertexAI model
        self.model = self.init_client()
        self.model_name = model_name
        self.max_tokens = max_tokens

    def init_client(self):
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
        Make a prediction with the Anthropic model and return an Attempt object
        """
        start_time = datetime.utcnow()
        
        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(messages)
        end_time = datetime.utcnow()

        # Calculate costs based on Anthropic's pricing
        # These rates should be moved to a config file in production
        input_cost_per_token = 0.0000163  # $0.0163/1K tokens for Claude 3 Sonnet
        output_cost_per_token = 0.0000551  # $0.0551/1K tokens for Claude 3 Sonnet
        
        prompt_cost = response.usage.input_tokens * input_cost_per_token
        completion_cost = response.usage.output_tokens * output_cost_per_token

        # Convert input messages to choices
        input_choices = [
            Choice(
                index=i,
                message=Message(
                    role=msg["role"],
                    content=msg["content"]
                )
            )
            for i, msg in enumerate(messages)
        ]

        # Convert Anthropic response to our schema
        response_choices = [
            Choice(
                index=len(input_choices),
                message=Message(
                    role="assistant",
                    content=content.text if content.type == "text" else json.dumps(content.input)
                )
            )
            for content in response.content
            if content.type in ["text", "tool_use"]
        ]

        # Combine input and response choices
        all_choices = input_choices + response_choices

        # Create metadata using our Pydantic models
        metadata = AttemptMetadata(
            model=self.model_name,
            provider="anthropic",
            start_timestamp=start_time,
            end_timestamp=end_time,
            choices=all_choices,
            kwargs={
                "max_tokens": self.max_tokens,
            },
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                completion_tokens_details=CompletionTokensDetails(
                    reasoning_tokens=0,  # Anthropic doesn't provide this breakdown
                    accepted_prediction_tokens=response.usage.output_tokens,
                    rejected_prediction_tokens=0  # Anthropic doesn't provide this
                )
            ),
            cost=Cost(
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                total_cost=prompt_cost + completion_cost
            )
        )

        attempt = Attempt(
            metadata=metadata,
            answer=response.content[0].text if response.content else ""
        )

        return attempt.answer

    def chat_completion(self, messages, tools=[]):
        """
        Make a raw API call to Anthropic and return the response
        """
        return self.model.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=tools
        )
    
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        tools = [
            {
                "name": "extract_json",
                "description": "Extracts JSON from the response.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            },
                            "description": "A list of lists of integers extracted from the response."
                        }
                    },
                    "required": ["response"]
                }
            }
        ]

        text = f"Extract JSON of the test output from the following response: {input_response}"

        query = f"""
        <document>
        {text}
        </document>

        Use the extract_json tool.
        """

        response = self.chat_completion(
            messages=[{"role": "user", "content": query}],
            tools=tools
        )

        json_response = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "extract_json":
                json_entities = content.input
                break

        if json_entities:
            return json_entities['response']
        else:
            return None
        
if __name__ == "__main__":
    adapter = AnthropicAdapter("claude-3-5-sonnet-20240620")
    print(type(adapter.extract_json_from_response("[[1, 2, 3], [4, 5, 6]]")))