import pytest
from src.adapters.provider import ProviderAdapter

# Create a dummy subclass since ProviderAdapter is not instantiable on its own.
class DummyProvider(ProviderAdapter):
    def __init__(self, model_name: str, max_tokens: int = 4024):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def init_model(self):
        return None

    def make_prediction(self, prompt: str) -> str:
        return prompt

    def chat_completion(self, messages, tools=[]):
        return None

@pytest.fixture
def provider():
    return DummyProvider("dummy_model", max_tokens=1000)

def test_extract_json_from_valid_array(provider):
    input_response = "[[1, 2, 3], [4, 5, 6]]"
    result = provider.extract_json_from_response(input_response)
    assert result == [[1, 2, 3], [4, 5, 6]]

def test_extract_json_from_response_object(provider):
    input_response = 'Some text {"response": [[7, 8, 9]]} and more text'
    result = provider.extract_json_from_response(input_response)
    assert result == [[7, 8, 9]]

def test_extract_json_with_no_valid_json(provider):
    input_response = "This string has no JSON content."
    result = provider.extract_json_from_response(input_response)
    assert result is None

def test_extract_json_with_invalid_then_valid(provider):
    # The first candidate is invalid JSON, so it should be skipped until a valid JSON array is found.
    input_response = "Invalid JSON attempt: {not valid json} then valid: [[10, 20, 30]] extra text."
    result = provider.extract_json_from_response(input_response)
    assert result == [[10, 20, 30]]

def test_extract_json_with_dict_without_response(provider):
    # Valid JSON object that does not include a "response" key should be ignored.
    input_response = 'Some text {"data": [[1, 2, 3]]} more text'
    result = provider.extract_json_from_response(input_response)
    assert result is None

def test_extract_json_multiple_candidates(provider):
    # Multiple valid JSON substrings; the first valid one is a dictionary without "response"
    # so it should skip it and return the later valid JSON array.
    input_response = 'First valid: {"no_response": [1,2,3]} then valid: [[4,5,6], [7,8,9]]'
    result = provider.extract_json_from_response(input_response)
    assert result == [[4, 5, 6], [7, 8, 9]]

def test_extract_json_with_leading_text(provider):
    # A JSON array embedded within leading and trailing text should be correctly extracted.
    input_response = "Before text [[11, 12], [13, 14]] trailing text."
    result = provider.extract_json_from_response(input_response)
    assert result == [[11, 12], [13, 14]]

def test_extract_json_from_empty_input(provider):
    # An empty input should return None.
    input_response = ""
    result = provider.extract_json_from_response(input_response)
    assert result is None

def test_multiple_valid_responses(provider):
    # When multiple valid responses exist, the first valid one should be returned.
    input_response = 'Start {"response": [[10, 20]]} middle {"response": [[30, 40]]} end'
    result = provider.extract_json_from_response(input_response)
    assert result == [[10, 20]]

def test_extract_json_with_long_input(provider):
    # A long input with lots of noise should still return the valid JSON response.
    long_input = ("noise " * 1000) + ' {"response": [[99, 100], [101, 102]]}' + (" extra noise " * 1000)
    result = provider.extract_json_from_response(long_input)
    assert result == [[99, 100], [101, 102]]

def test_provider_subclasses_json_parsing(monkeypatch):
    # Set dummy API keys for providers that require them.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_anthropic_key")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_openai_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_google_key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy_deepseek_key")

    from src.adapters.anthropic import AnthropicAdapter
    from src.adapters.open_ai import OpenAIAdapter
    from src.adapters.gemini import GeminiAdapter
    from src.adapters.deepseek import DeepseekAdapter

    # Instantiate the adapters with dummy parameters.
    adapters = [
        AnthropicAdapter("dummy_model", max_tokens=1000),
        OpenAIAdapter("dummy_model", max_tokens=1000),
        GeminiAdapter("dummy_model", max_tokens=1000),
        DeepseekAdapter("dummy_model", max_tokens=1000)
    ]

    # Define test inputs and their expected outputs.
    json_array_input = "[[1, 2, 3], [4, 5, 6]]"
    json_object_input = 'Random text {"response": [[7, 8, 9]]} more text'
    complex_array_input = """[
        {
            "attempt_1": [
                [8, 8, 8, 0],
                [8, 0, 0, 8],
                [0, 8, 8, 0],
                [8, 0, 0, 8],
                [0, 8, 0, 0],
                [8, 8, 0, 0]
            ],
            "attempt_2": [
                [8, 8, 8, 0],
                [8, 0, 0, 8],
                [0, 8, 8, 0],
                [8, 0, 0, 8],
                [0, 8, 0, 0],
                [8, 8, 0, 0]
            ]
        }
    ]"""

    expected_array = [[1, 2, 3], [4, 5, 6]]
    expected_response = [[7, 8, 9]]
    expected_complex = [{"attempt_1": [[8, 8, 8, 0], [8, 0, 0, 8], [0, 8, 8, 0], [8, 0, 0, 8], [0, 8, 0, 0], [8, 8, 0, 0]], 
                        "attempt_2": [[8, 8, 8, 0], [8, 0, 0, 8], [0, 8, 8, 0], [8, 0, 0, 8], [0, 8, 0, 0], [8, 8, 0, 0]]}]

    for adapter in adapters:
        result = adapter.extract_json_from_response(json_array_input)
        assert result == expected_array, f"Adapter {adapter.__class__.__name__} failed on array input."

        result2 = adapter.extract_json_from_response(json_object_input)
        assert result2 == expected_response, f"Adapter {adapter.__class__.__name__} failed on object input."
        
        result3 = adapter.extract_json_from_response(complex_array_input)
        assert result3 == expected_complex, f"Adapter {adapter.__class__.__name__} failed on complex array input." 