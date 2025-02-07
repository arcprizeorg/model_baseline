import abc
from typing import List, Dict, Tuple
import json

class ProviderAdapter(abc.ABC):
    @abc.abstractmethod
    def chat_completion(self, message: str) -> str:
        pass

    @abc.abstractmethod
    def extract_json_from_response(self, input_response: str) -> List[List[int]]:
        pass

    def extract_json_from_response(self, input_response: str) -> list:
        """
        Extract JSON from any substring within the input_response using standard parsing.

        Valid responses are defined as follows:
          - A JSON object containing a "response" key is valid; the corresponding value is returned.
          - A top-level JSON array is valid and returned as-is.

        Any JSON object that does not have a "response" key is considered invalid, and all nested candidates
        within its boundary are ignored.

        When multiple valid responses are present, the first valid response (in reading order) is returned.
        """
        import re, json
        decoder = json.JSONDecoder()
        skip_until = -1
        for match in re.finditer(r'[\[{]', input_response):
            if match.start() < skip_until:
                continue
            try:
                obj, end_index = decoder.raw_decode(input_response[match.start():])
                if isinstance(obj, dict):
                    if "response" in obj:
                        return obj["response"]
                    else:
                        # Mark the region spanned by this JSON object so nested candidates are ignored.
                        skip_until = match.start() + end_index
                        continue
                if isinstance(obj, list):
                    if match.start() < skip_until:
                        continue
                    return obj
            except json.JSONDecodeError:
                continue
        return None