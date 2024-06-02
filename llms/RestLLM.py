import requests
from llms.LLMInterface import LLMInterface

class RestLLM(LLMInterface):
    def __init__(self, base_url):
        self.base_url = base_url

    def generate_response(self, prompt, system_message):
        payload = {
            'prompt': prompt,
            'system_message': system_message
        }

        print("PROMPT::\n")
        print(prompt)
        response = requests.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()
        print("RESPONSE::\n")
        print(response_json)
        return response_json['response']
