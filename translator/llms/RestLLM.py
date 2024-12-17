import requests
import json
from translator.llms.LLMInterface import LLMInterface

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

    def stream_response(self, prompt, system_message):
        data = {
            "prompt": prompt,
            "system_message": system_message,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/stream", headers=headers, json=data, stream=True)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if 'data: ' in decoded_line:
                    data = decoded_line[len('data: '):]
                    if data.strip() == "[DONE]":
                        break
                    if data:
                        message = json.loads(data)['choices'][0]['delta']
                        if 'content' in message:
                            content_part = message['content']
                            yield content_part
