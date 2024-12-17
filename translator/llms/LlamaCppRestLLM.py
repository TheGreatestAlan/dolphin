import json
import requests
from translator.llms.LLMInterface import LLMInterface


class LlamaCppRestLLM(LLMInterface):
    def __init__(self, base_url, timeout=600):
        self.base_url = base_url
        self.timeout = timeout  # Default timeout of 600 seconds

    def generate_response(self, prompt, system_message):
        # Adjust payload format to include messages as an array
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        }

        print("PROMPT::\n", prompt)
        response = requests.post(self.base_url + "/chat/completions", json=payload, timeout=self.timeout)
        response.raise_for_status()
        response_json = response.json()
        print("RESPONSE::\n", response_json)

        # Adjust to retrieve the assistant's message
        return response_json.get('choices', [{}])[0].get('message', {}).get('content', "")

    def stream_response(self, prompt, system_message):
        # Use the adjusted payload format for streaming response
        payload = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{self.base_url}/stream", headers=headers, json=payload, stream=True, timeout=self.timeout
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if 'data: ' in decoded_line:
                    data = decoded_line[len('data: '):]
                    if data.strip() == "[DONE]":
                        break
                    if data:
                        message = json.loads(data).get('choices', [{}])[0].get('message', {}).get('content', "")
                        yield message
