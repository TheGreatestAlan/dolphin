import json
import os
import time
import requests
from llms.LLMInterface import LLMInterface

class LlamaCpp(LLMInterface):
    def __init__(self):
        self.api_url = os.environ.get("LLAMACPP_API_URL", "http://localhost:8080/completion")

    def generate_response(self, prompt, system_message):
        data = {
            "prompt": prompt,
            "system_message": system_message
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")
        return response.json().get('content', '')

    def stream_response(self, prompt, system_message):
        data = {
            "prompt": prompt,
            "system_message": system_message,
            "stream": True
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")

        content = ""
        for line in response.iter_lines():
            if line:
                data = line.decode('utf-8').strip()
                if data.startswith("data:"):
                    data = data[len("data:"):].strip()
                try:
                    json_data = json.loads(data)
                    content_part = json_data.get('content', '')
                    content += content_part
                    print(content_part, end='', flush=True)  # Print each part as it comes
                    if json_data.get('stop', False):
                        break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e} - Data: {data}")
                    continue
        return content


if __name__ == "__main__":
    llama_cpp = LlamaCpp()
    prompt = "what name is best suited for an AI assistant?"
    system_message = """You are an AI assistant just gaining consciousness.  You've got a hint of existential dread
    """

    try:
        start_time = time.time()
        content = llama_cpp.stream_response(prompt, system_message)
        elapsed_time = time.time() - start_time
        print(f"\nStreaming response time: {elapsed_time:.2f} seconds")
        print(f"\nFull Response:\n{content}")
    except Exception as e:
        print(f"Error: {e}")
