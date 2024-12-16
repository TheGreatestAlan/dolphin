import json

import requests

from agent_server.llms.EncryptedKeyStore import EncryptedKeyStore
from agent_server.llms.LLMInterface import LLMInterface


class FireworksAiRestLLM(LLMInterface):
    def __init__(self, api_token, model="accounts/fireworks/models/llama-v3p1-8b-instruct"):
        self.api_token = api_token
        self.model = model
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def generate_response(self, prompt, system_message):
#        print('SYSTEM_MESSAGE:::\n' +  system_message)
#        print('PROMPT_MESSAGE:::\n' +  prompt)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            res = data["choices"][0]["message"]["content"]
#            print(res)
            return res
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")


    def stream_response(self, prompt, system_message):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": True
        }

        with requests.post(self.url, json=payload, headers=self.headers, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = line.decode("utf-8")
                        if data == "data: [DONE]":
                            yield self.END_STREAM
                            break
                        else:
                            # Parse JSON and yield only the content
                            json_data = json.loads(data[6:])  # Strip off "data: "
                            if "choices" in json_data and "delta" in json_data["choices"][0]:
                                content = json_data["choices"][0]["delta"].get("content")
                                if content:
                                    yield content
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

import sys

def main():
    keystore = EncryptedKeyStore()
    api_token = keystore.get_api_key("FIREWORKS_API_KEY")
    model = FireworksAiRestLLM(api_token)

    prompt = "tell me a story about Hasan Piker"
    system_message = "You are a helpful assistant."

    # Use the stream_response method to get the response in real time
    try:
        print("Streaming response:", end=" ", flush=True)
        for response_chunk in model.stream_response(prompt, system_message):
            # Print each chunk without a newline, ChatGPT style
            sys.stdout.write(response_chunk + " ")
            sys.stdout.flush()
        print("\n")  # Ensure a newline at the end of the response
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()