import os
import time
import requests
import json

from llms.LLMInterface import LLMInterface


class ChatGPT4(LLMInterface):
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        self.api_url = os.environ.get("API_URL")

    def generate_response(self, prompt, system_message):
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")
        return response.json()['choices'][0]['message']['content']

    def stream_response(self, prompt, system_message):
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "stream": True  # Request streaming response
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")

        buffer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if 'data: ' in decoded_line:
                    data = decoded_line[len('data: '):]
                    if data.strip() == "[DONE]":
                        yield "[DONE]"  # Yield the end marker before breaking
                        break  # Terminate the stream processing after yielding [DONE]
                    if data:
                        message = json.loads(data)['choices'][0]['delta']
                        if 'content' in message:
                            content_part = message['content']
                            buffer += content_part
                            yield content_part  # Yield each part of the content as it arrives
        return buffer

# Example usage
if __name__ == "__main__":
    gpt4 = ChatGPT4()
    prompt = "Write me the conversation between Sansa, Arya, and Jon, when they finally met after being separated in the godswoods that everyone wanted from Game of Thrones"
    system_message = "You are George R.R. Martin, a famous writer."

    try:
        start_time = time.time()
        response_start_time = None
        for content_part in gpt4.stream_response(prompt, system_message):
            if response_start_time is None:
                response_start_time = time.time()
                first_response_time = response_start_time - start_time
                print(f"\nTime to first response: {first_response_time:.2f} seconds")
            print(content_part, end="", flush=True)  # Print each part of the content as it arrives
        elapsed_time = time.time() - start_time
        print(f"\nStreaming response time: {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
