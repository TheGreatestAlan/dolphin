import os
import time
import requests

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
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"Received line: {decoded_line}")  # Debug print statement
                if 'data: ' in decoded_line:
                    data = decoded_line[len('data: '):]
                    if data.strip() == "[DONE]":
                        break
                    if data:
                        message = eval(data)['choices'][0]['delta']
                        if 'content' in message:
                            yield message['content']


# Example usage
if __name__ == "__main__":
    gpt4 = ChatGPT4()
    prompt = "Write me the conversation between Sansa, Arya, and Jon, when they finally met after being separated in the godswoods that everyone wanted from Game of Thrones"
    system_message = "You are George R.R. Martin, a famous writer."

    try:
        start_time = time.time()
        for content_part in gpt4.stream_response(prompt, system_message):
            print(content_part, end="", flush=True)
        elapsed_time = time.time() - start_time
        print(f"\nStreaming response time: {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
