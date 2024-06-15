import json
import os
import requests

from llms.LLMInterface import LLMInterface

class ChatpGPT4(LLMInterface):
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
            "stream": True
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.api_url, headers=headers, json=data, stream=True)
        if response.status_code == 200:
            buffer = ""
            try:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            line = line[6:]  # Remove the 'data: ' prefix
                        decoded_line = json.loads(line)
                        if 'choices' in decoded_line:
                            chunk = decoded_line['choices'][0]['delta']['content']
                            if chunk:  # Only process non-empty content
                                buffer += chunk
                                words = buffer.split()
                                if len(words) > 1:
                                    for word in words[:-1]:
                                        yield word
                                    buffer = words[-1] + " "
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Last line received: {line}")
            if buffer:  # Yield any remaining content in buffer
                yield buffer.strip()
        else:
            raise Exception(f"Failed to get valid response: {response.status_code} {response.text}")

# Example usage
if __name__ == "__main__":
    gpt4 = ChatpGPT4()
    prompt = "add a screwdriver to container 15"
    system_message = """You are chatGPT-4, a well-trained LLM used to assist humans.
    /set system You are a helpful assistant with access to the following inventory system functions. Use them if required - examples:
    1. If asked 'add a hammer to drawer 5', you should update the inventory of drawer 5 by adding a hammer.
    2. If asked 'remove a screwdriver from drawer 10', you should update the inventory of drawer 10 by removing a screwdriver.
    3. If asked 'add two pencils to drawer 3', you should update the inventory of drawer 3 by adding two pencils.

    [
        {"name": "add_inventory", "description": "Add items to the inventory of a specific drawer", "parameters": {"type": "object", "properties": {"drawer_number": {"type": "integer", "description": "The number of the drawer to update"}, "items_to_add": {"type": "array", "items": {"type": "string"}, "description": "List of items to add to the drawer"}}, "required": ["drawer_number", "items_to_add"]}},
        {"name": "delete_inventory", "description": "Delete items from the inventory of a specific drawer", "parameters": {"type": "object", "properties": {"drawer_number": {"type": "integer", "description": "The number of the drawer to update"}, "items_to_delete": {"type": "array", "items": {"type": "string"}, "description": "List of items to delete from the drawer"}}, "required": ["drawer_number", "items_to_delete"]}}
    ]
    """

    try:
        for word in gpt4.stream_response(prompt, system_message):
            print(word, end=' ', flush=True)  # Print each word followed by a space
    except Exception as e:
        print(f"Error: {e}")
