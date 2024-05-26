from abc import ABC, abstractmethod
import os
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer


# Abstract base class for text generation
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, conversation_id, prompt, system_message):
        pass


class LocalLLM(LLMInterface):
    def __init__(self, model_path, cache_directory):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", cache_dir=cache_directory).to(
            'cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, cache_dir=cache_directory)

    def generate_response(self, conversation_id, prompt, system_message):
        input_ids = self.tokenizer.encode(system_message + prompt, return_tensors='pt').to('cuda')
        response_ids = self.model.generate(input_ids, max_length=1000)
        return conversation_id, self.tokenizer.decode(response_ids[0], skip_special_tokens=True)


class ChatpGPT4(LLMInterface):
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url

    def generate_response(self, conversation_id, prompt, system_message):
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
        return conversation_id, response.json()['choices'][0]['message']['content']


# Example usage
if __name__ == "__main__":

    api_key = os.getenv('OPEN_AI_API_KEY')
    url = "https://api.openai.com/v1/chat/completions"
    gpt4 = ChatpGPT4(api_key, url)
    conversation_id = "conv1"
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
        conv_id, response = gpt4.generate_response(conversation_id, prompt, system_message)
        print(f"Conversation ID: {conv_id}\nResponse:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
