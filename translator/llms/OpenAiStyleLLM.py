import json
import requests
from translator.llms.LLMInterface import LLMInterface


def process_message_history(message_history, system_message, current_user_message=None):
    """
    Processes the message history by:
    - Removing previous system messages.
    - Ensuring the current user message is not duplicated.
    - Adding the provided system message at the start.
    - Appending the current user message at the end if provided.
    """
    if not message_history:
        message_history = []

    # Remove all previous system messages
    filtered_history = [msg for msg in message_history if msg["role"] != "system"]

    # Ensure the current user message is not duplicated
    if current_user_message:
        filtered_history = [msg for msg in filtered_history if
                            not (msg["role"] == "user" and msg["content"] == current_user_message)]

    # Add the system message at the start
    processed_history = [{"role": "system", "content": system_message}] + filtered_history

    # Append the current user message at the end
    if current_user_message:
        processed_history.append({"role": "user", "content": current_user_message})

    return processed_history


class OpenAIStyleLLM:
    def __init__(self, api_token, model, url):
        """
        Initializes the OpenAIStyleLLM with API token, model, and URL.
        """
        self.model = model
        self.url = url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }

    def generate_response(self, prompt, system_message, message_history=None):
        """
        Generates a single response (non-streaming).
        """
        # Process the message history with the current prompt
        messages = process_message_history(message_history, system_message, prompt)

        payload = self.generate_payload(messages, stream=False)
        response = self.make_request(payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def stream_response(self, prompt, system_message, message_history=None):
        """
        Streams the response in real time.
        """
        # Process the message history with the current prompt
        messages = process_message_history(message_history, system_message, prompt)

        payload = self.generate_payload(messages, stream=True)
        response = self.make_request(payload, True)  # Don't use 'with' here to keep the response open
        try:
            for line in response.iter_lines():
                if line:
                    data = line.decode("utf-8")
                    if data == "data: [DONE]":
                        yield LLMInterface.END_STREAM
                        break
                    else:
                        # Parse JSON and yield only the content
                        json_data = json.loads(data[6:])  # Strip off "data: "
                        if "choices" in json_data and "delta" in json_data["choices"][0]:
                            content = json_data["choices"][0]["delta"].get("content")
                            if content:
                                yield content
        finally:
            response.close()  # Ensure the response is properly closed

    def generate_payload(self, messages, stream=False):
        """
        Generates the payload for the API request.
        """
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": stream
        }

    def make_request(self, payload, stream=False):
        """
        Sends the API request with the given payload.
        """
        response = requests.post(self.url, json=payload, headers=self.headers, stream=stream)
        if response.status_code == 200:
            return response
        else:
            response.close()  # Ensure response is closed in case of an error
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
