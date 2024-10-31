import requests
import json
from agent_server.llms.LLMInterface import LLMInterface


class OllamaLLM(LLMInterface):
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name

    def generate_response(self, prompt, system_message):
        payload = {
            "model": self.model_name,
            "system": system_message,
            "prompt": prompt,
            "stream": False
        }

        print("PROMPT::\n", prompt)
        response = requests.post(self.base_url + "/api/generate", json=payload)
        response.raise_for_status()
        response_json = response.json()
        print("RESPONSE::\n", response_json)

        # Adjusting to access the content based on Ollama response format
        return response_json.get('response', {})

    def stream_response(self, prompt, system_message):
        data = {
            "model": self.model_name,
            "system": system_message,
            "prompt": prompt,
            "stream": True
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
                        message = json.loads(data).get('response', {})
                        yield message


def main():
    base_url = "http://localhost:11434"
    model_name = "qwen2.5:3b"

    # Initialize the OllamaLLM with the base URL and model name
    ollama_llm = OllamaLLM(base_url, model_name)

    # Define the system message and user prompt
    system_message = (
        "You are responsible for interpreting user requests related to retrieving the contents of a specific "
        "container and generating a JSON response based on the provided action and parameters.\n\n"
        "### Action Definition: get_container\n"
        "- **Description**: Retrieves the contents of a specific container.\n"
        "- **Parameters**:\n"
        "  - `container_id` (string): The identifier of the container.\n\n"
        "### Instructions:\n"
        "1. For any query that asks for the contents of a container, respond using the `get_container` action.\n"
        "2. Structure your response in JSON format with the specified parameters, using the example as a guide.\n"
        "3. Only use the parameters provided in the action definition.\n\n"
        "**Example**:\n"
        "- **User Query**: \"What's in container 5?\"\n"
        "- **Response**:\n"
        "  ```json\n"
        "  {\n"
        "    \"action\": \"get_container\",\n"
        "    \"parameters\": {\n"
        "      \"container_id\": \"5\"\n"
        "    }\n"
        "  }\n"
        "  ```\n\n"
        "Please generate JSON responses for queries related to retrieving container contents according to this format."
    )
    user_prompt = "Can you tell me what's in container 5?"

    # Generate a response from the Ollama LLM
    response = ollama_llm.generate_response(user_prompt, system_message)

    # Print the final response
    print("Final Response:\n", response)


if __name__ == "__main__":
    main()
