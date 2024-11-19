import json
import requests
from openai import OpenAI

from agent_server.llms.LLMInterface import LLMInterface

class OptiLLM(LLMInterface):
    def __init__(self, model="fireworks_ai/accounts/fireworks/models/qwen2p5-72b-instruct"):
        self.model = model
        self.url = "http://localhost:8000/v1/"
        self.client = OpenAI(api_key="api_key", base_url=self.url)

    def generate_response(self, prompt, system_message):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            extra_body={"optillm_approach": "moa"}
        )
        # Extract the assistant's message content
        assistant_message = response.choices[0].message.content
        return assistant_message

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


def main():
    model = OptiLLM()

    prompt = "tell me a story about Hasan Piker"
    system_message = "You are a helpful assistant."

    try:
        response = model.generate_response(prompt, system_message)
        print(response)  # Print the assistant's response
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()