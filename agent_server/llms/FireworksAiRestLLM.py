import requests

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
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def stream_response(self, prompt, system_message):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,
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
                            yield data
            else:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")

def main():
     api_token = "your_fireworks_api_token"
     model = FireworksAiRestLLM(api_token)
     print(model.generate_response("What is the weather today?", "You are a helpful assistant."))

if __name__ == "__main__":
    main()
