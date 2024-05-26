import os
import requests
import json

class ChatGPTCorrector:
    def __init__(self):
        self.api_key = os.getenv('OPEN_AI_API_KEY')
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def call_chatgpt(self, prompt):
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are to respond with a JSON format. Please ensure the JSON is correctly formatted and contains no syntax errors."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}, {response.text}"

    def correct_json(self, malformed_json):
        corrected = self.call_chatgpt(malformed_json)
        try:
            json_content = corrected['choices'][0]['message']['content']
            json_object = json.loads(json_content)  # This ensures the response is valid JSON
            return json.dumps(json_object, indent=4)  # Format and return the JSON
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return f"Failed to validate JSON: {e}"

# Example usage
malformed_json = "{\"name\": \"add_items\", \"arguments\": '{\"container_id\": \"13\", \"items\": [\"pencil\", \"pencil\"]}\"}";

corrector = ChatGPTCorrector()
result = corrector.correct_json(malformed_json)
print(result)
