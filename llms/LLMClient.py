import requests

from llms.LLMInterface import LLMInterface


class LLMClient(LLMInterface):
    def __init__(self, base_url):
        self.base_url = base_url

    def start_session(self, prompt, system_message):
        payload = {
            "prompt": prompt,
            "system_message": system_message
        }
        response = requests.post(f"{self.base_url}/api/v1/session", json=payload)
        response.raise_for_status()
        return response.json().get('session_id')

    def generate_response(self, session_id, prompt, system_message):
        payload = {
            "prompt": prompt,
            "system_message": system_message,
            "session_id": session_id
        }
        response = requests.post(f"{self.base_url}/api/v1/generate", json=payload)
        response.raise_for_status()
        return response.json()

    def end_session(self, session_id):
        response = requests.delete(f"{self.base_url}/api/v1/session/{session_id}")
        response.raise_for_status()
