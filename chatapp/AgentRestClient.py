import json

import requests


class AgentRestClient:
    def __init__(self, agent_url):
        self.agent_url = agent_url
        self.session_id = None

    def start_session(self):
        try:
            response = requests.post(f"{self.agent_url}/start_session")
            response.raise_for_status()
            jsonRes = response.json()
            self.session_id = jsonRes['session_id']
            return self.session_id
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to start session: {e}")

    def send_prompt(self, prompt):
        if not self.session_id:
            raise Exception("No active session")

        payload = {
            'session_id': self.session_id,
            'user_message': prompt
        }

        try:
            response = requests.post(f"{self.agent_url}/message_agent", json=payload)
            response.raise_for_status()
            return "Message sent to chatapp."
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to send message: {e}")

    def stream_response(self):
        if not self.session_id:
            raise Exception("No active session")

        try:
            response = requests.get(f"{self.agent_url}/stream/{self.session_id}", stream=True)
            response.raise_for_status()
            yield from self._process_stream(response)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to stream response: {e}")


    def _process_stream(self, response):
        for chunk in response.iter_lines():
            if chunk:
                chunk_decoded = chunk.decode('utf-8')
                try:
                    data = json.loads(chunk_decoded.split("data: ")[1])
                    message = data.get("message")
                    if message:
                        yield message
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Failed to parse chunk: {e}")

    def end_session(self):
        if not self.session_id:
            raise Exception("No active session")

        payload = {
            'session_id': self.session_id
        }

        try:
            response = requests.delete(f"{self.agent_url}/end_session", json=payload)
            response.raise_for_status()
            self.session_id = None
            return "Session ended."
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to end session: {e}")


# Example usage:
if __name__ == '__main__':
    agent_client = AgentRestClient("http://127.0.0.1:5000")
    session_id = agent_client.start_session()
    print(f"Started session: {session_id}")

    user_message = "Hello, how are you?"
    print(f"Sending message: {user_message}")
    response = agent_client.send_prompt(user_message)
    print(f"Response: {response}")

    print("Streaming response:")
    for chunk in agent_client.stream_response():
        print(chunk, end='', flush=True)

    agent_client.end_session()
    print("Session ended.")
