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
            return "Message sent to agent."
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to send message: {e}")

    def poll_response(self):
        if not self.session_id:
            raise Exception("No active session")

        try:
            response = requests.get(f"{self.agent_url}/poll_response", params={'session_id': self.session_id})
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to poll response: {e}")

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
