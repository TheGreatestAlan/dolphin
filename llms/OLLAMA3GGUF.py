import requests
import json

class Ollama3LLM:
    def __init__(self, server_url):
        self.server_url = server_url
        self.conversations = {}

    def generate_response(self, prompt, system_message):
        payload = {
            "prompt": prompt,
            "system_message": system_message,
            "max_new_tokens": 1500,  # Equivalent to max_length
            "temperature": 0.7,
            "top_p": 0.9
        }

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(f"{self.server_url}/api/v1/generate", headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            generated_text = response_data.get("results", [{}])[0].get("text", "").strip()
            return generated_text
        else:
            raise Exception(f"Failed to generate response: {response.status_code}, {response.text}")

    def generate_response_with_session(self, prompt, system_message, session_id):
        # Initialize session if not already present
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        # Add the new prompt to the conversation history
        self.conversations[session_id].append(f"User: {prompt}")

        # Construct the full context by joining the session history
        context = "\n".join(self.conversations[session_id])

        payload = {
            "prompt": context,
            "system_message": system_message,
            "max_new_tokens": 1500,  # Equivalent to max_length
            "temperature": 0.7,
            "top_p": 0.9
        }

        headers = {
            'Content-Type': 'application/json'
        }

        # Send the request with the full context
        response = requests.post(f"{self.server_url}/api/v1/generate", headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            response_data = response.json()
            generated_text = response_data.get("results", [{}])[0].get("text", "").strip()

            # Add the generated response to the conversation history
            self.conversations[session_id].append(f"Bot: {generated_text}")
            return generated_text
        else:
            raise Exception(f"Failed to generate response: {response.status_code}, {response.text}")

    def create_session(self):
        # Create a new session ID based on the number of existing sessions
        session_id = str(len(self.conversations) + 1)
        self.conversations[session_id] = []
        return session_id

# Example usage
if __name__ == "__main__":
    server_url = "http://localhost:5001"  # Adjust this to your local KoboldCpp server URL
    ollama3 = Ollama3LLM(server_url)

    # Using the single response method
    prompt = "In this list, where is the turmeric? 1: turmeric, parsley 2: pencils 3: phone 4: food"
    system_message = "You are an inventory scanning guy."
    response = ollama3.generate_response(prompt, system_message)
    print(response)

    # Using the session-based response method
    session_id = ollama3.create_session()
    print(f"Session ID: {session_id}")
    response_with_session = ollama3.generate_response_with_session(prompt, system_message, session_id)
    print(response_with_session)
