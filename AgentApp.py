from flask import Flask, request, jsonify
import os
import logging

from llms.OLLAMA3GGUF import Ollama3LLM


class AgentApp:
    def __init__(self, model_server_url, system_message_file, host='0.0.0.0', port=5000):
        self.model = Ollama3LLM(model_server_url)
        self.system_message_file = system_message_file
        self.app = Flask(__name__)
        self.host = host
        self.port = port

        # Configure logging
        logging.basicConfig(level=logging.INFO)

        # Define endpoints
        self.app.add_url_rule('/create_session', 'create_session', self.create_session, methods=['POST'])
        self.app.add_url_rule('/generate', 'generate', self.generate_response, methods=['POST'])

    def read_system_message(self):
        logging.info(f"Reading system message from {self.system_message_file}")
        if os.path.exists(self.system_message_file):
            with open(self.system_message_file, 'r') as file:
                return file.read().strip()
        else:
            raise FileNotFoundError(f"System message file {self.system_message_file} not found.")

    def create_session(self):
        session_id = self.model.create_session()
        return jsonify({"session_id": session_id})

    def generate_response(self):
        data = request.json
        session_id = data.get('session_id')
        user_message = data.get('user_message')

        if not session_id or not user_message:
            return jsonify({"error": "session_id and user_message are required"}), 400

        try:
            system_message = self.read_system_message()
            prompt = f"{system_message}\n\nUser: {user_message}"
            response = self.model.generate_response_with_session(prompt, system_message, session_id)
            return jsonify({"response": response})
        except FileNotFoundError as e:
            logging.error(str(e))
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logging.error(str(e))
            return jsonify({"error": str(e)}), 500

    def run(self):
        self.app.run(host=self.host, port=self.port)

# Example usage
if __name__ == "__main__":
    model_server_url = "http://localhost:5001"  # Adjust this to your local KoboldCpp server URL
    system_message_file = "./prompt/SystemPrompt.txt"  # Path to your system message file
    agent_app = AgentApp(model_server_url, system_message_file)
    agent_app.run()
