from flask import jsonify
import os
import json


class ChatHandler:
    def __init__(self, sessions, sessions_file_path='sessions.json'):
        self.sessions = sessions
        self.sessions_file_path = sessions_file_path

    def save_sessions_to_file(self):
        with open(self.sessions_file_path, 'w') as file:
            json.dump(self.sessions, file)
        print("Sessions successfully saved to file.")

    def load_sessions_from_file(self):
        if os.path.exists(self.sessions_file_path):
            with open(self.sessions_file_path, 'r') as file:
                self.sessions = json.load(file)
            print("Sessions successfully loaded from file.")
        else:
            print("Sessions file not found, starting with an empty session.")

    def send_message(self, session_id, content):
        self.sessions[session_id].append({
            "prompt": "Agent",
            "response": content
        })
        self.save_sessions_to_file()
        print(f"Sending message to user: {content}")
        return {"status": "200"}

    def poll_response(self, session_id):
        if len(self.sessions[session_id]) == 0:
            return '', 204

        latest_response = self.sessions[session_id][-1]
        return jsonify(latest_response)

    def start_session(self):
        session_id = os.urandom(16).hex()
        self.sessions[session_id] = []
        self.save_sessions_to_file()
        return jsonify({"session_id": session_id})

    def end_session(self, session_id):
        del self.sessions[session_id]
        self.save_sessions_to_file()
        return jsonify({"message": "Session ended successfully"})
