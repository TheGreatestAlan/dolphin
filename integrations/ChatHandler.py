import json
import os

from flask import jsonify
from langchain import ContextManager, Memory, SlidingWindowContext

from FunctionResponse import FunctionResponse, Status


class ChatHandler:
    def __init__(self, sessions, sessions_file_path='sessions.json'):
        self.sessions = sessions
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.stream_listeners = {}
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages

        # Initialize LangChain Context Manager and Memory
        self.context_manager = ContextManager(memory=Memory())
        self.context_window = SlidingWindowContext(window_size=5)  # Adjust window size as needed

        self.load_sessions_from_file()

    def save_sessions_to_file(self):
        try:
            with open(self.sessions_file_path, 'w') as file:
                json.dump(self.sessions, file)
            print("Sessions successfully saved to file.")
        except IOError as err:
            print(f"Error saving sessions to file: {err}")

    def load_sessions_from_file(self):
        try:
            if not os.path.exists(self.sessions_file_path):
                self.sessions = {}
                self.save_sessions_to_file()

            with open(self.sessions_file_path, 'r') as file:
                self.sessions = json.load(file)
            print("Sessions successfully loaded from file.")
        except (IOError, json.JSONDecodeError) as err:
            print(f"Error loading sessions from file: {err}")
            self.sessions = {}  # Reset or initialize if loading fails

    def cache_result(self, session_id, content):
        if session_id not in self.result_cache:
            self.result_cache[session_id] = ""
        self.result_cache[session_id] += "\n" + content.response

    def send_message(self, session_id, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Retrieve the cached result for the session, if any
        full_content = content
        if session_id in self.result_cache:
            full_content += f"\nResult: {self.result_cache[session_id]}"
            del self.result_cache[session_id]

        self.sessions[session_id].append({"message": full_content})
        self.save_sessions_to_file()
        print(f"Sending message to user: {full_content}")
        return FunctionResponse(Status.SUCCESS, "completed")

    def poll_response(self, session_id):
        if len(self.sessions.get(session_id, [])) == 0:
            return jsonify({}), 204  # Ensure valid JSON response

        latest_response = self.sessions[session_id][-1]
        return jsonify(latest_response)

    def receive_stream_data(self, session_id, data_chunk, message_id):
        """Process received stream data by appending to session and notifying listeners."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Initialize buffer if not present
        if message_id not in self.temp_buffers:
            self.temp_buffers[message_id] = ""

        # Accumulate chunks to buffer
        self.temp_buffers[message_id] += data_chunk

        # Strip the buffer of whitespace and check for end marker
        if self.temp_buffers[message_id].strip().endswith("[DONE]"):
            complete_message = self.temp_buffers.pop(message_id).replace("[DONE]", "").strip()
            self.sessions[session_id].append({"message": complete_message})
            self.finalize_message(session_id, complete_message)

        self.notify_listeners(session_id, data_chunk)

    def finalize_message(self, session_id, message):
        """Finalize the message and update the context."""
        self.context_manager.add_to_context(session_id, message)  # Add complete message to context

    def notify_listeners(self, session_id, data):
        """Notify listeners with the given data."""
        if session_id in self.stream_listeners:
            formatted_data = f"data: {json.dumps({'message': data})}\n\n"
            for listener in self.stream_listeners[session_id]:
                listener.send(formatted_data)  # Ensure listener.send handles SSE formatting

    def register_listener(self, session_id, listener):
        self.stream_listeners[session_id] = listener

    def unregister_listener(self, session_id):
        if session_id in self.stream_listeners:
            del self.stream_listeners[session_id]

    def start_session(self):
        session_id = os.urandom(16).hex()
        self.sessions[session_id] = []
        self.save_sessions_to_file()
        self.context_manager.create_context(session_id)  # Initialize context for the session
        return session_id

    def end_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions_to_file()
            self.context_manager.clear_context(session_id)  # Clear context for the session
        return True
