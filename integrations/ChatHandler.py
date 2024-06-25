import json
import os
import threading
import time

from flask import jsonify

from FunctionResponse import FunctionResponse, Status

class ChatHandler:
    def __init__(self, sessions, sessions_file_path='sessions.json'):
        self.sessions = sessions
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.stream_listeners = {}  # Ensure this is initialized


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

    def receive_stream_data(self, session_id, data_chunk):
        """Process received stream data by appending to session and notifying listeners."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"message": data_chunk})
        self.notify_listeners(session_id, data_chunk)

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
        return session_id

    def end_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions_to_file()
        return True

    def start_counting(self, session_id):
        """Starts a background task that sends each count message individually to simulate real-time data generation."""

        def count():
            # First phase of counting from 1 to 10
            for i in range(1, 11):
                self.receive_stream_data(session_id, f"Count: {i}")
                time.sleep(0.5)  # Sleep for 0.5 seconds between counts

            # Send a done message after the first phase
            self.receive_stream_data(session_id, "[DONE]")

            # Pause between phases
            time.sleep(2)  # Optional pause between the two phases

            # Second phase of counting from 11 onwards
            for i in range(11, 21):
                self.receive_stream_data(session_id, f"Count: {i}")
                time.sleep(2)  # Sleep for 2 seconds between counts

            # Send a done message after the second phase
            self.receive_stream_data(session_id, "[DONE]")

            # Continue counting beyond 20 if necessary
            count_index = 21
            while True:  # Or some condition to stop
                self.receive_stream_data(session_id, f"Count: {count_index}")
                count_index += 1
                time.sleep(2)  # Continue with 2 second intervals
                if count_index % 10 == 0:  # Every ten counts, send a done message
                    self.receive_stream_data(session_id, "[DONE]")

        thread = threading.Thread(target=count)
        thread.daemon = True  # Set as a daemon so it won't prevent the program from exiting
        thread.start()
