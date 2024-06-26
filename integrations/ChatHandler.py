import json
import os
import time

from flask import jsonify, Response
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from FunctionResponse import FunctionResponse, Status


class ChatHandler:
    def __init__(self, sessions_file_path='sessions.json'):
        self.sessions = {}
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.stream_listeners = {}
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages
        self.memories = {}  # Dictionary to hold ConversationBufferMemory instances

        self.load_sessions_from_file()

    def save_sessions_to_file(self):
        try:
            # Save sessions and their corresponding memory states
            sessions_data = {
                "sessions": self.sessions,
                "memories": {session_id: self.serialize_memory(memory) for session_id, memory in self.memories.items()}
            }
            with open(self.sessions_file_path, 'w') as file:
                json.dump(sessions_data, file)
            print("Sessions successfully saved to file.")
        except IOError as err:
            print(f"Error saving sessions to file: {err}")

    def load_sessions_from_file(self):
        try:
            if not os.path.exists(self.sessions_file_path):
                self.sessions = {}
                self.save_sessions_to_file()

            with open(self.sessions_file_path, 'r') as file:
                sessions_data = json.load(file)
                self.sessions = sessions_data.get("sessions", {})
                # Load memory states for each session
                self.memories = {
                    session_id: self.deserialize_memory(memory_data)
                    for session_id, memory_data in sessions_data.get("memories", {}).items()
                }
            print("Sessions successfully loaded from file.")
        except (IOError, json.JSONDecodeError) as err:
            print(f"Error loading sessions from file: {err}")
            self.sessions = {}  # Reset or initialize if loading fails

    def cache_result(self, session_id, content):
        if session_id not in self.result_cache:
            self.result_cache[session_id] = ""
        self.result_cache[session_id] += "\n" + content.response

    def send_message(self, session_id, content, role="AI"):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Retrieve the cached result for the session, if any
        full_content = content
        if session_id in self.result_cache:
            full_content += f"\nResult: {self.result_cache[session_id]}"
            del self.result_cache[session_id]

        if full_content.strip():
            self.sessions[session_id].append({"role": role, "message": full_content})
            self.save_sessions_to_file()
            print(f"Sending message to user: {full_content}")
            return FunctionResponse(Status.SUCCESS, "completed")
        else:
            print("Ignored empty message.")

    def poll_response(self, session_id):
        if len(self.sessions.get(session_id, [])) == 0:
            return jsonify({}), 204  # Ensure valid JSON response

        latest_response = self.sessions[session_id][-1]
        return jsonify(latest_response)

    def listen_to_stream(self, session_id):
        """Continuously listen and stream data for a given session."""

        def generate():
            yield "data: {\"message\": \"Connection established.\"}\n\n"
            last_index = -1  # Initialize to indicate no messages sent yet

            while True:
                # Stream from the completed messages
                if session_id in self.sessions:
                    session_messages = self.sessions[session_id]
                    while last_index < len(session_messages) - 1:
                        last_index += 1
                        message = session_messages[last_index]
                        yield f"data: {json.dumps(message)}\n\n"

                # Stream live data from the buffer
                if session_id in self.temp_buffers:
                    buffer = self.temp_buffers[session_id]
                    if buffer:
                        yield f"data: {json.dumps({'message': buffer})}\n\n"

                time.sleep(1)

        return Response(generate(), mimetype='text/event-stream')

    def receive_stream_data(self, session_id, data_chunk,message_id, role="AI"):
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
            self.sessions[session_id].append({"role": role, "message": complete_message})
            self.finalize_message(session_id, complete_message, role)

        self.notify_listeners(session_id, data_chunk)

    def finalize_message(self, session_id, message, role):
        """Finalize the message and update the context."""
        if message:
            if session_id in self.memories:
                if role == "Human":
                    self.memories[session_id].save_context({"input": message}, {"output": ""})
                elif role == "AI":
                    self.memories[session_id].save_context({"input": ""}, {"output": message})

    def store_human_context(self, session_id, message):
        """Store a human message in the session context from an external source."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        if message.strip():
            self.sessions[session_id].append({"role": "Human", "message": message})
            self.finalize_message(session_id, message, "Human")
            self.save_sessions_to_file()
        else:
            print("Ignored empty human message.")

    def get_context(self, session_id):
        """Retrieve the message context for a given session."""
        if session_id in self.memories:
            context = self.memories[session_id].load_memory_variables({})
            return context.get("history", "")
        return ""

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
        self.memories[session_id] = ConversationBufferMemory()  # Create memory for new session
        return session_id

    def end_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions_to_file()
            if session_id in self.memories:
                del self.memories[session_id]  # Remove memory for ended session
        return True

    def serialize_memory(self, memory):
        """Serialize a ConversationBufferMemory to a dictionary."""
        messages = [{"role": "Human" if isinstance(msg, HumanMessage) else "AI", "content": msg.content}
                    for msg in memory.chat_memory.messages]
        return {"messages": messages}

    def deserialize_memory(self, memory_data):
        """Deserialize a dictionary to a ConversationBufferMemory."""
        memory = ConversationBufferMemory()
        for msg in memory_data.get("messages", []):
            if msg["role"] == "Human":
                memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
            else:
                memory.chat_memory.add_message(AIMessage(content=msg["content"]))
        return memory
