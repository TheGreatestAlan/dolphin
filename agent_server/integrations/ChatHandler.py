import json
import os

from agent_server.integrations.StreamManager import StreamManager

from typing import Final
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from agent_server.llms.LLMInterface import LLMInterface

class ChatSession:
    def __init__(self, username: Final[str], session_id: Final[str], chat_handler):
        self._username: Final[str] = username
        self._session_id: Final[str] = session_id
        self._chat_handler = chat_handler

    def parse_llm_stream(self, data_chunk, message_id):
        if self._username not in self._chat_handler.temp_buffers:
            self._chat_handler.temp_buffers[self._username] = {}

        if message_id not in self._chat_handler.temp_buffers[self._username]:
            self._chat_handler.temp_buffers[self._username][message_id] = ""

        # Accumulate data chunk
        self._chat_handler.temp_buffers[self._username][message_id] += data_chunk

        self._chat_handler.stream_manager.add_to_text_buffer(self._session_id, data_chunk)
        # Check for END_STREAM token
        if self._chat_handler.temp_buffers[self._username][message_id].strip().endswith(LLMInterface.END_STREAM):
            complete_message = self._chat_handler.temp_buffers[self._username][message_id].replace(LLMInterface.END_STREAM, "").strip()
            self._chat_handler.temp_buffers[self._username].pop(message_id)  # Clear the temp buffer for this message

            # Finalize the message
            self.finalize_message(complete_message, "AI")

            # If no more messages remain for this session, clear the session from temp_buffers
            if not self._chat_handler.temp_buffers[self._username]:
                self._chat_handler.temp_buffers.pop(self._username)

    def finalize_message(self, message, role):
        if message:
            timestamp = datetime.utcnow().isoformat() + 'Z'
            if self._username in self._chat_handler.memories:
                if role == "Human":
                    print(f"HUMAN ({timestamp}): {message}")
                    self._chat_handler.memories[self._username].save_context({"input": message}, {"output": ""})
                elif role == "AI":
                    print(f"AI ({timestamp}): {message}")
                    self._chat_handler.memories[self._username].save_context({"input": ""}, {"output": message})

    def store_human_context(self, message):
        if self._username not in self._chat_handler.users:
            self._chat_handler.users[self._username] = []

        if message.strip():
            timestamp = datetime.utcnow().isoformat() + 'Z'
            self._chat_handler.users[self._username].append({"role": "Human", "message": message, "timestamp": timestamp})
            self.finalize_message(message, "Human")
            self._chat_handler.save_sessions_to_file()
        else:
            print("Ignored empty human message.")

    def get_context(self):
        if self._username in self._chat_handler.memories:
            context = self._chat_handler.memories[self._username].load_memory_variables({})
            return context.get("history", "")
        return ""

    def get_full_session_history(self):
        return self._chat_handler.users.get(self._username, [])

    def get_current_chat(self):
        session_history = self._chat_handler.users.get(self._username, [])
        if not session_history:
            return []

        current_chat = [session_history[0]]
        for i in range(1, len(session_history)):
            prev_message = session_history[i - 1]
            current_message = session_history[i]

            # Handle missing timestamps gracefully
            if "timestamp" not in prev_message or "timestamp" not in current_message:
                continue

            prev_timestamp = datetime.fromisoformat(prev_message["timestamp"].rstrip('Z'))
            current_timestamp = datetime.fromisoformat(current_message["timestamp"].rstrip('Z'))
            time_diff = current_timestamp - prev_timestamp
            if time_diff <= timedelta(minutes=30):
                current_chat.append(current_message)
            else:
                current_chat = [current_message]

        print(current_chat)
        return current_chat

class ChatHandler:
    def __init__(self,stream_manager: StreamManager, sessions_file_path='sessions.json'):
        self.users = {}
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.memories = {}  # Dictionary to hold ConversationBufferMemory instances
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages by session_id
        self.load_sessions_from_file()
        self.stream_manager = stream_manager

    def create_session(self, username, session_id):
        self.get_or_create_user(username)
        return ChatSession(username, session_id, self)

    def save_sessions_to_file(self):
        try:
            sessions_data = {
                "sessions": self.users,
                "memories": {session_id: self.serialize_memory(memory) for session_id, memory in
                             self.memories.items()}
            }
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.sessions_file_path), exist_ok=True)
            with open(self.sessions_file_path, 'w') as file:
                json.dump(sessions_data, file)
        except IOError as err:
            print(f"Error saving sessions to file: {err}")

    def load_sessions_from_file(self):
        try:
            if not os.path.exists(self.sessions_file_path):
                # File doesn't exist, initialize and create it
                print(f"File {self.sessions_file_path} not found. Creating a new one.")
                self.users = {}
                self.save_sessions_to_file()
                return

            with open(self.sessions_file_path, 'r') as file:
                sessions_data = json.load(file)
                self.users = sessions_data.get("sessions", {})
                self.memories = {
                    session_id: self.deserialize_memory(memory_data)
                    for session_id, memory_data in sessions_data.get("memories", {}).items()
                }
            print("Sessions successfully loaded from file.")
        except (IOError, json.JSONDecodeError) as err:
            print(f"Error loading sessions from file: {err}")

    def cache_result(self, session_id, content):
        if session_id not in self.result_cache:
            self.result_cache[session_id] = ""
        self.result_cache[session_id] += "\n" + content.response

    def parse_llm_stream(self, username, session_id, data_chunk, message_id):
        """Parse incoming LLM stream data and immediately send it back without waiting for END_STREAM."""
        if username not in self.temp_buffers:
            self.temp_buffers[username] = {}

        if message_id not in self.temp_buffers[username]:
            self.temp_buffers[username][message_id] = ""

        # Accumulate data chunk
        self.temp_buffers[username][message_id] += data_chunk

        self.stream_manager.add_to_text_buffer(session_id, data_chunk)
        # Check for END_STREAM token
        if self.temp_buffers[username][message_id].strip().endswith(LLMInterface.END_STREAM):
            complete_message = self.temp_buffers[username][message_id].replace(LLMInterface.END_STREAM, "").strip()
            self.temp_buffers[username].pop(message_id)  # Clear the temp buffer for this message

            # Finalize the message
            self.finalize_message(username, complete_message, "AI")

            # If no more messages remain for this session, clear the session from temp_buffers
            if not self.temp_buffers[username]:
                self.temp_buffers.pop(username)

    def finalize_message(self, username, message, role):
        """Finalize the message and update the context."""
        if message:
            timestamp = datetime.utcnow().isoformat() + 'Z'
            if username in self.memories:
                if role == "Human":
                    print(f"HUMAN ({timestamp}): {message}")
                    self.memories[username].save_context({"input": message}, {"output": ""})
                elif role == "AI":
                    print(f"AI ({timestamp}): {message}")
                    self.memories[username].save_context({"input": ""}, {"output": message})

    def store_human_context(self, username, message):
        """Store a human message in the session context from an external source."""
        if username not in self.users:
            self.users[username] = []

        if message.strip():
            timestamp = datetime.utcnow().isoformat() + 'Z'
            self.users[username].append({"role": "Human", "message": message, "timestamp": timestamp})
            self.finalize_message(username, message, "Human")
            self.save_sessions_to_file()
        else:
            print("Ignored empty human message.")

    def get_context(self, username):
        """Retrieve the message context for a given session."""
        if username in self.memories:
            context = self.memories[username].load_memory_variables({})
            return context.get("history", "")
        return ""

    def get_full_session_history(self, username):
        """Retrieve the entire conversation history for a given user."""
        return self.users.get(username, [])

    def get_current_chat(self, username):
        """Retrieve the current chat where there is no time gap of greater than 30 minutes between messages."""
        session_history = self.users.get(username, [])
        if not session_history:
            return []

        current_chat = [session_history[0]]
        for i in range(1, len(session_history)):
            prev_message = session_history[i - 1]
            current_message = session_history[i]

            # Handle missing timestamps gracefully
            if "timestamp" not in prev_message or "timestamp" not in current_message:
                continue

            prev_timestamp = datetime.fromisoformat(prev_message["timestamp"].rstrip('Z'))
            current_timestamp = datetime.fromisoformat(current_message["timestamp"].rstrip('Z'))
            time_diff = current_timestamp - prev_timestamp
            if time_diff <= timedelta(minutes=30):
                current_chat.append(current_message)
            else:
                current_chat = [current_message]

        print(current_chat)
        return current_chat

    def get_or_create_user(self, username):
        if username and username in self.users:
            # If session already exists, return the existing session ID and pick up where it left off
            print(f"Resuming session: {username}")
            return username

        self.users[username] = []
        self.save_sessions_to_file()
        self.memories[username] = ConversationBufferMemory()  # Create memory for new session
        print(f"Starting new session: {username}")
        return username

    def end_session(self, session_id):
        if session_id in self.users:
            del self.users[session_id]
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
