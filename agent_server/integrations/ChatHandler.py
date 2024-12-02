import json
import os

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

from agent_server.FunctionResponse import FunctionResponse, Status
from agent_server.integrations.StreamManager import StreamManager
from agent_server.llms.LLMInterface import LLMInterface


class ChatHandler:
    def __init__(self,stream_manager: StreamManager, sessions_file_path='sessions.json'):
        self.users = {}
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.memories = {}  # Dictionary to hold ConversationBufferMemory instances
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages by session_id
        self.load_sessions_from_file()
        self.stream_manager = stream_manager

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
            print(f"Sessions successfully saved to file at {self.sessions_file_path}.")
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

    def send_message(self, session_id, content, role="AI"):
        if session_id not in self.users:
            self.users[session_id] = []

        # Retrieve the cached result for the session, if any
        full_content = content
        if session_id in self.result_cache:
            full_content += f"\nResult: {self.result_cache[session_id]}"
            del self.result_cache[session_id]

        if full_content.strip():
            self.users[session_id].append({"role": role, "message": full_content})
            self.save_sessions_to_file()
            print(f"Sending message to user: {full_content}")
            return FunctionResponse(Status.SUCCESS, "completed")
        else:
            print("Ignored empty message.")

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
        else:
            # Send the chunk to the user
            self.send_message(username, data_chunk)


    def finalize_message(self, username, message, role):
        """Finalize the message and update the context."""
        if message:
            if username in self.memories:
                if role == "Human":
                    print("HUMAN:" + message)
                    self.memories[username].save_context({"input": message}, {"output": ""})
                elif role == "AI":
                    print("AI:" + message)
                    self.memories[username].save_context({"input": ""}, {"output": message})

    def store_human_context(self, username, message):
        """Store a human message in the session context from an external source."""
        if username not in self.users:
            self.users[username] = []

        if message.strip():
            self.users[username].append({"role": "Human", "message": message})
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
