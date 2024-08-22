import json
import os

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

from FunctionResponse import FunctionResponse, Status
from integrations.StreamManager import StreamManager


#HEMMINGWAY BRIDGE:
# OK you finished the StreamManager piece, and you think the concerns have been separated
# Now you need to go into the chat handler, figure out exactly what it's doing, and make
# sure that it's passing back the right things.  I believe it'll handle a lot of the
# history, and role stuff. After that you'll need to look into perhaps some sort of
# notifier in the flast rest stuff and see how you're going to go from a queue to
# actually streaming over a rest service
class ChatHandler:
    def __init__(self, sessions_file_path='sessions.json'):
        self.sessions = {}
        self.sessions_file_path = sessions_file_path
        self.result_cache = {}
        self.memories = {}  # Dictionary to hold ConversationBufferMemory instances
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages by session_id

        #model_name = "tts_models/en/jenny/jenny"
        #tts_handler = CoquiTTSHandler(model_name)
        self.stream_manager = StreamManager()  # Initialize the StreamManager

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

    def listen_to_text_stream(self, session_id):
        return self.stream_manager.listen_to_text_stream(session_id)

    def listen_to_audio_stream(self, session_id):
        return self.stream_manager.listen_to_audio_stream(session_id)

    def receive_stream_data(self, session_id, data_chunk, message_id, role="AI"):
        """Process received stream data by appending to session and notifying listeners."""
        self.stream_manager.receive_stream_data(session_id, data_chunk, message_id, role)

    def parse_llm_stream(self, session_id, data_chunk, message_id):
        """Parse incoming LLM stream data and accumulate it until the complete message is received."""
        if session_id not in self.temp_buffers:
            self.temp_buffers[session_id] = {}

        if message_id not in self.temp_buffers[session_id]:
            self.temp_buffers[session_id][message_id] = ""

        self.temp_buffers[session_id][message_id] += data_chunk

        # Check if the message is complete
        if self.temp_buffers[session_id][message_id].strip().endswith("[DONE]"):
            complete_message = self.temp_buffers[session_id][message_id].replace("[DONE]", "").strip()
            self.temp_buffers[session_id].pop(message_id)  # Clear the temp buffer
            self.stream_manager.add_completed_message(session_id, complete_message)

    def finalize_message(self, session_id, message, role):
        """Finalize the message and update the context."""
        if message:
            if session_id in self.memories:
                if role == "Human":
                    print("HUMAN:" + message)
                    self.memories[session_id].save_context({"input": message}, {"output": ""})
                elif role == "AI":
                    print("AI:" + message)
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


import threading
import sounddevice as sd

if __name__ == "__main__":
    # Instantiate the ChatHandler
    chat_handler = ChatHandler()

    # Start a new session
    session_id = chat_handler.start_session()

    # Define a player function that will continuously listen and play audio samples
    def audio_player():
        for audio_sample, sample_rate in chat_handler.listen_to_audio_stream(session_id):
            if audio_sample is not None:
                print("Playing audio sample...")
                sd.play(audio_sample, samplerate=24000)
                sd.wait()
            else:
                break

    # Start the audio player in a separate thread
    player_thread = threading.Thread(target=audio_player)
    player_thread.start()

    # Send a test sentence to be streamed as audio
    test_sentence = "This is a test sentence for streaming text-to-speech."
    chat_handler.send_message(session_id, test_sentence)

    # Simulate receiving and streaming the audio
    chat_handler.receive_stream_data(session_id, test_sentence, message_id="test_message_id")

    # Finalize the message
    chat_handler.finalize_message(session_id, test_sentence, role="AI")

    print(f"Test sentence '{test_sentence}' has been processed and streamed as audio.")

    # Wait for the player thread to finish
    player_thread.join()
    print("Audio playback complete.")