import time
import json
from flask import Response
import sounddevice as sd
from tts.OpenAITTS import OpenAITTS
import threading

class StreamManager:
    def __init__(self):
        self.text_buffers = {}  # Dictionary to hold stream buffers for messages by session_id
        self.tts_instances = {}  # Dictionary to hold OpenAITTS instances by session_id
        self.stream_listeners = {}  # Dictionary to hold listeners for text streams
        self.audio_listeners = {}  # Dictionary to hold listeners for audio streams

    def add_to_text_buffer(self, session_id, data_chunk):
        """Add a data chunk to the stream buffer for a specific session if a listener is subscribed."""
        if session_id in self.text_buffers:
            self.text_buffers[session_id].append(data_chunk)
        # If there is no buffer, do nothing

    def get_tts_instance(self, session_id):
        """Retrieve or create an OpenAITTS instance for the session."""
        if session_id not in self.tts_instances:
            self.tts_instances[session_id] = OpenAITTS()
        return self.tts_instances[session_id]

    def start_text_streaming(self, session_id):
        """Continuously stream text data for a given session."""
        def generate():
            while True:
                if session_id in self.text_buffers:
                    while self.text_buffers[session_id]:
                        text_chunk = self.text_buffers[session_id].pop(0)
                        yield f"data: {json.dumps({'message': text_chunk})}\n\n"
                time.sleep(1)

        return Response(generate(), mimetype='text/event-stream')

    def listen_to_text_stream(self, session_id):
        """Listen to the text stream and start streaming text data for the given session."""
        if session_id in self.stream_listeners:
            return self.start_text_streaming(session_id)
        # Do nothing if there are no listeners

    def listen_to_audio_stream(self, session_id):
        """Return the audio buffer of the OpenAITTS instance for a given session."""
        tts_instance = self.get_tts_instance(session_id)
        return tts_instance.get_audio_buffer()

    def unregister_listener(self, session_id, listener, audio=False):
        """Unregister a listener from a specific session's text or audio stream."""
        listeners = self.audio_listeners if audio else self.stream_listeners
        if session_id in listeners and listener in listeners[session_id]:
            listeners[session_id].remove(listener)
            if not listeners[session_id]:  # Remove session if no listeners remain
                del listeners[session_id]
                self.cleanup_resources(session_id, audio)

    def cleanup_resources(self, session_id, audio=False):
        """Clean up resources like buffers and TTS instances if no listeners remain."""
        if audio:
            if session_id in self.tts_instances:
                self.tts_instances[session_id].stop()  # Stop the OpenAITTS instance
                del self.tts_instances[session_id]
        else:
            if session_id in self.text_buffers:
                del self.text_buffers[session_id]  # Remove text buffer
        if session_id in self.temp_buffers:
            del self.temp_buffers[session_id]  # Remove temp buffer

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
            self.add_completed_message(session_id, complete_message)

    def receive_stream_data(self, session_id, data_chunk, message_id):
        """Receive and parse stream data for LLM, accumulating until a complete message is received."""
        self.parse_llm_stream(session_id, data_chunk, message_id)

