import time
import json
from flask import Response
from queue import Queue
from typing import Optional

from tts.SpeechInterfaces import TTSInterface


# HEMMINGWAY BRIDGE
# 2024-08-21
# CREATE NEW STREAM MANAGER THAT SEPARATES OUT THE CONCERNS MORE.  STREAM MANAGER SHOULDN'T
# BE ADDING THINGS TO THE STREAM.  IT SHOULD JUST BE SUPPLYING AND MANAGING THE BUFFERS THAT
# HOLD THOSE THINGS.  WE ALSO NEED TO CONSIDER THAT EACH BUFFER IS PROBABLY A SEPARATE THREAD
# BUT YOU"LL NEED TO THINK THAT THROUGH.  BUT I BET A LOT OF THE CLEANUP ON THE THREADING HAPPENS
# HERE
class StreamManager:
    def __init__(self, tts: Optional[TTSInterface] = None):
        self.stream_buffers = {}  # Dictionary to hold stream buffers for messages by session_id
        self.audio_buffers = {}  # Dictionary to hold audio buffers by session_id
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages by session_id
        self.stream_listeners = {}  # Dictionary to hold listeners for text streams
        self.audio_listeners = {}  # Dictionary to hold listeners for audio streams
        self.tts = tts  # Instance of a class implementing TTSInterface

    def add_to_stream_buffer(self, session_id, data_chunk):
        """Add a data chunk to the stream buffer for a specific session."""
        if session_id not in self.stream_buffers:
            self.stream_buffers[session_id] = []
        self.stream_buffers[session_id].append(data_chunk)

    def add_to_audio_buffer(self, session_id, audio_chunk):
        """Add an audio chunk to the audio buffer for a specific session."""
        if session_id not in self.audio_buffers:
            self.audio_buffers[session_id] = []
        self.audio_buffers[session_id].append(audio_chunk)

    def start_text_streaming(self, session_id):
        """Continuously stream text data for a given session."""
        def generate():
            while True:
                if session_id in self.stream_buffers:
                    while self.stream_buffers[session_id]:
                        text_chunk = self.stream_buffers[session_id].pop(0)
                        yield f"data: {json.dumps({'message': text_chunk})}\n\n"
                time.sleep(1)

        return Response(generate(), mimetype='text/event-stream')

    def generate_audio_stream(self, session_id):
        """Generator to stream audio data for a given session."""
        chunk_size = 1024
        while True:
            if session_id in self.audio_buffers:
                while self.audio_buffers[session_id]:
                    audio_chunk = self.audio_buffers[session_id].pop(0)
                    yield audio_chunk  # Yield raw audio data directly
            time.sleep(1)

    def start_audio_streaming(self, session_id):
        """Flask route handler to stream audio data."""
        if session_id not in self.audio_buffers:
            return "Session ID not found", 404

        return Response(self.generate_audio_stream(session_id), mimetype='audio/mpeg')

    def listen_to_text_stream(self, session_id):
        """Listen to the text stream and start streaming text data for the given session."""
        return self.start_text_streaming(session_id)

    def listen_to_audio_stream(self, session_id):
        """Listen to the audio stream and start streaming audio data for the given session."""
        return self.start_audio_streaming(session_id)

    def register_listener(self, session_id, listener, audio=False):
        """Register a new listener for a specific session's text or audio stream."""
        if audio:
            if session_id not in self.audio_listeners:
                self.audio_listeners[session_id] = []
            self.audio_listeners[session_id].append(listener)
        else:
            if session_id not in self.stream_listeners:
                self.stream_listeners[session_id] = []
            self.stream_listeners[session_id].append(listener)

    def unregister_listener(self, session_id, listener, audio=False):
        """Unregister a listener from a specific session's text or audio stream."""
        listeners = self.audio_listeners if audio else self.stream_listeners
        if session_id in listeners and listener in listeners[session_id]:
            listeners[session_id].remove(listener)
            if not listeners[session_id]:  # Remove session if no listeners remain
                del listeners[session_id]

    def notify_listeners(self, session_id, data, audio=False):
        """Notify all listeners of new data for a specific session."""
        listeners = self.audio_listeners if audio else self.stream_listeners
        if session_id in listeners:
            formatted_data = f"data: {json.dumps({'message': data})}\n\n"
            for listener in listeners[session_id]:
                listener.send(formatted_data)  # Ensure listener.send handles SSE formatting

    def receive_stream_data(self, session_id, data_chunk, message_id, role="AI"):
        """Process received stream data by appending to buffers and notifying listeners."""
        if session_id not in self.temp_buffers:
            self.temp_buffers[session_id] = {}

        if message_id not in self.temp_buffers[session_id]:
            self.temp_buffers[session_id][message_id] = ""

        self.temp_buffers[session_id][message_id] += data_chunk
        self.add_to_stream_buffer(session_id, data_chunk)

        # Check if there are listeners for the audio stream and if TTS is available
        if session_id in self.audio_listeners and self.audio_listeners[session_id] and self.tts:
            audio_chunk = self.tts.text_to_speech(data_chunk)
            self.add_to_audio_buffer(session_id, audio_chunk)  # MP3 data is added directly

        if self.temp_buffers[session_id][message_id].strip().endswith("[DONE]"):
            complete_message = self.temp_buffers[session_id].pop(message_id).replace("[DONE]", "").strip()
            # Notify text listeners
            self.notify_listeners(session_id, complete_message)
            # Notify audio listeners if TTS was used
            if session_id in self.audio_listeners and self.audio_listeners[session_id]:
                self.notify_listeners(session_id, complete_message, audio=True)
