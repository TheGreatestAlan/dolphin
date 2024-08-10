import time
import json
from flask import Response

class StreamManager:
    def __init__(self):
        self.stream_buffers = {}  # Dictionary to hold stream buffers for messages by session_id
        self.temp_buffers = {}  # Dictionary to hold temporary buffers for messages by session_id
        self.stream_listeners = {}  # Dictionary to hold listeners for each session

    def add_to_stream_buffer(self, session_id, data_chunk):
        """Add a data chunk to the stream buffer for a specific session."""
        if session_id not in self.stream_buffers:
            self.stream_buffers[session_id] = []
        self.stream_buffers[session_id].append(data_chunk)

    def start_streaming(self, session_id):
        """Continuously stream data for a given session."""
        print("Starting Stream")

        def generate():
            print("generating")
            while True:
                if session_id in self.stream_buffers:
                    while self.stream_buffers[session_id]:
                        data_chunk = self.stream_buffers[session_id].pop(0)
                        yield f"data: {json.dumps({'message': data_chunk})}\n\n"
                time.sleep(1)

        return Response(generate(), mimetype='text/event-stream')

    def listen_to_stream(self, session_id):
        """Listen to the stream and start streaming data for the given session."""
        return self.start_streaming(session_id)

    def register_listener(self, session_id, listener):
        """Register a new listener for a specific session."""
        if session_id not in self.stream_listeners:
            self.stream_listeners[session_id] = []
        self.stream_listeners[session_id].append(listener)

    def unregister_listener(self, session_id, listener):
        """Unregister a listener from a specific session."""
        if session_id in self.stream_listeners and listener in self.stream_listeners[session_id]:
            self.stream_listeners[session_id].remove(listener)
            if not self.stream_listeners[session_id]:  # Remove session if no listeners remain
                del self.stream_listeners[session_id]

    def notify_listeners(self, session_id, data):
        """Notify all listeners of new data for a specific session."""
        if session_id in self.stream_listeners:
            formatted_data = f"data: {json.dumps({'message': data})}\n\n"
            for listener in self.stream_listeners[session_id]:
                listener.send(formatted_data)  # Ensure listener.send handles SSE formatting

    def receive_stream_data(self, session_id, data_chunk, message_id, role="AI"):
        """Process received stream data by appending to buffers and notifying listeners."""
        if session_id not in self.temp_buffers:
            self.temp_buffers[session_id] = {}

        if message_id not in self.temp_buffers[session_id]:
            self.temp_buffers[session_id][message_id] = ""

        self.temp_buffers[session_id][message_id] += data_chunk
        self.add_to_stream_buffer(session_id, data_chunk)

        if self.temp_buffers[session_id][message_id].strip().endswith("[DONE]"):
            complete_message = self.temp_buffers[session_id].pop(message_id).replace("[DONE]", "").strip()
            # Finalize the message (you would typically call ChatHandler's finalize_message here)
            self.notify_listeners(session_id, complete_message)
