import queue

from agent_server.tts.OpenAITTS import OpenAITTS


class StreamManager:
    def __init__(self):
        self.text_buffers = {}  # Dictionary to hold queues for messages by session_id
        self.tts_instances = {}  # Dictionary to hold OpenAITTS instances by session_id
        self.stream_threads = {}  # Dictionary to hold streaming threads by session_id
        self.stop_events = {}  # Dictionary to hold stop events for each thread

    def add_to_text_buffer(self, session_id, data_chunk):
        """Add a data chunk to the queue for a specific session."""
        if session_id in self.text_buffers:
            self.text_buffers[session_id].put(data_chunk)
        if session_id in self.tts_instances:
            self.get_tts_instance(session_id).add_text_to_queue(data_chunk)

    def get_tts_instance(self, session_id):
        """Retrieve or create an OpenAITTS instance for the session."""
        if session_id not in self.tts_instances:
            self.tts_instances[session_id] = OpenAITTS()
        return self.tts_instances[session_id]


    def listen_to_text_stream(self, session_id):
        """Retrieve the queue for the given session, creating it if necessary."""
        if session_id not in self.text_buffers:
            self.text_buffers[session_id] = queue.Queue()  # Initialize a new queue if it doesn't exist
        return self.text_buffers[session_id]

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

    def end_text_stream(self, session_id):
        """End the text stream for a specific session and clean up resources."""
        if session_id in self.stream_threads:
            # Set the stop event to signal the thread to stop
            self.stop_events[session_id].set()
            # Optionally join the thread to ensure it's completed
            self.stream_threads[session_id].join()
            # Clean up resources associated with the text stream
            self.cleanup_resources(session_id, audio=False)
            # Remove the thread and stop event after cleanup
            del self.stream_threads[session_id]
            del self.stop_events[session_id]
        else:
            # If no thread exists, just clean up resources
            self.cleanup_resources(session_id, audio=False)

    def cleanup_resources(self, session_id, audio=False):
        """Clean up resources like queues and TTS instances if no listeners remain."""
        if audio:
            if session_id in self.tts_instances:
                self.tts_instances[session_id].stop()  # Stop the OpenAITTS instance
                del self.tts_instances[session_id]
        else:
            if session_id in self.text_buffers:
                del self.text_buffers[session_id]  # Remove the text queue
