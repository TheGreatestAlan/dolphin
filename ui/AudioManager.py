import threading

class AudioManager:
    def __init__(self):
        self.audio_lock = threading.Lock()
        self.record_event = threading.Event()
        self.record_event.set()  # Default to giving audio resources to the AudioRecorder

    def acquire_audio(self):
        self.record_event.clear()
        self.audio_lock.acquire()

    def release_audio(self):
        if self.audio_lock.locked():
            self.audio_lock.release()
            self.record_event.set()
