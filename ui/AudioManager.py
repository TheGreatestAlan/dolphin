import threading

class AudioManager:
    def __init__(self):
        self.audio_lock = threading.Lock()
        self.record_event = threading.Event()
        self.record_event.set()  # Default to giving audio resources to the AudioRecorder

    def acquire_audio(self):
        print("Acquiring audio resources...")
        self.record_event.clear()
        self.audio_lock.acquire()
        print("Audio resources acquired by Speech.")

    def release_audio(self):
        print("Releasing audio resources...")
        if self.audio_lock.locked():
            self.audio_lock.release()
            self.record_event.set()
            print("Audio resources released by Speech.")
        else:
            print("Audio resources were not locked.")
