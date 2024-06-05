from threading import Lock

class AudioManager:
    def __init__(self):
        self.lock = Lock()

    def acquire_audio(self):
        self.lock.acquire()
        print("Audio resource acquired")

    def release_audio(self):
        self.lock.release()
        print("Audio resource released")
