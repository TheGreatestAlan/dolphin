import queue
from abc import ABC, abstractmethod

class TTSInterface(ABC):
    @abstractmethod
    def text_to_speech(self, text: str):
        pass

    @abstractmethod
    def stream_text_to_speech(self, text_stream: queue.Queue):
        pass

class AudioOutputInterface(ABC):
    @abstractmethod
    def play_audio(self, audio_fp):
        pass
