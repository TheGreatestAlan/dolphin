

import queue
from abc import ABC, abstractmethod

class TTSInterface(ABC):

    @abstractmethod
    def add_text_to_queue(self, text: str):
        pass
    def get_audio_buffer(self):
        pass
class AudioOutputInterface(ABC):
    @abstractmethod
    def play_audio(self, audio_fp):
        pass