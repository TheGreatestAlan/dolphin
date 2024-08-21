

import queue
from abc import ABC, abstractmethod

class TTSInterface(ABC):

    @abstractmethod
    def stream_text_to_speech(self, text_stream: queue.Queue):
        pass
class AudioOutputInterface(ABC):
    @abstractmethod
    def play_audio(self, audio_fp):
        pass