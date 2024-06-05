import io
import queue
import threading
from gtts import gTTS

from tts.SpeachInterfaces import TTSInterface


class GTTSHandler(TTSInterface):
    def __init__(self, lang='en'):
        self.lang = lang

    def text_to_speech(self, text: str):
        tts = gTTS(text=text, lang=self.lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp

    def stream_text_to_speech(self, text_stream: queue.Queue):
        def stream_worker():
            while True:
                text = text_stream.get()
                if text is None:  # None is a signal to stop streaming
                    break
                audio_fp = self.text_to_speech(text)
                audio_fp.seek(0)
                yield audio_fp

        threading.Thread(target=stream_worker).start()
