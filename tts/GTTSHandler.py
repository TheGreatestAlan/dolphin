import os
import queue
import threading
import pyaudio
import wave
from gtts import gTTS
from pydub import AudioSegment

from tts.SpeachInterfaces import TTSInterface


class GTTSHandler(TTSInterface):
    def __init__(self, lang='en'):
        self.lang = lang

    def text_to_speech(self, text: str):
        tts = gTTS(text=text, lang=self.lang)
        filename_mp3 = "output.mp3"
        filename_wav = "output.wav"
        tts.save(filename_mp3)

        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(filename_mp3)
        audio.export(filename_wav, format="wav")

        return filename_wav

    def stream_text_to_speech(self, text_stream: queue.Queue):
        def stream_worker():
            while True:
                text = text_stream.get()
                if text is None:  # None is a signal to stop streaming
                    break
                yield self.text_to_speech(text)

        return stream_worker()

    def play_audio(self, filename: str):
        chunk = 1024
        wf = wave.open(filename, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()
