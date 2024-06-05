import threading
import queue
import time

from tts.SpeachInterfaces import TTSInterface, AudioOutputInterface
from ui.AudioManager import AudioManager

class Speech:
    def __init__(self, tts_handler: TTSInterface, audio_output: AudioOutputInterface, audio_manager: AudioManager):
        self.tts_handler = tts_handler
        self.audio_output = audio_output
        self.audio_manager = audio_manager
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        self.periodic_thread = threading.Thread(target=self._speak_periodically)
        self.periodic_thread.daemon = True
        self.periodic_thread.start()

    def speak(self, text: str):
        print(f"Queueing text: {text}")
        self.text_queue.put(text)

    def stream_speak(self, text_stream: queue.Queue):
        for audio_fp in self.tts_handler.stream_text_to_speech(text_stream):
            self.audio_output.play_audio(audio_fp)

    def _process_queue(self):
        while True:
            text = self.text_queue.get()
            if text is None:
                break
            print(f"Processing text: {text}")
            self.audio_manager.acquire_audio()
            audio_fp = self.tts_handler.text_to_speech(text)
            self.audio_output.play_audio(audio_fp)
            self.audio_manager.release_audio()

    def _speak_periodically(self):
        while True:
            time.sleep(5)
            self.speak("This is a periodic message every 5 seconds.")

    def wait_until_done(self):
        self.text_queue.put(None)
        self.thread.join()
