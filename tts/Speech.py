import queue
import threading
import re
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
        self.chunk_buffer = ""

    def speak(self, text: str):
        print(f"Queueing text: {text}")
        self.text_queue.put(text)

    def stream_speak(self, chunk: str):
        self._process_chunk(chunk)

    def _process_chunk(self, chunk: str):
        self.chunk_buffer += chunk
        # Match words and spaces, keep trailing partial words in buffer
        words = re.findall(r'\S+|\s+', self.chunk_buffer)

        # Check if the last matched item is an incomplete word
        last_word = words[-1] if words else ""
        if last_word and not re.match(r'\s', last_word):
            self.chunk_buffer = last_word  # Keep the partial word in buffer
            words = words[:-1]  # Remove the partial word from the list
        else:
            self.chunk_buffer = ""  # Clear buffer if last word is complete

        # Enqueue complete words
        for word in words:
            if re.match(r'\S+', word):
                self.text_queue.put(word)

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

    def wait_until_done(self):
        self.text_queue.put(None)
        self.thread.join()
