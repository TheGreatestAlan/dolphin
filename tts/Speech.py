import queue
import threading

from tts.SpeachInterfaces import TTSInterface, AudioOutputInterface


class Speech:
    def __init__(self, tts_handler: TTSInterface, audio_output: AudioOutputInterface, audio_manager=None):
        self.tts_handler = tts_handler
        self.audio_output = audio_output
        self.audio_manager = audio_manager
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        print("Speech processor thread started")

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
            audio_fp = self.tts_handler.text_to_speech(text)
            if self.audio_manager:
                self.audio_manager.acquire_audio()
            self.audio_output.play_audio(audio_fp)
            if self.audio_manager:
                self.audio_manager.release_audio()

    def wait_until_done(self):
        self.text_queue.put(None)
        self.thread.join()