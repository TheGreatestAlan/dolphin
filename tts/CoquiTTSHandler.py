from TTS.api import TTS
import pyaudio
import queue
import threading
import numpy as np

from tts.SpeachInterfaces import TTSInterface

class CoquiTTSHandler(TTSInterface):
    def __init__(self, model_name):
        self.tts = TTS(model_name=model_name, progress_bar=False).to("cpu")  # Use "cuda" if GPU is available
        self.audio_queue = queue.Queue()
        self.audio_player_thread = threading.Thread(target=self.stream_audio)
        self.audio_player_thread.daemon = True
        self.audio_player_thread.start()

    def text_to_speech(self, text: str):
        # Streaming audio data to the audio queue
        audio_generator = self.tts.tts_streaming(text=text)
        for audio_chunk in audio_generator:
            self.audio_queue.put(audio_chunk.numpy())  # Convert tensor to numpy array and enqueue

    def stream_text_to_speech(self, text_stream: queue.Queue):
        while not text_stream.empty():
            sentence = text_stream.get()
            if sentence is None:
                break
            self.text_to_speech(sentence)

    def stream_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,  # Make sure this format matches the data from xTTS
                        channels=1,  # Mono output
                        rate=22050,  # Sample rate (adjust if needed)
                        output=True)
        while True:
            audio_chunk = self.audio_queue.get()
            if audio_chunk is None:
                break
            stream.write(audio_chunk.astype(np.int16).tobytes())  # Play audio
        stream.stop_stream()
        stream.close()
        p.terminate()

    @staticmethod
    def main():
        model_name = "tts_models/en/jenny/jenny"
        tts_handler = CoquiTTSHandler(model_name)

        # Text to synthesize, divided into sentences
        poem = '''In twilight's embrace, the shadows play,
        Soft whispers float in the moon's ballet.
        Velvet skies and stars alight,
        Seduce the silence of the night.

        Linger here in dreams' soft lace,
        Where time dissolves in love's sweet trace.
        Each word a kiss, each breath a sigh,
        Underneath the vast, starlit sky.

        Come close, let go, no need to hide,
        In whispers low, let love confide.
        With every beat, with every rhyme,
        Feel the pulse of timeless time.'''
        sentences = poem.split('. ')

        text_queue = queue.Queue()
        for sentence in sentences:
            text_queue.put(sentence + '.')  # Adding period

        tts_handler.stream_text_to_speech(text_queue)

if __name__ == "__main__":
    CoquiTTSHandler.main()
