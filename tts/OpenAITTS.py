import os
import time
from io import BytesIO
import queue
import threading

import numpy as np
from openai import OpenAI
from pydub import AudioSegment
from tts.SpeechInterfaces import TTSInterface
import sounddevice as sd


class OpenAITTS(TTSInterface):
    def __init__(self):
        # Read API key from environment variables
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = OpenAI(api_key=api_key)

        # Initialize text and audio buffers
        self.text_buffer = queue.Queue()
        self.audio_buffer = queue.Queue()

        # Start the text processing in a separate thread
        self.process_thread = threading.Thread(target=self.process_text_queue)
        self.process_thread.daemon = True  # Daemonize thread so it exits when the main program exits
        self.process_thread.start()

    def get_audio_buffer(self):
        """Return the audio buffer so external code can access the audio data."""
        return self.audio_buffer

    def add_text_to_queue(self, text: str):
        """Add text to the text buffer for processing."""
        self.text_buffer.put(text)

    def play_audio_stream(self, text: str):
        """Convert text to audio using OpenAI's TTS and return audio samples."""
        start_time = time.time()

        # Create the TTS request and stream the response
        with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text
        ) as response:
            audio_data = response.read()

        # Calculate the latency
        latency = time.time() - start_time
        print(f"Latency from request to first sound: {latency:.2f} seconds")

        # Convert the audio data using pydub
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")

        # Convert audio_segment to numpy array for playback or further processing
        samples = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate

        return samples, sample_rate

    def process_text_queue(self):
        """Continuously process the text buffer, convert text to audio, and add to the audio buffer."""
        while True:
            try:
                # Get text from the queue, blocking until something is available
                text = self.text_buffer.get(timeout=5)

                if text is None:  # Sentinel value to stop processing
                    break

                # Convert text to audio
                samples, sample_rate = self.play_audio_stream(text)

                # Put the audio data onto the audio buffer
                self.audio_buffer.put((samples, sample_rate))

            except queue.Empty:
                continue  # Continue checking the queue if it's empty

    def stop(self):
        """Stop the processing thread by sending a sentinel value."""
        self.text_buffer.put(None)
        self.process_thread.join()  # Ensure the thread has stopped before continuing


def audio_player(audio_buffer):
    """Continuously play audio from the buffer."""
    while True:
        audio_chunk = audio_buffer.get()
        if audio_chunk is None:  # Sentinel value to stop the thread
            break
        audio_samples, sample_rate = audio_chunk
        sd.play(audio_samples, samplerate=sample_rate)
        sd.wait()


if __name__ == "__main__":
    tts_player = OpenAITTS()

    # Get the audio buffer and play the audio
    audio_buffer = tts_player.get_audio_buffer()

    # Start the audio player in a separate thread
    audio_thread = threading.Thread(target=audio_player, args=(audio_buffer,))
    audio_thread.daemon = True  # Daemonize thread so it exits when the main program exits
    audio_thread.start()

    # Add some text to the queue
    tts_player.add_text_to_queue("Hello, how are you today?")
    tts_player.add_text_to_queue("This is a streaming text-to-speech test.")
    tts_player.add_text_to_queue("Goodbye!")

    # Simulate the main thread doing other work
    time.sleep(12)  # Adjust as needed

    # Stop the TTS player and audio thread
    tts_player.stop()
    audio_buffer.put(None)  # Send sentinel to stop the audio player
    audio_thread.join()  # Ensure the audio thread has stopped before continuing
