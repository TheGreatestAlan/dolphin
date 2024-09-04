import os
import time
from io import BytesIO
import queue
import threading

import numpy as np
from openai import OpenAI
from pydub import AudioSegment
from agent_server.tts.SpeechInterfaces import TTSInterface


class OpenAITTS(TTSInterface):
    def __init__(self):
        # Read API key from environment variables
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = OpenAI(api_key=api_key)

        # Initialize text, sentence, and audio buffers
        self.text_buffer = queue.Queue()
        self.sentence_buffer = queue.Queue()
        self.audio_buffer = queue.Queue()

        # Start the text processing in separate threads
        self.process_text_thread = threading.Thread(target=self.process_text_queue)
        self.process_text_thread.daemon = True  # Daemonize thread so it exits when the main program exits
        self.process_text_thread.start()

        self.process_sentence_thread = threading.Thread(target=self.process_sentence_queue)
        self.process_sentence_thread.daemon = True  # Daemonize thread so it exits when the main program exits
        self.process_sentence_thread.start()

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
        print(sample_rate)

        return samples, sample_rate

    def process_text_queue(self):
        """Continuously process the text buffer, form sentences, and add to the sentence buffer."""
        sentence = []
        while True:
            try:
                # Get text fragment from the text buffer
                text = self.text_buffer.get(timeout=5)
                if text is None:  # Sentinel value to stop processing
                    break

                # Accumulate fragments into a sentence
                sentence.append(text)

                # If a sentence-ending punctuation is detected, send the sentence to the sentence buffer
                if any(punct in text for punct in ".!?"):
                    full_sentence = ''.join(sentence)
                    self.sentence_buffer.put(full_sentence)
                    sentence = []  # Reset for the next sentence

            except queue.Empty:
                continue  # Continue checking the queue if it's empty

    def process_sentence_queue(self):
        """Continuously process the sentence buffer, convert to audio, and add to the audio buffer."""
        while True:
            try:
                # Get a full sentence from the sentence buffer
                sentence = self.sentence_buffer.get(timeout=5)
                if sentence is None:  # Sentinel value to stop processing
                    break

                # Convert sentence to audio
                samples, sample_rate = self.play_audio_stream(sentence)

                # Put the audio data onto the audio buffer
                self.audio_buffer.put((samples, sample_rate))

            except queue.Empty:
                continue  # Continue checking the queue if it's empty

    def stop(self):
        """Stop the processing threads by sending sentinel values."""
        self.text_buffer.put(None)
        self.sentence_buffer.put(None)
        self.process_text_thread.join()
        self.process_sentence_thread.join()
