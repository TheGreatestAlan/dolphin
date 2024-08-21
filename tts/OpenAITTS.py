from abc import ABC, abstractmethod
import os
import time
from io import BytesIO
import queue
import numpy as np
from openai import OpenAI
from pydub import AudioSegment
from tts.SpeechInterfaces import TTSInterface


class OpenAITTS(TTSInterface):
    def __init__(self):
        # Read API key from environment variables
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables.")
        self.client = OpenAI(api_key=api_key)

    def play_audio_stream(self, text: str):
        # Start timing the request
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

        # Convert audio_segment to numpy array for further processing
        samples = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate

        return samples, sample_rate

    def stream_text_to_speech(self, text_queue: queue.Queue, audio_buffer: queue.Queue):
        while True:
            try:
                # Get text from the queue
                text = text_queue.get(timeout=5)  # wait for 5 seconds if queue is empty

                if text is None:  # Sentinel value to stop streaming
                    break

                # Convert the text to audio samples
                samples, sample_rate = self.play_audio_stream(text)

                # Put the audio samples and sample rate into the audio buffer
                audio_buffer.put((samples, sample_rate))

            except queue.Empty:
                print("No more text to process.")
                break
