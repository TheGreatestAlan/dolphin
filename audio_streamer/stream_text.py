import os
import time

import numpy as np
import sounddevice as sd
from openai import OpenAI
from pydub import AudioSegment
from io import BytesIO

#HEMMINGWAY BRIDGE, need to figure out what type of file this is so that I can stream it properly
# it's already different because it's an mp3 instead of a wav.  It wants to use ffmpeg, but do I need to?
from pydub.utils import which


class OpenAITTSPlayer:
    def __init__(self):

        # Read API key from environment variables
        api_key = os.getenv('OPENAI_API_KEY', '')
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
        # Manually set the path to ffmpeg and ffprobe
        AudioSegment.converter = which("ffmpeg")
        AudioSegment.ffmpeg = which("ffmpeg")
        AudioSegment.ffprobe = which("ffprobe")

        # Convert the audio data using pydub
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")

        # Convert audio_segment to numpy array for playback with sounddevice
        samples = np.array(audio_segment.get_array_of_samples())
        sample_rate = audio_segment.frame_rate
        channels = audio_segment.channels

        # Play the audio
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

# Example usage
if __name__ == "__main__":
    tts_player = OpenAITTSPlayer()
    tts_player.play_audio_stream("Hello world! This is a streaming test.")
