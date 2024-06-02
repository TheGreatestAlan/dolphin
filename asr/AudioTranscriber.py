import subprocess
import time

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os


class AudioTranscriber:
    def __init__(self, cache_dir="M:/model_cache/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA:" + torch.cuda.is_available().__str__())

        # Load the pre-trained model and tokenizer
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        self.model.to(self.device)

    def convert_to_wav(self, filename):
        """Converts an audio file to WAV format using ffmpeg directly."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified audio file does not exist: {filename}")

        basename, ext = os.path.splitext(filename)
        wav_filename = basename + '.wav'

        if ext.lower() != '.wav':
            # Construct the command to convert the file to WAV format
            ffmpeg_path = r"M:\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe"
            command = [ffmpeg_path, '-i', filename, wav_filename]

            # Execute the command
            subprocess.run(command, check=True)

        # Return the path to the WAV file
        return wav_filename


    def transcribe_audio(self, filename="recording.wav"):
        # Start the timer
        start_time = time.time()

        # Determine if conversion is needed and convert if necessary
        # wav_filename = self.convert_to_wav(filename)

        # Load the audio file
        speech, rate = librosa.load(filename, sr=16000)
        input_values = self.tokenizer(speech, return_tensors='pt').input_values
        input_values = input_values.to(self.device)
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.decode(predicted_ids[0])

        # Stop the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Transcription time: {elapsed_time:.2f} seconds")

        return transcription


# Create an instance of the AudioTranscriber
transcriber = AudioTranscriber()

# Path to the audio file you want to transcribe
audio_file_path = "M:/model_cache/6_2/temp_detected_voice_1717365225.wav"

# Check if the audio file exists
if os.path.exists(audio_file_path):
    # Transcribe the audio file
    transcription = transcriber.transcribe_audio(audio_file_path)
    print("Transcription:", transcription)
else:
    print(f"The specified audio file does not exist: {audio_file_path}")