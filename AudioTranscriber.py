import subprocess

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os

class AudioTranscriber:
    def __init__(self, cache_dir="M:/model_cache/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Determine if conversion is needed and convert if necessary
        wav_filename = self.convert_to_wav(filename)

        #Load the audio file
        speech, rate = librosa.load(wav_filename, sr=16000)
        input_values = self.tokenizer(speech, return_tensors='pt').input_values
        input_values = input_values.to(self.device)
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.decode(predicted_ids[0])
        return transcription
