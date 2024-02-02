import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


class AudioTranscriber:
    def __init__(self, cache_dir="M:/model_cache/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the pre-trained model and tokenizer
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", cache_dir=cache_dir)
        self.model.to(self.device)

    def transcribe_audio(self, filename="recording.wav"):
        # Load the audio file
        speech, rate = librosa.load(filename, sr=16000)
        input_values = self.tokenizer(speech, return_tensors='pt').input_values
        input_values = input_values.to(self.device)
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.decode(predicted_ids[0])
        return transcription

