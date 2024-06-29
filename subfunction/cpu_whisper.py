import whisper
import os
from pydub import AudioSegment

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        # Load the model
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, filename="recording.wav"):
        """Transcribes the given audio file using Whisper."""
        # Ensure the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified audio file does not exist: {filename}")

        # Convert the audio file to WAV format using PyDub
        audio = AudioSegment.from_file(filename)
        wav_path = "temp.wav"
        audio.export(wav_path, format="wav")

        # Transcribe the audio file
        result = self.model.transcribe(wav_path)
        transcription = result['text']

        # Remove the temporary WAV file
        os.remove(wav_path)

        return transcription

# Example usage
if __name__ == "__main__":
    # Create an instance of the WhisperTranscriber
    transcriber = WhisperTranscriber(model_name="base")

    # Path to the audio file you want to transcribe
    audio_file_path = "M:/model_cache/6_2/temp_detected_voice_1717365225.wav"

    # Check if the audio file exists
    if os.path.exists(audio_file_path):
        # Transcribe the audio file
        transcription = transcriber.transcribe_audio(audio_file_path)
        print("Transcription:", transcription)
    else:
        print(f"The specified audio file does not exist: {audio_file_path}")
