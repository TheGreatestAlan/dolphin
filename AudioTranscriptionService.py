from asr.AudioTranscriber import AudioTranscriber


class AudioTranscriptionService:
    def __init__(self, cache_dir="M:/model_cache/"):
        # Initialize the AudioTranscriber with the specified cache directory
        self.transcriber = AudioTranscriber(cache_dir=cache_dir)

    def transcribe(self, file_path):
        # Transcribe the provided audio file
        try:
            transcription = self.transcriber.transcribe_audio("M:\\workspace\\dolphin\\uploads\\recorded_audio2372643121406819241.3gp");
            print(f"Transcription: {transcription}")
        except Exception as e:
            print(f"An error occurred during transcription: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace 'path_to_your_audio_file' with the actual path to the audio file you want to transcribe
    audio_file_path = 'path_to_your_audio_file'
    transcription_service = AudioTranscriptionService()
    transcription_service.transcribe(audio_file_path)
