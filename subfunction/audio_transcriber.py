from subfunction.cpu_whisper import WhisperTranscriber


class AudioTranscriber:
    def __init__(self):
        print("Initializing AudioTranscriber...")
        self.audioTranscriber = WhisperTranscriber()


    def transcribe_audio_file(self, filepath):
        transcription = self.audioTranscriber.transcribe_audio(filepath)
        if transcription.strip():
            print(f"Transcription result: {transcription}")
        return transcription


if __name__ == "__main__":
    transcriber = AudioTranscriber()
    transcription = transcriber.transcribe_audio_file("C:\\workspace\\dolphin\\subfunction\\recordings\\temp_detected_voice_1719615056.wav")
    print(f"Transcription: {transcription}")
