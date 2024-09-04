import sounddevice as sd
import soundfile as sf

from agent_server.tts.SpeechInterfaces import AudioOutputInterface


class SoundDeviceAudioOutput(AudioOutputInterface):
    def play_audio(self, audio_fp: str):
        # Load the audio file
        data, samplerate = sf.read(audio_fp, dtype='float32')

        # Play the audio file
        sd.play(data, samplerate)
        sd.wait()  # Wait until the file is done playing

# Example usage:
# audio_output = SoundDeviceAudioOutput()
# audio_output.play_audio("path_to_audio_file.wav")
