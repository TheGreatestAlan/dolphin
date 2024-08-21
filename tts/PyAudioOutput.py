import pyaudio
import wave

from tts.SpeechInterfaces import AudioOutputInterface


class PyAudioOutput(AudioOutputInterface):
    def play_audio(self, audio_fp):
        if isinstance(audio_fp, str):  # Check if audio_fp is a file path
            wf = wave.open(audio_fp, 'rb')
        else:
            raise ValueError("Expected audio_fp to be a file path")

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        chunk = 1024
        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
