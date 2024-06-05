import io
import pyaudio
import wave
from pydub import AudioSegment

from tts.SpeachInterfaces import AudioOutputInterface


class PyAudioOutput(AudioOutputInterface):
    def play_audio(self, audio_fp: io.BytesIO):
        audio_fp.seek(0)
        try:
            wf = wave.open(audio_fp, 'rb')
            print("Playing WAV audio")
        except wave.Error:
            print("Converting MP3 to WAV")
            audio = AudioSegment.from_file(audio_fp, format="mp3")
            wav_fp = io.BytesIO()
            audio.export(wav_fp, format="wav")
            wav_fp.seek(0)
            audio_fp = wav_fp
            wf = wave.open(audio_fp, 'rb')

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
        print("Audio playback finished")
