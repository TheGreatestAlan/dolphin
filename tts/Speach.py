import queue

from tts.SpeachInterfaces import TTSInterface, AudioOutputInterface


class Speech:
    def __init__(self, tts_handler: TTSInterface, audio_output: AudioOutputInterface):
        self.tts_handler = tts_handler
        self.audio_output = audio_output

    def speak(self, text: str):
        audio_fp, format = self.tts_handler.text_to_speech(text)
        self.audio_output.play_audio(audio_fp, format=format)

    def stream_speak(self, text_stream: queue.Queue):
        self.tts_handler.stream_text_to_speech(text_stream, self.audio_output)
