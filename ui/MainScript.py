from tts.GTTSHandler import GTTSHandler
from tts.PyAudioOutput import PyAudioOutput
from tts.Speach import Speech
from audio_recorder import AudioRecorder
from ui.AudioManager import AudioManager


def main():
    audio_manager = AudioManager()

    tts_handler = GTTSHandler(lang='en')
    audio_output = PyAudioOutput()
    speech = Speech(tts_handler=tts_handler, audio_output=audio_output, audio_manager=audio_manager)

    recorder = AudioRecorder(speech=speech, audio_manager=audio_manager)

    recorder.run_gui()

if __name__ == "__main__":
    main()
