from tts.GTTSHandler import GTTSHandler
from tts.PyAudioOutput import PyAudioOutput
from tts.Speech import Speech
from ui.AudioManager import AudioManager


class TestSpeech:
    def __init__(self):
        tts_handler = GTTSHandler(lang='en')
        audio_output = PyAudioOutput()
        self.speech = Speech(tts_handler=tts_handler, audio_output=audio_output, audio_manager=AudioManager())

    def run_test(self):
        test_texts = [
            "TESTING TESTING ONE TWO FOUR",
            "HELLO WORLD",
            "THIS IS A TEST"
        ]
        for text in test_texts:
            self.speech.speak(text)

        # Wait for the speech processing to complete
        self.speech.wait_until_done()

if __name__ == "__main__":
    tester = TestSpeech()
    tester.run_test()
