from tts.GTTSHandler import GTTSHandler
from tts.PyAudioOutput import PyAudioOutput
from tts.Speach import Speech


class TestSpeech:
    def __init__(self):
        tts_handler = GTTSHandler(lang='en')
        audio_output = PyAudioOutput()
        self.speech = Speech(tts_handler=tts_handler, audio_output=audio_output)

    def run_test(self):
        test_text = "TESTING TESTING ONE TWO FOUR"
        self.speech.speak(test_text)

if __name__ == "__main__":
    tester = TestSpeech()
    tester.run_test()
