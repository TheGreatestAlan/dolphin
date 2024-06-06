import time as tm
import queue
from threading import Thread, Event

from ui.audio_recorder import AudioRecorder


class TestAudioRecorder:
    def __init__(self):
        self.record_event = Event()
        self.record_event.set()

    def release_audio(self):
        self.record_event.clear()

    def send_to_agent(self, transcription):
        print(f"Transcription sent to agent: {transcription}")

if __name__ == "__main__":
    # Initialize the audio manager and voice assistant stubs
    audio_manager = TestAudioRecorder()
    voice_assistant = TestAudioRecorder()

    # Create the audio recorder instance
    audio_recorder = AudioRecorder(audio_manager=audio_manager, voice_assistant=voice_assistant)

    # Start the recording in a separate thread to allow interaction
    recording_thread = Thread(target=audio_recorder.start_recording)
    recording_thread.start()

    try:
        while recording_thread.is_alive():
            tm.sleep(0.1)
    except KeyboardInterrupt:
        audio_recorder.continue_recording = False
        recording_thread.join()
