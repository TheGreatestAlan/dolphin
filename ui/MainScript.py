import os
from threading import Thread

from tts.GTTSHandler import GTTSHandler
from tts.PyAudioOutput import PyAudioOutput
from tts.Speech import Speech
from audio_recorder import AudioRecorder
from ui.AudioManager import AudioManager
from agent.AgentRestClient import AgentRestClient
from ui.VoiceAssistantGUI import VoiceAssistantGUI
from ui.VoiceAssistant import VoiceAssistant

def main():
    audio_manager = AudioManager()

    tts_handler = GTTSHandler(lang='en')
    audio_output = PyAudioOutput()
    speech = Speech(tts_handler=tts_handler, audio_output=audio_output, audio_manager=audio_manager)

    agent_client = AgentRestClient(os.environ.get("AGENT_URL", "http://127.0.0.1:5000"))

    gui = VoiceAssistantGUI()

    voice_assistant = VoiceAssistant(agent_client, speech, gui)
    audio_recorder = AudioRecorder(audio_manager=audio_manager, voice_assistant=voice_assistant, gui=gui)

    record_thread = Thread(target=audio_recorder.start_recording)
    gui_thread = Thread(target=gui.run)

    record_thread.start()
    gui_thread.start()

    record_thread.join()
    gui_thread.join()

if __name__ == "__main__":
    main()
