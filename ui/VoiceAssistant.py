from threading import Thread

class VoiceAssistant:
    def __init__(self, agent_client, speech, gui):
        self.agent_client = agent_client
        self.speech = speech
        self.gui = gui
        self.session_id = self.start_session()
        self.listen_to_stream()  # Start listening to the stream once the session is established

    def start_session(self):
        try:
            session_id = self.agent_client.start_session()
            print(f"Started session with ID: {session_id}")
            return session_id
        except Exception as e:
            raise RuntimeError(f"Failed to start session with agent: {e}")

    def send_to_agent(self, transcription):
        thread = Thread(target=self._send_to_agent, args=(transcription,))
        thread.start()

    def _send_to_agent(self, transcription):
        try:
            self.agent_client.send_prompt(transcription)
        except Exception as e:
            print(f"Failed to send prompt to agent: {e}")

    def listen_to_stream(self):
        thread = Thread(target=self._stream_agent_response)
        thread.start()

    def _stream_agent_response(self):
        try:
            for chunk in self.agent_client.stream_response():
                if chunk:
                    print(f"Agent response chunk: {chunk}")
                    self.gui.update_chat("Agent", chunk)
                    if self.speech:
                        print(f"Speaking response chunk: {chunk}")
                        self.speech.speak(chunk)
        except Exception as e:
            print(f"Failed to stream response from agent: {e}")

    def update_chat(self, speaker, message):
        self.gui.update_chat(speaker, message)

    def run_gui(self):
        self.gui.run()
