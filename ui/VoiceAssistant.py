from threading import Thread

class VoiceAssistant:
    def __init__(self, agent_client, speech, gui):
        self.agent_client = agent_client
        self.speech = speech
        self.gui = gui
        self.session_id = self.start_session()

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
            self.poll_agent_response()
        except Exception as e:
            print(f"Failed to send prompt to agent: {e}")

    def poll_agent_response(self):
        thread = Thread(target=self._poll_agent_response)
        thread.start()

    def _poll_agent_response(self):
        try:
            while True:
                response = self.agent_client.poll_response()
                if response:
                    print(f"Agent response: {response}")
                    self.gui.update_chat("Agent", response)
                    if self.speech:
                        print(f"Speaking response: {response}")
                        self.speech.speak(response)
                    break
        except Exception as e:
            print(f"Failed to poll response from agent: {e}")

    def update_chat(self, speaker, message):
        self.gui.update_chat(speaker, message)

    def run_gui(self):
        self.gui.run()
