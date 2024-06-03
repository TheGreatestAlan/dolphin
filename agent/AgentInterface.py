class AgentInterface:
    def start_session(self):
        raise NotImplementedError

    def send_prompt(self, prompt):
        raise NotImplementedError

    def poll_for_response(self):
        raise NotImplementedError

    def end_session(self):
        raise NotImplementedError

