from abc import ABC, abstractmethod

class LLMInterface(ABC):

    END_STREAM = "[DONE]"

    @abstractmethod
    def generate_response(self, prompt, system_message, message_history):
        pass

    def stream_response(self, prompt, system_message, message_history):
        pass