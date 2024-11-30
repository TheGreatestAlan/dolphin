from abc import ABC, abstractmethod

class LLMInterface(ABC):

    END_STREAM = "[DONE]"

    @abstractmethod
    def generate_response(self, prompt, system_message):
        pass

    def stream_response(self, prompt, system_message):
        pass