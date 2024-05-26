from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, conversation_id, prompt, system_message):
        pass
