from abc import ABC, abstractmethod

class TextGeneratorInterface(ABC):
    @abstractmethod
    def generate_response(self, conversation_id, prompt, system_message):
        pass
