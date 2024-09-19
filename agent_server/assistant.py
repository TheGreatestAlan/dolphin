from abc import ABC, abstractmethod

class Assistant(ABC):
    @abstractmethod
    def message_assistant(self, session_id: str, username: str, user_message: str):
        """Process the message for the assistant."""
        pass
