from abc import ABC, abstractmethod

from agent_server.integrations.ChatHandler import ChatSession


class Assistant(ABC):
    @abstractmethod
    def message_assistant(self, chat_session:ChatSession, user_message: str):
        """Process the message for the assistant."""
        pass
