from abc import ABC, abstractmethod
from typing import Generator

class PersonalityAgent(ABC):
    @abstractmethod
    def generate_acknowledgment(self, user_message: str) -> Generator[str, None, None]:
        """
        Generates an acknowledgment response to the user's message, filtered through the assistant's personality.

        Args:
            user_message (str): The user's input message.

        Returns:
            Generator[str, None, None]: A stream of acknowledgment messages.
        """
        pass

    @abstractmethod
    def generate_final_response(self, username: str, reasoning_result: str, chat_handler) -> Generator[str, None, None]:
        """
        Generates the final response to the user, incorporating the reasoning result and filtered through the assistant's personality.

        Args:
            username (str): The username of the user.
            reasoning_result (str): The result returned by the ReasoningAgent.
            chat_handler: The chat handler for accessing conversation history.

        Returns:
            Generator[str, None, None]: A stream of final response messages.
        """
        pass
