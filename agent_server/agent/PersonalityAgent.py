# agent_server/agent/PersonalityAgent.py

from abc import ABC, abstractmethod

class PersonalityAgent(ABC):
    @abstractmethod
    def generate_acknowledgment(self, user_message: str) -> str:
        """
        Generates an acknowledgment response to the user's message, filtered through the assistant's personality.

        Args:
            user_message (str): The user's input message.

        Returns:
            str: The acknowledgment message.
        """
        pass

    @abstractmethod
    def generate_final_response(self, username: str, reasoning_result: str, chat_handler) -> str:
        """
        Generates the final response to the user, incorporating the reasoning result and filtered through the assistant's personality.

        Args:
            username (str): The username of the user.
            reasoning_result (str): The result returned by the ReasoningAgent.
            chat_handler: The chat handler for accessing conversation history.

        Returns:
            str: The final response message.
        """
        pass
