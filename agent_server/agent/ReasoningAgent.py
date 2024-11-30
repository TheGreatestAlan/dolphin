from abc import ABC, abstractmethod
from typing import Dict

class ReasoningAgent(ABC):
    @abstractmethod
    def process_request(self, user_input: str) -> str:
        """
        Processes the user's input and returns a structured result.

        Args:
            user_input (str): The user's input message.

        Returns:
            Dict: A dictionary containing the reasoning result, which may include actions taken,
                  observations, and any analysis needed by the ChatAgent.
        """
        pass
