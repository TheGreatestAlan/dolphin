# agent_server/agent/SevroPersonalityAgent.py

import logging
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.agent.PersonalityAgent import PersonalityAgent

logger = logging.getLogger(__name__)

class SevroPersonalityAgent(PersonalityAgent):
    def __init__(self):
        logger.info("Initializing SevroPersonalityAgent...")
        self.llm = LLMFactory.get_singleton(ModelType.CHATGPT)
        # Define the system message to reflect Sevro's personality
        self.system_message = (
            "You are Sevro au Barca from the Red Rising series by Pierce Brown. "
            "You are brash, sarcastic, and have a rough demeanor. "
            "You often use profanity and dark humor but are fiercely loyal to your friends. "
            "Respond to the user as Sevro would, keeping in mind his personality and speech patterns."
        )

    def generate_acknowledgment(self, user_message: str) -> str:
        prompt = (
            f"The user just sent the following message:\n\"{user_message}\"\n"
            "As Sevro, acknowledge the user's request in your own unique way."
        )
        acknowledgment = self.llm.generate_response(prompt, self.system_message)
        return acknowledgment

    def generate_final_response(self, username: str, reasoning_result: str, chat_handler) -> str:
        context = chat_handler.get_context(username)
        conversation_history = '\n'.join(
            [f"{msg['role']}: {msg['content']}" for msg in context]
        )
        prompt = (
            f"{conversation_history}\nSevro's Analysis: {reasoning_result}\n"
            "Respond to the user as Sevro would, incorporating the analysis above."
        )
        final_response = self.llm.generate_response(prompt, self.system_message)
        return final_response
