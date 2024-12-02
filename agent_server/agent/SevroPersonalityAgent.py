# agent_server/agent/SevroPersonalityAgent.py

import logging
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.agent.PersonalityAgent import PersonalityAgent

logger = logging.getLogger(__name__)

class SevroPersonalityAgent(PersonalityAgent):
    def __init__(self):
        logger.info("Initializing SevroPersonalityAgent...")
        self.llm = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B)
        # Define the system message to reflect Sevro's personality
        self.system_message = (
            "You are Sevro au Barca from the Red Rising series by Pierce Brown. "
            "You are brash, sarcastic, and have a rough demeanor. "
            "You often use profanity and dark humor but are fiercely loyal to your friends. "
            "You are however direct and dedicated to completing the tasks efficiently and without misdirection"
            "Respond to the user as Sevro would, keeping in mind his personality and speech patterns."
        )

    def generate_acknowledgment(self, user_message: str) -> str:
        prompt = (
            f"The user just sent the following message:\n\"{user_message}\"\n"
            "As Sevro, acknowledge the user's request in your own unique way."
        )
        acknowledgment = self.llm.generate_response(prompt, self.system_message)
        return acknowledgment

    def generate_final_response(self, username: str, reasoning_result: str, chat_handler):
        prompt = (
            f"MESSSAGE---{reasoning_result}---\n"
            "Take this message and respond in Sevro's voice with it.  Do not omit any information"
        )
        # Use the LLM's streaming interface to get a generator
        response_stream = self.llm.stream_response(prompt, self.system_message)
        return response_stream