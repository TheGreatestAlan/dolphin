import logging
import re
import json
import os  # Import the os module
import uuid
import sys
import time
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.agent.PersonalityAgent import PersonalityAgent

logger = logging.getLogger(__name__)

class SevroPersonalityAgent():
    def __init__(self):
        logger.info("Initializing SevroPersonalityAgent...")
        self.llm = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B)

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory (one level above)
        prompt_dir = os.path.abspath(os.path.join(script_dir, '../prompt'))

        # Build the paths to the personality.txt and system_message.txt files
        personality_path = os.path.join(prompt_dir, 'SevroPersonality.txt')
        system_message_path = os.path.join(prompt_dir, 'ChatBotPrompt.txt')

        # Read the personality and system message from text files
        with open(personality_path, 'r') as f:
            self.personality = f.read()

        with open(system_message_path, 'r') as f:
            self.system_message = f.read()

        # Combine the personality and system message
        self.full_system_message = f"{self.personality}\n\n{self.system_message}"

    def process_user_message(self, user_message: str):
        """
        Process the user's message and determine the appropriate response.
        Returns a generator for streaming the response.
        """
        # Build the prompt to send to the LLM
        prompt = (
            f"The user just sent the following message:\n\"{user_message}\"\n"
            "Please analyze the message according to your responsibilities."
        )

        # Generate the assistant's response as a stream
        response_generator = self.llm.stream_response(prompt, self.full_system_message)

        # Return the response generator
        return response_generator


    def generate_final_response(self, reasoning_result: str):
        """
        Generates the final response to the user after the ReAct module has processed the task.
        Returns a generator for streaming the response.
        """
        # Build the prompt
        prompt = (
            f"The ReAct module has returned the following result:\n\"{reasoning_result}\"\n"
            "As per your responsibilities, provide the final response to the user in your character's voice, enclosed within [conversation][/conversation] tags."
            "Ensure you do not omit any information, and keep your response concise."
        )

        # Use the LLM's streaming interface to get a generator
        response_generator = self.llm.stream_response(prompt, self.full_system_message)

        # Return the response generator
        return self.parse_response_stream(response_generator)

if __name__ == "__main__":
    # Set up basic logging to console
    logging.basicConfig(level=logging.INFO)

    # Create an instance of the agent
    agent = SevroPersonalityAgent()

    # Simulate user input
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Process the user input and get a response generator
        response_generator = agent.process_user_message(user_input)

        # Print the assistant's response as it streams
        print("Assistant: ", end="", flush=True)
        for token in response_generator:
            print(token, end="", flush=True)
            # Sleep a bit to simulate streaming (optional)
            time.sleep(0.05)
        print()  # Newline after the assistant's response
