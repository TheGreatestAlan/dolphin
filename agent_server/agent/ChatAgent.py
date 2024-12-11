import json
import logging
import os
import uuid

from agent_server.integrations.ChatHandler import ChatSession
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.llms.LLMInterface import LLMInterface

logger = logging.getLogger(__name__)

class ChatAgent():
    def __init__(self):
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

    def process_user_message(self, user_message: str, chat_session:ChatSession):
        prompt = (
            f"The user just sent the following message:\n\"{user_message}\"\n"
            "Please analyze the message according to your responsibilities."
        )

        # Generate the assistant's response as a stream
        response_generator = self.llm.stream_response(prompt, self.full_system_message)

        # Handle streaming and parsing
        return self.handle_response(response_generator, chat_session)

    def handle_response(self, response_generator, chat_session: ChatSession):
        """
        Handles streaming and parsing of the response.
        Detects and processes '[conversation]' and '[task_request]' tags.
        Returns a tuple (conversation_message, task_request_content).
        """
        in_conversation = False
        task_request_content = None
        message_id = str(uuid.uuid4())

        # Debugging variable to capture unedited raw response
        raw_response = ""

        # Temporary buffer to handle split tags
        temp_buffer = ""

        for chunk in response_generator:
            temp_buffer += chunk
            raw_response += chunk  # Append each chunk to raw debug variable

            # Check if the start tag `[conversation]` is present
            if not in_conversation and '[conversation]' in temp_buffer:
                in_conversation = True
                # Remove everything up to and including the start tag
                temp_buffer = temp_buffer.split('[conversation]', 1)[1]

            if in_conversation:
                if '[/conversation]' in temp_buffer:
                    # Handle closing tag
                    content, temp_buffer = temp_buffer.split('[/conversation]', 1)
                    chat_session.parse_llm_stream(content + LLMInterface.END_STREAM, message_id)
                    in_conversation = False  # Reset conversation state
                else:
                    # Stream intermediate content
                    chat_session.parse_llm_stream(temp_buffer, message_id)
                    temp_buffer = ""  # Clear the buffer after streaming content

        # Extract task_request if it exists
        if '[task_request]' in raw_response and '[/task_request]' in raw_response:
            start = raw_response.find('[task_request]') + len('[task_request]')
            end = raw_response.find('[/task_request]')
            task_request_content = raw_response[start:end].strip()

        return task_request_content

    def generate_final_response(self, reasoning_result: str, chat_session:ChatSession):
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
        message_id = str(uuid.uuid4())

        for chunk in response_generator:
            chat_session.parse_llm_stream(chunk, message_id)