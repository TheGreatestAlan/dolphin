import json
import logging
import os
import uuid

from agent_server.integrations.ChatHandler import ChatSession
from agent_server.llms.LLMFactory import LLMFactory, ModelType

logger = logging.getLogger(__name__)

# HEMMINGWAY BRIDGE
# K you've updated the chat related stuff and got it working.  Looks like
# you're not storing the response of the chat in the history, so it thinks
# there's just a series of user requests which haven't been answered.

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
        """
        Process the user's message and handle streaming and parsing internally.
        If stream_handler is provided, it will be used to stream content.
        Returns a tuple (conversation_message, task_json).
        """
        # Build the prompt
        prompt = (
            f"The user just sent the following message:\n\"{user_message}\"\n"
            "Please analyze the message according to your responsibilities."
        )

        # Generate the assistant's response as a stream
        response_generator = self.llm.stream_response(prompt, self.full_system_message)

        # Handle streaming and parsing
        task_json = self.handle_response(response_generator, chat_session)
        if not task_json:
            return None

        try:
            return json.loads(task_json).get("task_request")
        except json.JSONDecodeError:
            # Handle invalid JSON here
            logger.error("Problem decoding task:" + task_json)
            return None

    def handle_response(self, response_generator, chat_session: ChatSession):
        """
        Handles streaming and parsing of the response.
        Detects and processes '[conversation]' and '[task_request]' tags.
        Returns a tuple (conversation_message, task_request_content).
        """
        finding_buffer = ""
        in_conversation = False
        task_request_content = None
        message_id = str(uuid.uuid4())

        # Debugging variable to capture unedited raw response
        raw_response_debug = ""

        for chunk in response_generator:
            finding_buffer += chunk
            raw_response_debug += chunk  # Append each chunk to raw debug variable
            logger.debug("CHUNK: " + chunk)

            # Check for conversation tags
            if not in_conversation and '[conversation]' in finding_buffer:
                in_conversation = True
                finding_buffer = finding_buffer.split('[conversation]', 1)[1]

            if in_conversation:
                if '[/conversation]' in finding_buffer:
                    # Extract conversation content
                    content, remaining = finding_buffer.split('[/conversation]', 1)
                    chat_session.parse_llm_stream(content, message_id)
                    finding_buffer = remaining
                    in_conversation = False
                else:
                    continue  # Wait for the closing tag, continue collecting

            # Check for task_request tag
            if '[task_request]' in finding_buffer:
                # Extract task_request content
                task_start = finding_buffer.find('[task_request]')
                task_end = finding_buffer.find('[/task_request]')
                if task_end != -1:
                    task_request_content = finding_buffer[task_start + len('[task_request]'):task_end].strip()
                    logger.info(f"Received task request: {task_request_content}")
                    finding_buffer = finding_buffer[task_end + len('[/task_request]'):]
                    break  # Assuming only one task request is expected
                else:
                    continue  # Wait for the closing tag, continue collecting

        # Save raw response for debugging purposes
        self.raw_response_debug = raw_response_debug  # Make it available as a class attribute
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