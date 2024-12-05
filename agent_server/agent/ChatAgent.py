import json
import logging
import os
import uuid

from agent_server.integrations.ChatHandler import ChatSession
from agent_server.llms.LLMFactory import LLMFactory, ModelType

logger = logging.getLogger(__name__)

# HEMMINGWAY BRIDGE
# K, so you're still reworking the ChatAgent, you just reworked it to run with
# a chat session in the chat handler.  You'll probably need to just delete the
# llm_assistant since chat_handler doesn't do the same stuff anymore.  Then go ahead
# and get ChatAgent running.  Looks like you'll probably need to figure out how
# to parse the stream right, somethings wrong there.  Then in the Assistant Orchestrator
# you can finally test the ChatAgent as the first point of contact instead of immediately
# going into the reasoning loop

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

        return task_json

    def handle_response(self, response_generator, chat_session:ChatSession):
        """
        Handles streaming and parsing of the response.
        If stream_handler is provided, streams content via stream_handler.
        If not, accumulates the content and returns it.
        Returns a tuple (conversation_message, task_json).
        """
        finding_buffer = ""
        in_conversation = False
        task_json = None
        message_id = str(uuid.uuid4())


        for chunk in response_generator:
            finding_buffer += chunk

            # Check for conversation start tag
            if not in_conversation and '[conversation]' in finding_buffer:
                in_conversation = True
                finding_buffer = finding_buffer.split('[conversation]', 1)[1]

            # Process conversational content
            if in_conversation:
                if '[/conversation]' in finding_buffer:
                    content, remaining = finding_buffer.split('[/conversation]', 1)
                    chat_session.parse_llm_stream(content, message_id)
                    finding_buffer = remaining
                    in_conversation = False
                else:
                    # Stream or accumulate the content as it arrives
                    finding_buffer = ""
            else:
                # Check for JSON start
                json_start = finding_buffer.find('{')
                if json_start != -1:
                    json_str = finding_buffer[json_start:]
                    try:
                        task_request = json.loads(json_str)
                        if task_request.get('response_type') == 'task_request':
                            task_json = task_request
                            logger.info(f"Received task request: {task_json['task_summary']}")
                            break  # Assuming the task JSON comes at the end
                    except json.JSONDecodeError:
                        # JSON is incomplete, continue collecting
                        pass
                    finding_buffer = ""

        return task_json

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