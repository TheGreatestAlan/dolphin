import uuid
import logging

from agent_server.agent.PersonalityAgent import PersonalityAgent
from agent_server.agent.ReasoningAgent import ReasoningAgent
from agent_server.assistant import Assistant
from agent_server.integrations.ChatHandler import ChatHandler

logger = logging.getLogger(__name__)

class AssistantOrchestrator(Assistant):
    def __init__(self, reasoning_agent: ReasoningAgent, personality_agent: PersonalityAgent, chat_handler: ChatHandler):
        logger.info("Initializing AssistantOrchestrator...")
        self.reasoning_agent = reasoning_agent
        self.personality_agent = personality_agent
        self.chat_handler = chat_handler

    def message_assistant(self, session_id: str, username: str, user_message: str):
        if not user_message:
            logger.warning(f"Empty message received for session {session_id}, skipping processing.")
            return

        logger.info(f"Processing user message for session {session_id}: {user_message}")

        try:
            # Store the user message in the chat history
            logger.debug(f"Storing user message in chat history for username '{username}'.")
            self.chat_handler.store_human_context(username, user_message)

            message_id = str(uuid.uuid4())
            logger.debug(f"Generated message ID for acknowledgment: {message_id}")

            # Generate an acknowledgment response to the user
            # logger.debug(f"Generating acknowledgment response for user message: {user_message}")
            # acknowledgment = self.personality_agent.generate_acknowledgment(user_message)
            # for data_chunk in acknowledgment:
            #     self.chat_handler.parse_llm_stream(username, session_id, data_chunk, message_id)

            #logger.info(f"Acknowledgment sent for session {session_id}, message ID: {message_id}")

            # Pass the request to the ReasoningAgent (synchronously)
            logger.debug(f"Passing user message to ReasoningAgent for processing.")
            reasoning_result = self.reasoning_agent.process_request(user_message)
            logger.info(f"ReasoningAgent returned result for session {session_id}: {reasoning_result}")

            # Generate the final response to the user, filtered through the PersonalityAgent
            response_id = str(uuid.uuid4())
            logger.debug(f"Generated response ID for final response: {response_id}")
            final_response_stream = self.personality_agent.generate_final_response(
                username, reasoning_result, self.chat_handler)
            for data_chunk in final_response_stream:
                self.chat_handler.parse_llm_stream(username, session_id, data_chunk, response_id)
            logger.info(f"Final response sent for session {session_id}, response ID: {response_id}")

        except Exception as e:
            logger.exception(f"Error processing user message for session {session_id}: {e}")
