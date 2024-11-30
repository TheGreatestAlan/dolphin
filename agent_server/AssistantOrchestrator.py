import uuid
import logging

from agent_server.agent.PersonalityAgent import PersonalityAgent
from agent_server.agent.ReasoningAgent import ReasoningAgent
from agent_server.integrations.ChatHandler import ChatHandler

logger = logging.getLogger(__name__)

class AssistantOrchestrator:
    def __init__(self, reasoning_agent: ReasoningAgent, personality_agent: PersonalityAgent):
        logger.info("Initializing AssistantOrchestrator...")
        self.reasoning_agent = reasoning_agent
        self.personality_agent = personality_agent
        self.chat_handler = ChatHandler()

    def handle_user_message(self, session_id: str, username: str, user_message: str):
        if not user_message:
            return

        logger.info(f"Processing user message for session {session_id}: {user_message}")

        try:
            # Store the user message in the chat history
            self.chat_handler.store_human_context(username, user_message)

            # Generate an acknowledgment response to the user
            acknowledgment = self.personality_agent.generate_acknowledgment(user_message)
            ack_message_id = str(uuid.uuid4())
            self.chat_handler.stream_message(username, acknowledgment, ack_message_id)
            self.chat_handler.finalize_message(username, acknowledgment, "AI", ack_message_id)

            # Pass the request to the ReasoningAgent
            reasoning_result = self.reasoning_agent.process_request(user_message)

            # Generate the final response to the user, filtered through the PersonalityAgent
            final_response = self.personality_agent.generate_final_response(username, reasoning_result, self.chat_handler)
            final_message_id = str(uuid.uuid4())
            self.chat_handler.stream_message(username, final_response, final_message_id)
            self.chat_handler.finalize_message(username, final_response, "AI", final_message_id)

        except Exception as e:
            logger.exception(f"Error processing user message for session {session_id}: {e}")
            error_message = f"An error occurred: {str(e)}"
            error_message_id = str(uuid.uuid4())
            self.chat_handler.finalize_message(username, error_message, "AI", error_message_id)
