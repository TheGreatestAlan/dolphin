import uuid
import logging

from agent_server.agent.ChatAgent import ChatAgent
from agent_server.agent.ReasoningAgent import ReasoningAgent
from agent_server.assistant import Assistant
from agent_server.integrations.ChatHandler import ChatSession

logger = logging.getLogger(__name__)

class AssistantOrchestrator(Assistant):
    def __init__(self, reasoning_agent: ReasoningAgent, chat_agent: ChatAgent, ):
        logger.info("Initializing AssistantOrchestrator...")
        self.reasoning_agent = reasoning_agent
        self.chat_agent = chat_agent

    def message_assistant(self, chat_session:ChatSession, user_message: str):
        if not user_message:
            return

        try:
            # Store the user message in the chat history
            chat_session.store_human_context(user_message)
            context = chat_session.get_current_chat()

            final_response_stream = self.chat_agent.process_user_message(context, chat_session)

            message_id = str(uuid.uuid4())
            logger.debug(f"Generated message ID for acknowledgment: {message_id}")

            reasoning_result = self.reasoning_agent.process_request(context)

            # Generate the final response to the user, filtered through the PersonalityAgent
            response_id = str(uuid.uuid4())
            logger.debug(f"Generated response ID for final response: {response_id}")
            self.chat_agent.generate_final_response(
                reasoning_result, chat_session)

        except Exception as e:
            logger.exception(f"Error processing user message {e}")
