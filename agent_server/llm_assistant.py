import json
import uuid
import logging
import os
from flask import jsonify

from agent_server.integrations.SmartFindingInventoryClient import SmartFindingInventoryClient
from agent_server.assistant import Assistant
from agent_server.InventoryFunctionGenerator import InventoryFunctionGenerator
from agent_server.integrations.InventoryRestClient import InventoryClient
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery
from agent_server.integrations.StreamManager import StreamManager
from agent_server.integrations.local_device_action import LocalDeviceAction
from llms.RestLLM import RestLLM
from llms.ChatGPT4 import ChatGPT4
from FunctionMapper import FunctionMapper
from agent_server.integrations.ChatHandler import ChatHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# HEMMINGWAY BRIDGE
# looks like the process isn't following up correctly on reading the response from the llm
# and prompting it back.  Follow the logic through by asking it for the time, put in break
# points and follow the response.

class LLMAssistant(Assistant):
    def __init__(self, chat_handler: ChatHandler, stream_manager: StreamManager):
        logger.info("Initializing LLMAssistant...")
        llm_type = os.getenv('LLM_TYPE', '')
        llm_url = os.getenv('LLM_URL', '')
        self.stream_manager = stream_manager
        self.temp_buffers = {} #Dictionary to hold streaming text

        if llm_type == 'RestLLM':
            logger.info("Using RestLLM")
            self.llm_client = RestLLM(llm_url)
        elif llm_type == 'ChatGPT':
            logger.info("Using ChatGPT4")
            self.llm_client = ChatGPT4()
        else:
            logger.error(f"Unsupported LLM_TYPE: {llm_type}")
            raise ValueError(f"Unsupported LLM_TYPE: {llm_type}")

        sessions_file_path = '../chatapp/sessions.json'
        self.MAX_NESTING_LEVEL = 3
        self.chat_handler = chat_handler

        # Initialize Inventory, FunctionMapper, and ChatHandler
        rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
        smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, self.llm_client)
        function_generator = InventoryFunctionGenerator(self.llm_client)
        knowledge_query = KnowledgeQuery(self.llm_client)
        local_device_action = LocalDeviceAction(stream_manager)
        self.function_mapper = FunctionMapper(smart_finding_inventory_client, function_generator, self.chat_handler,
                                              knowledge_query, local_device_action)

        self.SYSTEM_MESSAGE = ''
        self.FUNCTION_LIST = ''

        self.read_system_message()
        self.read_function_list()

    def read_system_message(self, file_path='./prompt/SystemPrompt.txt'):
        global SYSTEM_MESSAGE
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
        try:
            with open(absolute_path, 'r') as file:
                SYSTEM_MESSAGE = file.read()
            logger.info("System message read successfully.")
        except FileNotFoundError:
            logger.error(f"File not found: {absolute_path}")
        except Exception as e:
            logger.exception(f"Error reading system message: {e}")

    def read_function_list(self, file_path='./prompt/functionList.txt'):
        global FUNCTION_LIST
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
        try:
            with open(absolute_path, 'r') as file:
                FUNCTION_LIST = file.read()
            logger.info("Function list read successfully.")
        except FileNotFoundError:
            logger.error(f"File not found: {absolute_path}")
        except Exception as e:
            logger.exception(f"Error reading function list: {e}")

    def stream_immediate_response(self, response_generator, session_id, username):
        logger.info(f"Streaming immediate response for session {session_id}.")
        in_content = False
        finding_buffer = ""
        complete_buffer = ""
        message_id = str(uuid.uuid4())

        def is_unescaped_quote(buffer, position):
            if position == 0:
                return True
            return buffer[position - 1] != '\\'

        for chunk in response_generator:
            finding_buffer += chunk
            complete_buffer += chunk

            if not in_content:
                immediate_start = finding_buffer.find('"immediate_response": {')
                if immediate_start != -1:
                    content_start = finding_buffer.find('"content": "', immediate_start)
                    if content_start != -1:
                        content_start += len('"content": "')
                        finding_buffer = finding_buffer[content_start:]
                        in_content = True

            if in_content:
                content_end = -1
                pos = 0
                while pos < len(finding_buffer):
                    pos = finding_buffer.find('"', pos)
                    if pos == -1:
                        break
                    if is_unescaped_quote(finding_buffer, pos):
                        content_end = pos
                        break
                    pos += 1

                if content_end != -1:
                    content_to_stream = finding_buffer[:content_end]
                    self.parse_llm_stream(session_id, username, content_to_stream, message_id)
                    finding_buffer = finding_buffer[content_end + 1:]
                    in_content = False
                    self.parse_llm_stream(session_id, username, "[DONE]", message_id)
                else:
                    self.parse_llm_stream(session_id, username, finding_buffer, message_id)
                    finding_buffer = ""

        complete_buffer = complete_buffer.replace('[DONE]', '')
        return complete_buffer

    def parse_llm_stream(self, session_id, username, data_chunk, message_id):

        self.stream_manager.add_to_text_buffer(session_id, data_chunk)

        """Parse incoming LLM stream data and immediately send it back without waiting for [DONE]."""
        if session_id not in self.temp_buffers:
            self.temp_buffers[session_id] = {}

        if message_id not in self.temp_buffers[session_id]:
            self.temp_buffers[session_id][message_id] = ""

        # Accumulate data chunk
        self.temp_buffers[session_id][message_id] += data_chunk

        # If the message is complete, finalize it
        if self.temp_buffers[session_id][message_id].strip().endswith("[DONE]"):
            complete_message = self.temp_buffers[session_id][message_id].replace("[DONE]", "").strip()
            self.temp_buffers[session_id].pop(message_id)  # Clear the temp buffer for this message

            # Finalize the message
            self.chat_handler.finalize_message(username, complete_message, "AI")

            # If no more messages remain for this session, clear the session from temp_buffers
            if not self.temp_buffers[session_id]:
                self.temp_buffers.pop(session_id)


    def handle_llm_response(self, response, session_id, username, streaming=False, nesting_level=0):
        logger.info(
            f"Handling LLM response for session {session_id}, streaming={streaming}, nesting_level={nesting_level}, response={response}.")
        if nesting_level > self.MAX_NESTING_LEVEL:
            logger.warning(f"Max nesting level reached for session {session_id}.")
            return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

        try:
            if streaming:
                response = self.stream_immediate_response(response, session_id, username)
                response_dict = json.loads(response)
                if 'immediate_response' in response_dict:
                    del response_dict['immediate_response']
                if 'action' in response_dict:
                    show_result_to_user = response_dict.get('action', {}).get('show_results_to_user', False)
                    return self.process_response_content(json.dumps(response_dict), show_result_to_user, session_id, username, nesting_level)
                else:
                    return
            else:
                full_response = ''.join(response())
                if "\"name\": \"send_message\"" not in full_response:
                    return self.process_response_content(full_response, session_id, username, nesting_level)
                else:
                    return
        except Exception as e:
            logger.exception(f"Error handling LLM response for session {session_id}: {e}")
            return jsonify({"error": str(e)})

    def process_response_content(self, generated_text, show_results_to_user, session_id, username, nesting_level):
        logger.info(f"Processing response content for session {session_id}, nesting_level={nesting_level}, generated_text={generated_text}")
        if nesting_level > self.MAX_NESTING_LEVEL:
            return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

        try:
            response_json = json.loads(generated_text)
            if "action" not in response_json or "self_message" not in response_json:
                raise ValueError("Incorrect response format")

            function_response = self.function_mapper.handle_function_call(response_json, session_id)
            if show_results_to_user:
                message_id = str(uuid.uuid4())
                self.parse_llm_stream(session_id, username, function_response.response, message_id)
                self.parse_llm_stream(session_id, username, "[DONE]", message_id)
                return

            context = self.chat_handler.get_context(session_id)
            next_chunk = self.send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'],
                                                  function_response, context, True)
            return self.handle_llm_response(next_chunk, session_id, username, True, nesting_level + 1)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error processing response content: {e}")
            error_response = {
                "self_message": response_json['self_message'] if 'self_message' in response_json else '',
                "action": {
                    "action": "self_message",
                    "parameters": {
                        "content": f"Invalid request, incorrect response format of this message: {generated_text}, return "
                                   f"valid json"
                    }
                }
            }
            context = self.chat_handler.get_context(session_id)

            return self.handle_llm_response(
                self.send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'],
                                         json.dumps(error_response), context),
                session_id,
                username,
                True,
                nesting_level + 1
            )

    def send_message_to_llm(self, system_message, user_message, self_message, action_response, context=None,
                            streaming=False):
        log_dict = {
            "user_message": user_message,
            "self_message": self_message,
            "action_response": action_response,
            "conversation_history": context or []
        }
        prompt_dict = {
            "system_message": system_message,
            "user_message": user_message,
            "self_message": self_message,
            "action_response": action_response,
            "conversation_history": context or []
        }
        log = json.dumps(log_dict, indent=4)
        prompt = json.dumps(prompt_dict, indent=4)

        logger.info(f"Sending message to LLM for streaming={streaming}. message={log}")
        if streaming:
            return self.llm_client.stream_response(prompt, system_message)
        else:
            return self.llm_client.generate_response(prompt, system_message)

    def message_assistant(self, session_id, username, user_message):
        if not user_message:
            return
        logger.info(f"User message received for session {session_id}: {user_message}")
        self.chat_handler.store_human_context(username, user_message)
        context = self.chat_handler.get_context(username)
        # Use the streaming option in send_message_to_llm
        response_stream = self.send_message_to_llm(
            SYSTEM_MESSAGE + "\n" + FUNCTION_LIST,
            user_message,
            None,
            None,
            context,
            streaming=True
        )
        self.handle_llm_response(response_stream, session_id, username, True, )
