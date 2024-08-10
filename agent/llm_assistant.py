import json
import uuid

from flask import jsonify
import os

from SmartFindingInventoryClient import SmartFindingInventoryClient
from agent.assistant import Assistant
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
from integrations.InventoryRestClient import InventoryClient
from integrations.KnowledgeQuery import KnowledgeQuery
from llms.RestLLM import RestLLM
from llms.ChatGPT4 import ChatGPT4  # Import the ChatGPT4 class
from FunctionMapper import FunctionMapper
from integrations.ChatHandler import ChatHandler


class LLMAssistant(Assistant):
    def __init__(self, chat_handler: ChatHandler):
        # Determine which LLM client to use based on environment variable
        llm_type = os.getenv('LLM_TYPE', 'RestLLM')  # Default to 'RestLLM' if not specified
        llm_url = os.getenv('LLM_URL', 'http://127.0.0.1:5002')

        if llm_type == 'RestLLM':
            self.llm_client = RestLLM(llm_url)
        elif llm_type == 'ChatGPT':
            self.llm_client = ChatGPT4()  # Assuming ChatGPT4 does not require a URL
        else:
            raise ValueError(f"Unsupported LLM_TYPE: {llm_type}")

        sessions_file_path = 'sessions.json'
        self.MAX_NESTING_LEVEL = 3
        self.chat_handler = chat_handler

        # Initialize Inventory, FunctionMapper, and ChatHandler
        rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
        smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, self.llm_client)
        function_generator = InventoryFunctionGenerator(self.llm_client)
        knowledge_query = KnowledgeQuery(self.llm_client)
        self.function_mapper = FunctionMapper(smart_finding_inventory_client, function_generator, self.chat_handler,
                                              knowledge_query)

        # Ensure the system message is read at startup
        self.SYSTEM_MESSAGE = ''
        self.FUNCTION_LIST = ''

        self.read_system_message()
        self.chat_handler.load_sessions_from_file()
        self.read_function_list()

    def read_system_message(self, file_path='../prompt/SystemPrompt.txt'):
        global SYSTEM_MESSAGE
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
        try:
            with open(absolute_path, 'r') as file:
                SYSTEM_MESSAGE = file.read()
        except FileNotFoundError:
            print(f"File not found: {absolute_path}")
        except Exception as e:
            print(f"Error reading system message: {e}")

    def read_function_list(self, file_path='../prompt/functionList.txt'):
        global FUNCTION_LIST
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
        try:
            with open(absolute_path, 'r') as file:
                FUNCTION_LIST = file.read()
        except FileNotFoundError:
            print(f"File not found: {absolute_path}")
        except Exception as e:
            print(f"Error reading system message: {e}")

    def stream_immediate_response(self, response_generator, session_id):
        in_content = False
        finding_buffer = ""  # Buffer used for finding content markers
        complete_buffer = ""  # Buffer to build the entire message
        message_id = str(uuid.uuid4())

        def is_unescaped_quote(buffer, position):
            # Check if the quote is unescaped
            if position == 0:
                return True  # The first character is an unescaped quote
            return buffer[position - 1] != '\\'

        for chunk in response_generator:
            finding_buffer += chunk
            complete_buffer += chunk

            if not in_content:
                # Look for the start of "immediate_response"
                immediate_start = finding_buffer.find('"immediate_response": {')
                if immediate_start != -1:
                    # Ensure we are within the "immediate_response"
                    content_start = finding_buffer.find('"content": "', immediate_start)
                    if content_start != -1:
                        content_start += len('"content": "')
                        finding_buffer = finding_buffer[content_start:]  # Trim buffer to start at content
                        in_content = True

            if in_content:
                # Find the position of the unescaped ending quote
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
                    # Extract and send the content up to the end marker
                    content_to_stream = finding_buffer[:content_end]
                    self.chat_handler.receive_stream_data(session_id, content_to_stream, message_id)
                    finding_buffer = finding_buffer[content_end + 1:]  # Keep the remaining buffer
                    in_content = False  # Reset flag after handling content
                    self.chat_handler.receive_stream_data(session_id, "[DONE]", message_id)
                else:
                    # Stream the content as it arrives without the final part
                    self.chat_handler.receive_stream_data(session_id, finding_buffer, message_id)
                    finding_buffer = ""  # Clear buffer after streaming

        # Remove the [DONE] flag if present
        complete_buffer = complete_buffer.replace('[DONE]', '')

        # Return the entire response as a JSON string
        return complete_buffer

    def handle_llm_response(self, response, session_id, streaming=False, nesting_level=0):
        if nesting_level > self.MAX_NESTING_LEVEL:
            return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

        try:
            if streaming:
                # Pass the buffer to a dedicated function for streaming processing
                response = self.stream_immediate_response(response, session_id)
                response_dict = json.loads(response)

                # Remove the 'immediate_response' key from the dictionary
                if 'immediate_response' in response_dict:
                    del response_dict['immediate_response']

                # Pass the modified response dictionary to process_response_content
                if 'action' in response_dict:
                    return self.process_response_content(json.dumps(response_dict), session_id, nesting_level)
                else:
                    return

            else:
                # Full response handling for non-streaming responses
                full_response = ''.join(response())
                if "\"name\": \"send_message\"" not in full_response:
                    return self.process_response_content(full_response, session_id, nesting_level)
                else:
                    return

        except Exception as e:
            print(e)
            return jsonify({"error": str(e)})  # Generic error handling

    def process_response_content(self, generated_text, session_id, nesting_level):
        print(generated_text)
        if nesting_level > self.MAX_NESTING_LEVEL:
            return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

        try:
            response_json = json.loads(generated_text)
            if "action" not in response_json or "self_message" not in response_json:
                raise ValueError("Incorrect response format")

            # Handling actions and potentially recursive operations
            function_response = self.function_mapper.handle_function_call(response_json, session_id)

            action = response_json.get("action")
            parameters = action.get("parameters", {})
            show_results_to_user = parameters.pop("showResultsToUser", False)
            if show_results_to_user:
                return

            # Retrieve context from self.chat_handler
            context = self.chat_handler.get_context(session_id)

            # Recursive call to process further based on the self_message and function response
            next_chunk = self.send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'],
                                                  function_response,
                                                  context, True)
            return self.handle_llm_response(next_chunk, session_id, True, nesting_level + 1)

        except (ValueError, KeyError, json.JSONDecodeError) as e:
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
                True,
                nesting_level + 1
            )

    def send_message_to_llm(self, system_message, user_message, self_message, action_response, context=None,
                            streaming=False):
        prompt_dict = {
            "system_message": system_message,
            "user_message": user_message,
            "self_message": self_message,
            "action_response": action_response,
            "conversation_history": context or []
        }
        prompt = json.dumps(prompt_dict, indent=4)

        if streaming:
            return self.llm_client.stream_response(prompt, system_message)
        else:
            return self.llm_client.generate_response(prompt, system_message)

    def message_assistant(self, session_id, user_message):
        self.chat_handler.store_human_context(session_id, user_message)
        context = self.chat_handler.get_context(session_id)
        # Use the streaming option in send_message_to_llm
        response_stream = self.send_message_to_llm(
            SYSTEM_MESSAGE + "\n" + FUNCTION_LIST,
            user_message,
            None,
            None,
            context,
            streaming=True
        )
        self.handle_llm_response(response_stream, session_id, True, )
