import json
import os
import uuid

from agent_server.FunctionResponse import FunctionResponse, Status
from agent_server.InventoryFunctionGenerator import InventoryFunctionGenerator
from agent_server.integrations.ChatHandler import ChatHandler
from agent_server.integrations.Inventory import Inventory
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery


class FunctionMapper:
    def __init__(self, inventory: Inventory, function_generator: InventoryFunctionGenerator, chat_handler: ChatHandler, knowledge_query: KnowledgeQuery):
        self.inventory = inventory
        self.function_generator = function_generator
        self.chat_handler = chat_handler
        self.knowledge_query = knowledge_query
        self.cache = {}  # Cache to store results temporarily

        # Define expected method signatures
        self.expected_signatures = {
            "get_inventory": "get_inventory()",
            "find_location": "find_location(item_name: str)",
            "get_container": "get_container(container_id: str)",
            "create_items": "create_items(container: str, items: list)",
            "delete_items": "delete_items(container: str, items: list)",
            "send_message": "send_message(session_id: str, content: str)",
            "list_actions": "list_actions()",
            "poll_response": "poll_response(session_id: str)",
            "start_session": "start_session()",
            "end_session": "end_session(session_id: str)",
            "knowledge_query": "knowledge_query(query: str)"
        }

    def wrap_to_action_response(self, function_response: FunctionResponse, action_name:str) -> dict:
        action_name = action_name  # Replace with actual action name
        status = function_response.status.name
        value = function_response.response
        return {
            "action_name": action_name,
            "status": status,
            "value": value
        }

    def handle_function_call(self, prompt: dict, session_id: str):
        try:
            if "action" not in prompt:
                raise ValueError("Invalid function call format: missing 'action'")

            action = prompt.get("action")
            parameters = action.get("parameters", {})
            action_name = action.get("action")

            action_mapping = {
                "get_inventory": lambda _: self.inventory.get_inventory(),
                "find_location": lambda params: self.inventory.find_location(params["item_name"]),
                "get_container": lambda params: self.inventory.get_container(params["container_id"]),
                "create_items": lambda params: self.inventory.create_items(params["container"], params["items"]),
                "delete_items": lambda params: self.inventory.delete_items(params["container"], params["items"]),
                "send_message": lambda params: self.chat_handler.send_message(session_id, params["content"]),
                "list_actions": lambda _: self.list_functions(),
                "poll_response": lambda params: self.chat_handler.poll_response(session_id),
                "start_session": lambda _: self.chat_handler.get_or_create_user(),
                "end_session": lambda params: self.chat_handler.end_session(session_id),
                "knowledge_query": lambda params: self.knowledge_query.query(params["query"])
            }

            if action_name not in action_mapping:
                return {'action_name': action_name, 'response': f"Unknown action: {action_name}. Expected actions: {list(action_mapping.keys())}"}

            required_params = {
                "find_location": ["item_name"],
                "get_container": ["container_id"],
                "create_items": ["container", "items"],
                "delete_items": ["container", "items"],
                "send_message": ["content"],
                "knowledge_query": ["query"]
            }

            if action_name in required_params:
                for param in required_params[action_name]:
                    if param not in parameters:
                        return {
                            'action_name': action_name,
                            'response': f"Missing required parameter '{param}' for action '{action_name}'. Expected signature: {self.expected_signatures[action_name]}"
                        }

            response = action_mapping[action_name](parameters)

            return self.wrap_to_action_response(response, action_name)

        except KeyError as e:
            action_name = prompt.get("action", {}).get("action", "unknown")
            return {'action_name': action_name, 'response': f"KeyError: {str(e)}. Expected signature: {self.expected_signatures.get(action_name, 'unknown')}"}
        except Exception as e:
            print("Error:", str(e))
            return {'action_name': 'error', 'response': str(e)}

    def list_functions(self) -> FunctionResponse:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, 'prompt/functionList.txt')
            with open(file_path, 'r') as file:
                functions = json.load(file)
            return FunctionResponse(Status.SUCCESS, functions)
        except FileNotFoundError:
            return FunctionResponse(Status.FAILURE, {"error": "functions.json file not found"})
        except json.JSONDecodeError as e:
            return FunctionResponse(Status.FAILURE,{"error": f"Error decoding JSON: {str(e)}"})
        except Exception as e:
            return FunctionResponse(Status.Failure,{"error": str(e)})
