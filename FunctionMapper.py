import json
import os

from FunctionResponse import FunctionResponse
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
from integrations.ChatHandler import ChatHandler
from integrations.Inventory import Inventory


class FunctionMapper:
    def __init__(self, inventory: Inventory, function_generator: InventoryFunctionGenerator, chat_handler: ChatHandler):
        self.inventory = inventory
        self.function_generator = function_generator
        self.chat_handler = chat_handler
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
            "end_session": "end_session(session_id: str)"
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
            show_results_to_user = parameters.pop("showResultsToUser", False)

            action_mapping = {
                "get_inventory": lambda _: self.inventory.get_inventory(),
                "find_location": lambda params: self.inventory.find_location(params["item_name"]),
                "get_container": lambda params: self.inventory.get_container(params["container_id"]),
                "create_items": lambda params: self.inventory.create_items(params["container"], params["items"]),
                "delete_items": lambda params: self.inventory.delete_items(params["container"], params["items"]),
                "send_message": lambda params: self.chat_handler.send_message(session_id, params["content"]),
                "list_actions": lambda _: self.list_functions(),
                "poll_response": lambda params: self.chat_handler.poll_response(session_id),
                "start_session": lambda _: self.chat_handler.start_session(),
                "end_session": lambda params: self.chat_handler.end_session(session_id)
            }

            if action_name not in action_mapping:
                return {'action_name': action_name, 'response': f"Unknown action: {action_name}. Expected actions: {list(action_mapping.keys())}"}

            required_params = {
                "find_location": ["item_name"],
                "get_container": ["container_id"],
                "create_items": ["container", "items"],
                "delete_items": ["container", "items"],
                "send_message": ["content"],
            }

            if action_name in required_params:
                for param in required_params[action_name]:
                    if param not in parameters:
                        return {
                            'action_name': action_name,
                            'response': f"Missing required parameter '{param}' for action '{action_name}'. Expected signature: {self.expected_signatures[action_name]}"
                        }

            response = action_mapping[action_name](parameters)

            if show_results_to_user:
                self.cache[session_id] = response

            return self.wrap_to_action_response(response, action_name)

        except KeyError as e:
            action_name = prompt.get("action", {}).get("action", "unknown")
            return {'action_name': action_name, 'response': f"KeyError: {str(e)}. Expected signature: {self.expected_signatures.get(action_name, 'unknown')}"}
        except Exception as e:
            print("Error:", str(e))
            return {'action_name': 'error', 'response': str(e)}

    def list_functions(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, './prompt/functionList.txt')
            with open(file_path, 'r') as file:
                functions = json.load(file)
            return functions
        except FileNotFoundError:
            return {"error": "functions.json file not found"}
        except json.JSONDecodeError as e:
            return {"error": f"Error decoding JSON: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
