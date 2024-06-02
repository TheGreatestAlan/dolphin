import json
import os
from integrations.Inventory import Inventory
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
from integrations.ChatHandler import ChatHandler

class FunctionMapper:
    def __init__(self, inventory: Inventory, function_generator: InventoryFunctionGenerator, chat_handler: ChatHandler):
        self.inventory = inventory
        self.function_generator = function_generator
        self.chat_handler = chat_handler

    def handle_function_call(self, prompt: dict, session_id: str):
        try:
            # Validate function call structure
            if "action" not in prompt:
                raise ValueError("Invalid function call format: missing 'action'")

            # Extract action and parameters
            action = prompt.get("action")
            parameters = action.get("parameters", {})

            # Debug: Print the entire prompt, action, and parameters
            print("Prompt:", prompt)
            print("Action:", action)
            print("Parameters:", parameters)

            # Define a mapping of actions to InventoryClient methods
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

            # Define expected method signatures
            expected_signatures = {
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

            # Debug: Check if action is in the action_mapping
            action_name = action.get("action")
            print("Action Name:", action_name)
            print("Session id:", session_id)
            if action_name not in action_mapping:
                return {'action_name': action_name, 'response': f"Unknown action: {action_name}. Expected actions: {list(action_mapping.keys())}"}

            required_params = {
                "find_location": ["item_name"],
                "get_container": ["container_id"],
                "create_items": ["container", "items"],
                "delete_items": ["container", "items"],
                "send_message": ["content"],
            }

            # Debug: Print required parameters for the action
            print("Required Parameters:", required_params.get(action_name, "None"))

            # Check for missing required parameters
            if action_name in required_params:
                for param in required_params[action_name]:
                    if param not in parameters:
                        return {
                            'action_name': action_name,
                            'response': f"Missing required parameter '{param}' for action '{action_name}'. Expected signature: {expected_signatures[action_name]}"
                        }

            # Add session_id to parameters where needed
            #if action_name in ["poll_response", "end_session", "send_message"]:
            #    parameters["session_id"] = session_id

            # Execute the action
            response = action_mapping[action_name](parameters)
            return {'action_name': action_name, 'response': response}

        except KeyError as e:
            action_name = prompt.get("action", {}).get("action", "unknown")
            return {'action_name': action_name, 'response': f"KeyError: {str(e)}. Expected signature: {expected_signatures.get(action_name, 'unknown')}"}
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
