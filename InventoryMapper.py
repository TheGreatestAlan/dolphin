from flask import jsonify
from Inventory import Inventory
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator

class InventoryMapper:
    def __init__(self, inventory: Inventory, function_generator: InventoryFunctionGenerator):
        self.inventory = inventory
        self.function_generator = function_generator

    def handle_text_inventory(self, prompt: str):
        try:
            function_call = self.function_generator.generate_function_call(prompt)

            # Debug: Print the function call
            print("Function Call:", function_call)

            # Validate function call structure
            if "action" not in function_call or "parameters" not in function_call:
                raise ValueError("Invalid function call format: missing 'action' or 'parameters'")

            action = function_call.get("action")
            parameters = function_call.get("parameters", {})

            # Define a mapping of actions to InventoryClient methods
            action_mapping = {
                "get_inventory": lambda _: self.inventory.get_inventory(),
                "find_location": lambda params: self.inventory.find_location(params["item_name"]),
                "find_container": lambda params: self.inventory.find_container(params["container_id"]),
                "create_items": lambda params: self.inventory.create_items(params["container"], params["items"]),
                "delete_items": lambda params: self.inventory.delete_items(params["container"], params["items"])
            }

            # Debug: Check if action is in the action_mapping
            print("Action:", action)
            if action not in action_mapping:
                raise ValueError(f"Unknown action: {action}")

            required_params = {
                "find_location": ["item_name"],
                "find_container": ["container_id"],
                "create_items": ["container", "items"],
                "delete_items": ["container", "items"]
            }

            # Debug: Print parameters and required parameters for the action
            print("Parameters:", parameters)
            if action in required_params:
                for param in required_params[action]:
                    if param not in parameters:
                        raise ValueError(f"Missing required parameter '{param}' for action '{action}'")

            # Execute the action
            response = action_mapping[action](parameters)
            return jsonify({'response': response})

        except Exception as e:
            print("Error:", str(e))
            return jsonify({'error': str(e)}), 500
