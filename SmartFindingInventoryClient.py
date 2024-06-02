import json

from FunctionResponse import FunctionResponse, Status
from integrations.Inventory import Inventory
from llms.LLMInterface import LLMInterface


class SmartFindingInventoryClient(Inventory):
    def __init__(self, inventory: Inventory, text_generator: LLMInterface):
        self.inventory = inventory
        self.text_generator = text_generator

    def map_to_function_response(self, response):
        if response is None:
            return FunctionResponse("FAILURE", "An error occurred")
        elif 200 <= response.status_code < 300:
            # Successful response (may include 204 No Content)
            try:
                data = response.json()
                return FunctionResponse("SUCCESS", data)
            except ValueError:
                # Likely an empty response body
                return FunctionResponse("SUCCESS", "Action completed successfully")
        else:
            # Handle error responses
            error_msg = f"Error {response.status_code}: {response.text}"
            return FunctionResponse("FAILURE", error_msg)

    def get_inventory(self) -> FunctionResponse:
        function_res = self.inventory.get_inventory()
        return function_res

    def find_location(self, item_name) -> FunctionResponse:
        # Get the entire inventory
        inventory = self.inventory.get_inventory().response

        prompt = f"From this list, where is the {item_name}?\n{inventory}"
        system_message = "You are an inventory searching specialist"

        # Create the JSON object to be passed to the LLM
        json_payload = {
            "prompt": prompt,
            "system_message": system_message
        }

        # Print out the JSON
        print(json.dumps(json_payload, indent=4))

        # Use the text generator to find the location
        response = self.text_generator.generate_response(prompt, system_message)

        return FunctionResponse(Status.SUCCESS, response)

    def get_container(self, container_id) -> FunctionResponse:
        return self.inventory.get_container(container_id)

    def create_items(self, container, items) -> FunctionResponse:
        return self.inventory.create_items(container, items)

    def delete_items(self, container, items) -> FunctionResponse:
        return self.inventory.delete_items(container, items)
