import uuid

from Inventory import Inventory
from llms.LLMInterface import LLMInterface

class SmartFindingInventoryClient(Inventory):
    def __init__(self, inventory: Inventory, text_generator: LLMInterface):
        self.inventory = inventory
        self.text_generator = text_generator

    def get_inventory(self):
        return self.inventory.get_inventory()

    def find_location(self, item_name):
        # Get the entire inventory
        inventory = self.get_inventory()

        # Prepare the prompt for the text generator
        inventory_list = "\n".join([f"{idx + 1}: {item}" for idx, item in enumerate(inventory)])
        prompt = f"From this list, where is the {item_name}?\n{inventory_list}"
        system_message = "You are an inventory searching specialist"

        # Use the text generator to find the location
        response = self.text_generator.generate_response(prompt, system_message)

        return response

    def find_container(self, container_id):
        return self.inventory.find_container(container_id)

    def create_items(self, container, items):
        return self.inventory.create_items(container, items)

    def delete_items(self, container, items):
        return self.inventory.delete_items(container, items)
