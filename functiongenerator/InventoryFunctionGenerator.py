import json
import uuid
from typing import Any

class InventoryFunctionGenerator:
    def __init__(self, text_generator: Any):
        self.text_generator = text_generator

    def generate_function_call(self, prompt: str) -> dict:
        system_message = (
            "You are chatGPT-4, a well-trained LLM used to assist humans. "
            "You must respond only with a valid JSON object representing a function call. Do not include any other text or explanation in your response. "
            "Here are some examples of how you should respond:\n\n"
            "1. If asked 'add a hammer to container 5', respond with:\n"
            "{\"action\": \"create_items\", \"parameters\": {\"container\": \"5\", \"items\": [\"hammer\"]}}\n"
            "2. If asked 'remove a screwdriver from container 10', respond with:\n"
            "{\"action\": \"delete_items\", \"parameters\": {\"container\": \"10\", \"items\": [\"screwdriver\"]}}\n"
            "3. If asked 'retrieve the entire inventory', respond with:\n"
            "{\"action\": \"get_inventory\", \"parameters\": {}}\n"
            "4. If asked 'find the location of item named screwdriver', respond with:\n"
            "{\"action\": \"find_location\", \"parameters\": {\"item_name\": \"screwdriver\"}}\n"
            "5. If asked 'find the location of container 15', respond with:\n"
            "{\"action\": \"find_container\", \"parameters\": {\"container_id\": \"15\"}}\n\n"
            "Respond only with the JSON object, without any additional text."
        )

        conversation_id = str(uuid.uuid4())
        conversation_id, response_text = self.text_generator.generate_response(conversation_id, prompt, system_message)

        # Extract JSON part from the response
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            raise Exception("The response is not valid JSON")

        return response_json
