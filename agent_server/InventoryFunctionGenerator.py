import json

from llms.LLMInterface import LLMInterface


class InventoryFunctionGenerator:
    def __init__(self, text_generator: LLMInterface):
        self.text_generator = text_generator

    def generate_function_call(self, prompt: str) -> dict:
        system_message = (
            "You are chatGPT-4, a well-trained LLM used to assist humans. You must respond only with a valid JSON "
            "object representing a function call. Do not include any other text or explanation in your response."
            "Here are some examples of how you should respond:\n\n"
            "1. If asked 'add a hammer to container 5', respond with:\n"
            "{\"action\": \"create_items\", \"parameters\": {\"container\": \"5\", \"items\": [\"hammer\"]}}\n"
            "2. If asked 'remove a screwdriver from container 10', respond with:\n"
            "{\"action\": \"delete_items\", \"parameters\": {\"container\": \"10\", \"items\": [\"screwdriver\"]}}\n"
            "3. If asked 'retrieve the entire inventory', respond with:\n"
            "{\"action\": \"get_inventory\", \"parameters\": {}}\n"
            "4. If asked 'find the location of item named screwdriver', respond with:\n"
            "{\"action\": \"find_location\", \"parameters\": {\"item_name\": \"screwdriver\"}}\n"
            "5. If asked 'What's in container 5?', respond with:\n"
            "{\"action\": \"get_container\", \"parameters\": {\"container_id\": \"5\"}}\n"
            "6. If asked 'Oops I messed up, delete the phone from container 6', respond with:\n"
            "{\"action\": \"delete_items\", \"parameters\": {\"container\": \"6\", \"items\": [\"phone\"]}}\n"
            "7. If asked 'In container 7 add a screwdriver, gorilla glue, nerf balls, and a fan', respond with:\n"
            "{\"action\": \"create_items\", \"parameters\": {\"container\": \"7\", \"items\": [\"screwdriver\", "
            "\"gorilla glue\", \"nerf balls\", \"fan\"]}}\n\n"
            "Respond only with the JSON object, without any additional text."
        )

        response_text = self.text_generator.generate_response(prompt, system_message)
        # Extract JSON part from the response
        try:
            response_json = json.loads(response_text)
        except json.JSONDecodeError:
            raise Exception("The response is not valid JSON")

        return response_json