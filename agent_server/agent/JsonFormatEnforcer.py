import json
from agent_server.llms.LLMInterface import LLMInterface


def build_system_message(json_definition: str) -> str:
    """Constructs a system message using the provided function definition."""
    return (
        "You are an intelligent English to JSON interpreter responsible for taking the output of an LLM and enforcing "
        "the text into expected JSON. "
        "\nPlease take the incorrect input and put it into the expected JSON response:\n" +
        json_definition + "\n"
        "RESPOND ONLY WITH JSON"
    )


class JsonFormatEnforcer:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    def create_json(self, json_definition: str, user_request: str) -> dict:
        system_message = build_system_message(json_definition)

        # Call the LLM and generate the JSON response
        response_json = self.llm_interface.generate_response(user_request, system_message)

        # Parse and return the JSON response
        try:
            json.loads(response_json)
            return response_json
        except json.JSONDecodeError:
            # If JSON parsing fails, attempt to correct the JSON using the format_json method
            corrected_json = self.format_json(response_json)
            try:
                return corrected_json
            except json.JSONDecodeError as e:
                # If parsing still fails, raise an exception with details
                raise ValueError(f"Failed to parse JSON after correction: {e}")

    def format_json(self, invalid_json: str) -> str:
        """Attempts to correct invalid JSON using the LLM."""
        # Build the prompt for the LLM
        system_message = (
            "You are a JSON formatting assistant. "
            "Your task is to correct the following invalid JSON so that it is valid and well-formatted. "
            "Do not include any additional text or explanations; only return the corrected JSON.\n\n"
            "Invalid JSON:\n"
        )

        # Prepare the prompt with the invalid JSON
        prompt = system_message + invalid_json

        # Call the LLM to get the corrected JSON
        corrected_json = self.llm_interface.generate_response(prompt, "")

        return corrected_json
