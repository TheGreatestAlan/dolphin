import json
from pathlib import Path

from agent_server.agent.UnknownFunctionError import UnknownFunctionError
from agent_server.function.FunctionName import FunctionName
from agent_server.llms.LLMInterface import LLMInterface


def _load_function_definition(definition_path: str) -> str:
    """Reads the system message content from the specified text file."""
    try:
        with Path(definition_path).open("r") as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError(f"Function definition file not found at: {definition_path}")


def build_system_message(function_definition: str) -> str:
    """Constructs a system message using the provided function definition."""
    return (
        "You are responsible for interpreting user requests and generating a JSON response based on "
        "the provided action and parameters.\n\n" + function_definition +
        "\nPlease generate JSON responses for queries related to retrieving container contents according to this format. "
        "YOU WILL RESPOND WITH ONLY JSON AND NOTHING MORE."
    )


class JsonFunctionCreator:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    def create_json(self, function_name: str, user_request: str) -> dict:
        """Generates JSON for a specified function based on a user request."""
        # Check if the function_name is valid by matching it against the Enum
        if not FunctionName.has_value(function_name):
            raise UnknownFunctionError(f"Unknown function name: {function_name}")

        # Retrieve the corresponding FunctionName Enum and load the system message
        function_enum = FunctionName[function_name.upper()]
        function_definition = _load_function_definition(function_enum.definition_path)
        system_message = build_system_message(function_definition)

        # Call the LLM and generate the JSON response
        response_json = self.llm_interface.generate_response(user_request, system_message)

        # Parse and return the JSON response
        try:
            return json.loads(response_json)
        except json.JSONDecodeError:
            # If JSON parsing fails, attempt to correct the JSON using the format_json method
            corrected_json = self.format_json(response_json)
            try:
                return json.loads(corrected_json)
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
