import json

from agent_server.function.FunctionName import FunctionName
from agent_server.llms.LLMInterface import LLMInterface


class FunctionChooser:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

    def choose_function(self, user_request: str) -> str:
        """Chooses the most appropriate function based on the user's request."""
        # Generate a system message listing available functions
        system_message = self.build_system_message()

        # Call the LLM to interpret the user request and select the function
        response_json = self.llm_interface.generate_response(user_request, system_message)

        # Parse and return the chosen function name from the LLM response
        chosen_function = json.loads(response_json).get("chosen_function")

        # Allow "none" as a valid response if no function matches
        if chosen_function == "none" or (chosen_function and FunctionName.has_value(chosen_function)):
            return chosen_function
        else:
            raise ValueError(f"Unrecognized function returned by LLM: {chosen_function}")

    def build_system_message(self) -> str:
        """Constructs a system message listing all available functions."""
        function_list = "\n".join([f"- {fn.function_name}" for fn in FunctionName])
        return (
            "You are responsible for choosing the most appropriate function from the list below based on the user's request.\n\n"
            "### Available Functions:\n"
            f"{function_list}\n\n"
            "Based on the user's query, respond with the function name in JSON format, like this:\n"
            '{ "chosen_function": "<function_name>" }\n\n'
            "If no function matches, respond with:\n"
            '{ "chosen_function": "none" }'
        )
