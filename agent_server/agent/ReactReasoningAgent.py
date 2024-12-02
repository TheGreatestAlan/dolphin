import json
from pathlib import Path
import re

from agent_server.agent.ReasoningAgent import ReasoningAgent
from agent_server.agent.UnknownFunctionError import UnknownFunctionError
from agent_server.function import functions
from agent_server.function.FunctionMapper import FunctionMapper
from agent_server.agent.JsonFormatEnforcer import JsonFormatEnforcer
from agent_server.agent.JsonFunctionCreator import JsonFunctionCreator
from agent_server.function.function_definitions import generate_json_definitions
from agent_server.llms.LLMFactory import LLMFactory, ModelType

# HEMMINGWAY BRIDGE
# Ok, so you've got a pretty good react agent running.  I guess you could design a way to test
# in an automated way different answers to the different models that you can configure the agent
# to use.

# For now though, I think this is your next task.  You need to have a chat agent that talks to
# the user, passes the user requests to the react agent.  You need to consider how a conversation
# will be translated to user requests that the react agent will see.  Do they get the whole conversation?
# maybe the chat client will give the react agent a set of context as well.  Either way, it probably
# needs to be a separate agent and llm, that will eventually be pulled in by the rest_orchestrator.
# That chat agent will need the ability to stream, so maybe get the fireworks ai streaming working.
class ReactException(Exception):
    """Custom exception for errors in the ReActAgent steps."""
    pass


class ReActReasoningAgent(ReasoningAgent):
    def __init__(self):
        self.json_function_creator = JsonFunctionCreator(LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_8B))

        # Resolve paths to prompt files
        script_dir = Path(__file__).resolve().parent
        general_prompt_path = (script_dir / '../prompt/ReActGeneralPrompt.txt').resolve()
        planning_prompt_path = (script_dir / '../prompt/ReActPlanningPrompt.txt').resolve()
        observation_prompt_path = (script_dir / '../prompt/ReActObservationPrompt.txt').resolve()

        # Load the general system prompt
        self.general_prompt = self._load_prompt_from_file(general_prompt_path)

        # Load the step-specific prompts
        self.planning_prompt = self._load_prompt_from_file(planning_prompt_path)
        self.observation_prompt = self._load_prompt_from_file(observation_prompt_path)
        self.json_format_enforcer = JsonFormatEnforcer(LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_2_11B))

        # Load available actions
        self.available_actions = self._generate_available_actions()

        self.chat_handler = None
        self.function_mapper = FunctionMapper()

        self.plan_llm = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_70B)
        self.observation_llm = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_70B)

    def _generate_available_actions(self) -> str:
        # Get the function definitions
        functions_data = generate_json_definitions()

        # Extract actions and their descriptions
        actions = [
            f"- {func['action']}: {func['description']}"
            for func in functions_data.get("available_actions", [])
            if 'action' in func and 'description' in func
        ]

        return "\nAvailable actions:\n" + "\n".join(actions)

    def _load_prompt_from_file(self, file_path: Path) -> str:
        try:
            with file_path.open('r') as file:
                return file.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt file not found at: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading prompt file: {e}")

    def _load_actions_from_functions_json(self, functions_json_path: Path) -> str:
        try:
            with functions_json_path.open('r') as file:
                functions_data = json.load(file)
            # Extract actions and concatenate them to the prompt
            actions = [
                f"- {func['action']}"
                for func in functions_data.get("available_actions", [])
                if 'action' in func
            ]
            return "\nAvailable actions:\n" + "\n".join(actions)
        except FileNotFoundError:
            raise ValueError(f"Functions JSON file not found at: {functions_json_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in functions JSON file.")
        except Exception as e:
            raise ValueError(f"Error reading functions JSON file: {e}")

    def _validate_and_parse_json(self, json_string: str, expected_format: str = None) -> dict:
        """
        Validates and parses a JSON string. If invalid, attempts to enforce the expected format.
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            if expected_format:
                try:
                    return json.loads(self.json_format_enforcer.create_json(expected_format, json_string))
                except json.JSONDecodeError:
                    raise ReactException("Error enforcing JSON format.")
            else:
                raise ReactException("Invalid JSON format.")

    def _format_conversation(self, conversation: list) -> str:
        formatted_conversation = ""
        for msg in conversation:
            role = msg['role']
            content = msg['content']
            formatted_conversation += f"{role.capitalize()}: {content}\n"
        return formatted_conversation.strip()

    def _generate_thought(self, conversation: list) -> str:
        # Exclude the main system prompt
        prompt_conversation = conversation[1:]  # Exclude the main system prompt

        # Concatenate the general prompt and the planning prompt
        system_message = self.general_prompt + "\n" + self.planning_prompt + "\n" + self.available_actions

        # Format the conversation for the LLM
        prompt = self._format_conversation(prompt_conversation)

        # Generate the assistant's thought process
        assistant_thought = self.plan_llm.generate_response(prompt, system_message)

        return assistant_thought

    def _generate_observation(self, conversation: list, user_request) -> bool:
        # Prepare conversation content for observation check
        prompt_conversation = conversation[1:]  # Exclude the main system prompt
        formatted_user_request = '***USER REQUEST***'+ user_request + '***USER REQUEST***'
        system_message = self.observation_prompt
        prompt = self._format_conversation(prompt_conversation)
        prompt = prompt + formatted_user_request

        # Generate the observation response and parse JSON for `is_answered`
        observation_response = self.observation_llm.generate_response(prompt, system_message)

        expected_format = '''
        {
          "is_answered": true or false,  // true if you have enough information to answer, false if more actions are needed
          "answer": "this is the answer",  // optional, if is_answered is true
        }
        '''

        observation_data = self._validate_and_parse_json(observation_response, expected_format)
        return observation_data

    def _needs_action(self, assistant_response: str) -> bool:
        try:
            # Attempt to extract JSON from code block if present
            code_block_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(code_block_pattern, assistant_response, re.DOTALL)
            action_json = match.group(1) if match else assistant_response.strip()

            # Validate and parse JSON content
            action_data = self._validate_and_parse_json(action_json)
            return 'action' in action_data
        except ReactException:
            return False

    def _extract_action(self, assistant_response: str) -> tuple:
        try:
            # Attempt to extract JSON from a code block if present
            code_block_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(code_block_pattern, assistant_response, re.DOTALL)
            action_json = match.group(1) if match else assistant_response.strip()

            # Validate and parse JSON content, enforce format if needed
            expected_format = '''
            {
              "action": "<action_name>",
              "action_prompt": "<natural_language_description>"
            }'''
            action_data = self._validate_and_parse_json(action_json, expected_format)

            action_name = action_data.get("action")
            action_prompt = action_data.get("action_prompt", "")

            if action_name.lower() == "no_action":
                params = None
            else:
                try:
                    params = self.json_function_creator.create_json(action_name, action_prompt)
                except UnknownFunctionError as e:
                    raise ReactException("Unknown function:" + action_name)

            return action_name, params
        except ReactException as e:
            raise ReactException(f"Error in _extract_action: {e}")

    def _perform_action(self, action: str, params: dict) -> str:
        # Execute the action and return the result
        function_response = self.function_mapper.handle_function_call(params, None)

        return function_response

    def _is_confident(self, assistant_observation: str) -> bool:
        try:
            # Parse the assistant's observation as JSON
            observation_data = self._validate_and_parse_json(assistant_observation)
            final_answer = observation_data.get("final_answer")
            # Check if 'final_answer' is not None (i.e., not null)
            return final_answer is not None
        except ReactException:
            return False

    def _generate_final_response(self, conversation: list) -> str:
        # System prompt for the final response step
        final_response_system_prompt = (
            "You have gathered sufficient information to answer the user's request. "
            "Using only the collected action results and conversation history, "
            "synthesize a final response that directly addresses the user's question. "
            "Do not include any additional reasoning or assumptions.  BE CONCISE AND EXACT WITH YOUR ANSWER"
        )

        # Exclude initial system prompt to focus on action results and conversation history
        prompt_conversation = conversation[1:]  # Skip the initial general prompt
        formatted_conversation = self._format_conversation(prompt_conversation)

        # Combine the system prompt with the formatted conversation
        system_message = f"{self.general_prompt}\n{final_response_system_prompt}"

        # Generate the final response from the assistant
        final_response = LLMFactory.get_singleton(ModelType.OPTILLM).generate_response(formatted_conversation, system_message)

        return final_response

    def _format_conversation(self, conversation: list) -> str:
        formatted_conversation = ""
        for idx, msg in enumerate(conversation, 1):
            role = msg['role']
            content = msg['content']
            formatted_conversation += f"{'-' * 50}\nMessage {idx} - {role.upper()}:\n{content}\n"
        formatted_conversation += '-' * 50  # Add a separator at the end
        return formatted_conversation

    def process_request(self, user_input: str) -> str:
        max_retries = 5
        retry_count = 0
        # Initialize the conversation with the general system prompt and user input
        conversation = [
            {'role': 'system', 'content': self.general_prompt},
            {'role': 'user', 'content': user_input}
        ]
        while True:
            try:
                assistant_thought = self._generate_thought(conversation)
                conversation.append({'role': 'assistant-plan', 'content': assistant_thought})

                action, params = self._extract_action(assistant_thought)

                if action.lower() != 'no_action':
                    action_result = self._perform_action(action, params)
                    conversation.append({'role': 'action_result', 'content': action_result})

                observation = self._generate_observation(conversation, user_input)
                conversation.append({'role': 'assistant-observation', 'content': observation})

                if observation.get("is_answered"):
                    return observation.get("answer")

            except ReactException as e:
                conversation.append({'role': 'exception', 'content': str(e)})
                retry_count += 1
                if retry_count >= max_retries:
                    return "An error occurred: Maximum retries exceeded. Exiting."
                else:
                    continue

def main():
    reactAgent = ReActReasoningAgent()
    reactAgent.process_request(
        "From my inventory, explain who I am as a person"
    )


if __name__ == "__main__":
    main()
