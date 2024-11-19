import json
import os
from pathlib import Path

from agent_server.function.FunctionMapper import FunctionMapper
from agent_server.agent.JsonFormatEnforcer import JsonFormatEnforcer
from agent_server.agent.JsonFunctionCreator import JsonFunctionCreator
from agent_server.llms.LLMFactory import LLMFactory, ModelType

# Hemmingway Bridge
# K you're going to need to change the function list that you're loading in the
# plan step.  It's loading the full function descriptions, a huge waste of tokens
# instead you'll probably want a way to pull the function names and descriptions
# from an enum or something
#
# You're going to also need to rework functionmapper so it works, you were in the middle
# of a refactor on that

# YOUR FIRST TASK IS TO SEPARATE OUT THE JSON EXTRACTION AND CALL IT BASICALLY WHEREVER IT'S NEEDED
# WHICH IS EVERYWHERE

class ReactException(Exception):
    """Custom exception for errors in the ReActAgent steps."""
    pass


class ReActAgent:
    def __init__(self):
        self.json_function_creator = JsonFunctionCreator(LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_8B))

        # Resolve paths to prompt files
        script_dir = Path(__file__).resolve().parent
        general_prompt_path = (script_dir / '../prompt/ReActGeneralPrompt.txt').resolve()
        planning_prompt_path = (script_dir / '../prompt/ReActPlanningPrompt.txt').resolve()
        observation_prompt_path = (script_dir / '../prompt/ReActObservationPrompt.txt').resolve()
        functions_json_path = (script_dir / '../prompt/functionList.json').resolve()

        # Load the general system prompt
        self.general_prompt = self._load_prompt_from_file(general_prompt_path)

        # Load the step-specific prompts
        self.planning_prompt = self._load_prompt_from_file(planning_prompt_path)
        self.observation_prompt = self._load_prompt_from_file(observation_prompt_path)
        self.json_format_enforcer = JsonFormatEnforcer(LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_8B))

        # Load available actions
        self.available_actions = self._load_actions_from_functions_json(functions_json_path)

        self.chat_handler = None
        self.function_mapper = FunctionMapper()

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
        assistant_thought = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(prompt,
                                                                                                        system_message)

        return assistant_thought

    def _generate_observation(self, conversation: list) -> bool:
        # Prepare conversation content for observation check
        prompt_conversation = conversation[1:]  # Exclude the main system prompt
        system_message = self.observation_prompt
        prompt = self._format_conversation(prompt_conversation)

        # Generate the observation response and parse JSON for `is_answered`
        observation_response = LLMFactory.get_singleton(ModelType.OPTILLM).generate_response(prompt, system_message)

        try:
            # Parse observation JSON to check if enough information has been gathered
            observation_data = json.loads(observation_response)
            is_answered = observation_data.get("is_answered", False)
        except json.JSONDecodeError:
            is_answered = False  # Default to False if parsing fails

        return is_answered

    def _needs_action(self, assistant_response: str) -> bool:
        import re
        try:
            # Attempt to extract JSON from code block if present
            code_block_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(code_block_pattern, assistant_response, re.DOTALL)
            if match:
                # Extract JSON content from the code block
                action_json = match.group(1)
            else:
                # No code block found; assume the entire response is JSON
                action_json = assistant_response.strip()
            # Attempt to parse the JSON content
            action_data = json.loads(action_json)
            # Check if 'action' key is present
            return 'action' in action_data
        except json.JSONDecodeError:
            # If JSON decoding fails, no action is needed
            return False
        except Exception as e:
            raise ReactException(f"Error in _generate_observation: {e}")

    def _extract_action(self, assistant_response: str) -> tuple:
        import re
        try:
            # Attempt to extract JSON from a code block if present
            code_block_pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(code_block_pattern, assistant_response, re.DOTALL)
            if match:
                # Extract JSON content from the code block
                action_json = match.group(1)
            else:
                # No code block found; assume the entire response is JSON
                action_json = assistant_response.strip()

            # Attempt to parse the JSON content
            try:
                action_data = json.loads(action_json)
            except json.JSONDecodeError:
                try:
                    expected_format = '''
{
  "action": "<action_name>",
  "action_prompt": "<natural_language_description>"
}'''
                    action_data = json.loads(self.json_format_enforcer.create_json(expected_format, action_json))
                except json.JSONDecodeError:
                    raise ReactException(f"Error in _extract_action: ")

            action_name = action_data.get("action")
            action_prompt = action_data.get("action_prompt", "")

            if action_name.lower() == "no_action":
                params = None
            else:
                params = self.json_function_creator.create_json(action_name, action_prompt)

            return action_name, params
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
        except Exception as e:
            raise ReactException(f"Error in _extract_action: {e}")

    def _perform_action(self, action: str, params: dict) -> str:
        # Execute the action and return the result
        function_response = self.function_mapper.handle_function_call(params, None)

        print(f"Result of {action} with parameters {params}: " + json.dumps(function_response))

        return function_response

    def _is_confident(self, assistant_observation: str) -> bool:
        try:
            # Parse the assistant's observation as JSON
            observation_data = json.loads(assistant_observation)
            final_answer = observation_data.get("final_answer")
            # Check if 'final_answer' is not None (i.e., not null)
            return final_answer is not None
        except json.JSONDecodeError:
            # If parsing fails, assume not confident
            return False

    def _generate_observation(self, conversation: list) -> bool:
         # Use a simplified observation prompt to determine completeness
         prompt_conversation = conversation[1:]
         system_message = self.observation_prompt
         prompt = self._format_conversation(prompt_conversation)

         assistant_observation = LLMFactory.get_singleton(ModelType.OPTILLM).generate_response(prompt, system_message)
         observation_data = json.loads(assistant_observation)

         # Check if the final answer is present
         return observation_data.get("final_answer") is not None

    def _generate_final_response(self, conversation: list) -> str:
        # System prompt for the final response step
        final_response_system_prompt = (
            "You have gathered sufficient information to answer the user's request. "
            "Using only the collected action results and conversation history, "
            "synthesize a final response that directly addresses the user's question. "
            "Do not include any additional reasoning or assumptions."
        )

        # Exclude initial system prompt to focus on action results and conversation history
        prompt_conversation = conversation[1:]  # Skip the initial general prompt
        formatted_conversation = self._format_conversation(prompt_conversation)

        # Combine the system prompt with the formatted conversation
        system_message = f"{self.final_response_prompt}\n{final_response_system_prompt}"

        # Generate the final response from the assistant
        final_response = LLMFactory.get_singleton(ModelType.OPTILLM).generate_response(formatted_conversation, system_message)

        return final_response

    def process_request(self, user_input: str) -> str:
        # Process request with new observe and finalize response steps
        conversation = [
            {'role': 'system', 'content': self.general_prompt},
            {'role': 'user', 'content': user_input}
        ]

        while True:
            assistant_thought = self._generate_thought(conversation)
            conversation.append({'role': 'assistant-plan', 'content': assistant_thought})

            action, params = self._extract_action(assistant_thought)
            action_result = self._perform_action(action, params)
            conversation.append({'role': 'action_result', 'content': action_result})

            if self._generate_observation(conversation):
                final_answer = self._generate_final_response(conversation)
                print(final_answer)
                break


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

                if self._generate_observation(conversation):
                    final_answer = self._generate_final_response(conversation)
                    print(final_answer)
                    break

            except ReactException as e:
                print(f"ReactException caught: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    print("Maximum retries exceeded. Exiting.")
                    break
                else:
                    print(f"Retrying... ({retry_count}/{max_retries})")
                    # Optionally adjust the conversation or state before retrying
                    continue

def main():
    reactAgent = ReActAgent()
    reactAgent.process_request(
        "What would jack black most likely want out of my inventory? Give me specific items from my inventory")


if __name__ == "__main__":
    main()
