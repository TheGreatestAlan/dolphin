import json
import os
from pathlib import Path

from agent_server.FunctionMapper import FunctionMapper
from agent_server.InventoryFunctionGenerator import InventoryFunctionGenerator
from agent_server.agent.JsonFunctionCreator import JsonFunctionCreator
from agent_server.integrations.InventoryRestClient import InventoryClient
from agent_server.integrations.KnowledgeQuery import KnowledgeQuery
from agent_server.integrations.SmartFindingInventoryClient import SmartFindingInventoryClient
from agent_server.integrations.local_device_action import LocalDeviceAction
from agent_server.llms.LLMFactory import LLMFactory, ModelType
from agent_server.llms.LLMInterface import LLMInterface

# HEMMINGWAY BRIDGE
# You got some of the react steps running, you're at the Observation phase
# You noticed that it was doing logic in the observation phase.  That's wrong
# It's job is to look at the steps, actions and responses given, and determine
# if that's enough information to answer the question.  If it is to package th
# into the final answer, if not to continue the cycle.

class ReActAgent:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.json_function_creator = JsonFunctionCreator(llm_interface)

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

        # Load available actions
        self.available_actions = self._load_actions_from_functions_json(functions_json_path)

        self.chat_handler = None
        rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
        smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, llm_interface)
        function_generator = InventoryFunctionGenerator(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_8B))
        knowledge_query = KnowledgeQuery(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_1_8B))
        local_device_action = LocalDeviceAction(None)
        self.function_mapper = FunctionMapper(smart_finding_inventory_client, function_generator, self.chat_handler,
                                              knowledge_query, local_device_action)


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
        assistant_thought = self.llm_interface.generate_response(prompt, system_message)
        return assistant_thought

    def _generate_observation(self, conversation: list) -> str:
        # Exclude the main system prompt
        prompt_conversation = conversation[1:]  # Exclude the main system prompt

        # Concatenate the general prompt and the observation prompt
        system_message = self.general_prompt + "\n" + self.observation_prompt

        # Format the conversation for the LLM
        prompt = self._format_conversation(prompt_conversation)

        # Generate the assistant's observation
        assistant_observation = self.llm_interface.generate_response(prompt, system_message)
        return assistant_observation

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
        except Exception:
            # For any other exceptions, assume no action is needed
            return False

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
            action_data = json.loads(action_json)
            action_name = action_data.get("action")
            action_prompt = action_data.get("action_prompt", "")

            # Use JsonFunctionCreator to generate the action parameters
            params = self.json_function_creator.create_json(action_name, action_prompt)

            return action_name, params
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON decoding error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to extract action from assistant's response: {e}")

    def _perform_action(self, action: str, params: dict) -> str:
        # Execute the action and return the result
        function_response = self.function_mapper.handle_function_call(params, None)

        print(f"Result of {action} with parameters {params}: " + json.dumps(function_response))

        return function_response

    def _is_confident(self, assistant_observation: str) -> bool:
        # Evaluate if the agent is confident enough to provide the final answer
        # For example, check if the observation contains a specific keyword
        return "[FINAL_ANSWER]" in assistant_observation

    def _generate_final_response(self, conversation: list) -> str:
        # Exclude the main system prompt
        prompt_conversation = conversation[1:]  # Exclude the main system prompt

        # Use only the general prompt for the final response
        system_message = self.general_prompt

        # Format the conversation for the LLM
        prompt = self._format_conversation(prompt_conversation)

        # Generate the final response
        final_response = self.llm_interface.generate_response(prompt, system_message)
        return final_response

    def process_request(self, user_input: str) -> str:
        # Initialize the conversation with the general system prompt and user input
        conversation = [
            {'role': 'system', 'content': self.general_prompt},
            {'role': 'user', 'content': user_input}
        ]

        while True:
            # Step 1: Planning
            assistant_thought = self._generate_thought(conversation)
            conversation.append({'role': 'assistant', 'content': assistant_thought})

            # Determine if an action is needed
            if self._needs_action(assistant_thought):
                # Extract the action and parameters
                action, params = self._extract_action(assistant_thought)

                # Step 2: Action Execution
                action_result = self._perform_action(action, params)
                conversation.append({'role': 'action_result', 'content': action_result})

                # Step 3: Observation
                assistant_observation = self._generate_observation(conversation)
                conversation.append({'role': 'assistant', 'content': assistant_observation})
                break
                #
                # # Step 4: Decide to Repeat or Conclude
                # if self._is_confident(assistant_observation):
                #     # Step 5: Generate Final Response
                #     final_response = self._generate_final_response(conversation)
                #     conversation.append({'role': 'assistant', 'content': final_response})
                #     return final_response
                # else:
                #     # Loop back to planning with updated conversation
                #     continue
            else:
                # No action needed; provide the final answer
                final_response = assistant_thought
                return final_response

def main():
        reactAgent = ReActAgent(LLMFactory.create_llm(ModelType.FIREWORKS_LLAMA_3_70B))
        reactAgent.process_request("what is twice barrack obama's age?")

if __name__ == "__main__":
    main()

