import json
from pathlib import Path

from agent_server.llms.LLMInterface import LLMInterface

# HEMMINGWAY BRIDGE
# You've got some bones on a react agent, you're building it through and this remains
# untested.  You for sure need to handle the action creation step correctly, the
# way you're doing it in the sequencer.

# probably take this one at a time, pick a question, "Give me a list of all the
# locations for cooking supplies I need for pho." And just run that through
# one step at a time, observing the input and outputs of each step.

class ReActAgent:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

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
        return "[ACTION]" in assistant_response and "[/ACTION]" in assistant_response

    def _extract_action(self, assistant_response: str) -> tuple:
        try:
            action_start = assistant_response.index("[ACTION]") + len("[ACTION]")
            action_end = assistant_response.index("[/ACTION]")
            action_json = assistant_response[action_start:action_end].strip()
            action_data = json.loads(action_json)
            action_name = action_data.get("action")
            params = action_data.get("params", {})
            return action_name, params
        except (ValueError, json.JSONDecodeError):
            raise ValueError("Failed to extract action from assistant's response.")

    def _perform_action(self, action: str, params: dict) -> str:
        # Execute the action and return the result
        # Placeholder implementation; replace with your actual action execution logic
        result = f"Result of {action} with parameters {params}"
        return result

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

                # Step 4: Decide to Repeat or Conclude
                if self._is_confident(assistant_observation):
                    # Step 5: Generate Final Response
                    final_response = self._generate_final_response(conversation)
                    conversation.append({'role': 'assistant', 'content': final_response})
                    return final_response
                else:
                    # Loop back to planning with updated conversation
                    continue
            else:
                # No action needed; provide the final answer
                final_response = assistant_thought
                return final_response
