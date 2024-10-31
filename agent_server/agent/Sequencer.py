import json
from pathlib import Path

from agent_server.llms.LLMInterface import LLMInterface


class Sequencer:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

        # Resolve the path to the prompt file
        script_dir = Path(__file__).resolve().parent
        prompt_file_path = (script_dir / '../prompt/SequencerSystemPrompt.txt').resolve()
        functions_json_path = (script_dir / '../prompt/functionList.json').resolve()

        # Load and prepare the system prompt
        self.system_prompt = self._load_prompt_from_file(prompt_file_path)
        self.system_prompt += self._load_actions_from_functions_json(functions_json_path)

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
            actions = [f"- {func['action']}" for func in functions_data.get("available_actions", []) if
                       'action' in func]
            return "\nAvailable actions:\n" + "\n".join(actions)
        except FileNotFoundError:
            raise ValueError(f"Functions JSON file not found at: {functions_json_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in functions JSON file.")
        except Exception as e:
            raise ValueError(f"Error reading functions JSON file: {e}")

    def parse_tasks(self, user_request: str) -> list:
        response_json = self.llm_interface.generate_response(user_request, self.system_prompt)
        return self._parse_response_to_tasks(response_json)

    def _parse_response_to_tasks(self, response_json: str) -> list:
        try:
            tasks = json.loads(response_json)
            if isinstance(tasks, list):
                return tasks
            raise ValueError("Expected a list of tasks in the response format.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in LLM response.")
