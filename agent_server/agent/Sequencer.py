import json
from pathlib import Path
from agent_server.llms.LLMInterface import LLMInterface

#HEMMINGWAY BRIDGE
# You have a sequencer, you need to decide how complex you want it to handle
# It's not doing a good job of identifying compound tasks and breaking thme
# down.  You could run the compound tasks through the function chooser, if
# it fails, then attempt to break it down further.
class Sequencer:
    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

        # Resolve the absolute path of the prompt file based on the script's directory
        script_dir = Path(__file__).resolve().parent
        prompt_file_path = (script_dir / '../prompt/SequencerSystemPrompt.txt').resolve()

        self.system_prompt = self._load_prompt_from_file(prompt_file_path)

    def _load_prompt_from_file(self, file_path: Path) -> str:
        try:
            with file_path.open('r') as file:
                return file.read()
        except FileNotFoundError:
            raise ValueError(f"Prompt file not found at: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading prompt file: {e}")

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
