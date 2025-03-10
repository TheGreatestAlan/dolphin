import unittest

from agent_server.agent.Sequencer import Sequencer
from agent_server.llms.OllamaRestLLM import OllamaLLM


class TestSequencer(unittest.TestCase):
    def setUp(self):
        base_url = "http://localhost:11434"
        model_name = "qwen2.5:3b"

        # Instantiate the OllamaLLM with the mocked parameters
        self.ollama_llm = OllamaLLM(base_url, model_name)

        self.sequencer = Sequencer(self.ollama_llm)

    def test_parse_tasks_success(self):
        user_request = "what's in container 4?"
        tasks = self.sequencer.parse_tasks(user_request)

        # Expected output matches the mocked response structure
        expected_tasks = [
            {"task": "what's in container 4?", "function": "get_container", "order": 1},
        ]
        self.assertEqual(tasks, expected_tasks)

if __name__ == '__main__':
    unittest.main()
