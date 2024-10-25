import unittest
from unittest.mock import MagicMock

from agent_server.agent.JsonFunctionCreator import JsonFunctionCreator
from agent_server.llms.LLMInterface import LLMInterface
from agent_server.llms.OllamaRestLLM import OllamaLLM


class TestJsonFunctionCreator(unittest.TestCase):
    def setUp(self):
        base_url = "http://localhost:11434"
        model_name = "qwen2.5:3b"

        # Instantiate the OllamaLLM with the mocked parameters
        self.ollama_llm = OllamaLLM(base_url, model_name)


        # Instantiate the JsonFunctionCreator with the OllamaLLM instance
        self.json_creator = JsonFunctionCreator(self.ollama_llm)

    def test_create_json_for_get_container(self):
        # Define the function name and user request
        function_name = "get_container"
        user_request = "What's in container 5?"

        # Call create_json to generate the JSON for the get_container function
        result = self.json_creator.create_json(function_name, user_request)

        # Expected JSON output based on the mock response from the LLM
        expected_result = {
            "action": "get_container",
            "parameters": {
                "container_id": "5"
            }
        }

        # Assert that the result matches the expected JSON
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
