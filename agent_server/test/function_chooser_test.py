import unittest

from agent_server.agent.FunctionChooser import FunctionChooser
from agent_server.llms.OllamaRestLLM import OllamaLLM


class TestFunctionChooserIntegration(unittest.TestCase):
    def setUp(self):
        # Set up the base URL and model name for the Ollama LLM instance
        base_url = "http://localhost:11434"  # Replace with actual base URL if different
        model_name = "qwen2.5:3b"  # Replace with actual model name in Ollama

        # Initialize the Ollama LLM with the provided URL and model name
        self.ollama_llm = OllamaLLM(base_url, model_name)

        # Instantiate the FunctionChooser with the OllamaLLM instance
        self.function_chooser = FunctionChooser(self.ollama_llm)

    def test_choose_function_get_container(self):
        # Define a user request that should match the "get_container" function
        user_request = "Can you tell me what's in container 5?"

        # Use the FunctionChooser to select the function
        chosen_function = self.function_chooser.choose_function(user_request)

        # Assert that the chosen function is "get_container"
        self.assertEqual(chosen_function, "get_container")

    def test_choose_function_invalid_request(self):
        # Define a user request that should not match any known function
        user_request = "This is a random question that doesn't match any function."

        # Use the FunctionChooser to select the function
        chosen_function = self.function_chooser.choose_function(user_request)

        # Assert that the chosen function is "none"
        self.assertEqual(chosen_function, "none")

if __name__ == "__main__":
    unittest.main()
