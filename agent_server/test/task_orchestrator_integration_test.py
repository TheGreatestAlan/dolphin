import os
import unittest
from agent_server.llms.OllamaRestLLM import OllamaLLM
from agent_server.task_orchestrator import TaskOrchestrator


class TestTaskOrchestratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the Ollama instance as the LLMInterface implementation
        cls.ollama_llm = OllamaLLM(base_url="http://localhost:11434", model_name="qwen2.5:3b")
        os.environ["ORGANIZER_SERVER_URL"]= "http://127.0.0.1:8080"


        # Instantiate the TaskOrchestrator directly, which initializes and holds all agents
        cls.task_orchestrator = TaskOrchestrator(cls.ollama_llm)

    @classmethod
    def tearDownClass(cls):
        # Properly shut down the executor in TaskOrchestrator after tests complete
        cls.task_orchestrator.shutdown()

    def test_process_user_request_success(self):
        # Example user request that involves multiple tasks
        user_request = "what's in container 8?";

        # Process the request through the TaskOrchestrator
        results = self.task_orchestrator.process_user_request(user_request)

        # Check that results are a list of JSON objects (one per task)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0, "Expected at least one task result")
        print(results);

        # Verify each result has a JSON structure indicating success or error
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertTrue(
                "error" in result or ("action" in result and "parameters" in result),
                "Each task should have an 'error' or be a valid JSON action with parameters"
            )

    def test_process_user_request_no_function_match(self):
        # Input with a task that should not match any function
        user_request = "Translate this text to Spanish."

        # Process the request and verify the function chooser finds no match
        results = self.task_orchestrator.process_user_request(user_request)

        # Expecting an error message since no suitable function should match
        self.assertEqual(len(results), 1)
        self.assertIn("error", results[0])
        self.assertIn("No suitable function found", results[0]["error"])


if __name__ == '__main__':
    unittest.main()
