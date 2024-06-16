import unittest
import time
import os

from llms.Ollama3 import Ollama3LLM
from llms.llamacpp import LlamaCpp


class TestLlamaCppStreaming(unittest.TestCase):
    SYSTEM_MESSAGE = ''

    @classmethod
    def read_system_message(cls, file_path='../prompt/SystemPrompt.txt'):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
        try:
            with open(absolute_path, 'r') as file:
                cls.SYSTEM_MESSAGE = file.read()
        except FileNotFoundError:
            print(f"File not found: {absolute_path}")
        except Exception as e:
            print(f"Error reading system message: {e}")

    @classmethod
    def setUpClass(cls):
        cls.read_system_message()

    def setUp(self):
        self.llama_cpp = LlamaCpp()

    def test_stream_response(self):
        prompt = "\"user_message\":\"can you show me what's in the inventory?\""
        system_message = self.SYSTEM_MESSAGE

        try:
            start_time = time.time()
            content = self.llama_cpp.stream_response(prompt, system_message)
            elapsed_time = time.time() - start_time
            print(f"\nStreaming response time: {elapsed_time:.2f} seconds")
            print(f"\nFull Response:\n{content}")
        except Exception as e:
            self.fail(f"Streaming failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
