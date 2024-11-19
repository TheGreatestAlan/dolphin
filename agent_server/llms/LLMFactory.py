import os
from enum import Enum, auto
from typing import Dict

from agent_server.llms.EncryptedKeystore import EncryptedKeyStore
from agent_server.llms.FireworksAiRestLLM import FireworksAiRestLLM
from agent_server.llms.LLMInterface import LLMInterface
from agent_server.llms.ChatGPT4 import ChatGPT4
from agent_server.llms.OllamaRestLLM import OllamaLLM
from agent_server.llms.OptiLLM import OptiLLM


# Define the ModelType Enum
class ModelType(Enum):
    CHATGPT4 = auto()
    FIREWORKS_LLAMA_3_1_8B = auto()
    FIREWORKS_LLAMA_3_1_405B = auto()
    FIREWORKS_LLAMA_3_70B = auto()
    FIREWORKS_LLAMA_3_2_11B = auto()
    FIREWORKS_LLAMA_3_2_3B = auto()
    OLLAMA_QWEN = auto()
    OPTILLM = auto()


class LLMFactory:
    _instances: Dict[ModelType, LLMInterface] = {}

    @staticmethod
    def create_llm(model_type: ModelType) -> LLMInterface:
        """Always create a new instance of the LLM."""
        key_store = EncryptedKeyStore('keys.json.enc')
        if model_type == ModelType.CHATGPT4:
            api_key = key_store.get_api_key("CHATGPT4_API_KEY")
            return ChatGPT4(api_key=api_key)

        elif model_type == ModelType.FIREWORKS_LLAMA_3_1_8B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key)

        elif model_type == ModelType.FIREWORKS_LLAMA_3_1_405B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key, model='accounts/fireworks/models/llama-v3p1-405b-instruct')

        elif model_type == ModelType.FIREWORKS_LLAMA_3_2_3B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key, model='accounts/fireworks/models/llama-v3p2-3b-instruct')

        elif model_type == ModelType.FIREWORKS_LLAMA_3_2_11B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key,
                                      model='accounts/fireworks/models/llama-v3p2-11b-vision-instruct')

        elif model_type == ModelType.FIREWORKS_LLAMA_3_70B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key, model='accounts/fireworks/models/llama-v3-70b-instruct-hf')

        elif model_type == ModelType.OLLAMA_QWEN:
            base_url = "http://localhost:11434"  # Adjust as needed
            model_name = "qwen2.5:3b"  # Adjust as needed
            api_key = key_store.get_api_key("OLLAMA_API_KEY")
            return OllamaLLM(base_url=base_url, model_name=model_name, api_key=api_key)

        elif model_type == ModelType.OPTILLM:
            return OptiLLM()

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def get_singleton(model_type: ModelType) -> LLMInterface:
        """Get or create the singleton for the specified model type."""
        # Check if the singleton already exists
        if model_type not in LLMFactory._instances:
            # Create and store the singleton if it does not exist
            LLMFactory._instances[model_type] = LLMFactory.create_llm(model_type)

        return LLMFactory._instances[model_type]


# Example usage
if __name__ == "__main__":
    # Example usage to create a new instance (non-singleton)
    new_llm_instance = LLMFactory.create_llm(ModelType.CHATGPT4)
    prompt = "Explain the theory of relativity."
    system_message = "You are a knowledgeable assistant."
    response = new_llm_instance.generate_response(prompt, system_message)
    print("Response from new instance:", response)

    # Example usage to get a singleton instance (shared instance)
    singleton_llm_instance = LLMFactory.get_singleton(ModelType.CHATGPT4)
    response = singleton_llm_instance.generate_response(prompt, system_message)
    print("Response from singleton instance:", response)
