import os
from enum import Enum, auto

from agent_server.llms.EncryptedKeystore import EncryptedKeyStore
from agent_server.llms.FireworksAiRestLLM import FireworksAiRestLLM
from agent_server.llms.LLMInterface import LLMInterface
from agent_server.llms.ChatGPT4 import ChatGPT4
from agent_server.llms.OllamaRestLLM import OllamaLLM


# Define the ModelType Enum
class ModelType(Enum):
    CHATGPT4 = auto()
    FIREWORKS_LLAMA_3_1_8B = auto()
    OLLAMA_QWEN = auto()


class LLMFactory:
    @staticmethod
    def create_llm(model_type: ModelType) -> LLMInterface:
        key_store = EncryptedKeyStore('keys.json.enc')

        if model_type == ModelType.CHATGPT4:
            api_key = key_store.get_api_key("CHATGPT4_API_KEY")
            return ChatGPT4(api_key=api_key)

        elif model_type == ModelType.FIREWORKS_LLAMA_3_1_8B:
            api_key = key_store.get_api_key("FIREWORKS_API_KEY")
            return FireworksAiRestLLM(api_token=api_key)

        elif model_type == ModelType.OLLAMA_QWEN:
            base_url = "http://localhost:11434"  # Adjust as needed
            model_name = "qwen2.5:3b"  # Adjust as needed
            api_key = key_store.get_api_key("OLLAMA_API_KEY")
            return OllamaLLM(base_url=base_url, model_name=model_name, api_key=api_key)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Example instantiation
    llm_instance = LLMFactory.create_llm(ModelType.CHATGPT4)
    prompt = "Explain the theory of relativity."
    system_message = "You are a knowledgeable assistant."

    # Generate response
    response = llm_instance.generate_response(prompt, system_message)
    print("Response:", response)
