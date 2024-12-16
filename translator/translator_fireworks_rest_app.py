import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
import requests

from agent_server.llms.EncryptedKeyStore import EncryptedKeyStore
from agent_server.llms.LLMFactory import LLMFactory, ModelType

app = Flask(__name__)

# Configuration
MODEL_TYPE = ModelType.FIREWORKS_LLAMA_3_70B  # Centralize the model type here
SUPPORTED_LANGUAGES = ["English", "German", "French", "Italian", "Portuguese", "Hindi", "Spanish", "Thai"]  # Centralize supported languages

# Read API Base URL and API Key from environment variables
key_store = EncryptedKeyStore('keys.json.enc')
API_KEY = key_store.get_api_key("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai"

rating_threshold = 7

# Keep instructions in English and just refer to the target language in the prompt.
INSTRUCTIONS = {
    "persona_system_message": (
        "You are a knowledgeable and charming conversation partner who always responds in clear, "
        "appealing {target_language}. Provide useful information and interesting conversation. "
        "Keep responses to no more than a paragraph or two at a time."
    ),
    "translation_system_message": (
        "You are a concise and direct translator who translates to {target_language}. If the message is incorrect, "
        "provide the corrected message and, if necessary, a brief explanation. Ignore diacritical marks."
    ),
    "rating_system_message": (
        "You are a language evaluator. Evaluate the correctness of the following {target_language} sentence "
        "from 1 to 10, where 1 is completely incorrect and 10 is completely correct. "
        "Do not provide any explanation—only the integer."
    )
}

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(endpoint):
    if endpoint == "api/tags":
        return jsonify(
            {
                "models": [
                    {
                        "name": f"llama3.2:3b_{language.lower()}",
                        "model": f"llama3.2:3b_{language.lower()}",
                        "modified_at": "2024-10-24T08:28:31.8474952-06:00",
                        "size": 2019393189,
                        "digest": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": "3.2B",
                            "quantization_level": "Q4_K_M"
                        },
                    }
                    for language in SUPPORTED_LANGUAGES
                ]
            }
        ), 200

    endpoint_mapping = {
        "api/chat": "/inference/v1/api/chat",
        "chat/completions": "/inference/v1/chat/completions",
        "api/generate": "/inference/v1/api/generate",
    }

    target_path = endpoint_mapping.get(endpoint, f"/{endpoint}")
    target_url = f"{BASE_URL}{target_path}"

    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    headers['Authorization'] = f"Bearer {API_KEY}"
    data = request.get_json() if request.is_json else request.data
    params = request.args

    try:
        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid message format"}), 400

        # Determine model and language
        model_name = data.get("model_name", f"llama3.2:3b_{SUPPORTED_LANGUAGES[0].lower()}")  # Default to the first language
        target_language = extract_language_from_model_name(model_name)

        last_message = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        response = LLMFactory.get_singleton(MODEL_TYPE).stream_response(last_message, "None")
        return generate_ollama_response(response, model_name)

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500


def extract_language_from_model_name(model_name: str) -> str:
    # Extract language from model name (e.g., "llama3.2:3b_italian" -> "Italian")
    if "_" in model_name:
        language_code = model_name.split("_")[-1].capitalize()
        if language_code.lower() in [lang.lower() for lang in SUPPORTED_LANGUAGES]:
            return language_code.capitalize()
    return SUPPORTED_LANGUAGES[0]  # Default to the first language


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
