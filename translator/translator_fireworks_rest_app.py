import json
import os
from datetime import datetime

from flask import Flask, request, Response, jsonify
import requests

from agent_server.llms.EncryptedKeyStore import EncryptedKeyStore
from agent_server.llms.LLMFactory import LLMFactory, ModelType

app = Flask(__name__)

# Read API Base URL and API Key from environment variables
key_store = EncryptedKeyStore('keys.json.enc')
API_KEY = key_store.get_api_key("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai"
model_name = "llama3.2:3b"
rating_threshold = 7

# You can dynamically change this from "Spanish" to "Italian" or any other language
TARGET_LANGUAGE = "Italian"

# Keep instructions in English and just refer to the target language in the prompt.
# Use placeholders where the target language is mentioned, then format the strings.
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
                        "name": model_name,
                        "model": "llama3.2:3b",
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

        last_message = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(last_message, "None")
        return generate_ollama_response(response)

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid message format"}), 400

    user_message = messages[-1].get("content", "").strip()

    # If user requests translation of last assistant message
    if user_message.lower() in ["traducir", "translate"]:
        last_assistant_message = get_last_assistant_message(messages)
        if not last_assistant_message:
            return jsonify({"error": "No assistant message found to translate."}), 400
        return english_translation(last_assistant_message, stream=True)

    # Otherwise, check language correctness rating
    language_attempt_rating = rate_language_attempt(user_message)
    if language_attempt_rating <= rating_threshold:
        return translate_to_target_language(user_message, True)
    else:
        return converse_in_target_language(user_message)


def converse_in_target_language(user_message: str):
    """
    Use the chosen language persona and system instructions to continue the conversation.
    """
    system_message = INSTRUCTIONS["persona_system_message"].format(target_language=TARGET_LANGUAGE)
    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        prompt=user_message,
        system_message=system_message
    )
    return generate_ollama_response(response)


def get_last_assistant_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return None


def rate_language_attempt(message: str) -> int:
    system_message = INSTRUCTIONS["rating_system_message"].format(target_language=TARGET_LANGUAGE)
    prompt = f"Rate the following {TARGET_LANGUAGE} sentence correctness from 1 to 10:\n\n{message}"

    rating_response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(
        prompt=prompt,
        system_message=system_message
    )

    # Attempt to parse the rating
    try:
        rating = int(rating_response.strip())
    except ValueError:
        rating = 1

    rating = max(1, min(rating, 10))
    return rating


def english_translation(message: str, stream: bool):
    system_message = "You are a concise and direct English translator. You will not be yappy."
    prompt = f"Translate the following text to English:\n\n{message}"

    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        prompt=prompt,
        system_message=system_message
    )
    return generate_ollama_response(response)


def translate_to_target_language(message: str, stream: bool):
    system_message = INSTRUCTIONS["translation_system_message"].format(target_language=TARGET_LANGUAGE)
    prompt = f"Translate the following text to {TARGET_LANGUAGE}:\n\n{message}"

    if stream:
        response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
            prompt=prompt,
            system_message=system_message
        )
        return generate_ollama_response(response)
    else:
        completion = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(
            prompt=prompt,
            system_message=system_message
        )
        return jsonify({"translation": completion})


def generate_ollama_response(response):
    def generate_chunks():
        yielded_chunks = False
        for chunk in response:
            if chunk:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")

                created_at = datetime.utcnow().isoformat() + "Z"
                yield_obj = json.dumps({
                    "model": model_name,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "done": False
                }) + "\n"
                yield yield_obj
                yielded_chunks = True

        created_at = datetime.utcnow().isoformat() + "Z"
        final_obj = {
            "model": model_name,
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": True,
            "done_reason": "stop"
        }
        yield json.dumps(final_obj) + "\n"

    return Response(generate_chunks(), content_type="application/json")


@app.route('/translate_to_english', methods=['POST'])
def translate_to_english_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    translation_prompt = [
        {'role': 'system', 'content': "You are a concise and direct English translator."},
        {'role': 'user', 'content': f"Translate the following text to English:\n\n{message}"}
    ]

    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        last_message="",
        conversation=translation_prompt
    )
    return generate_ollama_response(response)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
