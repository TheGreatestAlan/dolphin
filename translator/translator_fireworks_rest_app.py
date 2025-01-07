import json
import os
from datetime import datetime

from flask import Flask, request, Response, jsonify
import requests

from translator.llms.EncryptedKeyStore import EncryptedKeyStore
from translator.llms.LLMFactory import LLMFactory, ModelType

app = Flask(__name__)

key_store = EncryptedKeyStore('keys.json.enc')
API_KEY = key_store.get_api_key("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai"
LLAMA_MODEL = ModelType.FIREWORKS_LLAMA_3_70B
QWEN_MODEL = ModelType.FIREWORKS_QWEN_72B
# Languages supported by QWEN
QWEN_LANGUAGES = {"Chinese", "English", "French", "Spanish", "Portuguese",
                  "German", "Italian", "Russian", "Japanese", "Korean",
                  "Vietnamese", "Thai", "Arabic"}

LLAMA_LANGUAGES = {"German", "French", "Italian", "Portuguese", "Hindi", "Spanish", "Thai"}

SUPPORTED_LANGUAGES = QWEN_LANGUAGES | LLAMA_LANGUAGES

rating_threshold = 6

# Keep instructions in English and just refer to the target language in the prompt.
INSTRUCTIONS = {
    "persona_system_message": (
        "You are a knowledgeable and charming conversation partner who always responds in clear, "
        "appealing {target_language}. Provide useful information and interesting conversation. "
        "Keep responses to no more than a paragraph or two at a time."
    ),
    "translation_system_message": (
        "You are a concise and direct translator who translates to {target_language}. If the message is incorrect, "
        "provide the corrected message and, if necessary, a brief explanation."
    ),
    "rating_system_message": (
        "You are a language evaluator. Evaluate the correctness of the following {target_language} sentence "
        "Do not take into account diacritical marks or incorrect leading punctuation."
        "from 1 to 10, where 1 is completely incorrect and 10 is completely correct. "
        "for example, if the sentence is not in {target_language} you would rank that a 1"
        "Do not provide any explanation—only the integer."
    )
}

@app.route('/api/version', methods=['GET'])
def get_version():
    return jsonify({"version": "0.3.14"}), 200

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(endpoint):
    if endpoint == "api/models" or "api/tags":
        res = jsonify(
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
        return res
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

    try:
        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid message format"}), 400

        # Determine model and language
        # If no model_name is provided, default to one
        model_name = data.get("model_name", "llama3.2:3b_italian")
        language = extract_language_from_model_name(model_name)
        model = get_model(language)

        last_message = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        response = LLMFactory.get_singleton(model).stream_response(last_message, "None")
        return generate_ollama_response(response, model_name)

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/chat', methods=['POST'])
def handle_chat():
    data = request.get_json()
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid message format"}), 400

    # Extract model_name from request, default if not present
    model_name = data.get("model", "llama3.2:3b_italian")
    target_language = extract_language_from_model_name(model_name)

    user_message = messages[-1].get("content", "").strip()

    # If user requests translation of last assistant message
    if user_message.lower() in ["translate"]:
        last_assistant_message = get_last_assistant_message(messages)
        if not last_assistant_message:
            return jsonify({"error": "No assistant message found to translate."}), 400
        return english_translation(last_assistant_message, stream=True, model_name=model_name)

    # Otherwise, check language correctness rating
    language_attempt_rating = rate_language_attempt(user_message, target_language)
    if language_attempt_rating <= rating_threshold:
        return translate_to_target_language(user_message, True, target_language, model_name)
    else:
        return converse_in_target_language(messages, target_language, model_name)


def extract_language_from_model_name(model_name: str) -> str:
    if "_" in model_name:
        language_code = model_name.split("_")[-1].capitalize()
        supported_languages = SUPPORTED_LANGUAGES
        if language_code.lower() in [lang.lower() for lang in supported_languages]:
            return language_code.capitalize()
    return "English"


def converse_in_target_language(messages: list, target_language: str, model_name: str):
    # Extract the latest user message
    latest_message = None
    for message in reversed(messages):
        if message["role"] == "user":
            latest_message = message["content"]
            break

    if not latest_message:
        raise ValueError("No user message found in the messages list.")

    # System message tailored for the target language
    system_message = f"You are a knowledgeable and charming assistant who always responds in {target_language}. Provide helpful, concise responses."
    model = get_model(target_language)

    # Generate response (this assumes you have an LLM factory or interface similar to earlier examples)
    response = LLMFactory.get_singleton(model).stream_response(
        latest_message,
        system_message,
        messages
    )
    return generate_ollama_response(response, model_name)


def get_last_assistant_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return None


def rate_language_attempt(message: str, target_language: str) -> int:
    system_message = INSTRUCTIONS["rating_system_message"].format(target_language=target_language)
    prompt = f"Rate the following {target_language} sentence correctness from 1 to 10:\n\n{message}"

    model = get_model(target_language)

    rating_response = LLMFactory.get_singleton(model).generate_response(
        prompt=prompt,
        system_message=system_message
    )

    try:
        rating = int(rating_response.strip())
        print(rating)
    except ValueError:
        rating = 1

    return max(1, min(rating, 10))

def get_model(target_language: str):
    normalized_language = target_language.strip().lower()  # Normalize the input
    normalized_llama_languages = {lang.strip().lower() for lang in LLAMA_LANGUAGES}
    normalized_qwen_languages = {lang.strip().lower() for lang in QWEN_LANGUAGES}

    if normalized_language in normalized_llama_languages:
        return LLAMA_MODEL
    elif normalized_language in normalized_qwen_languages:
        return QWEN_MODEL
    else:
        raise ValueError(f"Unsupported language: {target_language}")

def english_translation(message: str, stream: bool, model_name: str):
    system_message = "You are a concise and direct English translator. You will not be yappy."
    prompt = f"Translate the following text to English:\n\n{message}"
    language = extract_language_from_model_name(model_name)
    model = get_model(language)

    response = LLMFactory.get_singleton(model).stream_response(
        prompt=prompt,
        system_message=system_message
    )
    return generate_ollama_response(response, model_name)

def translate_to_target_language(message: str, stream: bool, target_language: str, model_name: str):
    system_message = INSTRUCTIONS["translation_system_message"].format(target_language=target_language)
    prompt = f"Translate the following text to {target_language}:\n\n{message}"
    model = get_model(target_language)

    if stream:
        response = LLMFactory.get_singleton(model).stream_response(
            prompt=prompt,
            system_message=system_message
        )
        return generate_ollama_response(response, model_name)
    else:
        completion = LLMFactory.get_singleton(model).generate_response(
            prompt=prompt,
            system_message=system_message
        )
        return jsonify({"translation": completion})

def generate_ollama_response(response, model_name):
    def generate_chunks():
        for chunk in response:
            if "[DONE]" in chunk:
                chunk = chunk.replace("[DONE]", "")
                created_at = datetime.utcnow().isoformat() + "Z"
                final_obj = {
                    "model": model_name,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "done": True,
                    "done_reason": "stop"
                }
                final_object = json.dumps(final_obj) + "\n"
                yield final_object
                break
            if chunk:
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

    return Response(generate_chunks(), content_type="application/json")


@app.route('/translate_to_english', methods=['POST'])
def translate_to_english_endpoint():
    data = request.get_json()
    message = data.get("message", "")
    model_name = data.get("model_name", "llama3.2:3b_italian")
    translation_prompt = [
        {'role': 'system', 'content': "You are a concise and direct English translator."},
        {'role': 'user', 'content': f"Translate the following text to English:\n\n{message}"}
    ]
    language = extract_language_from_model_name(model_name)
    model = get_model(language)

    response = LLMFactory.get_singleton(model).stream_response(
        last_message="",
        conversation=translation_prompt
    )
    return generate_ollama_response(response, model_name)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    port = int(os.environ.get('REST_PORT', 8080))
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
