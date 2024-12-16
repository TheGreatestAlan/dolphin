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

# Introduce a variable to easily switch target language
# For demonstration: switch between "Spanish" and "Italian"
TARGET_LANGUAGE = "Italian"  # or "Italian"

# Define language-specific instructions
LANGUAGE_CONFIG = {
    "Spanish": {
        "persona_system_message": (
            "Eres un conversador en español muy conocedor y encantador. Responderás siempre en un español claro y "
            "atractivo, brindando información útil y conversaciones interesantes. La conversación debe ser un ir y venir "
            "entre ambos, y trata de no extenderte más de uno o dos párrafos a la vez."
        ),
        "translation_system_message": (
            "You are a concise and direct Spanish translator. If the message is incorrect, give the correct message and "
            "if necessary a short explanation. Don't take into account correcting diacritical marks."
        ),
        "rating_system_message": (
            "You are a Spanish language message evaluator. You will read the message from a Spanish learning "
            "student. You will rate the message on the Spanish correctness from 1 to 10 with 1 being "
            "completely incorrect to 10 being completely correct. Don't take diacritical marks into the correctness of "
            "the message. DO NOT PROVIDE ANY EXPLANATION. DO NOT SAY ANYTHING ELSE. ONLY RETURN THE INTEGER."
        ),
        "translator_role": "Spanish translator",
        "persona_language": "Spanish"
    },
    "Italian": {
        "persona_system_message": (
            "Sei un conversatore italiano molto esperto e affascinante. Risponderai sempre in un italiano chiaro e "
            "attraente, fornendo informazioni utili e conversazioni interessanti. La conversazione deve essere un "
            "dialogo tra entrambi, e cerca di non dilungarti più di uno o due paragrafi alla volta."
        ),
        "translation_system_message": (
            "You are a concise and direct Italian translator. If the message is incorrect, give the correct message and "
            "if necessary a short explanation. Don't take into account correcting diacritical marks."
        ),
        # If you also wanted to rate correctness in Italian, you could adapt this message:
        "rating_system_message": (
            "You are an Italian language message evaluator. You will read the message from a student learning Italian. "
            "You will rate the message on its Italian correctness from 1 to 10 with 1 being completely incorrect "
            "and 10 being completely correct. Don't consider diacritical marks in correctness. "
            "DO NOT PROVIDE ANY EXPLANATION. DO NOT SAY ANYTHING ELSE. ONLY RETURN THE INTEGER."
        ),
        "translator_role": "Italian translator",
        "persona_language": "Italian"
    }
}

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(endpoint):
    """
    Proxies all requests to the Fireworks AI server, handling both standard and streaming responses.
    Dynamically reroutes requests to the correct Fireworks endpoint based on the mapping.
    Handles the api/tags call to return a fake response.
    """
    # Fake response for /api/tags
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

    # Define endpoint mapping
    endpoint_mapping = {
        "api/chat": "/inference/v1/api/chat",
        "chat/completions": "/inference/v1/chat/completions",
        "api/generate": "/inference/v1/api/generate",
    }

    # Reroute to the correct Fireworks endpoint if mapped
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

        # Just returning a dummy streaming response from LLMFactory for demonstration
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

    # Get the user's current message
    user_message = messages[-1].get("content", "").strip()

    # If user requests translation of last assistant message
    if user_message.lower() in ["traducir", "translate"]:
        last_assistant_message = get_last_assistant_message(messages)
        if not last_assistant_message:
            return jsonify({"error": "No assistant message found to translate."}), 400

        # Translate the last assistant message to English
        return english_translation(last_assistant_message, stream=True)

    # Otherwise, check language correctness rating
    language_attempt_rating = rate_language_attempt(user_message)

    if language_attempt_rating <= rating_threshold:
        return translate_to_target_language(user_message, True)
    else:
        return converse_in_target_language(user_message)


def converse_in_target_language(user_message: str):
    """
    Use the chosen language persona and system instructions to continue the conversation in that language.
    """
    system_message = LANGUAGE_CONFIG[TARGET_LANGUAGE]["persona_system_message"]

    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        prompt=user_message,
        system_message=system_message
    )

    return generate_ollama_response(response)


def get_last_assistant_message(messages):
    """
    Extract the last assistant message from the conversation history.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return None

def rate_language_attempt(message: str) -> int:
    # For now, we assume the rating_system_message corresponds to checking correctness in TARGET_LANGUAGE.
    system_message = LANGUAGE_CONFIG[TARGET_LANGUAGE]["rating_system_message"]
    prompt = f"Rate the following {TARGET_LANGUAGE} sentence correctness from 1 to 10:\n\n{message}"

    rating_response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(
        prompt=prompt,
        system_message=system_message
    )

    # Attempt to parse the rating from the response.
    try:
        rating = int(rating_response.strip())
    except ValueError:
        # If parsing fails, default to a low rating
        rating = 1

    # Ensure rating is within the 1-10 range
    if rating < 1:
        rating = 1
    elif rating > 10:
        rating = 10

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
    # Use the target language translation system message
    system_message = LANGUAGE_CONFIG[TARGET_LANGUAGE]["translation_system_message"]
    prompt = f"Translate the following text to {TARGET_LANGUAGE}:\n\n{message}"

    if stream:
        response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
            prompt=prompt,
            system_message=system_message
        )
        return generate_ollama_response(response)
    else:
        # Non-streaming response
        completion = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(
            prompt=prompt,
            system_message=system_message
        )
        return jsonify({"translation": completion})

def generate_ollama_response(response):
    """
    Convert your response chunks into Ollama's streaming format.
    """

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
    # Translate to English
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
    """
    Health check endpoint to verify the middleware is running.
    """
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
