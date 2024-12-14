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
        # Retrieve the last assistant message. Implement this function as needed.
        last_assistant_message = get_last_assistant_message(messages)
        if not last_assistant_message:
            return jsonify({"error": "No assistant message found to translate."}), 400

        # Translate the last assistant message to English
        return english_translation(last_assistant_message, stream=True)

    # Otherwise, check language
    is_english = check_if_english(user_message)

    if is_english:
        # If English, first translate user message to Spanish to maintain a Spanish conversation
        # Once translated, the Spanish message becomes the 'prompt' for the Spanish conversationalist
        # For simplicity, we do a synchronous translation first (non-streaming)
        return spanish_translation(user_message, True)
    else:
        # If not English, just converse in Spanish directly
        return converse_in_spanish(user_message)


def converse_in_spanish(user_message: str):
    """
    Use the Spanish persona and system instructions to continue the conversation in Spanish.
    """
    system_message = (
        "Eres un conversador en español muy conocedor y encantador. Responderás siempre en un español claro y "
        "atractivo, brindando información útil y conversaciones interesantes. La conversación debe ser un ir y venir "
        "entre ambos, y trata de no extenderte más de uno o dos párrafos a la vez."

    )

    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        prompt=user_message,
        system_message=system_message
    )

    return generate_ollama_response(response)


def get_last_assistant_message(messages):
    """
    Extract the last assistant message from the conversation history.
    For example, look backwards through the messages and find the most recent with 'role': 'assistant'.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return None

def check_if_english(message: str) -> bool:
    system_message = "You are a language detector."
    prompt = f"Is the following text in English? Respond only with 'true' or 'false':\n\nEnglish Candidate:{message}"

    # Use a non-streaming completion
    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).generate_response(
        prompt=prompt,
        system_message=system_message
    )

    is_english = response.strip().lower() == "true"
    return is_english

def english_translation(message: str, stream: bool):
    system_message = "You are a concise and direct English translator. You will not be yappy."
    prompt = f"Translate the following text to English:\n\n{message}"

    response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_1_405B).stream_response(
        prompt=prompt,
        system_message=system_message
    )
    return generate_ollama_response(response)

def spanish_translation(message: str, stream: bool):
    system_message = "You are a concise and direct Spanish translator."
    prompt = f"Translate the following text to Spanish:\n\n{message}"

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
    Convert your response chunks into Ollama's streaming format:
    - Each chunk: JSON with "model", "created_at", "message", and "done": false
    - Final chunk: JSON with "done": true and possibly additional metadata.
    """


    def generate_chunks():
        # Keep track of whether we have yielded any chunks
        yielded_chunks = False

        for chunk in response:
            if chunk:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")

                # Current timestamp in ISO 8601 format
                created_at = datetime.utcnow().isoformat() + "Z"

                # Yield a JSON object with the required fields
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

        # After all chunks are done, output the final JSON object
        # If you have actual metrics, insert them here. Otherwise, omit or use dummy values.
        created_at = datetime.utcnow().isoformat() + "Z"
        final_obj = {
            "model": model_name,
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": True,
            "done_reason": "stop",
            # You can omit the durations if you don't have them:
            # "total_duration": 737571100,
            # "load_duration": 42606400,
            # "prompt_eval_count": 26,
            # "prompt_eval_duration": 216262000,
            # "eval_count": 8,
            # "eval_duration": 476584000
        }

        # If we never yielded any chunks, you still need to provide a final object
        yield json.dumps(final_obj) + "\n"

    return Response(generate_chunks(), content_type="application/json")

def handle_llama32_3b(data, headers, params):
    """
    Handles requests for the llama3.2:3b model.
    Checks if the message is in English and processes accordingly.
    If the message is in English, streams the Spanish translation back to the user.
    """
    # Extract the message content
    messages = data.get("messages", [])
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "Invalid message format"}), 400

    user_message = messages[-1].get("content", "").strip()  # Get the last message content

    # Call the LLM to check if the message is in English
    if user_message == 'translato':
        return english_translation(messages[-2].get("content","").strip(), True, headers, params)
    else:
        is_english = check_if_english(user_message)

    # Log the result of the language check
    app.logger.debug(f"Message: '{user_message}', Is English: {is_english}")

    if is_english:
        app.logger.debug("Message is in English. Streaming Spanish translation.")
        # Stream the Spanish translation
        return spanish_translation(user_message, data.get("stream"), headers, params)

    app.logger.debug("Message is not in English. Proceeding with original request.")

    # Proceed with the original LLM call
    target_url = f"{BASE_URL}/api/chat"
    try:
        response = requests.request(
            method="POST",
            url=target_url,
            headers=headers,
            json=data,
            params=params,
            stream=data.get("stream")
        )

        # Handle streaming response
        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        return Response(generate(), status=response.status_code, headers=dict(response.headers))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error in handle_llama32_3b: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/translate_to_english', methods=['POST'])
def translate_to_english():
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
