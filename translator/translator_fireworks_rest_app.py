import json
import os
from flask import Flask, request, Response, jsonify
import requests

from agent_server.llms.EncryptedKeyStore import EncryptedKeyStore
from agent_server.llms.LLMFactory import LLMFactory, ModelType

app = Flask(__name__)

# Read API Base URL and API Key from environment variables
key_store = EncryptedKeyStore('keys.json.enc')
API_KEY = key_store.get_api_key("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai"

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
                        "name": "llama3.2:3b",
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

    # Proxy the request
    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    headers['Authorization'] = f"Bearer {API_KEY}"  # Add API key to headers
    data = request.get_json() if request.is_json else request.data
    params = request.args

    try:

        # response = requests.request(
        #     method=request.method,
        #     url=target_url,
        #     headers=headers,
        #     json=data if request.is_json else None,
        #     params=params,
        #     stream=(endpoint != "api/tags")  # Stream for all but /api/tags
        # )
        # Extract messages from the incoming data
        messages = data.get("messages", [])

        # Ensure that 'messages' is a list of dictionaries and access the last message's content
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid message format"}), 400

        # Safely access the last message content
        last_message = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""

        # Use the extracted content
        response = LLMFactory.get_singleton(ModelType.FIREWORKS_LLAMA_3_70B).stream_response(last_message, "None")

        return generate_ollama_response(response)

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500

import json

import json
from datetime import datetime
from flask import Response

def generate_ollama_response(response):
    """
    Convert your response chunks into Ollama's streaming format:
    - Each chunk: JSON with "model", "created_at", "message", and "done": false
    - Final chunk: JSON with "done": true and possibly additional metadata.
    """

    model_name = "llama3.2:3b"  # Replace with the actual model name if you have it

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
                yield json.dumps({
                    "model": model_name,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "done": False
                }) + "\n"
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

def english_translation(message, stream, headers, params):
    """
    Streams the Spanish translation of the given English message in the expected Ollama format.
    """
    translation_prompt = {
        "model": "llama3.2:3b",
        "messages": [
            {
                'role': 'system',
                'content': f"You are a concise and direct translating Spanish English expert.  You will not be yappy."
            },
            {
            'role':'user',
            'content':f"Translate the following text to English:\n\n{message}"
        }
        ],
        "stream": stream
    }

    target_url = f"{BASE_URL}/api/chat"
    try:
        response = requests.request(
            method="POST",
            url=target_url,
            headers=headers,
            json=translation_prompt,
            params=params,
        )

        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        if stream:
            return Response(generate(), status=response.status_code, headers=dict(response.headers))
        else:
            return Response(status=response.status_code, headers=dict(response.headers))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error during Spanish translation streaming: {e}")
        return jsonify({"error": str(e)}), 500

def spanish_translation(message, stream, headers, params):
    """
    Streams the Spanish translation of the given English message in the expected Ollama format.
    """
    translation_prompt = {
        "model": "llama3.2:3b",
        "messages": [{
            'role':'user',
            'content':f"Translate the following text to Spanish:\n\n{message}"
        }],
        "stream": stream
    }

    target_url = f"{BASE_URL}/api/chat"
    try:
        response = requests.request(
            method="POST",
            url=target_url,
            headers=headers,
            json=translation_prompt,
            params=params,
        )

        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk

        if stream:
            return Response(generate(), status=response.status_code, headers=dict(response.headers))
        else:
            return Response(status=response.status_code, headers=dict(response.headers))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error during Spanish translation streaming: {e}")
        return jsonify({"error": str(e)}), 500

def check_if_english(message):
    """
    Checks if the given message is in English by calling the /api/generate endpoint.
    Generates its own headers for the request and returns a boolean.
    """
    detection_prompt = {
        "model": "llama3.2:3b",
        "prompt": f"Is the following text in English? Respond only with 'true' or 'false':\n\nEnglish Candate:{message}",
        "stream": False  # Ensure a single JSON response
    }

    detection_url = f"{BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}  # Generate new headers

    try:
        response = requests.post(
            url=detection_url,
            headers=headers,  # Use the generated headers
            json=detection_prompt
        )

        if response.status_code == 200:
            response_json = response.json()
            is_english = response_json.get("response", "false").strip().lower() == "true"
            return is_english
        else:
            app.logger.error(f"LLM language detection failed: {response.status_code}, {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error during language detection: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint to verify the middleware is running.
    """
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
