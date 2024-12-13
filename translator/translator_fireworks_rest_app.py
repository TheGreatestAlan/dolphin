import os
from flask import Flask, request, Response, jsonify
import requests

from agent_server.llms.EncryptedKeyStore import EncryptedKeyStore

app = Flask(__name__)

# Read API Base URL and API Key from environment variables
key_store = EncryptedKeyStore('keys.json.enc')
API_KEY = key_store.get_api_key("FIREWORKS_API_KEY")
BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(endpoint):
    """
    Proxies all requests to the Fireworks AI server, handling both standard and streaming responses.
    """
    target_url = f"{BASE_URL}/{endpoint}"
    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    headers['Authorization'] = f"Bearer {API_KEY}"  # Add API key to headers
    data = request.get_json() if request.is_json else request.data
    params = request.args

    try:
        # Intercept requests targeting the llama3.2:3b model
        if endpoint == "api/chat" and request.is_json:
            if data and data.get("model") == "llama3.2:3b":
                return handle_llama32_3b(data, headers, params)

        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            json=data if request.is_json else None,
            params=params,
            stream = (endpoint != "api/tags")  # Stream for all but /api/tags

        )

        # Handle streaming responses
        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive chunks
                    yield chunk

        return Response(generate(), status=response.status_code, headers=dict(response.headers))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500

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
