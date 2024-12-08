from flask import Flask, request, Response, jsonify
import requests

app = Flask(__name__)

# Ollama API Base URL
OLLAMA_API_BASE_URL = "http://localhost:11434"

@app.route('/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_request(endpoint):
    """
    Proxies all requests to the Ollama server, handling both standard and streaming responses.
    """
    target_url = f"{OLLAMA_API_BASE_URL}/{endpoint}"
    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    data = request.get_json() if request.is_json else request.data
    params = request.args

    try:
        # Intercept requests targeting the llama3.2:3b model
        if endpoint == "api/chat" and request.is_json:
            if data and data.get("model") == "llama3.2:3b":
                return handle_llama32_3b(data, headers, params)

        # Use stream=True for requests
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            json=data if request.is_json else None,
            params=params,
            stream=(endpoint != "api/tags")  # Stream for all but /api/tags
        )

        # Intercept and modify response for /api/tags
        if endpoint == "api/tags" and response.status_code == 200:
            tags = response.json()  # Get existing models
            if "models" in tags and isinstance(tags["models"], list):
                # Add the fake Spanish translator model
                spanish_translator_model = {
                    "name": "spanish-translator",
                    "model": "spanish-translator",
                    "modified_at": "2024-12-08T00:00:00.0000000-06:00",
                    "size": 123456789,  # Fake size
                    "digest": "fake-digest-spanish-translator",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "custom",
                        "families": [
                            "custom"
                        ],
                        "parameter_size": "1.0B",
                        "quantization_level": "Q4_K_M"
                    }
                }
                tags["models"].append(spanish_translator_model)
            else:
                app.logger.error("Unexpected response format for /api/tags")
                return jsonify({"error": "Unexpected response format"}), 500

            return jsonify(tags), 200

        # Handle streaming responses for other endpoints
        def generate():
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # Filter out keep-alive chunks
                    yield chunk

        # Return the streamed response for non-taggable endpoints
        return Response(generate(), status=response.status_code, headers=dict(response.headers))

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error proxying request to {target_url}: {e}")
        return jsonify({"error": str(e)}), 500


def handle_llama32_3b(data, headers, params):
    """
    Handles requests for the llama3.2:3b model.
    Reroute or modify the request as needed.
    """
    # Modify the model to reroute to a different model
    data["model"] = "qwen2.5:3b"  # Example: Redirect to another valid model

    # Make a request to the Ollama server
    target_url = f"{OLLAMA_API_BASE_URL}/api/chat"
    try:
        response = requests.request(
            method="POST",
            url=target_url,
            headers=headers,
            json=data,
            params=params,
            stream=True
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


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint to verify the middleware is running.
    """
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
