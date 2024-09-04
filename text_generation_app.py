import os
import time
from flask import Flask, request, jsonify, Response

from agent_server.llms.ChatGPT4 import ChatGPT4
from llm.llamacpp import LlamaCpp

# Create Flask app for text generation server
app = Flask(__name__)


def create_text_generator():
    generator_type = os.getenv('TEXT_GENERATOR_TYPE')
    if generator_type == 'ChatGPT4':
        return ChatGPT4()
    elif generator_type == 'Ollama3':
        return LlamaCpp()
    else:
        raise ValueError("Unsupported TEXT_GENERATOR_TYPE. Supported types are: 'ChatGPT4', 'Ollama3'")


# Instantiate the text generator
text_generator = create_text_generator()


@app.route('/stream', methods=['POST'])
def stream_text():
    start_time = time.time()
    try:
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')

        def generate():
            try:
                for response in text_generator.stream_response(prompt, system_message):
                    yield f"data: {response}\n\n"
            except Exception as e:
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate_text():
    start_time = time.time()
    try:
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')
        response = text_generator.generate_response(prompt, system_message)
        elapsed_time = time.time() - start_time
        return jsonify({'response': response, 'processing_time': elapsed_time})
    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Use 5001 as the default port if not set in environment variables
    os.environ['FLASK_SKIP_DOTENV'] = 'true'  # Add this line to bypass dotenv loading
    app.run(host='0.0.0.0', port=port)
