import os
import time
from flask import Flask, request, jsonify

from llms.ChatGPT4 import ChatpGPT4
from llms.Ollama3 import Ollama3LLM

# Create Flask app for text generation server
app = Flask(__name__)

def create_text_generator():
    generator_type = os.getenv('TEXT_GENERATOR_TYPE')
    if generator_type == 'ChatGPT4':
        return ChatpGPT4()
    elif generator_type == 'Ollama3':
        return Ollama3LLM()
    else:
        raise ValueError("Unsupported TEXT_GENERATOR_TYPE. Supported types are: 'ChatGPT', 'Ollama3'")

# Instantiate the text generator
text_generator = create_text_generator()

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
    app.run(host='0.0.0.0', port=port)
