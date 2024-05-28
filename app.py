from AudioTranscriber import AudioTranscriber
from InventoryMapper import InventoryMapper
from flask import Flask, request, jsonify
import time
import os
from werkzeug.utils import secure_filename

from InventoryRestClient import InventoryClient
from SmartFindingInventoryClient import SmartFindingInventoryClient
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
import uuid

from llms.RestLLM import RestLLM

app = Flask(__name__)
app.logger.setLevel("INFO")
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# URL of the text generation server
chat_gpt_url = "http://127.0.0.1:5002"
#ollama_gpt_url = "http://localhost:5002"
chatgpt_rest_llm = RestLLM(chat_gpt_url)
#ollama_rest_llm = RestLLM(ollama_gpt_url)

rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, chatgpt_rest_llm)
function_generator = InventoryFunctionGenerator(chatgpt_rest_llm)
inventory_mapper = InventoryMapper(smart_finding_inventory_client, function_generator)
audio_transcriber = AudioTranscriber()

@app.route('/match_items', methods=['POST'])
def match_items():
    start_time = time.time()
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        known_items = request.json.get('known_items', [])
        mentioned_items = request.json.get('mentioned_items', [])

        system_message = (
            "You are chatGPT-4, a well-trained LLM used to assist humans. "
            "You must respond only with a valid JSON object. Do not include any other text or explanation in your response. "
            "Here are some examples of how you should respond:\n\n"
            "1. If asked 'add a hammer to container 5', respond with:\n"
            "{\"action\": \"add_inventory\", \"parameters\": {\"container_number\": 5, \"items_to_add\": [\"hammer\"]}}\n"
            "2. If asked 'remove a screwdriver from container 10', respond with:\n"
            "{\"action\": \"delete_inventory\", \"parameters\": {\"container_number\": 10, \"items_to_delete\": [\"screwdriver\"]}}\n"
            "3. If asked 'add two pencils to container 3', respond with:\n"
            "{\"action\": \"add_inventory\", \"parameters\": {\"container_number\": 3, \"items_to_add\": [\"pencil\", \"pencil\"]}}\n\n"
            "Remember, respond with only the JSON object."
        )

        prompt = f"Match the following known items: {known_items} with these mentioned items: {mentioned_items}."
        result = rest_llm.generate_response(conversation_id, prompt, system_message)
        elapsed_time = time.time() - start_time
        return jsonify(
            {'conversation_id': result['conversation_id'], 'matched_items': result['response'], 'processing_time': elapsed_time})
    except Exception as e:
        app.logger.error(f"Error processing items: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/inventory_function', methods=['POST'])
def inventory_function():
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', None)

        result = rest_llm.generate_response(conversation_id, prompt, system_message)
        return jsonify(
            {'conversation_id': result['conversation_id'], 'response': result['response'], 'processing_time': result['processing_time']})
    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate_text_endpoint():
    start_time = time.time()
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')
        result = rest_llm.generate_response(conversation_id, prompt, system_message)
        return jsonify({'conversation_id': result['conversation_id'], 'response': result['response']})
    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Now you can use the saved file path to transcribe
        transcription = audio_transcriber.transcribe_audio(file_path)  # Assuming this method exists and works correctly
        print(transcription)
        # Assuming generate_response can handle the prompt and optionally conversation_id and system_message
        import uuid

        # Assuming text_generator is already instantiated and available
        result = rest_llm.generate_response(
            conversation_id=str(uuid.uuid4()),  # Generate a new conversation ID or manage it appropriately
            prompt="Audio Transcription ::: " + transcription,
            system_message="""You are GPT-4, a language model being used in tandem with a speech to text model to clean up the transcription from context. You're leveraging your knowledge of English and what makes sense to be said to clean up the response. You will return only with the cleaned up response. You will offer no other comment besides the cleaned up response. Offering comment apart from the cleaned up response will destroy the system you are integrating with. Examples:

        1.Prompt: Audio Transcription ::: Hullo Hullo how are you doing
        Response: <Cleaned_Transcription>Hello Hello how are you doing</Cleaned_Transcription>

        2.Prompt: Audio Transcription ::: Place Pika che and container one!
        Response: <Cleaned_Transcription>Place Pikachu in container one!</Cleaned_Transcription>

        3.Prompt: Audio Transcription ::: Wur ganna be late for the meeting!
        Response: <Cleaned_Transcription>We're going to be late for the meeting!</Cleaned_Transcription>

        4.Prompt: Audio Transcription ::: It's raining cats and dogs out there, isn't it
        Response: <Cleaned_Transcription>It's raining cats and dogs out there, isn't it</Cleaned_Transcription>

        4.Prompt: Audio Transcription :::THE LEAT THE PHONE FROM DRUWER FOURTEEN
        Response: <Cleaned_Transcription>Delete the phone from container fourteen</Cleaned_Transcription>"""
        )

        print(result['response'])
        return jsonify({'transcription': result['response']})


@app.route('/text_inventory', methods=['POST'])
def text_inventory():
    try:
        prompt = request.json.get('prompt', '')
        response = inventory_mapper.handle_text_inventory(prompt)
        return response
    except Exception as e:
        app.logger.error(f"Error handling text inventory request: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
