from flask import Flask, request, jsonify
import time
import os
from werkzeug.utils import secure_filename
from AudioTranscriber import AudioTranscriber
from text_generator import TextGenerator
import uuid

app = Flask(__name__)
app.logger.setLevel("INFO")
UPLOAD_FOLDER = 'uploads'  # Define a folder to save uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Assuming the TextGenerator class is in text_generator.py and the model path is correct
text_generator = TextGenerator()
audio_transcriber = AudioTranscriber()

@app.route('/match_items', methods=['POST'])
def match_items():
    start_time = time.time()
    try:
        # Generate a unique conversation ID or use the provided one
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        known_items = request.json.get('known_items', [])
        mentioned_items = request.json.get('mentioned_items', [])

        # Revised system message
        system_message = (
            "You are a language model trained to assist in matching known inventory items with mentioned items, recognizing synonyms and similar terms. Provide responses only as an array of strings. Here are some examples: \n\n"
            "1. Known items: ['black trousers', 'blue shirt'], Mentioned items: ['black pants'] -> ['black trousers']\n"
            "2. Known items: ['wool hat', 'cotton gloves'], Mentioned items: ['woolen cap'] -> ['wool hat']\n"
            "3. Known items: ['gold necklace', 'silver earrings'], Mentioned items: ['golden chain'] -> ['gold necklace']\n"
            "4. Known items: ['canvas shoes', 'suede jacket'], Mentioned items: ['denim pants', 'jacket'] -> ['suede jacket']\n"
            "5. Known items: ['blue hat', 'red socks'], Mentioned items: ['blue cap'] -> ['blue hat']\n"
            "Respond with matched items only, based on the input arrays of known and mentioned items, considering synonyms and similar terms."
        )

        # Simplified prompt
        prompt = f"Match the following known items: {known_items} with these mentioned items: {mentioned_items}."

        # Assuming text_generator is a previously initialized instance capable of handling generate_response
        conversation_id, latest_response = text_generator.generate_response(conversation_id, prompt, system_message)
        elapsed_time = time.time() - start_time

        # Parse the response to get the matched items
        matched_items = latest_response
        return jsonify({'conversation_id': conversation_id, 'matched_items': matched_items, 'processing_time': elapsed_time})
    except Exception as e:
        app.logger.error(f"Error processing items: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/inventory_function', methods=['POST'])
def inventory_function():
    # Define the default system_message
    default_system_message = (
        "You are chatGPT-4, a well-trained LLM used to assist humans. /set system You are a helpful assistant with access to the following inventory system functions. Use them if required - examples: \n\n"
        "1. If asked 'add a hammer to drawer 5', you should update the inventory of drawer 5 by adding a hammer.\n"
        "2. If asked 'remove a screwdriver from drawer 10', you should update the inventory of drawer 10 by removing a screwdriver.\n"
        "3. If asked 'add two pencils to drawer 3', you should update the inventory of drawer 3 by adding two pencils.\n\n"
        "[{\"name\": \"add_inventory\", \"description\": \"Add items to the inventory of a specific drawer\", \"parameters\": {\"type\": \"object\", \"properties\": {\"drawer_number\": {\"type\": \"integer\", \"description\": \"The number of the drawer to update\"}, \"items_to_add\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of items to add to the drawer\"}}, \"required\": [\"drawer_number\", \"items_to_add\"]}, "
        "{\"name\": \"delete_inventory\", \"description\": \"Delete items from the inventory of a specific drawer\", \"parameters\": {\"type\": \"object\", \"properties\": {\"drawer_number\": {\"type\": \"integer\", \"description\": \"The number of the drawer to update\"}, \"items_to_delete\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of items to delete from the drawer\"}}, \"required\": [\"drawer_number\", \"items_to_delete\"]}]"
    )

    start_time = time.time()
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        # Use the provided system_message or the default if none provided
        system_message = request.json.get('system_message', default_system_message)

        # Assuming text_generator is a previously initialized instance capable of handling generate_response
        conversation_id, latest_response = text_generator.generate_response(conversation_id, prompt, system_message)
        elapsed_time = time.time() - start_time
        return jsonify({'conversation_id': conversation_id, 'response': latest_response, 'processing_time': elapsed_time})
    except Exception as e:
        app.logger.error(f"Error generating text: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_text():
    start_time = time.time()
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')
        conversation_id, latest_response = text_generator.generate_response(conversation_id, prompt, system_message)
        time.time() - start_time
        return jsonify({'conversation_id': conversation_id, 'response': latest_response})
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
        conversation_id, latest_response = text_generator.generate_response(
            conversation_id=str(uuid.uuid4()),  # Generate a new conversation ID or manage it appropriately
            prompt="Audio Transcription ::: " + transcription,
            system_message="""You are GPT-4, a language model being used in tandem with a speech to text model to clean up the transcription from context. You're leveraging your knowledge of English and what makes sense to be said to clean up the response. You will return only with the cleaned up response. You will offer no other comment besides the cleaned up response. Offering comment apart from the cleaned up response will destroy the system you are integrating with. Examples:

        1.Prompt: Audio Transcription ::: Hullo Hullo how are you doing
        Response: <Cleaned_Transcription>Hello Hello how are you doing</Cleaned_Transcription>

        2.Prompt: Audio Transcription ::: Place Pika che and drawer one!
        Response: <Cleaned_Transcription>Place Pikachu in drawer one!</Cleaned_Transcription>

        3.Prompt: Audio Transcription ::: Wur ganna be late for the meeting!
        Response: <Cleaned_Transcription>We're going to be late for the meeting!</Cleaned_Transcription>

        4.Prompt: Audio Transcription ::: It's raining cats and dogs out there, isn't it
        Response: <Cleaned_Transcription>It's raining cats and dogs out there, isn't it</Cleaned_Transcription>

        4.Prompt: Audio Transcription :::THE LEAT THE PHONE FROM DRUWER FOURTEEN
        Response: <Cleaned_Transcription>Delete the phone from drawer fourteen</Cleaned_Transcription>"""

        )

        print(latest_response)
        return jsonify({'transcription': latest_response})


if __name__ == '__main__':
    app.logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
