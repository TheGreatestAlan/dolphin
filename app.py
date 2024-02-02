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


@app.route('/generate', methods=['POST'])
def generate_text():
    start_time = time.time()
    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')
        conversation_id, latest_response = text_generator.generate_response(conversation_id, prompt, system_message)
        elapsed_time = time.time() - start_time
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
        conversation_id, latest_response = text_generator.generate_response(
            prompt="Please attempt to from context clean up the response from the audio transcriber.  Audio Transcription ::: " + transcription + " ::: respond with this template: Cleaned Transcription||| <Transcription>",
            conversation_id=str(uuid.uuid4()),  # Generate a new conversation ID or manage it appropriately
            system_message="You are dolphin, a language model being used in tandem with a speech to text model to clean up the transcription from context. You're leveraging you're knowledge of English and what makes sense to be said to clean up the response.  You will return only with the cleaned up response.  You will offer no other comment besides the cleaned up response.  Offering comment apart from the cleaned up response will destroy the system you are integrating with."
        )
        print(latest_response)
        return jsonify({'transcription': latest_response})


if __name__ == '__main__':
    app.logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
