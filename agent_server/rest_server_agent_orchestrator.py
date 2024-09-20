import logging
import os
import json
import queue
from threading import Thread
from llm_assistant import LLMAssistant
from assistant import Assistant
from integrations.ChatHandler import ChatHandler
from flask import Flask, request, jsonify, Response

from agent_server.integrations.StreamManager import StreamManager

app = Flask(__name__)

sessions_file_path = '../orchestration/sessions.json'
stream_manager = StreamManager()
chat_handler = ChatHandler(sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler, stream_manager)
active_threads = {}
# Initialize session-to-username mapping
user_sessions = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL = 60  # Heartbeat interval set to 5 seconds

def stream_text_in_thread(session_id):
    text_queue = stream_manager.listen_to_text_stream(session_id)

    def generate():
        try:
            while session_id in active_threads:  # Continue streaming as long as the session is active
                try:
                    # Wait for a new message for up to HEARTBEAT_INTERVAL seconds
                    text_chunk = text_queue.get(timeout=HEARTBEAT_INTERVAL)
                    yield f"data: {json.dumps({'message': text_chunk})}\n\n"
                except queue.Empty:
                    # If no message has been sent within the heartbeat interval, send a heartbeat
                    yield f"data: {json.dumps({'heartbeat': 'keep-alive'})}\n\n"
        finally:
            # Clean-up resources if the client is disconnected
            clean_up_resources(session_id)

    return Response(generate(), mimetype='text/event-stream')



@app.route('/streamaudio/<session_id>')
def stream_audio(session_id):
    audio_buffer = stream_manager.listen_to_audio_stream(session_id)

    def generate_audio():
        while session_id in active_threads:  # Continue streaming as long as the session is active
            try:
                audio_chunk = audio_buffer.get(timeout=HEARTBEAT_INTERVAL)  # Block until a new audio chunk or timeout
            except queue.Empty:
                continue  # Check if the session is still active after timeout

            if audio_chunk is None:  # Sentinel value to stop the stream
                break
            samples, sample_rate = audio_chunk
            # Convert numpy array to bytes
            audio_bytes = samples.tobytes()
            yield audio_bytes

    def stream_audio_thread():
        return Response(generate_audio(), mimetype='audio/raw')

    # Start the streaming in a new thread
    audio_thread = Thread(target=stream_audio_thread)
    audio_thread.daemon = True
    audio_thread.start()

    # Return the response to initiate the stream
    return stream_audio_thread()



@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Generate a unique session_id here in the main class
    session_id = os.urandom(16).hex()

    # Start a new session in the ChatHandler using the username
    chat_handler.get_or_create_user(username)

    # Map the session_id to the username
    user_sessions[session_id] = username

    # Return session_id and heartbeat interval
    return jsonify({"session_id": session_id, "heartbeat_interval": HEARTBEAT_INTERVAL})


@app.route('/stream/<session_id>')
def stream_text(session_id):
    """Start a thread for streaming text and return the response."""
    stream_thread = Thread(target=stream_text_in_thread, args=(session_id,))
    stream_thread.start()

    # Track the thread in the active registry
    active_threads[session_id] = stream_thread

    return stream_text_in_thread(session_id)




@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in user_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    # End the session in the ChatHandler and remove the session
    chat_handler.end_session(session_id)
    clean_up_resources(session_id)

    # Remove the session_id from the user_sessions map
    user_sessions.pop(session_id, None)

    return '', 200



def clean_up_resources(session_id):
    if session_id in active_threads:
        logger.info("Cleaning up resource for session: " + session_id)
        active_threads.pop(session_id, None)  # Safely remove session from active threads
    stream_manager.end_streams(session_id)  # Custom method to stop streams

    # Remove session_id from user_sessions if it's still present (extra safety)
    user_sessions.pop(session_id, None)

@app.route('/message_agent', methods=['POST'])
def message_agent():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    # Validate session_id
    if not session_id or session_id not in user_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    # Forward the message to the assistant via session_id
    assistant.message_assistant(session_id, user_sessions.get(session_id), user_message)

    return jsonify({"status": "Message received, processing started"}), 202


if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'
    app.run(debug=True, host=os.environ['REST_ADDRESS'], port=os.environ['REST_PORT'], use_reloader=False)
