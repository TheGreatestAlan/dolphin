import logging
import os
import json
import queue
from datetime import time
from threading import Thread
from llm_assistant import LLMAssistant
from assistant import Assistant
from integrations.ChatHandler import ChatHandler
from flask import Flask, request, jsonify, Response

from agent_server.integrations.StreamManager import StreamManager

# HEMMINGWAY BRIDGE:
# K we're handling the separation of user and sessionid.  You have, untested in here
# the implementation of a user logging in with a userId and then using the returned
# session id for the rest of the calls.  Make sure that when a user sends a message
# that the username is looked up here by sessionid and then sent to the assistant

# the assistant needs to then have both the username and the session, and send both
# I'm thinking you pull the stream manager into the assistant and out of the chat
# handler.  Then it streams directly to the stream manager, and once the message is
# completed it's saved by the chat handler.

# Chat handler should only know about usernames now, and no longer sessions

# also make sure that streammanager is handled correctly

app = Flask(__name__)

sessions_file_path = '../orchestraion/sessions.json'
stream_manager = StreamManager()
chat_handler = ChatHandler(stream_manager, sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler)
active_threads = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL = 5  # Heartbeat interval set to 5 seconds

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


# Initialize session-to-username mapping
user_sessions = {}

@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    # Generate a unique session_id here in the main class
    session_id = os.urandom(16).hex()

    # Start a new session in the ChatHandler using the username
    chat_handler.start_session(session_id, username)

    # Map the session_id to the username
    user_sessions[session_id] = username

    # Check if there's an active thread for this session and clean up if needed
    if session_id in active_threads:
        logger.info(f"Cleaning up existing thread for session {session_id}")
        clean_up_resources(session_id)

    # Return session_id and heartbeat interval
    return jsonify({"session_id": session_id, "heartbeat_interval": HEARTBEAT_INTERVAL})


@app.route('/stream/<session_id>')
def stream_text(session_id):
    """Start a thread for streaming text and return the response."""
    stream_thread = Thread(target=stream_text_in_thread, args=(session_id,))
    stream_thread.start()

    # Track the thread in the active registry
    active_threads[session_id] = stream_thread

    # Start monitoring the thread's health
    monitor_thread = Thread(target=monitor_thread_health, args=(session_id, stream_thread))
    monitor_thread.start()

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


def monitor_thread_health(session_id, thread):
    while thread.is_alive():
        time.sleep(5)  # Check every 5 seconds
    clean_up_resources(session_id)


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
    assistant.message_assistant(session_id, user_message)
    chat_handler.store_human_context(session_id, user_message)  # Add to queue

    return jsonify({"status": "Message received, processing started"}), 202


if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'
    app.run(debug=True, host=os.environ['REST_ADDRESS'], port=os.environ['REST_PORT'], use_reloader=False)
