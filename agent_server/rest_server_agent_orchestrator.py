import os
import json
from datetime import time
from threading import Thread
from llm_assistant import LLMAssistant
from assistant import Assistant
from integrations.ChatHandler import ChatHandler
from flask import Flask, request, jsonify, Response

from agent_server.integrations.StreamManager import StreamManager

# HEMMINGWAY BRIDGE:
# Trying to figure out when a stream dies, how to restart it
# K, you got the docker pull working correctly with additional scripting
# you need to build a shell version of this and put it on the server to use,
# then run it every time it restarts.

app = Flask(__name__)

sessions_file_path = '../orchestraion/sessions.json'
stream_manager = StreamManager()
chat_handler = ChatHandler(stream_manager, sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler)
active_threads = {}




def stream_text_in_thread(session_id):
    text_queue = stream_manager.listen_to_text_stream(session_id)

    def generate():
        while True:
            text_chunk = text_queue.get()  # Block until new text is available
            yield f"data: {json.dumps({'message': text_chunk})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

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

@app.route('/streamaudio/<session_id>')
def stream_audio(session_id):
    audio_buffer = stream_manager.listen_to_audio_stream(session_id)

    def generate_audio():
        while True:
            audio_chunk = audio_buffer.get()  # Block until a new audio chunk is available
            if audio_chunk is None:  # Sentinel value to stop the stream
                break
            samples, sample_rate = audio_chunk
            # Convert numpy array to bytes
            audio_bytes = samples.tobytes()
            chunk_size = len(audio_bytes)  # Measure the size of the chunk in bytes
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
    session_id = chat_handler.start_session()
    return jsonify({"session_id": session_id})

@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    chat_handler.end_session(session_id)
    return '', 200

def monitor_thread_health(session_id, thread):
    while thread.is_alive():
        time.sleep(5)  # Check every 5 seconds
    clean_up_resources(session_id)

def clean_up_resources(session_id):
    if session_id in active_threads:
        print("cleanint up resource:" + session_id)
        del active_threads[session_id]
    # Any other resource cleanup related to the stream
    stream_manager.end_text_stream(session_id)  # Custom method to stop streams


@app.route('/message_agent', methods=['POST'])
def message_agent():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    assistant.message_assistant(session_id, user_message)
    chat_handler.store_human_context(session_id, user_message)  # Add to queue

    return jsonify({"status": "Message received, processing started"}), 202

if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'
    app.run(debug=True, host=os.environ['REST_ADDRESS'], port=os.environ['REST_PORT'], use_reloader=False)
