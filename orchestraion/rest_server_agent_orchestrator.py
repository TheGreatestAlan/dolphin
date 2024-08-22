import os
import json
from threading import Thread
from agent.llm_assistant import LLMAssistant
from agent.assistant import Assistant
from integrations.ChatHandler import ChatHandler
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

sessions_file_path = 'sessions.json'
chat_handler = ChatHandler(sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler)

def stream_text_in_thread(session_id):
    text_queue = chat_handler.listen_to_text_stream(session_id)

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
    return stream_text_in_thread(session_id)

@app.route('/streamaudio/<session_id>')
def stream_audio(session_id):
    return chat_handler.listen_to_audio_stream(session_id)

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
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
