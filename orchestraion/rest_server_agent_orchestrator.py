import os

from agent.llm_assistant import LLMAssistant
from agent.assistant import Assistant
from integrations.ChatHandler import ChatHandler
from flask import Flask, request, jsonify

app = Flask(__name__)

sessions_file_path = 'sessions.json'
chat_handler = ChatHandler(sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler)


@app.route('/stream/<session_id>')
def stream(session_id):
    return chat_handler.listen_to_stream(session_id)


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

    return '', 200


@app.route('/message_agent', methods=['POST'])
def message_agent():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400
    assistant.message_assistant(session_id, user_message)
    # Correctly pass the generator to the Response object without calling it as a function
    return jsonify({"status": "Message received, processing started"}), 202


if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'  # Add this line to bypass dotenv loading
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
