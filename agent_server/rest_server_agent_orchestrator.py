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

# HEMMINGWAY BRIDGE:
# K we've implemented more agentic function, a singular function chooser, and a
# json parser agent.

# we're at the pieces now ahead of that, we've got a user interaction agent
# and a sequencer.  You're probably going to want to work these backwards, so get the sequencer up and running first
# then you can give it a complext question like ... I don't know come up with one.
# the sequencer can get you direct responses though

# 1. User Interaction Agent (Initial User Request Handler)
# Primary Focus: Handles the initial interpretation of the user’s intent and manages immediate conversational flow.
#
# Responsibilities:
#
# Determine Query Type: Conversational: If the query is casual or not actionable (e.g., “What can you do?” or “Tell
# me a joke”), it responds directly without involving other agents. Actionable: If the query seems to require a task,
# the Interaction Agent hands it off to the Sequencer. Clarify Ambiguous Requests: If the user’s query is unclear or
# lacks sufficient detail, it prompts the user for additional information. For instance: User Input: “Add something
# to the container.” Interaction Agent’s Response: “Could you specify what you’d like to add and to which container?”
# Provide User Feedback: It acknowledges receipt of a complex request and reassures the user that it’s being
# processed. This helps manage user expectations, especially if a multi-step process is involved. When It Passes to
# the Sequencer:
#
# If the query is identified as actionable and adequately clear, it hands the query over to the Sequencer for task
# breakdown. Example:
#
# Input: “Can you add a hammer to container 5 and find the location of item X?”
# Process:
# Recognizes this as an actionable request (not casual or conversational).
# Confirms it’s ready to process the task: “Got it! Working on adding the hammer and finding item X.”
# Hands off to the Sequencer.
#
#
# 2. Sequencer
# Primary Focus: Parses actionable requests into discrete, ordered tasks, and defines any dependencies.
#
# Responsibilities:
#
# Break Down Complex Requests: For requests with multiple steps, it isolates each atomic task (e.g., “add hammer to
# container 5” and “find the location of item X”). Identify Task Dependencies: Determines if tasks are independent (
# can be executed in parallel) or dependent (one task must follow another). Output Structured Task List: Generates a
# structured task list, providing each task in the correct order with dependency information for execution
# management. When It Passes to the FunctionChooser:
#
# Once each atomic task is identified and organized, the Sequencer hands each task to the FunctionChooser to map to
# specific functions. Example:
#
# Input from Interaction Agent: “Add a hammer to container 5 and find the location of item X.”
# Process:
# Breaks down the query into two tasks:
# Task 1: “Add hammer to container 5”
# Task 2: “Find location of item X”
# Assigns each task an order and determines that they can be executed independently.
# Passes each task individually to the FunctionChooser.

app = Flask(__name__)

sessions_file_path = '../orchestration/sessions.json'
stream_manager = StreamManager()
chat_handler = ChatHandler(sessions_file_path)
assistant: Assistant = LLMAssistant(chat_handler, stream_manager)
active_threads = {}
user_sessions = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HEARTBEAT_INTERVAL = 60  # Heartbeat interval set to 5 seconds

# Read BYPASS_LLM flag from the environment
BYPASS_LLM = os.getenv('BYPASS_LLM', 'false').lower() == 'true'


def stream_text_in_thread(session_id):
    text_queue = stream_manager.listen_to_text_stream(session_id)

    def generate():
        try:
            while session_id in active_threads:  # Continue streaming as long as the session is active
                try:
                    text_chunk = text_queue.get(timeout=HEARTBEAT_INTERVAL)
                    if text_chunk.startswith(StreamManager.FUNCTION):
                        yield f"data: {json.dumps({'function': text_chunk.replace(StreamManager.FUNCTION, '')})}\n\n"
                    else:
                        yield f"data: {json.dumps({'message': text_chunk})}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'heartbeat': 'keep-alive'})}\n\n"
        finally:
            clean_up_resources(session_id)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/streamaudio/<session_id>')
def stream_audio(session_id):
    audio_buffer = stream_manager.listen_to_audio_stream(session_id)

    def generate_audio():
        while session_id in active_threads:
            try:
                audio_chunk = audio_buffer.get(timeout=HEARTBEAT_INTERVAL)
            except queue.Empty:
                continue

            if audio_chunk is None:
                break
            samples, sample_rate = audio_chunk
            audio_bytes = samples.tobytes()
            yield audio_bytes

    def stream_audio_thread():
        return Response(generate_audio(), mimetype='audio/raw')

    audio_thread = Thread(target=stream_audio_thread)
    audio_thread.daemon = True
    audio_thread.start()

    return stream_audio_thread()


@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.json
    username = data.get('username')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    session_id = os.urandom(16).hex()
    chat_handler.get_or_create_user(username)
    user_sessions[session_id] = username

    return jsonify({"session_id": session_id, "heartbeat_interval": HEARTBEAT_INTERVAL})


@app.route('/stream/<session_id>')
def stream_text(session_id):
    stream_thread = Thread(target=stream_text_in_thread, args=(session_id,))
    stream_thread.start()

    active_threads[session_id] = stream_thread

    return stream_text_in_thread(session_id)


@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in user_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    chat_handler.end_session(session_id)
    clean_up_resources(session_id)
    user_sessions.pop(session_id, None)

    return '', 200


def clean_up_resources(session_id):
    if session_id in active_threads:
        logger.info("Cleaning up resource for session: " + session_id)
        active_threads.pop(session_id, None)
    stream_manager.end_streams(session_id)
    user_sessions.pop(session_id, None)


@app.route('/message_agent', methods=['POST'])
def message_agent():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    if not session_id or session_id not in user_sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    # Use a separate method to handle message processing based on the BYPASS_LLM flag
    if handle_bypass_llm(session_id, user_message):
        return jsonify({"status": "Message received, canned response sent"}), 202
    else:
        assistant.message_assistant(session_id, user_sessions.get(session_id), user_message)
        return jsonify({"status": "Message received, processing started"}), 202


def handle_bypass_llm(session_id, user_message):
    if BYPASS_LLM:
        logger.info(f"BYPASS_LLM is enabled. Logging message: {user_message}")
        current_directory = os.path.dirname(os.path.abspath(__file__))
        dev_dir = os.path.join(current_directory, 'dev')
        CANNED_RESPONSE_FILE = os.path.join(dev_dir, 'canned_response.txt')
        try:
            with open(CANNED_RESPONSE_FILE, 'r') as file:
                canned_response = file.read().strip()
        except FileNotFoundError:
            logger.error("Canned response file not found")
            return True

        # Send the canned response to the text stream
        stream_manager.add_to_text_buffer(session_id, canned_response)
        return True

    return False


if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'
    app.run(debug=True, host=os.environ['REST_ADDRESS'], port=os.environ['REST_PORT'], use_reloader=False)
