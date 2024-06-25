import json
import time

import requests
from flask import Flask, request, jsonify, Response
import os

from SmartFindingInventoryClient import SmartFindingInventoryClient
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
from integrations.InventoryRestClient import InventoryClient
from llms.RestLLM import RestLLM
from llms.ChatGPT4 import ChatGPT4  # Import the ChatGPT4 class
from FunctionMapper import FunctionMapper
from integrations.ChatHandler import ChatHandler

app = Flask(__name__)

# Determine which LLM client to use based on environment variable
llm_type = os.getenv('LLM_TYPE', 'RestLLM')  # Default to 'RestLLM' if not specified
llm_url = os.getenv('LLM_URL', 'http://127.0.0.1:5002')

if llm_type == 'RestLLM':
    llm_client = RestLLM(llm_url)
elif llm_type == 'ChatGPT':
    llm_client = ChatGPT4()  # Assuming ChatGPT4 does not require a URL
else:
    raise ValueError(f"Unsupported LLM_TYPE: {llm_type}")

sessions_file_path = 'sessions.json'
MAX_NESTING_LEVEL = 3

# Initialize Inventory, FunctionMapper, and ChatHandler
rest_inventory_client = InventoryClient(os.environ.get("ORGANIZER_SERVER_URL"))
smart_finding_inventory_client = SmartFindingInventoryClient(rest_inventory_client, llm_client)
function_generator = InventoryFunctionGenerator(llm_client)
sessions = {}
chat_handler = ChatHandler(sessions, sessions_file_path)
function_mapper = FunctionMapper(smart_finding_inventory_client, function_generator, chat_handler)

# Ensure the system message is read at startup
SYSTEM_MESSAGE = ''
FUNCTION_LIST = ''


def read_system_message(file_path='../prompt/SystemPrompt.txt'):
    global SYSTEM_MESSAGE
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
    try:
        with open(absolute_path, 'r') as file:
            SYSTEM_MESSAGE = file.read()
    except FileNotFoundError:
        print(f"File not found: {absolute_path}")
    except Exception as e:
        print(f"Error reading system message: {e}")


read_system_message()
chat_handler.load_sessions_from_file()


def readFunctionList(file_path='../prompt/functionList.txt'):
    global FUNCTION_LIST
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.normpath(os.path.join(script_dir, file_path))
    try:
        with open(absolute_path, 'r') as file:
            FUNCTION_LIST = file.read()
    except FileNotFoundError:
        print(f"File not found: {absolute_path}")
    except Exception as e:
        print(f"Error reading system message: {e}")

readFunctionList()

def handle_llm_response(response, session_id, streaming=False, nesting_level=0):
    if nesting_level > MAX_NESTING_LEVEL:
        return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

    try:
        if streaming:
            full_response = ''
            for part in response():
                full_response += part
                if part.endswith('[DONE]'):  # Assuming '[END]' is a marker indicating the end of the response
                    break
            # Now that we have the full response, we can process it
            process_response_content(full_response, session_id, nesting_level)
        else:
            # Full response handling
            process_response_content(response, session_id, nesting_level)
    except Exception as e:
        return jsonify({"error": str(e)})  # Generic error handling

def process_response_content(generated_text, session_id, nesting_level):
    if nesting_level > MAX_NESTING_LEVEL:
        return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

    try:
        response_json = json.loads(generated_text)
        if "action" not in response_json or "self_message" not in response_json:
            raise ValueError("Incorrect response format")

        # Handling actions and potentially recursive operations
        function_response = function_mapper.handle_function_call(response_json, session_id)
        if function_response.get('action_name') == 'send_message':
            return  # Exit if the action does not require further processing

        # Recursive call to process further based on the self_message and function response
        next_chunk = send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'], function_response)
        return process_response_content(next_chunk, session_id, nesting_level + 1)

    except (ValueError, KeyError, json.JSONDecodeError) as e:
        error_response = {
            "self_message": response_json['self_message'] if 'self_message' in response_json else '',
            "action": {
                "action": "self_message",
                "parameters": {
                    "content": f"Invalid request, incorrect response format of this message: {generated_text}, return valid json"
                }
            }
        }
        return handle_llm_response(
            send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'], json.dumps(error_response)),
            session_id,
            nesting_level + 1
        )



def send_message_to_llm(system_message, user_message, self_message, action_response, streaming=False):
    prompt_dict = {
        "user_message": user_message,
        "self_message": self_message,
        "action_response": action_response
    }
    prompt = json.dumps(prompt_dict, indent=4)

    if streaming:
        return llm_client.stream_response(prompt, system_message)
    else:
        return llm_client.generate_response(prompt, system_message)

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = chat_handler.start_session()
    return jsonify({"session_id": session_id})


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    try:
        generated_text = send_message_to_llm(
            SYSTEM_MESSAGE,
            user_message,
            None,
            None
        )
        response = handle_llm_response(generated_text)
        return jsonify({"response": response})
    except requests.RequestException as e:
        return jsonify({"error": "Failed to generate text", "details": str(e)}), 500


@app.route('/message_agent', methods=['POST'])
def message_agent():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    try:
        # Use the streaming option in send_message_to_llm
        response_stream = send_message_to_llm(
            SYSTEM_MESSAGE + "\n" + FUNCTION_LIST,
            user_message,
            None,
            None,
            streaming=True
        )

        # Return the streamed response to the client
        return Response(response_stream(), mimetype='text/event-stream')
    except requests.RequestException as e:
        return jsonify({"error": "Failed to generate text", "details": str(e)}), 500


@app.route('/poll_response', methods=['GET'])
def poll_response():
    session_id = request.args.get('session_id')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    if len(chat_handler.sessions[session_id]) == 0:
        return '', 204

    latest_response = chat_handler.sessions[session_id][-1]
    print("LATEST RESPONSE")
    print(jsonify(latest_response))
    return jsonify(latest_response)


@app.route('/stream/<session_id>')
def stream(session_id):
    def generate():
        # Initial connection message
        yield "data: {\"message\": \"Connection established.\"}\n\n"
        last_index = -1  # Initialize to indicate no messages sent yet

        # Continuously check for new messages
        while True:
            # Ensure the session exists and has messages
            if session_id not in chat_handler.sessions:
                yield "data: {\"error\": \"Session not found or ended.\"}\n\n"
                break  # Break the loop if session does not exist or ends

            session_messages = chat_handler.sessions[session_id]
            # Stream new messages if available
            while last_index < len(session_messages) - 1:
                last_index += 1
                message = session_messages[last_index]
                yield f"data: {json.dumps(message)}\n\n"

            # Delay next iteration to reduce CPU usage
            time.sleep(1)

    # Flask response object with the appropriate mimetype for SSE
    return Response(generate(), mimetype='text/event-stream')


@app.route('/receive_data/<session_id>', methods=['POST'])
def receive_data(session_id):
    # Ensure that the session exists before attempting to add data
    if session_id not in chat_handler.sessions:
        return jsonify({"error": "Session not found"}), 404

    # Read the incoming data chunk
    data_chunk = request.data.decode('utf-8')  # Decode data assuming it's sent as UTF-8

    # Process the received data through the ChatHandler
    chat_handler.receive_stream_data(session_id, data_chunk)

    # Optionally, you might want to confirm receipt or provide additional info
    return jsonify({"status": "Data received", "session_id": session_id})


@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    return '', 200


if __name__ == '__main__':
    os.environ['FLASK_SKIP_DOTENV'] = 'true'  # Add this line to bypass dotenv loading
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)

