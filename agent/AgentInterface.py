import json
import requests
from flask import Flask, request, jsonify
import os

from SmartFindingInventoryClient import SmartFindingInventoryClient
from functiongenerator.InventoryFunctionGenerator import InventoryFunctionGenerator
from integrations.InventoryRestClient import InventoryClient
from llms.RestLLM import RestLLM
from FunctionMapper import FunctionMapper
from integrations.ChatHandler import ChatHandler

app = Flask(__name__)

# Initialize the RestLLM instance with the appropriate URL
llm_url: str = os.getenv('LLM_URL', 'http://127.0.0.1:5002')
llm_client = RestLLM(llm_url)
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

def handle_llm_response(generated_text, session_id, nesting_level=0):
    if nesting_level > MAX_NESTING_LEVEL:
        return jsonify({"error": "Max nesting level reached, aborting to avoid infinite recursion."})

    try:
        try:
            response_json = json.loads(generated_text)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            response_json = json.loads("{" + generated_text + "}")
        if "action" not in response_json or "self_message" not in response_json:
            raise ValueError("Incorrect response format")

        function_response = function_mapper.handle_function_call(response_json, session_id)
        if function_response.get('action_name') == 'send_message':
            return

        return handle_llm_response(
            send_message_to_llm(SYSTEM_MESSAGE, None, response_json['self_message'], function_response), session_id,
            nesting_level + 1
        )

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


def send_message_to_llm(system_message, user_message, self_message, action_response):
    prompt_dict = {}
    if user_message:
        prompt_dict["user_message"] = user_message
    if self_message:
        prompt_dict["self_message"] = self_message
    if action_response:
        prompt_dict["action_response"] = action_response

    prompt = json.dumps(prompt_dict, indent=4)
    return llm_client.generate_response(prompt, system_message)


@app.route('/start_session', methods=['POST'])
def start_session():
    return chat_handler.start_session()


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
        generated_text = send_message_to_llm(
            SYSTEM_MESSAGE + "\n" + FUNCTION_LIST,
            user_message,
            None,
            None
        )
        handle_llm_response(generated_text, session_id)
        return '', 200
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


@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in chat_handler.sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    return chat_handler.end_session(session_id)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
