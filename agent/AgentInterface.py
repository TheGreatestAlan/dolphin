import requests
from flask import Flask, request, jsonify
import os

from llms.RestLLM import RestLLM

app = Flask(__name__)

# Initialize the RestLLM instance with the appropriate URL
llm_url = os.getenv('LLM_URL', 'http://localhost:5001')
llm_client = RestLLM(llm_url)
sessions = {}

# Function to read the system message from a file
def read_system_message(file_path='../prompt/SystemPrompt.txt'):
    with open(file_path, 'r') as file:
        return file.read()

@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = os.urandom(16).hex()
    sessions[session_id] = []
    return jsonify({"session_id": session_id})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('user_message', '')
    system_message = read_system_message()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    prompt = f"""
    system_message: {system_message}
    user_message: {user_message}
    self_message: This block contains your inner monologue, reflecting your private thoughts and planning. This should outline the steps needed before responding to the user.
    action_response: This block contains any responses from the actions you have previously called.
    """

    try:
        generated_text = llm_client.generate_response(prompt, system_message)
        sessions[session_id].append({
            "prompt": user_message,
            "response": generated_text
        })
        return jsonify({"response": generated_text})
    except requests.RequestException as e:
        return jsonify({"error": "Failed to generate text", "details": str(e)}), 500

@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    del sessions[session_id]
    return jsonify({"message": "Session ended successfully"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
