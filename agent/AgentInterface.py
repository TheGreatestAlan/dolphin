from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

KOBOLDAI_URL = "http://localhost:5001"  # Update this if your server runs on a different address or port

sessions = {}

@app.route('/start_session', methods=['POST'])
def start_session():
    response = requests.post(f"{KOBOLDAI_URL}/api/v1/session")
    if response.status_code == 200:
        session_id = response.json().get('session_id')
        sessions[session_id] = []
        return jsonify({"session_id": session_id})
    else:
        return jsonify({"error": "Failed to start session"}), 500

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    session_id = data.get('session_id')
    prompt = data.get('prompt')
    max_length = data.get('max_length', 150)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    n = data.get('n', 1)

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "session_id": session_id
    }

    response = requests.post(f"{KOBOLDAI_URL}/api/v1/generate", json=payload)
    if response.status_code == 200:
        generated_text = response.json()
        sessions[session_id].append({
            "prompt": prompt,
            "response": generated_text
        })
        return jsonify(generated_text)
    else:
        return jsonify({"error": "Failed to generate text"}), 500

@app.route('/end_session', methods=['DELETE'])
def end_session():
    data = request.json
    session_id = data.get('session_id')

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Invalid or missing session_id"}), 400

    response = requests.delete(f"{KOBOLDAI_URL}/api/v1/session/{session_id}")
    if response.status_code == 200:
        del sessions[session_id]
        return jsonify({"message": "Session ended successfully"})
    else:
        return jsonify({"error": "Failed to end session"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
