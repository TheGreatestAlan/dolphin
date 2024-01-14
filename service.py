from flask import Flask, request, jsonify
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid

app = Flask(__name__)
app.logger.setLevel("INFO")


# Local path where the model is stored
local_model_path = "M:\\workspace\\dolphin\\dolphin-2.5-mixtral-8x7b-GPTQ"

model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                             device_map="auto",
                                             trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)

conversations = {}


@app.route('/generate', methods=['POST'])
def generate_text():
    global conversations
    start_time = time.time()

    try:
        conversation_id = request.json.get('conversation_id', str(uuid.uuid4()))
        prompt = request.json.get('prompt', '')
        system_message = request.json.get('system_message', '')

        # Retrieve or initialize the conversation history
        conversation_history = conversations.get(conversation_id, [])

        # Update conversation history with the new input
        conversation_history.append(f"system\n{system_message}\nuser\n{prompt}")

        # Truncate history if necessary
        # ... (Your truncation logic here)

        # Construct the full prompt including history
        full_prompt = "\n".join(conversation_history) + "\nassistant\n"

        # Generate response
        input_ids = tokenizer(full_prompt, return_tensors='pt').input_ids.to('cuda')
        response_ids = model.generate(input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
                                      max_new_tokens=512)
        full_response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Extract only the latest response
        latest_response = full_response_text.split("assistant\n")[-1].strip()

        # Update conversation history with the latest response
        conversation_history.append(f"assistant\n{latest_response}")

        # Save the updated history
        conversations[conversation_id] = conversation_history

        elapsed_time = time.time() - start_time

        return jsonify({'conversation_id': conversation_id, 'response': latest_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.logger.info("Starting Flask application...")
    app.run(debug=True)
