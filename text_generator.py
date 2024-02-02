from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "M:\\workspace\\dolphin\\dolphin-2.5-mixtral-8x7b-GPTQ"

class TextGenerator:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map="auto",
                                                          trust_remote_code=False).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.conversations = {}

    def generate_response(self, conversation_id, prompt, system_message):
        conversation_history = self.conversations.get(conversation_id, [])
        conversation_history.append(f"system\n{system_message}\nuser\n{prompt}")
        # Truncation logic (if needed) goes here

        full_prompt = "\n".join(conversation_history) + "\nassistant\n"
        input_ids = self.tokenizer(full_prompt, return_tensors='pt').input_ids.to('cuda')
        response_ids = self.model.generate(input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40,
                                           max_new_tokens=512)
        full_response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        latest_response = full_response_text.split("assistant\n")[-1].strip()
        conversation_history.append(f"assistant\n{latest_response}")
        self.conversations[conversation_id] = conversation_history

        return conversation_id, latest_response
