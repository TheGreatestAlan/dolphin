import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaAssistant:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="auto", dtype=torch.bfloat16):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure Hugging Face token is set
        self.token = os.getenv("HUGGING_FACE_TOKEN")
        if not self.token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            use_auth_token=self.token
        ).to(self.device)
        self.tool = {
            "name": "search_web",
            "description": "Perform a web search for a given search terms.",
            "parameter": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The search queries for which the search is performed.",
                        "required": True,
                    }
                }
            },
        }

    def create_messages(self, user_message):
        system_message = {
            "role": "system",
            "content": f"You are a helpful assistant with access to the following functions. Use them if required - {str(self.tool)}",
        }
        user_message = {"role": "user", "content": user_message}
        return [system_message, user_message]

    def tokenize_messages(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        return input_ids

    def generate_response(self, input_ids, max_new_tokens=256, temperature=0.1):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("")
        ]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
        )
        return outputs

    def decode_response(self, response_ids):
        return self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

    def stream_response(self, user_message):
        messages = self.create_messages(user_message)
        input_ids = self.tokenize_messages(messages)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("")
        ]

        response_ids = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_text = self.tokenizer.decode(response_ids.sequences[0], skip_special_tokens=False)
        response_parts = generated_text.split(" ")  # Split response into parts

        for part in response_parts:
            yield part


# Example usage
if __name__ == "__main__":
    assistant = LlamaAssistant()
    user_message = "Today's news in Melbourne, just for your information, today is April 27, 2014."

    print("Generated Response:")
    response = assistant.get_response(user_message)
    print(response)

    print("\nStreaming Response:")
    for part in assistant.stream_response(user_message):
        print(part, end=" ")
