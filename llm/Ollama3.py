import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent_server.llms.LLMInterface import LLMInterface

class Ollama3LLM(LLMInterface):
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        self.model_cache_dir = "M:\\workspace\\dolphin"  # Hard-coded model cache directory

        # Verify if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {'GPU' if self.device.type == 'cuda' else 'CPU'}")

        # Load the model and tokenizer with specified cache directory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=self.model_cache_dir,
            use_auth_token=os.getenv("HUGGING_FACE_TOKEN")  # Use the token
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.model_cache_dir)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        self.conversations = {}

    def generate_response(self, prompt, system_message):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = input_ids.ne(self.pad_token_id).to(self.device)

        with torch.no_grad():
            response_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=1500,  # Adjust max length if the input is longer
                pad_token_id=self.pad_token_id,
                temperature=0.7,  # Adjust temperature for more coherent results
                top_p=0.9
            )

        generated_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)

        # Clean up the response by removing "system", "assistant" labels, and any special tokens
        # Find the positions of the tags
        start_tag = ""
        end_tag = ""

        # Extract the substring between the tags
        start_index = generated_text.rfind(start_tag)
        end_index = generated_text.rfind(end_tag)

        response = generated_text

        if start_index != -1 and end_index != -1 and end_index > start_index:
            response = generated_text[start_index + len(start_tag):end_index].strip()

            # Remove any trailing tags
            response = response.rstrip(start_tag).rstrip(end_tag).strip()

        return response

    def stream_response(self, prompt, system_message):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = input_ids.ne(self.pad_token_id).to(self.device)

        response_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=1500,  # Adjust max length if the input is longer
            pad_token_id=self.pad_token_id,
            temperature=0.7,  # Adjust temperature for more coherent results
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True
        )

        generated_text = self.tokenizer.decode(response_ids.sequences[0], skip_special_tokens=False)
        response_parts = generated_text.split(" ")  # Split response into parts

        for part in response_parts:
            yield part

# Example usage
if __name__ == "__main__":
    ollama3 = Ollama3LLM()
    prompt = "In this list, where is the turmeric? 1: turmeric, parsley 2: pencils 3: phone 4: food"
    system_message = "You are an inventory scanning guy."
    print("Generated Response:")
    print(ollama3.generate_response(prompt, system_message))

    print("\nStreaming Response:")
    for part in ollama3.stream_response(prompt, system_message):
        print(part, end=" ")
