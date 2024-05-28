import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llms.LLMInterface import LLMInterface

class Ollama3LLM(LLMInterface):
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.model_cache_dir = "M:\\workspace\\dolphin"  # Hard-coded model cache directory

        # Verify if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {'GPU' if self.device.type == 'cuda' else 'CPU'}")

        # Load the model and tokenizer with specified cache directory
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=self.model_cache_dir,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.model_cache_dir)

        self.conversations = {}

    def generate_response(self, prompt, system_message):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(
            self.device) if self.tokenizer.pad_token_id is not None else None

        with torch.no_grad():
            response_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=1000,
                                               pad_token_id=self.tokenizer.eos_token_id)

        generated_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)

        # Clean up the response by removing "system", "assistant" labels, and any special tokens
        # Find the positions of the tags
        start_tag = "<|end_header_id|>"
        end_tag = "<|eot_id|>"

        # Extract the substring between the tags
        start_index = generated_text.rfind(start_tag)
        end_index = generated_text.rfind(end_tag)

        response = generated_text

        if start_index != -1 and end_index != -1 and end_index > start_index:
            response = generated_text[start_index + len(start_tag):end_index].strip()

            # Remove any trailing tags
            response = response.rstrip(start_tag).rstrip(end_tag).strip()

        return response