from openai import OpenAI

from translator.llms.LLMInterface import LLMInterface
from translator.llms.MethodNotSupportedException import MethodNotSupportedException


class OptiLLM(LLMInterface):
    def __init__(self, model="fireworks_ai/accounts/fireworks/models/qwen2p5-72b-instruct"):
        self.model = model
        self.url = "http://localhost:8000/v1/"
        self.client = OpenAI(api_key="api_key", base_url=self.url)

    def generate_response(self, prompt, system_message):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            extra_body={"optillm_approach": "moa"}
        )
        # Extract the assistant's message content
        assistant_message = response.choices[0].message.content
        return assistant_message

    def stream_response(self, prompt, system_message):
        raise MethodNotSupportedException(
            "stream",
            "Streaming responses are not supported for OptiLLM due to the multi-inference nature of this implementation."
        )

def main():
    model = OptiLLM()

    prompt = "tell me a story about Hasan Piker"
    system_message = "You are a helpful assistant."

    try:
        response = model.generate_response(prompt, system_message)
        print(response)  # Print the assistant's response
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()