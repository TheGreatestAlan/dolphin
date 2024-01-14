from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Local path where the model is stored
local_model_path = "M:\\workspace\\dolphin\\dolphin-2.5-mixtral-8x7b-GPTQ"

model = AutoModelForCausalLM.from_pretrained(local_model_path,
                                             device_map="auto",
                                             trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=True)

prompt = "Write a story about llamas"
system_message = "You are a story writing assistant"
prompt_template=f'''system
{system_message}
user
{prompt}
assistant
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,  # Reduced
    do_sample=True,
    temperature=0.8,    # Adjusted
    top_p=0.8,          # Adjusted
    top_k=30,           # Adjusted
    repetition_penalty=1.1
)

print(pipe("Write a story about llamas")[0]['generated_text'])

