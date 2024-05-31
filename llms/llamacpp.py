import subprocess

def run_llama(prompt):
    # Define the path to the executable and the model
    executable_path = "M:\\llama_cpp\\llama-b3051-bin-win-cuda-cu11.7.1-x64\\main.exe"
    model_path = "M:\\llama_cpp\\models\\Llama-2-70B.gguf"

    # Define the command to run the executable with the model and prompt
    command = [
        executable_path,
        "-m", model_path,
        "-p", prompt,
    ]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output
    if result.returncode == 0:
        print("Output:\n", result.stdout)
    else:
        print("Error:\n", result.stderr)

# Example prompt
prompt = "Tell me a story about a brave knight."
run_llama(prompt)
