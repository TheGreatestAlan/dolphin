import subprocess
import time
import os
import re

class WhisperTranscriber:
    def __init__(self,
                 whisper_exe="../dev/whisper-cublas-11.8.0-bin-x64/main.exe",
                 model_path="../models/ggml-medium-32-2.en.bin"):
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve relative paths based on the script directory
        self.whisper_exe = os.path.abspath(os.path.join(script_dir, whisper_exe))
        self.model_path = os.path.abspath(os.path.join(script_dir, model_path))

        # Verify that the whisper executable exists
        if not os.path.exists(self.whisper_exe):
            raise FileNotFoundError(f"The specified whisper executable does not exist: {self.whisper_exe}")

        # Verify that the model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"The specified model file does not exist: {self.model_path}")

    def transcribe_audio(self, filename="recording.wav"):
        """Transcribes the given audio file using whisper.cpp."""
        # Start the timer
        start_time = time.time()

        # Ensure the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified audio file does not exist: {filename}")

        # Construct the command to transcribe the file
        command = [self.whisper_exe, '-m', self.model_path, filename]

        # Execute the command and capture the output
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except Exception as e:
            raise RuntimeError(f"Failed to execute whisper command: {e}")

        # Stop the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Transcription time: {elapsed_time:.2f} seconds")

        # Check if the command was successful
        if result.returncode != 0:
            raise RuntimeError(f"Whisper failed with error: {result.stderr}")

        # Extract the transcribed text by removing timestamps and extra spaces
        transcription = self._process_output(result.stdout)

        return transcription

    def _process_output(self, output):
        """Process the output to remove timestamps and extra spaces."""
        # Use a regular expression to remove timestamps
        processed_output = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', '', output)

        # Remove extra spaces and newlines
        processed_output = processed_output.strip()
        processed_output = re.sub(r'\s+', ' ', processed_output)

        return processed_output

# Example usage
if __name__ == "__main__":
    # Create an instance of the WhisperTranscriber
    transcriber = WhisperTranscriber()

    # Path to the audio file you want to transcribe
    audio_file_path = "M:/model_cache/6_2/temp_detected_voice_1717365225.wav"

    # Check if the audio file exists
    if os.path.exists(audio_file_path):
        # Transcribe the audio file
        transcription = transcriber.transcribe_audio(audio_file_path)
        print("Transcription:", transcription)
    else:
        print(f"The specified audio file does not exist: {audio_file_path}")
