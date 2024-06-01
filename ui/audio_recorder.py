import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
import requests
import os
import time
import re
import json

class AudioRecorder:
    def __init__(self, fs=16000, output_directory="M:/model_cache/"):
        self.server_url = os.getenv('SERVER_URL')
        self.username = os.getenv("USERNAME")
        self.password = os.getenv('PASSWORD')
        self.fs = fs
        self.output_directory = output_directory
        self.filename = None
        self.recording = False
        self.record_thread = None
        self.root = None
        self.myrecording = None
        self.response_text = None

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def record_audio(self):
        with sd.InputStream(samplerate=self.fs, channels=1, callback=self.audio_callback):
            while self.recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        if self.myrecording is None:
            self.myrecording = indata.copy()
        else:
            self.myrecording = np.vstack((self.myrecording, indata))

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.myrecording = None
            self.filename = f"recording_{int(time.time())}.wav"
            self.record_thread = Thread(target=self.record_audio)
            self.record_thread.start()

    def stop_recording(self):
        self.recording = False
        self.record_thread.join()
        filepath = os.path.join(self.output_directory, self.filename)
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        write(filepath, self.fs, self.myrecording)
        print("Recording stopped.")
        self.process_transcription(filepath)

    def process_transcription(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (self.filename, f)}
                response = requests.post(self.server_url + "/transcribe", files=files,
                                         auth=(self.username, self.password), verify=False)
                transcription_response = response.text
                match = re.search(r'<Cleaned_Transcription>(.*?)</Cleaned_Transcription>', transcription_response)
                if match:
                    cleaned_transcription = match.group(1)
                    self.send_transcription_to_inventory(cleaned_transcription)
                else:
                    self.root.after(0, self.show_response, "No transcription found")
        except Exception as e:
            self.root.after(0, self.show_response, f"Failed to send audio file: {e}")

    def send_transcription_to_inventory(self, transcription):
        data = {
            "prompt": transcription
        }
        response = requests.post(self.server_url + "/text_inventory", json=data, auth=(self.username, self.password),
                                 verify=False)
        self.root.after(0, self.show_response, response.text)

    def show_response(self, message):
        try:
            # Attempt to parse the message as JSON
            json_response = json.loads(message)
            if isinstance(json_response, dict):
                if "response" in json_response:
                    sorted_response = {
                        k: json_response["response"][k] for k in sorted(
                            json_response["response"].keys(),
                            key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x.lower())
                        )
                    }
                    formatted_message = json.dumps({"response": sorted_response}, indent=4)
                else:
                    formatted_message = json.dumps(json_response, indent=4)
            elif isinstance(json_response, list):
                formatted_message = json.dumps(json_response, indent=4)
            else:
                formatted_message = message.replace("\\n", "\n")
        except Exception:
            # If it's not JSON, show it as a string
            formatted_message = message.replace("\\n", "\n")

        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, formatted_message)
        self.response_text.config(state=tk.DISABLED)

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder")

        toggle_button = tk.Button(self.root, text="Toggle Recording", command=self.toggle_recording)
        toggle_button.pack(pady=20)

        self.response_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, height=15)
        self.response_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.root.mainloop()


if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run_gui()
