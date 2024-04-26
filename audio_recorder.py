import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import requests
import os
import time
import re

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
                response = requests.post(self.server_url + "/transcribe", files=files, auth=(self.username, self.password), verify=False)
                transcription_response = response.text
                match = re.search(r'<Cleaned_Transcription>(.*?)</Cleaned_Transcription>', transcription_response)
                if match:
                    cleaned_transcription = match.group(1)
                    self.send_inventory_command(cleaned_transcription)
                else:
                    self.root.after(0, self.show_response, "No transcription found")
        except Exception as e:
            self.root.after(0, self.show_response, f"Failed to send audio file: {e}")

    def send_inventory_command(self, transcription):
        data = {
            "prompt": transcription,
            "system_message": "You are chatGPT-4, a well-trained LLM used to assist humans. /set system You are a helpful assistant with access to the following inventory system functions. Use them if required - examples: \n\n1. If asked 'add a hammer to drawer 5', you should update the inventory of drawer 5 by adding a hammer.\n2. If asked 'remove a screwdriver from drawer 10', you should update the inventory of drawer 10 by removing a screwdriver.\n3. If asked 'add two pencils to drawer 3', you should update the inventory of drawer 3 by adding two pencils.\n\n[{\"name\": \"add_inventory\", \"description\": \"Add items to the inventory of a specific drawer\", \"parameters\": {\"type\": \"object\", \"properties\": {\"drawer_number\": {\"type\": \"integer\", \"description\": \"The number of the drawer to update\"}, \"items_to_add\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of items to add to the drawer\"}}, \"required\": [\"drawer_number\", \"items_to_add\"]}, {\"name\": \"delete_inventory\", \"description\": \"Delete items from the inventory of a specific drawer\", \"parameters\": {\"type\": \"object\", \"properties\": {\"drawer_number\": {\"type\": \"integer\", \"description\": \"The number of the drawer to update\"}, \"items_to_delete\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"List of items to delete from the drawer\"}}, \"required\": [\"drawer_number\", \"items_to_delete\"]}]"
        }
        response = requests.post(self.server_url + "/generate", json=data, auth=(self.username, self.password), verify=False)
        self.show_response(response.text)

    def show_response(self, message):
        messagebox.showinfo("Server Response", message)

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder")
        toggle_button = tk.Button(self.root, text="Toggle Recording", command=self.toggle_recording)
        toggle_button.pack(pady=20)
        self.root.mainloop()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run_gui()
