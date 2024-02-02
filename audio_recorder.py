import sounddevice as sd
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import requests
import os
import time

class AudioRecorder:
    def __init__(self, fs=16000, duration=5, output_directory="M:/model_cache/",
                 server_url="http://localhost:5000/transcribe"):
        self.fs = fs
        self.duration = duration
        self.output_directory = output_directory
        self.server_url = server_url
        self.filename = f"recording_{int(time.time())}.wav"  # Unique filename
        self.record_thread = None
        self.root = None  # Add a root attribute to store the Tk root window

    def record_audio(self):
        print("Recording...")
        myrecording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        filepath = os.path.join(self.output_directory, self.filename)
        write(filepath, self.fs, myrecording)
        print("Recording stopped.")
        self.send_file(filepath)

    def send_file(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                files = {'file': (self.filename, f)}
                response = requests.post(self.server_url, files=files)
            # Schedule the show_response method to run on the main thread
            self.root.after(0, self.show_response, response.text)
        except Exception as e:
            self.root.after(0, self.show_response, f"Failed to send file to server: {e}")

    def show_response(self, message):
        messagebox.showinfo("Server Response", message)

    def start_recording(self):
        if self.record_thread is None or not self.record_thread.is_alive():
            self.record_thread = Thread(target=self.record_audio)
            self.record_thread.start()

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Voice Recorder")
        record_button = tk.Button(self.root, text="Press to Record", command=self.start_recording)
        record_button.pack(pady=20)
        self.root.mainloop()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run_gui()
