import base64
import json
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
import requests
import os
import threading
import sounddevice as sd


from agent.AgentInterface import AgentInterface

class ChatApp(AgentInterface):
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Chat App")

        # Read AGENT_URL from environment variable
        self.agent_api_base_path = os.getenv("AGENT_API_BASE_PATH", "")
        self.agent_url = os.getenv('AGENT_URL', 'http://127.0.0.1:5000')

        # Read auth details from environment variables
        self.username = os.getenv('AGENT_API_USERNAME')
        self.password = os.getenv('AGENT_API_PASSWORD')
        self.skipSSL = os.getenv("SKIP_SSL", False)

        # Prepare the Authorization header if credentials are provided
        self.auth_headers = {}
        if self.username and self.password:
            auth_string = f"{self.username}:{self.password}"
            auth_base64 = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
            self.auth_headers = {
                "Authorization": f"Basic {auth_base64}"
            }

        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
        self.chat_window.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.prompt_entry = tk.Entry(root, width=80)
        self.prompt_entry.grid(row=1, column=0, padx=10, pady=10)
        self.prompt_entry.bind("<Return>", self.send_prompt)

        self.send_button = tk.Button(root, text="Send", command=self.send_prompt)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.audio_button = tk.Button(root, text="Start Audio", command=self.listen_to_audio_stream)
        self.audio_button.grid(row=1, column=2, padx=10, pady=10)

        self.message_history = []
        self.current_message = ""

        self.start_session()

    def start_session(self):
        self.append_chat("System", "Starting session...")
        thread = threading.Thread(target=self._start_session)
        thread.start()

    def _start_session(self):
        response = requests.post(f"{self.agent_url}{self.agent_api_base_path}/start_session", headers=self.auth_headers, verify=self.skipSSL)
        if response.ok:
            self.session_id = response.json()['session_id']
            self.append_chat("System", "Session started.")
            self.listen_to_stream()
        else:
            self.append_chat("System", "Failed to start session.")

    def send_prompt(self, event=None):
        prompt = self.prompt_entry.get()
        if prompt:
            self.prompt_entry.delete(0, tk.END)
            self.append_chat("You", prompt)

            # Start streaming immediately after sending the message
            thread = threading.Thread(target=self._send_prompt, args=(prompt,))
            thread.start()

    def _send_prompt(self, prompt):
        data = {
            'session_id': self.session_id,
            'user_message': prompt
        }
        try:
            response = requests.post(f"{self.agent_url}{self.agent_api_base_path}/message_agent", json=data, headers=self.auth_headers, verify=self.skipSSL)
            if not response.ok:
                self.append_chat("System", f"Failed to send message: {response.text}")
        except requests.exceptions.RequestException as e:
            self.append_chat("System", f"Failed to send message: {e}")

    def listen_to_stream(self):
        def stream():
            try:
                response = requests.get(f"{self.agent_url}{self.agent_api_base_path}/stream/{self.session_id}", stream=True, headers=self.auth_headers, verify=self.skipSSL)

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if 'data:' in decoded_line:
                            data_json = decoded_line.split('data: ')[1]
                            data = json.loads(data_json)
                            message = data.get('message')

                            if message == "[DONE]":
                                self.message_history.append(f"Agent: {self.current_message.strip()}")
                                self.current_message = ""
                                self.update_chat_display()
                            else:
                                self.current_message += message
                                self.update_chat_display()

            except requests.exceptions.RequestException as e:
                self.append_chat("System", f"Failed to connect to stream: {e}")

        thread = threading.Thread(target=stream)
        thread.daemon = True
        thread.start()

    def listen_to_audio_stream(self):
        def stream_audio():
            try:
                response = requests.get(f"{self.agent_url}{self.agent_api_base_path}/streamaudio/{self.session_id}", stream=True, headers=self.auth_headers, verify=self.skipSSL)

                audio_buffer = []  # Buffer to accumulate audio data
                buffer_size_threshold = 8192  # Adjust this threshold as needed

                for chunk in response.iter_content(chunk_size=None):  # Allow dynamic chunk size
                    if chunk:
                        audio_buffer.append(np.frombuffer(chunk, dtype=np.int16))

                        # Concatenate and play when buffer size exceeds the threshold
                        if sum(len(data) for data in audio_buffer) > buffer_size_threshold:
                            combined_audio = np.concatenate(audio_buffer)
                            self.play_audio_chunk(combined_audio)
                            audio_buffer = []  # Clear the buffer after playing

                # Play any remaining audio in the buffer
                if audio_buffer:
                    combined_audio = np.concatenate(audio_buffer)
                    self.play_audio_chunk(combined_audio)

            except requests.exceptions.RequestException as e:
                self.append_chat("System", f"Failed to connect to audio stream: {e}")

        thread = threading.Thread(target=stream_audio)
        thread.daemon = True
        thread.start()

    def play_audio_chunk(self, chunk):
        dtype = np.int16
        channels = 1
        sample_rate = 22050

        # Convert bytes to NumPy array
        audio_data = np.frombuffer(chunk, dtype=dtype)
        print(len(chunk))

        # Reshape for stereo channels if necessary
        audio_data = audio_data.reshape(-1, channels)


        # Play audio using sounddevice
        sd.play(audio_data, samplerate=sample_rate, blocking=True)
        sd.wait()  # Wait until the audio is played before continuing

    def update_chat_display(self):
        def update():
            self.chat_window.config(state='normal')
            self.chat_window.delete("1.0", tk.END)

            # Display message history
            for entry in self.message_history:
                self.chat_window.insert(tk.END, f"{entry}\n")

            # Display current agent message
            if self.current_message:
                self.chat_window.insert(tk.END, f"Agent: {self.current_message.strip()}\n")

            self.chat_window.config(state='disabled')
            self.chat_window.yview(tk.END)

        self.root.after(0, update)

    def append_chat(self, speaker, text, newline=True):
        def append():
            self.chat_window.config(state='normal')
            entry = f"{speaker}: {text}"
            if newline:
                entry += "\n"
            self.message_history.append(entry.strip())
            self.chat_window.delete("1.0", tk.END)

            # Display message history
            for entry in self.message_history:
                self.chat_window.insert(tk.END, f"{entry}\n")

            self.chat_window.config(state='disabled')
            self.chat_window.yview(tk.END)

        self.root.after(0, append)

    def end_session(self):
        if self.session_id:
            try:
                requests.delete(f"{self.agent_url}{self.agent_api_base_path}/end_session", json={'session_id': self.session_id}, headers=self.auth_headers, verify=self.skipSSL)
            except requests.exceptions.RequestException as e:
                self.append_chat("System", f"Failed to end session: {e}")
        self.root.destroy()

    def on_closing(self):
        self.end_session()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
