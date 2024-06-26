import json
import tkinter as tk
from tkinter import scrolledtext
import requests
import os
import threading

from agent.AgentInterface import AgentInterface


class ChatApp(AgentInterface):
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Chat App")

        # Read AGENT_URL from environment variable
        self.agent_url = os.getenv('AGENT_URL', 'http://127.0.0.1:5000')

        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled')
        self.chat_window.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.prompt_entry = tk.Entry(root, width=80)
        self.prompt_entry.grid(row=1, column=0, padx=10, pady=10)
        self.prompt_entry.bind("<Return>", self.send_prompt)

        self.send_button = tk.Button(root, text="Send", command=self.send_prompt)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.start_session()

    def start_session(self):
        self.append_chat("System", "Starting session...")
        thread = threading.Thread(target=self._start_session)
        thread.start()

    def _start_session(self):
        response = requests.post(f"{self.agent_url}/start_session")
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
            response = requests.post(f"{self.agent_url}/message_agent", json=data)
            if not response.ok:
                self.append_chat("System", f"Failed to send message: {response.text}")
        except requests.exceptions.RequestException as e:
            self.append_chat("System", f"Failed to send message: {e}")

    def listen_to_stream(self):
        def stream():
            try:
                response = requests.get(f"{self.agent_url}/stream/{self.session_id}", stream=True)
                current_message = ""  # Accumulate messages here

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if 'data:' in decoded_line:
                            data_json = decoded_line.split('data: ')[1]
                            data = json.loads(data_json)
                            message = data.get('message')

                            if message == "[DONE]":
                                self.append_chat("Agent", current_message)  # Display the complete message
                                current_message = ""  # Clear the current message after displaying
                            else:
                                current_message += message  # Append new part to the current message
                                self.update_chat_display("Agent", current_message)  # Update GUI with the current message

            except requests.exceptions.RequestException as e:
                self.append_chat("System", f"Failed to connect to stream: {e}")

        thread = threading.Thread(target=stream)
        thread.daemon = True
        thread.start()

    def update_chat_display(self, speaker, text):
        def update():
            self.chat_window.config(state='normal')
            lines = self.chat_window.get("1.0", tk.END).split("\n")

            # Check if the last line starts with "Agent:"
            if lines and any(line.startswith("Agent:") for line in lines):
                # Find the last line with "Agent:"
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].startswith("Agent:"):
                        lines.pop(i)
                        break
                # Re-insert the lines after removing the last "Agent:" line
                self.chat_window.delete("1.0", tk.END)
                self.chat_window.insert(tk.END, "\n".join(lines) + "\n")

            # Append new text with speaker label
            self.chat_window.insert(tk.END, f"{speaker}: {text}\n")
            self.chat_window.config(state='disabled')
            self.chat_window.yview(tk.END)

        self.root.after(0, update)

    def append_chat(self, speaker, text, newline=True):
        def append():
            self.chat_window.config(state='normal')
            entry = f"{speaker}: {text}"
            if newline:
                entry += "\n"
            self.chat_window.insert(tk.END, entry)
            self.chat_window.config(state='disabled')
            self.chat_window.yview(tk.END)

        self.root.after(0, append)

    def end_session(self):
        if self.session_id:
            try:
                requests.delete(f"{self.agent_url}/end_session", json={'session_id': self.session_id})
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
