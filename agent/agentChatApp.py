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
        try:
            response = requests.post(f"{self.agent_url}/start_session")
            response.raise_for_status()
            self.session_id = response.json()['session_id']
            self.append_chat("System", "Session started.")
        except requests.exceptions.RequestException as e:
            self.append_chat("System", f"Failed to start session: {e}")

    def send_prompt(self, event=None):
        prompt = self.prompt_entry.get()
        if prompt:
            self.prompt_entry.delete(0, tk.END)
            self.append_chat("You", prompt)
            thread = threading.Thread(target=self._send_prompt, args=(prompt,))
            thread.start()

    def _send_prompt(self, prompt):
        data = {
            'session_id': self.session_id,
            'user_message': prompt
        }
        try:
            response = requests.post(f"{self.agent_url}/message_agent", json=data)
            response.raise_for_status()
            self.append_chat("System", "Message sent to agent.")
            # Start a thread to poll for the agent's response
            thread = threading.Thread(target=self.poll_for_response)
            thread.start()
        except requests.exceptions.RequestException as e:
            self.append_chat("System", f"Failed to send message: {e}")

    def poll_for_response(self):
        while True:
            try:
                response = requests.get(f"{self.agent_url}/poll_response", params={'session_id': self.session_id})
                response.raise_for_status()
                if response.status_code == 200:
                    generated_text = response.text
                    self.append_chat("Agent", generated_text)
                    break
            except requests.exceptions.RequestException as e:
                self.append_chat("System", f"Failed to get response: {e}")
                break

    def append_chat(self, speaker, text):
        self.chat_window.config(state='normal')
        self.chat_window.insert(tk.END, f"{speaker}: {text}\n")
        self.chat_window.config(state='disabled')
        self.chat_window.yview(tk.END)

    def end_session(self):
        self.append_chat("System", "Ending session...")
        thread = threading.Thread(target=self._end_session)
        thread.start()

    def _end_session(self):
        data = {
            'session_id': self.session_id
        }
        try:
            response = requests.delete(f"{self.agent_url}/end_session", json=data)
            response.raise_for_status()
            self.append_chat("System", "Session ended.")
        except requests.exceptions.RequestException as e:
            self.append_chat("System", f"Failed to end session: {e}")

    def on_closing(self):
        self.end_session()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
