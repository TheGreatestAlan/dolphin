import tkinter as tk
from tkinter import scrolledtext

class VoiceAssistantGUI:
    def __init__(self):
        self.root = None
        self.response_text = None
        self.voice_level_canvas = None
        self.voiced_confidences = []
        self.max_confidence = 0.5  # Initial max confidence for scaling

    def update_chat(self, speaker, message):
        self.root.after(0, self._update_chat, speaker, message)

    def _update_chat(self, speaker, message):
        self.response_text.config(state=tk.NORMAL)
        self.response_text.insert(tk.END, f"{speaker}: {message}\n")
        self.response_text.config(state=tk.DISABLED)
        self.response_text.yview(tk.END)

    def update_voice_level_graph(self, new_confidence):
        if self.voice_level_canvas is not None:
            self.voice_level_canvas.delete("all")
            self.voiced_confidences = self.voiced_confidences[-100:]  # Keep the last 100 values

            if new_confidence > self.max_confidence:
                self.max_confidence = new_confidence

            for i, confidence in enumerate(self.voiced_confidences):
                x = i * 5
                y = 100 - (confidence / self.max_confidence) * 100
                self.voice_level_canvas.create_rectangle(x, y, x + 5, 100, fill="blue")

    def run(self):
        self.root = tk.Tk()
        self.root.title("Voice Assistant")

        self.response_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, height=15)
        self.response_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.voice_level_canvas = tk.Canvas(self.root, height=100, bg="white")
        self.voice_level_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.root.mainloop()
