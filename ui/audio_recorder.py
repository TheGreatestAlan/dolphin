import pyaudio
from scipy.io.wavfile import write
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
import os
import time as tm
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import itertools
import queue

from asr.WhisperTranscriber import WhisperTranscriber

# Initialize AudioTranscriber
audioTranscriber = WhisperTranscriber()

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_ttimestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def validate(model, inputs: torch.Tensor, sr: int):
    with torch.no_grad():
        outs = model(inputs, sr)
    return outs

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound

class AudioRecorder:
    def __init__(self, fs=16000, output_directory="M:/model_cache/6_2"):
        self.audioTranscriber = WhisperTranscriber()
        self.server_url = os.getenv('SERVER_URL')
        self.username = os.getenv("USERNAME")
        self.password = os.getenv('PASSWORD')
        self.fs = fs
        self.output_directory = os.path.normpath(output_directory)
        self.filename = None
        self.recording = False
        self.record_thread = None
        self.root = None
        self.myrecording = None
        self.response_text = None
        self.voiced_confidences = []
        self.continue_recording = True
        self.current_audio_chunk = []
        self.last_voice_time = None
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        # Start the transcription thread
        self.transcription_thread = Thread(target=self.process_transcription_queue)
        self.transcription_thread.start()

    def stop(self):
        input("Press Enter to stop the recording:")
        self.continue_recording = False
        self.transcription_queue.put(None)  # Signal the transcription thread to stop

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_int16 = np.frombuffer(in_data, np.int16)
        audio_float32 = int2float(audio_int16)
        new_confidence = validate(model, torch.from_numpy(audio_float32).unsqueeze(0), self.fs).item()
        self.voiced_confidences.append(new_confidence)

        current_time = tm.time()
        if new_confidence > 0.5:  # Voice detected
            self.current_audio_chunk.append(audio_int16)
            self.last_voice_time = current_time
        else:
            if self.current_audio_chunk and (current_time - self.last_voice_time) > 1.0:
                self.save_detected_voice()

        return (in_data, pyaudio.paContinue)

    def save_detected_voice(self):
        if self.current_audio_chunk:
            self.myrecording = np.concatenate(self.current_audio_chunk)
            self.current_audio_chunk = []
            temp_filename = f"temp_detected_voice_{int(tm.time())}.wav"
            temp_filepath = os.path.join(self.output_directory, temp_filename)
            write(temp_filepath, self.fs, self.myrecording)
            self.transcription_queue.put(temp_filepath)
            print(f"Saved detected voice to {temp_filepath}")

    def process_transcription_queue(self):
        while True:
            temp_filepath = self.transcription_queue.get()
            if temp_filepath is None:
                break
            self.transcribe_in_thread(temp_filepath)
            os.remove(temp_filepath)  # Clean up temporary file

    def transcribe_in_thread(self, temp_filepath):
        print("transcribing")
        transcription = self.audioTranscriber.transcribe_audio(temp_filepath)
        self.audio_queue.put(transcription)

    def start_recording(self):
        self.recording = True
        self.myrecording = None

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.fs,
                        input=True,
                        frames_per_buffer=1024,
                        stream_callback=self.audio_callback)

        self.continue_recording = True

        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'b-', animated=True)

        def init():
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            return ln,

        def update(frame):
            xdata.append(len(xdata))
            ydata.append(self.voiced_confidences[-1])
            ln.set_data(xdata, ydata)
            if len(xdata) > 100:
                ax.set_xlim(len(xdata) - 100, len(xdata))
            return ln,

        ani = FuncAnimation(fig, update, frames=itertools.count, init_func=init, blit=True, save_count=100)
        plt.show(block=False)

        stream.start_stream()

        stop_listener = Thread(target=self.stop)
        stop_listener.start()

        while self.continue_recording:
            plt.pause(0.1)
            if not self.audio_queue.empty():
                transcription = self.audio_queue.get()
                self.update_gui(transcription)

        stream.stop_stream()
        stream.close()
        p.terminate()
        plt.close(fig)

        self.recording = False
        if self.current_audio_chunk:
            self.save_detected_voice()

    def update_gui(self, message):
        self.root.after(0, self._update_gui, message)

    def _update_gui(self, message):
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, message)
        self.response_text.config(state=tk.DISABLED)

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder")

        self.response_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, height=15)
        self.response_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.record_thread = Thread(target=self.start_recording)
        self.record_thread.start()

        self.root.mainloop()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run_gui()
