import time as tm
import os
import numpy as np
import pyaudio
import torch
import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from scipy.io.wavfile import write
import queue

from agent.AgentRestClient import AgentRestClient
from asr.WhisperTranscriber import WhisperTranscriber

# Initialize WhisperTranscriber
audioTranscriber = WhisperTranscriber()

# Load the Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

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
        self.voice_level_canvas = None
        self.max_confidence = 0.5  # Initial max confidence for scaling

        # Initialize the AgentRestClient
        self.agent_client = AgentRestClient(os.environ.get("AGENT_URL", "http://127.0.0.1:5000"))
        self.session_id = self.start_session()

        # Start the transcription thread
        self.transcription_thread = Thread(target=self.process_transcription_queue)
        self.transcription_thread.start()

    def start_session(self):
        try:
            session_id = self.agent_client.start_session()
            print(f"Started session with ID: {session_id}")
            return session_id
        except Exception as e:
            raise RuntimeError(f"Failed to start session with agent: {e}")

    def stop(self):
        input("Press Enter to stop the recording:")
        self.continue_recording = False
        self.transcription_queue.put(None)  # Signal the transcription thread to stop

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_int16 = np.frombuffer(in_data, np.int16)
        audio_float32 = int2float(audio_int16)
        new_confidence = validate(model, torch.from_numpy(audio_float32).unsqueeze(0), self.fs).item()
        self.voiced_confidences.append(new_confidence)

        self.update_voice_level_graph(new_confidence)

        current_time = tm.time()
        if new_confidence > 0.5:  # Voice detected
            self.current_audio_chunk.append(audio_int16)
            self.last_voice_time = current_time
        else:
            if self.current_audio_chunk and (current_time - self.last_voice_time) > 1.0:
                self.save_detected_voice()

        return (in_data, pyaudio.paContinue)

    def update_voice_level_graph(self, new_confidence):
        self.voice_level_canvas.delete("all")
        self.voiced_confidences = self.voiced_confidences[-100:]  # Keep the last 100 values

        if new_confidence > self.max_confidence:
            self.max_confidence = new_confidence

        for i, confidence in enumerate(self.voiced_confidences):
            x = i * 5
            y = 100 - (confidence / self.max_confidence) * 100
            self.voice_level_canvas.create_rectangle(x, y, x + 5, 100, fill="blue")

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
        transcription = self.audioTranscriber.transcribe_audio(temp_filepath)
        if transcription.strip():
            self.audio_queue.put(transcription)
            self.send_to_agent(transcription)

    def send_to_agent(self, transcription):
        thread = Thread(target=self._send_to_agent, args=(transcription,))
        thread.start()

    def _send_to_agent(self, transcription):
        try:
            self.agent_client.send_prompt(transcription)
            self.poll_agent_response()
        except Exception as e:
            print(f"Failed to send prompt to agent: {e}")

    def poll_agent_response(self):
        thread = Thread(target=self._poll_agent_response)
        thread.start()

    def _poll_agent_response(self):
        try:
            while True:
                response = self.agent_client.poll_response()
                if response:
                    self.update_chat("Agent", response)
                    break
        except Exception as e:
            print(f"Failed to poll response from agent: {e}")

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
        stream.start_stream()

        stop_listener = Thread(target=self.stop)
        stop_listener.start()

        while self.continue_recording:
            tm.sleep(0.1)
            if not self.audio_queue.empty():
                transcription = self.audio_queue.get()
                self.update_chat("You", transcription)

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.recording = False
        if self.current_audio_chunk:
            self.save_detected_voice()

        # End the session when recording stops
        try:
            end_response = self.agent_client.end_session()
            print(end_response)
        except Exception as e:
            print(f"Failed to end session with agent: {e}")

    def update_chat(self, speaker, message):
        self.root.after(0, self._update_chat, speaker, message)

    def _update_chat(self, speaker, message):
        self.response_text.config(state=tk.NORMAL)
        self.response_text.insert(tk.END, f"{speaker}: {message}\n")
        self.response_text.config(state=tk.DISABLED)
        self.response_text.yview(tk.END)

    def run_gui(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder")

        self.response_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, height=15)
        self.response_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.voice_level_canvas = tk.Canvas(self.root, height=100, bg="white")
        self.voice_level_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.record_thread = Thread(target=self.start_recording)
        self.record_thread.start()

        self.root.mainloop()

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run_gui()
