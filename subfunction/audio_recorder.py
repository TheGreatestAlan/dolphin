import os
import queue
import time as tm
import numpy as np
import sounddevice as sd
from threading import Thread, Event
from scipy.io.wavfile import write
import tkinter as tk
from tkinter import ttk
import whisper

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound

class WhisperTranscriber:
    def __init__(self, model_name="base"):
        # Load the model
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, filename="recording.wav"):
        """Transcribes the given audio file using Whisper."""
        # Ensure the file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The specified audio file does not exist: {filename}")

        # Transcribe the audio file
        result = self.model.transcribe(filename)
        transcription = result['text']

        return transcription

class AudioRecorder:
    def __init__(self, fs=16000, output_directory="recordings", gui_update_queue=None):
        print("Initializing AudioRecorder...")
        self.fs = fs
        self.output_directory = os.path.normpath(output_directory)
        os.makedirs(self.output_directory, exist_ok=True)
        self.filename = None
        self.recording = False
        self.record_thread = None
        self.myrecording = None
        self.continue_recording = True
        self.current_audio_chunk = []
        self.audio_manager = self.DummyAudioManager()
        self.gui_update_queue = gui_update_queue
        self.transcriber = WhisperTranscriber()

    class DummyAudioManager:
        def __init__(self):
            self.record_event = Event()

        def release_audio(self):
            self.record_event.clear()

    def audio_callback(self, in_data, frames, time_info, status):
        if not self.audio_manager.record_event.is_set():
            return

        audio_int16 = in_data
        audio_float32 = int2float(audio_int16)
        self.current_audio_chunk.append(audio_int16)

        # Calculate the RMS (Root Mean Square) of the audio signal to get the volume level
        rms = np.sqrt(np.mean(audio_float32**2))
        if self.gui_update_queue:
            self.gui_update_queue.put(rms)

    def save_and_transcribe(self):
        if self.current_audio_chunk:
            self.myrecording = np.concatenate(self.current_audio_chunk)
            self.current_audio_chunk = []
            temp_filename = f"temp_recording_{int(tm.time())}.wav"
            temp_filepath = os.path.join(self.output_directory, temp_filename)
            write(temp_filepath, self.fs, self.myrecording)
            print(f"Saved recording to {temp_filepath}")
            transcription = self.transcriber.transcribe_audio(temp_filepath)
            print(f"Transcription result: {transcription}")

    def start_recording(self):
        self.recording = True
        self.myrecording = None

        self.continue_recording = True
        self.audio_manager.record_event.set()

        def audio_callback_for_sd(indata, frames, time, status):
            self.audio_callback(indata.copy(), frames, time, status)

        with sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype='int16',
                callback=audio_callback_for_sd,
                blocksize=512) as stream:
            while self.continue_recording:
                tm.sleep(0.1)

        self.audio_manager.release_audio()

        self.recording = False
        self.save_and_transcribe()

    def stop_recording(self):
        self.continue_recording = False

def start_recording(recorder):
    if not recorder.recording:
        recorder.record_thread = Thread(target=recorder.start_recording)
        recorder.record_thread.start()

def stop_recording(recorder):
    if recorder.recording:
        recorder.stop_recording()
        recorder.record_thread.join()

if __name__ == "__main__":
    gui_update_queue = queue.Queue()
    recorder = AudioRecorder(gui_update_queue=gui_update_queue)

    def start_stop_callback():
        if recorder.recording:
            stop_recording(recorder)
            start_stop_button.config(text="Start Recording")
        else:
            start_recording(recorder)
            start_stop_button.config(text="Stop Recording")

    def update_audio_level_display():
        try:
            rms = gui_update_queue.get_nowait()
            audio_level.set(rms * 100)  # Scale RMS value for the progress bar
        except queue.Empty:
            pass
        root.after(100, update_audio_level_display)

    root = tk.Tk()
    root.title("Audio Recorder")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    start_stop_button = ttk.Button(frame, text="Start Recording", command=start_stop_callback)
    start_stop_button.grid(row=0, column=0, padx=5, pady=5)

    audio_level = tk.DoubleVar()
    audio_level_display = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate", variable=audio_level)
    audio_level_display.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

    update_audio_level_display()

    root.mainloop()
