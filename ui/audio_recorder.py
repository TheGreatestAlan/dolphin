import time as tm
import os
import numpy as np
import pyaudio
import torch
from threading import Thread, Event
from scipy.io.wavfile import write
import queue

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
    def __init__(self, fs=16000, output_directory="M:/model_cache/6_2", audio_manager=None, voice_assistant=None, gui=None):
        self.audioTranscriber = WhisperTranscriber()
        self.fs = fs
        self.output_directory = os.path.normpath(output_directory)
        self.filename = None
        self.recording = False
        self.record_thread = None
        self.myrecording = None
        self.continue_recording = True
        self.current_audio_chunk = []
        self.last_voice_time = None
        self.transcription_queue = queue.Queue()

        self.audio_manager = audio_manager
        self.voice_assistant = voice_assistant
        self.gui = gui
        self.record_event = Event()
        self.record_event.set()

        # Start the transcription thread
        self.transcription_thread = Thread(target=self.process_transcription_queue)
        self.transcription_thread.start()
        print("Transcription processor thread started")

    def stop(self):
        input("Press Enter to stop the recording:")
        self.continue_recording = False
        self.transcription_queue.put(None)  # Signal the transcription thread to stop

    def pause_recording(self):
        print("Pausing recording...")
        self.record_event.clear()

    def resume_recording(self):
        print("Resuming recording...")
        self.record_event.set()

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.record_event.is_set():
            print("Recording paused")
            return (in_data, pyaudio.paContinue)

        audio_int16 = np.frombuffer(in_data, np.int16)
        audio_float32 = int2float(audio_int16)
        new_confidence = validate(model, torch.from_numpy(audio_float32).unsqueeze(0), self.fs).item()

        if self.gui:
            self.gui.update_voice_level_graph(new_confidence)

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
        transcription = self.audioTranscriber.transcribe_audio(temp_filepath)
        if transcription.strip():
            print(f"Transcription result: {transcription}")
            self.voice_assistant.send_to_agent(transcription)

    def start_recording(self):
        self.recording = True
        self.myrecording = None

        self.audio_manager.acquire_audio()
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

        stream.stop_stream()
        stream.close()
        p.terminate()
        self.audio_manager.release_audio()

        self.recording = False
        if self.current_audio_chunk:
            self.save_detected_voice()
