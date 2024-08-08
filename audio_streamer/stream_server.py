from flask import Flask, request, Response, send_file
import wave
import requests
import pyaudio
import threading
import time
import urllib.parse

app = Flask(__name__)

def generate_audio_stream(file_path):
    """Generator to stream audio file."""
    chunk_size = 1024
    try:
        with wave.open(file_path, 'rb') as audio_file:
            data = audio_file.readframes(chunk_size)
            while data:
                yield data
                data = audio_file.readframes(chunk_size)
    except FileNotFoundError:
        return

@app.route('/stream_audio', methods=['GET'])
def stream_audio():
    file_path = request.args.get('file')
    if not file_path:
        return "File path is required", 400

    return Response(generate_audio_stream(file_path), mimetype='audio/x-wav')

@app.route('/download_audio', methods=['GET'])
def download_audio():
    output_file = 'C:\\workspace\\dolphin\\tts\\output_1720285074522.wav'
    try:
        return send_file(output_file, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

def download_audio_from_server(url, output_file):
    print("Starting to download audio from server...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Audio file downloaded successfully to {output_file}.")
    else:
        print(f"Failed to get audio from server. Status code: {response.status_code}")

@app.route('/play_audio', methods=['GET'])
def play_audio():
    file_path = request.args.get('file')
    if not file_path:
        return "File path is required", 400
    stream_audio_file(file_path)
    return "Playback finished."

def get_audio_properties(file_path):
    """Get audio properties from the wave file."""
    with wave.open(file_path, 'rb') as wf:
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        frame_rate = wf.getframerate()
    return sample_width, channels, frame_rate

def stream_audio_file(server_url, sample_width, channels, frame_rate):
    response = requests.get(server_url, stream=True)
    if response.status_code != 200:
        print(f"Failed to get audio from server. Status code: {response.status_code}")
        return

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Create a stream to play audio
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True)

    # Stream audio in chunks
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            stream.write(chunk)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Playback finished.")

def start_server():
    app.run(host='0.0.0.0', port=5000)

def download_and_play_audio():
    time.sleep(1)  # Wait for the server to start
    file_path = "C:\\workspace\\dolphin\\tts\\output_1720285074522.wav"
    server_url = f"http://localhost:5000/stream_audio?file={urllib.parse.quote(file_path)}"

    # Get audio properties from the local file
    sample_width, channels, frame_rate = get_audio_properties(file_path)

    # Stream the audio file from the server
    stream_audio_file(server_url, sample_width, channels, frame_rate)

if __name__ == '__main__':
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.start()

    # Start the download and play process in the main thread
    download_and_play_audio()
