import requests
import pyaudio
import wave

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

def play_audio_file(file_path):
    wf = wave.open(file_path, 'rb')

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Playback finished.")

if __name__ == '__main__':
    server_url = "http://localhost:5000/stream_audio?file=C:\\workspace\\dolphin\\tts\\output_1720285074522.wav"
    output_file = "downloaded_audio.wav"
    output_file = 'C:\\workspace\\dolphin\\tts\\output_1720285074522.wav'


    # Download the audio file from the server
    #download_audio_from_server(server_url, output_file)

    # Play the downloaded audio file
    play_audio_file(output_file)
