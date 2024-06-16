import requests
import pyaudio
import wave


def get_tts(text, voice, output_file="output.wav"):
    url = "http://localhost:8020/api/tts"
    payload = {
        "text": text,
        "voice": voice
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Audio saved as {output_file}")
        return output_file
    else:
        print(f"Failed to generate TTS. Status code: {response.status_code}")
        print(response.text)
        return None


def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create an interface to PortAudio
    p = pyaudio.PyAudio()

    # Open a .Stream object to write the WAV file to
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read data in chunks
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    # Play the audio
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk_size)

    # Close and terminate the stream
    stream.close()
    p.terminate()


if __name__ == "__main__":
    text = "Hello, World!"
    voice = "en_us_male"
    output_file = "output.wav"

    audio_file = get_tts(text, voice, output_file)

    if audio_file:
        play_audio(audio_file)
