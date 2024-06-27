import queue
import threading
import re
from tts.GTTSHandler import GTTSHandler
from tts.PyAudioOutput import PyAudioOutput
from ui.AudioManager import AudioManager

class Speech:
    def __init__(self, tts_handler: GTTSHandler, audio_output: PyAudioOutput, audio_manager: AudioManager):
        self.tts_handler = tts_handler
        self.audio_output = audio_output
        self.audio_manager = audio_manager
        self.text_queue = queue.Queue()
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        self.chunk_buffer = ""

    def speak(self, text: str):
        print(f"Queueing text: {text}")
        self.text_queue.put(text)

    def stream_speak(self, chunk: str):
        self._process_chunk(chunk)

    def _process_chunk(self, chunk: str):
        # Filter out any chunks that contain '[DONE]'

        self.chunk_buffer += chunk.replace("[DONE]", "")

        # Check if the buffer ends with a backslash
        if self.chunk_buffer.endswith("\\"):
            return

        # Once the buffer does not end with a backslash, replace newlines with spaces
        self.chunk_buffer = self.chunk_buffer.replace("\n", " ")

        # Match words, punctuation, and spaces, keep trailing partial words in buffer
        words = re.findall(r'\S+|\s+', self.chunk_buffer)

        # Check if the last matched item is an incomplete word
        last_word = words[-1] if words else ""
        if last_word and not re.match(r'\s', last_word) and not re.match(r'[.,!?]', last_word):
            # If the last word is not followed by a space or punctuation, it's incomplete
            self.chunk_buffer = last_word  # Keep the partial word in buffer
            words = words[:-1]  # Remove the partial word from the list
        else:
            self.chunk_buffer = ""  # Clear buffer if last word is complete

        # Enqueue complete words including punctuation
        for word in words:
            if re.match(r'\S+', word):
                self.text_queue.put(word)

        # Special handling to enqueue the last word if it is the end of the input
        if not chunk or chunk.isspace() or not words:
            if self.chunk_buffer:
                self.text_queue.put(self.chunk_buffer)
                self.chunk_buffer = ""

    def _process_queue(self):
        self.audio_manager.acquire_audio()
        try:
            sentence_buffer = ""
            while True:
                text = self.text_queue.get()
                if text is None:
                    if sentence_buffer:
                        self._speak_sentence(sentence_buffer)
                    break

                sentence_buffer += text + " "
                if re.search(r'[.!?,]', text):  # Check if the text contains a sentence-ending punctuation
                    self._speak_sentence(sentence_buffer)
                    sentence_buffer = ""  # Clear buffer after speaking the sentence

            if sentence_buffer:  # If there is any remaining text, speak it as a whole
                self._speak_sentence(sentence_buffer)
        finally:
            self.audio_manager.release_audio()

    def _speak_sentence(self, sentence):
        if sentence.strip() and re.search(r'\w',
                                          sentence):  # Check if the sentence is not empty and contains alphanumeric characters
            # Replace newlines with spaces
            sentence = sentence.replace("\n", " ").replace("\\n", " ")
            audio_fp = self.tts_handler.text_to_speech(sentence)
            self.audio_output.play_audio(audio_fp)
        else:
            print("Empty sentence received, skipping TTS processing.")

    def wait_until_done(self):
        self.text_queue.put(None)
        self.thread.join()

# Main function for testing
if __name__ == "__main__":
    tts_handler = GTTSHandler(lang='en')
    audio_output = PyAudioOutput()
    audio_manager = AudioManager()

    speech = Speech(tts_handler, audio_output, audio_manager)

    # Simulated chunks
    chunks = [
        "Hey there! ",
        "This is ",
        "Jack Black. ",
        "Of course I ",
        "hear you, ",
        "loud and clear! ",
        "How can I ",
        "assist you ",
        "today?[DONE]"
    ]
    # Simulated chunks - one word at a time
    #sentence = "Hey there! This is Jack Black. Of course I hear you, loud and clear! How can I assist you today?"
    #chunks = sentence.split()

    for chunk in chunks:
        speech.stream_speak(chunk)  # Add space after each word to simulate normal speech pattern

    speech.wait_until_done()
