import torch
from TTS.api import TTS

class TextToSpeech:
    def __init__(self, model_path="tts_models/multilingual/multi-dataset/xtts_v2"):
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the TTS model
        self.tts = TTS(model_path).to(self.device)

    def list_available_models(self):
        return self.tts.list_models()

    def list_speakers(self):
        return self.tts.speakers

    def text_to_speech(self, text, speaker="Craig Gutsy", language="en", file_path="output.wav"):
        self.tts.tts_to_file(
            text=text,
            speaker=speaker,
            language=language,
            file_path=file_path
        )
