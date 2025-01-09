import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"


class TextToSpeech:
    def __init__(self, model_path="tts_models/en/ljspeech/tacotron2-DDC"):
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize the TTS model
        self.tts = TTS(model_path).to(self.device)

    def text_to_speech(self, text, file_path="output.wav"):
        # Convert text to speech and save to a file
        self.tts.tts_to_file(
            text=text,
            file_path=file_path
        )

#This XTTS2 model required more CPU/GPU
# class TextToSpeech:
#     def __init__(self, model_path="tts_models/multilingual/multi-dataset/xtts_v2"):
#         # Get device
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         # Initialize the TTS model
#         self.tts = TTS(model_path).to(self.device)

#     def list_available_models(self):
#         return self.tts.list_models()

#     def list_speakers(self):
#         return self.tts.speakers

#     def text_to_speech(self, text, speaker="Craig Gutsy", language="en", file_path="output.wav"):
#         self.tts.tts_to_file(
#             text=text,
#             speaker=speaker,
#             language=language,
#             file_path=file_path
#         )


# # Initialize TTS with the target model name for English
# tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

# # Run TTS
# tts.tts_to_file(text="This is a test message.", file_path="audio.wav")