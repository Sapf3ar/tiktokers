import scipy
import torch
import numpy as np

from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

from sbert_punc_case_ru import SbertPuncCase


class Processor:
    def __init__(
        self,
        device="cuda",
    ) -> None:
        self.device = device
        self.model: EncDecRNNTBPEModel = None

    def load_model(self):
        model = EncDecRNNTBPEModel.from_config_file("./weights/rnnt_model_config.yaml")
        ckpt = torch.load("./weights/rnnt_model_weights.ckpt", map_location=self.device)
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to(self.device)
        self.model = model
        print("ASR MODEL LOADED")
        vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=True)
        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = utils
        self.vad_iterator = VADIterator(vad, threshold=0.5, min_silence_duration_ms=500)
        print("VAD LOADED")
        punctuation_model = SbertPuncCase()
        punctuation_model.to(self.device)
        self.punctuation_model = punctuation_model
        print("PUNCTUATION LOADED")

    def process_audio(self, audio):
        window_size_samples = 512
        start=0
        end=0
        for j in range(0, len(audio), window_size_samples):
            chunk = audio[j: j+window_size_samples]
            if len(chunk) < window_size_samples:
                continue
            speech_timestamps = self.vad_iterator(chunk, return_seconds=False)
            if speech_timestamps is not None:
                if "start" in speech_timestamps:
                    start = speech_timestamps["start"]
                else:
                    end = speech_timestamps["end"]
                    print(f"{start} - {end}")
                if start < end:
                    speech = audio[start:end]

                    orig_text = self.model.transcribe(speech)[0][0]
                    if orig_text:
                        text = self.punctuation_model.punctuate(orig_text)
                        yield text
        self.vad_iterator.reset_states()
