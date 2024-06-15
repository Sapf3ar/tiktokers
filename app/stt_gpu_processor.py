import torch
import logging

from sbert_punc_case_ru import SbertPuncCase

from nemo.collections.asr.models import EncDecRNNTBPEModel


class Processor:
    def __init__(
        self,
        device="cuda",
    ) -> None:
        self.device = device
        self.model: EncDecRNNTBPEModel = None

    def load_model(self):
        logging.info("INITIALIZING ASR MODEL")
        model = EncDecRNNTBPEModel.from_config_file("./weights/rnnt_model_config.yaml")
        ckpt = torch.load("./weights/rnnt_model_weights.ckpt", map_location=self.device.split(":")[0])
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to(self.device)
        self.model = model
        logging.info("ASR MODEL INITIALIZED")
        logging.info("LOADING VAD")
        vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=True,
                                    onnx=True)
        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = utils
        self.get_speech_timestamps = get_speech_timestamps
        self.vad = vad
        logging.info("VAD LOADED")
        logging.info("LOADING PUNCTUATION MODEL")
        punctuation_model = SbertPuncCase()
        punctuation_model.to(self.device)
        self.punctuation_model = punctuation_model
        logging.info("PUNCTUATION MODEL LOADED")

    def process_audio(self, audio):
        speech_timestamps = self.get_speech_timestamps(audio,
                                                       self.vad,
                                                       threshold=0.5, 
                                                       min_speech_duration_ms=250, 
                                                       min_silence_duration_ms=500,
                                                       max_speech_duration_s=16,
                                                       window_size_samples=512,)
        for timestamp in speech_timestamps:
            start = timestamp["start"]
            end = timestamp["end"]
            speech = audio[start:end]
            orig_text = self.model.transcribe(speech)[0][0]
            if orig_text:
                text = self.punctuation_model.punctuate(orig_text)
                yield text
