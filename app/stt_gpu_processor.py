"""
stt_gpu_processor module

This module provides functionality to process speech-to-text using GPU,
including voice activity detection (VAD) and punctuation restoration.
"""

import torch
import logging

from sbert_punc_case_ru import SbertPuncCase
from nemo.collections.asr.models import EncDecRNNTBPEModel

class Processor:
    """
    Processor class

    This class handles the initialization and management of various components
    required for speech-to-text processing using GPU resources.
    """

    def __init__(self, device: str = "cuda") -> None:
        """
        Initialize the Processor class.

        Parameters
        ----------
        device : str, optional
            The device to use for processing ('cuda' for GPU, 'cpu' for CPU). Default is 'cuda'.
        """
        self.device = device
        self.model: EncDecRNNTBPEModel = None
        self.vad = None
        self.get_speech_timestamps = None
        self.punctuation_model = None

    def load_model(self) -> None:
        """
        Load the ASR model, VAD model, and punctuation model.

        This method initializes the automatic speech recognition (ASR) model,
        voice activity detection (VAD) model, and punctuation restoration model,
        setting them up for use in audio processing.

        Loads ASR model from a configuration file and checkpoint.
        Loads VAD and related utilities from a repository.
        Loads punctuation model and moves it to the specified device.

        Raises
        ------
        RuntimeError
            If any model fails to load.
        """
        logging.info("INITIALIZING ASR MODEL")
        default_asr_path = "./weights/rnnt_model_config.yaml"
        model = EncDecRNNTBPEModel.from_config_file(default_asr_path)
        ckpt = torch.load("./weights/rnnt_model_weights.ckpt",
                          map_location=self.device.split(":")[0])
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to(self.device)
        self.model = model
        logging.info("ASR MODEL INITIALIZED")

        logging.info("LOADING VAD")
        vad, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                    model="silero_vad",
                                    force_reload=True,
                                    onnx=True)
        self.get_speech_timestamps = utils[0]
        self.vad = vad
        logging.info("VAD LOADED")

        logging.info("LOADING PUNCTUATION MODEL")
        punctuation_model = SbertPuncCase()
        punctuation_model.to(self.device)
        self.punctuation_model = punctuation_model
        logging.info("PUNCTUATION MODEL LOADED")

    def process_audio(self, audio) -> str:
        """
        Process an audio file and convert it to text with punctuation.

        This method performs voice activity detection on the audio input, 
        transcribes the detected speech segments using the ASR model, 
        and restores punctuation in the transcribed text.

        Parameters
        ----------
        audio : np.ndarray
            The audio signal to be processed.

        Yields
        ------
        str
            The transcribed and punctuated text for detected speech segment.
        """
        timestamps = self.get_speech_timestamps(audio,
                                                self.vad,
                                                threshold=0.5,
                                                min_speech_duration_ms=250,
                                                min_silence_duration_ms=500,
                                                max_speech_duration_s=16,
                                                window_size_samples=512,)
        for timestamp in timestamps:
            start = timestamp["start"]
            end = timestamp["end"]
            speech = audio[start:end]
            orig_text = self.model.transcribe(speech)[0][0]
            if orig_text:
                text = self.punctuation_model.punctuate(orig_text)
                yield text
