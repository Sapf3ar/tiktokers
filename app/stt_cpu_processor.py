"""
stt_cpu_processor module

This module provides functionality to process speech-to-text using CPU,
including voice activity detection (VAD) and punctuation restoration.
"""

import logging
import torch

from sbert_punc_case_ru import SbertPuncCase
from decode_onnx import Onnx_asr_model


class Processor:
    """
    Processor class

    This class handles the initialization and management of various components
    required for speech-to-text processing using CPU resources.
    """

    def __init__(self, device: str = "cpu") -> None:
        """
        Initialize the Processor class.

        Parameters
        ----------
        device : str, optional
            The device to use for processing ('cpu' for CPU). Default is 'cpu'.
        """
        self.device = device
        self.model = None
        self.vad = None
        self.get_speech_timestamps = None
        self.punctuation_model = None

    def load_model(self) -> None:
        """
        Load the ASR model, VAD model, and punctuation model.

        This method initializes the automatic speech recognition (ASR) model,
        voice activity detection (VAD) model, and punctuation restoration model,
        setting them up for use in audio processing.

        Raises
        ------
        RuntimeError
            If any model fails to load.
        """
        logging.info("INITIALIZING ASR MODEL")
        self.model = Onnx_asr_model("weights")
        logging.info("ASR MODEL INITIALIZED")

        logging.info("LOADING VAD")
        vad, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
            onnx=True
        )
        self.get_speech_timestamps = utils[0]
        self.vad = vad
        logging.info("VAD LOADED")

        logging.info("LOADING PUNCTUATION MODEL")
        punctuation_model = SbertPuncCase()
        punctuation_model.to(self.device)
        self.punctuation_model = punctuation_model
        logging.info("PUNCTUATION MODEL LOADED")

    def process_audio(self, audio):
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
        timestamps = self.get_speech_timestamps(
            audio,
            self.vad,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            max_speech_duration_s=16,
            window_size_samples=512
        )
        for timestamp in timestamps:
            start = timestamp["start"]
            end = timestamp["end"]
            speech = audio[start:end]
            orig_text = self.model.transcribe(speech)
            if orig_text:
                text = self.punctuation_model.punctuate(orig_text)
                yield text
