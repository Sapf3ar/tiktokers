"""
Module for audio separation functionality.

This module contains the `MySeparator` class that uses a pre-trained model to separate
audio files into individual stems, focusing on vocals.

Classes:
--------
MySeparator:
    A class for audio music separation.
"""

from audio_separator.separator import Separator
import logging

class MySeparator:
    """
     A class for audio music separation.

    Attributes
    ----------
    audio_ext : str
        Default audio file extension for processing (default is ".wav").
    separator : Separator
        Instance of Separator for performing audio separation.

    Methods
    -------
    __init__(audio_ext=".wav")
        Initializes MySeparator with default parameters.
    load_model()
        Loads the pre-trained separation model.
    separate_audio(file_path: str) -> str
        Separates audio at `file_path` into individual stems and returns the path to the separated vocals.
    """

    def __init__(self, audio_ext=".wav") -> None:
        """
        Initializes MySeparator with default parameters.

        Parameters
        ----------
        audio_ext : str, optional
            Default audio file extension for processing (default is ".wav").
        """
        self.audio_ext = audio_ext
        self.separator = Separator(
            log_level=logging.ERROR,
            output_single_stem="vocals",
            sample_rate=48000,
            mdx_params={
                "hop_length": 1024,
                "segment_size": 256,
                "overlap": 0.25,
                "batch_size": 4,
                "enable_denoise": True,
            },
        )

    def load_model(self):
        """
        Loads the pre-trained separation model.
        """
        self.separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")

    def separate_audio(self, file_path: str) -> str:
        """
        Separates audio at `file_path` into individual stems and returns the path to the separated vocals.

        Parameters
        ----------
        file_path : str
            Path to the input audio file.

        Returns
        -------
        str
            Path to the separated vocals audio file.
        """
        output_file = self.separator.separate(file_path)
        return output_file[0]
