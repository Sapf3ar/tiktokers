import argparse
import copy
import json
import os
import pathlib
import shutil

import tqdm
from audio_separator.separator import Separator

import logging

class MySeparator:
    def __init__(
        self,
        audio_ext=".wav",
    ) -> None:
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
        self.separator.load_model(model_filename="UVR-MDX-NET-Inst_HQ_3.onnx")

    def separate_audio(self, file_path):
        output_file = self.separator.separate(file_path.as_posix())
        file_dir = os.path.dirname(file_path.as_posix())
        for _file in output_file:
            if os.path.exists(_file):
                shutil.move(_file, file_dir)
