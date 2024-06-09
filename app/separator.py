import argparse
import copy
import json
import os
import pathlib
import shutil
#os.environ["TQDM_DISABLE"] = "1"

import tqdm
from audio_separator.separator import Separator

import logging

class MySeparator:
    def __init__(
        self,
        audio_ext=".mp3",
    ) -> None:
        self.audio_ext = audio_ext
        self.separator = Separator(
            log_level=logging.ERROR,
            output_single_stem="vocals",
            sample_rate=24000,
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

    def set_new_ext(self, audio_ext):
        self.audio_ext = audio_ext

    def process_dataset(self, dataset_dir):
        files = list(pathlib.Path(dataset_dir).rglob(f"*{self.audio_ext}"))
        files.sort()

        if len(files) > 0:
            with tqdm.tqdm(total=len(files), desc=f"file={files[0]}", disable=False) as pbar:
                for file in files:
                    pbar.update(1)
                    
                    if "/parts/" in file.as_posix():
                        continue

                    if "/clean/" in file.as_posix():
                        continue

                    if os.path.exists(
                        file.as_posix().replace(
                            self.audio_ext,
                            "_(Vocals)_UVR-MDX-NET-Inst_HQ_3.wav",
                        )
                    ):
                        continue

                    pbar.set_description(f"file={file}; | 0%")

                    output_file = self.separator.separate(file.as_posix())
                    file_dir = os.path.dirname(file.as_posix())
                    for _file in output_file:
                        if os.path.exists(_file):
                            shutil.move(_file, file_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="DATASET_YT")
    parser.add_argument("--audio_ext", type=str, default=".mp3")

    args, _ = parser.parse_known_args()
    processor = MySeparator(
        audio_ext=args.audio_ext,
    )
    processor.load_model()
    processor.process_dataset(args.data_dir)
