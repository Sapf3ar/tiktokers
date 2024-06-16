import os
import subprocess
import logging

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)
import torchaudio


FILE_HANDLER = logging.FileHandler(filename="./out.log")
FILE_HANDLER.setFormatter(
    logging.Formatter("[%(asctime)s]\t[%(levelname)2s]:\t%(message)s")
)
FILE_HANDLER.setLevel(logging.INFO)

STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(
    logging.Formatter("[%(asctime)s]\t%(levelname)2s:\t%(message)s")
)
STREAM_HANDLER.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.INFO, handlers=[FILE_HANDLER, STREAM_HANDLER])


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


def download_if_not_exist(path, url):
    if not os.path.isfile(path):
        logging.info(f"File {path} not found")
        logging.info(f"Download from {url}")
        process = subprocess.Popen(
            "gdown --folder 'https://drive.google.com/drive/folders/1-Yl9KDIjuxaYlxsFJB8jO3htRbRU1CdH?usp=sharing'", shell=True, stdout=subprocess.PIPE)
        process.wait()
        if process.returncode != 0:
            logging.error("Download failed!")
            exit(1)


def download_models():
    download_if_not_exist(
        "weights/rnnt_model_weights.ckpt",
        "https://drive.google.com/drive/folders/1-Yl9KDIjuxaYlxsFJB8jO3htRbRU1CdH?usp=sharing"
        )


def check_models_files():
    is_tokenizer_exists = os.path.exists("./weights/tokenizer_all_sets")
    is_weights_exists = os.path.isfile("./weights/rnnt_model_weights.ckpt")
    if not (is_tokenizer_exists and is_weights_exists):
        download_models()


if __name__ == "__main__":
    check_models_files()
    from server import start_server
    start_server()
