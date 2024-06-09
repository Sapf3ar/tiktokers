import os
import subprocess
import logging


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


def download_if_not_exist(path, url):
    if not os.path.isfile(path):
        logging.info(f"File {path} not found")
        logging.info(f"Download from {url}")
        process = subprocess.Popen(
            f"gdown --folder 'https://drive.google.com/drive/folders/1-Yl9KDIjuxaYlxsFJB8jO3htRbRU1CdH?usp=sharing'", shell=True, stdout=subprocess.PIPE)
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
    if not (os.path.exists("./weights/tokenizer_all_sets") and
                os.path.isfile("./weights/rnnt_model_weights.ckpt")):
            download_models()


if __name__ == "__main__":
    check_models_files()
    from server import start_server
    start_server()
