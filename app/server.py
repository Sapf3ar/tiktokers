import os
import time
import grpc
import torch
import logging
import asyncio
import librosa
import numpy as np
from typing import Dict, Any
from scipy.io.wavfile import read, write

from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServer, add_InferenceServerServicer_to_server

from proccess_dataset import Processor
from separator import MySeparator


def start_server():
    global stt_runner
    global audio_separator
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Start stt model service on {device}")
    try:
        audio_separator = MySeparator()
        audio_separator.load_model()
        logging.info(f"Separator model loaded")
        stt_runner = Processor(skip_exist=True, device=device)
        stt_runner.load_model()
        logging.info(f"Stt runner models loaded")
        asyncio.run(serve())
    except Exception as e:
        logging.error(e)
        logging.shutdown()



class InferenceService(InferenceServer):
    def __init__(self) -> None:
        super().__init__()
        self.user_data: Dict[Any, Any] = {}

    async def inference(self,
                        request: InferenceRequest,
                        context) -> InferenceReply:
        logging.info("Received request")

        # read PCM16 wav format
        # '<i2' means little-endian signed 2-byte integer
        audio = np.frombuffer(request.audio, dtype='<i2')
        # convert to librosa "floating point time series"
        write("audio.wav", 48000, audio)

        start = time.time()
        audio_separator.separate_audio("audio.wav")
        logging.info(f"Separation done in {time.time() - start}s")

        sr, audio = read("audio.wav")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32) / 32768

        start = time.time()
        global_text = ".".join(text for text in stt_runner.process_audio(audio))
        logging.info(f"Stt done in {time.time() - start}s")

        return InferenceReply(pred=global_text)


async def serve():
    server = grpc.aio.server()
    add_InferenceServerServicer_to_server(InferenceService(), server)
    adddress = f"{os.environ['HOST']}:{os.environ['PORT']}"
    server.add_insecure_port(adddress)
    logging.info(f"Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    start_server()
