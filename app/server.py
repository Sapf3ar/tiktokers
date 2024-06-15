import os
import time
import grpc
import torch
import scipy
import logging
import asyncio
import librosa
import numpy as np
from typing import Dict, Any
from scipy.io.wavfile import read, write

from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerServicer, add_InferenceServerServicer_to_server

device = os.environ.get(
            'device',
            'cpu'
        )
use_gpu = 'cuda' in device.lower()

use_separator = os.environ.get(
            'use_separator',
            'False'
        )
use_separator = use_separator.lower() in ('True', 'true')

if use_gpu:
    from stt_gpu_processor import Processor
else:
    from stt_cpu_processor import Processor

if use_separator:
    from separator import MySeparator


def start_server():
    global stt_runner
    global audio_separator

    logging.info(f"Start stt model service on {device}")
    try:
        if use_separator:
            audio_separator = MySeparator()
            audio_separator.load_model()
            logging.info("Separator model loaded")
        stt_runner = Processor(device=device)
        stt_runner.load_model()
        logging.info("STT runner models loaded")
        asyncio.run(serve())
    except Exception as e:
        logging.error(e)
        logging.shutdown()


def convert(audio, sr, new_sr=16000):
    number_of_samples = round(len(audio) * float(new_sr) / sr)
    audio = scipy.signal.resample(audio, number_of_samples)

    if len(audio.shape) > 1:
        audio = (audio[:,0] + audio[:,1]) / 2

    return audio

class InferenceService(InferenceServerServicer):
    def __init__(self) -> None:
        super().__init__()
        self.user_data: Dict[Any, Any] = {}
        use_separator = os.environ.get(
            'use_separator',
            'False'
        )
        self.use_separator = use_separator.lower() in ('True', 'true')

    async def inference(self, request: InferenceRequest, context) -> InferenceReply:
        logging.info("Received request")

        # Read PCM16 WAV format
        audio = np.frombuffer(request.audio, dtype='<i2')
        sr = 48000

        if self.use_separator:
            write("audio.wav", 48000, audio)
            start = time.time()
            separated_audio_path = audio_separator.separate_audio("./audio.wav")
            logging.info(f"Separation done in {time.time() - start:.3f}s")
            sr, audio = read(separated_audio_path)

        audio = audio.astype(np.float32) / 32768
        audio = convert(audio, sr, new_sr=16000)

        start = time.time()
        global_text = " ".join([text for text in stt_runner.process_audio(audio)])
        logging.info(f"STT done in {time.time() - start:.3f}s")
        logging.info(f"STT pred: {global_text}")
        return InferenceReply(pred=global_text)


async def serve():
    server_options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50 MB
    ]
    server = grpc.aio.server(options=server_options)
    add_InferenceServerServicer_to_server(InferenceService(), server)
    address = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5021')}"
    logging.info(f"Addres: {address}")
    server.add_insecure_port(address)
    logging.info(f"Starting server on {address}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    start_server()
