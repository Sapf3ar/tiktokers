"""
STT Model Service Module

This module sets up a GRPC server for speech-to-text (STT) processing using either
CPU or GPU, with optional audio separation.
"""

import os
import time
import grpc
import scipy
import logging
import asyncio
import numpy as np
from typing import Dict, Any
from scipy.io.wavfile import read, write

from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerServicer, add_InferenceServerServicer_to_server

device = os.environ.get('device', 'cpu')
use_gpu = 'cuda' in device.lower()

use_separator = os.environ.get('use_separator', 'False').lower() in ('true', 'true')

if use_gpu:
    from stt_gpu_processor import Processor
else:
    from stt_cpu_processor import Processor

if use_separator:
    from separator import MySeparator

stt_runner = None
audio_separator = None

def start_server():
    """
    Initialize and start the STT model server.

    This function sets up the necessary models for STT processing, 
    including optional audio separation, and starts the GRPC server.
    """
    global stt_runner
    global audio_separator

    logging.info('Start stt model service on %s', device)
    try:
        if use_separator:
            audio_separator = MySeparator()
            audio_separator.load_model()
            logging.info('Separator model loaded')
        stt_runner = Processor(device=device)
        stt_runner.load_model()
        logging.info('STT runner models loaded')
        asyncio.run(serve())
    except Exception as e:
        logging.error(e)
        logging.shutdown()

def convert(audio, sr, new_sr=16000):
    """
    Convert the audio sample rate to the specified rate.

    Parameters
    ----------
    audio : np.ndarray
        The input audio array.
    sr : int
        The original sample rate of the audio.
    new_sr : int, optional
        The target sample rate (default is 16000).

    Returns
    -------
    np.ndarray
        The audio array resampled to the new sample rate.
    """
    number_of_samples = round(len(audio) * float(new_sr) / sr)
    audio = scipy.signal.resample(audio, number_of_samples)

    if len(audio.shape) > 1:
        audio = (audio[:, 0] + audio[:, 1]) / 2

    return audio

class InferenceService(InferenceServerServicer):
    """
    Inference Service Class

    This class implements the GRPC server service for handling
    inference requests for speech-to-text processing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.user_data: Dict[Any, Any] = {}
        self.use_separator = os.environ.get(
            'use_separator', 'False'
        ).lower() in ('true', 'true')

    async def inference(self, request: InferenceRequest, context) -> InferenceReply:
        """
        Handle an inference request.

        This method processes the audio input to perform STT and optionally
        separates the audio before processing.

        Parameters
        ----------
        request : InferenceRequest
            The request containing the audio data.
        context : grpc.ServicerContext
            The context for the GRPC call.

        Returns
        -------
        InferenceReply
            The reply containing the transcribed text.
        """
        logging.info('Received request')

        # Read PCM16 WAV format
        audio = np.frombuffer(request.audio, dtype='<i2')
        sr = 48000

        if self.use_separator:
            write('audio.wav', 48000, audio)
            start = time.time()
            separated_audio_path = audio_separator.separate_audio('audio.wav')
            logging.info('Separation done in %.3fs', time.time() - start)
            sr, audio = read(separated_audio_path)

        audio = audio.astype(np.float32) / 32768
        audio = convert(audio, sr, new_sr=16000)

        start = time.time()
        global_text = ' '.join([text for text in stt_runner.process_audio(audio)])
        logging.info('STT done in %.3fs', time.time() - start)
        logging.info('STT pred: %s', global_text)
        return InferenceReply(pred=global_text)

async def serve():
    """
    Start the GRPC server.

    This function configures and starts the GRPC server for handling
    inference requests.
    """
    server_options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)  # 50 MB
    ]
    server = grpc.aio.server(options=server_options)
    add_InferenceServerServicer_to_server(InferenceService(), server)
    address = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5021')}"
    logging.info('Address: %s', address)
    server.add_insecure_port(address)
    logging.info('Starting server on %s', address)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    start_server()
