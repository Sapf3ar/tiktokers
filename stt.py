import sys
sys.path.append('app')
from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerStub
import scipy
import asyncio
import numpy as np 
import grpc
import librosa


def convert(path, new_rate=48000) -> np.ndarray:
    audio, sr = librosa.load(path)

    # Resample data
    number_of_samples = round(len(audio) * float(new_rate) / sr)
    audio = scipy.signal.resample(audio, number_of_samples)

    if len(audio.shape) > 1:
        audio = (audio[:,0] + audio[:,1]) / 2

    audio = (audio * 32767).astype(np.int16)

    return audio


async def stt_request(audio : np.ndarray,
                    server_ : str = "195.242.24.229:5022",
                      ) -> str:
    async with grpc.aio.insecure_channel(server_) as channel:
        stub = InferenceServerStub(channel)

        res: InferenceReply = await stub.inference(
                InferenceRequest(audio=bytes(audio))
            )
        print(res.pred)
        return res.pred

async def process_video_stt(local_path_video:str) -> str:

  audio = convert(local_path_video)
  return await stt_request(audio)

if __name__ == "__main__":
  path = "test_clt.mp4"
  audio = convert(path)
  asyncio.run(stt_request(audio))

