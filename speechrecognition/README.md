# Speechrecognition backend

## How to start
docker building
```
cd docker
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml build
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml up -d speechrec
```
check logs
```
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml logs --follow
```

## How to send requests
```
from app.inference_pb2 import InferenceRequest, InferenceReply
from app.inference_pb2_grpc import InferenceServerStub
import numpy as np
import scipy

def convert(path, new_rate=48000) -> None:
    sr, audio = scipy.io.wavfile.read(path, mmap=False)

    # Resample data
    number_of_samples = round(len(audio) * float(new_rate) / sr)
    audio = scipy.signal.resample(audio, number_of_samples)

    if len(audio.shape) > 1:
        audio = (audio[:,0] + audio[:,1]) / 2

    if audio.max() <= 1.0:
        audio = (audio * 32767).astype(np.int16)

    scipy.io.wavfile.write(path, 48000, audio.astype(np.int16))


async def stt_request(path_to_audio : str,
                      server_ : str = "172.21.0.88:5021",
                      ) -> str:
    async with grpc.aio.insecure_channel(server_) as channel:
        stub = InferenceServerStub(channel)

        with open(path_to_audio, "rb") as f:
            audio = f.read()
            res: InferenceReply = await stub.inference(
                    InferenceRequest(audio=audio)
                )
            return res.pred


if __name__ == "__main__":
	path = "/path/to/audio.wav"
	convert(path)
	print(await stt_request(path))
```