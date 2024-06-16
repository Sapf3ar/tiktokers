# Документация OmniSearch

---
Проект предоставляет решение для извлечения и суммаризации ключевой информации из видео, используя мультимодальные данные: распознанную речь (ASR), визуально распознаваний текст (OCR) и описание видео (captioning). Решение позволяет индексировать видео и производить поиск релеватного контента по пользовательским запросам.

---

## Бизнес-логика решения

**Основные сценарии использования:**

1. **Загрузка и обработка видео:** Сервис принимает на вход видео извлекает информацию с помощью OCR, ASR, и Captioning микросервисов. Затем, с помощью LLM данные суммаризируются в емкое текстовое представление видео.
   
2. **Индексация:** Суммаризованное видео кодируется и вносится в векторную базу данных.

3. **Векторный поиск:** Пользовательский ввод кодируется и служит запросом в векторную базу данных.

---

## Описание технической реализации

**Логика бэкенда:**

- **API:** Использует FastAPI для создания веб-сервиса. Основные эндпоинты включают добавление задач на обработку видео, получение статуса задач, и выполнение поиска по суммаризациям.

- **STT (Speech-to-Text):** Модуль `stt.py` отвечает за преобразование речи в текст.
  - `convert(path, new_rate) -> np.ndarray`: Конвертирует аудиофайл в формат, подходящий для обработки.
  - `stt_request(audio: np.ndarray, server_: str) -> str`: Выполняет запрос к серверу для распознавания речи.
  - `process_video_stt(local_path_video: str) -> str`: Извлекает аудиотрек из видео и обрабатывает его для получения текста.

- **OCR (Optical Character Recognition):** Модуль `text_extractor.py` извлекает текст из видео.
  - `extract_text(video_link: str, sample_rate: float, confidence_threshold: float, max_image_size: int, max_frames: int, batch_size: int) -> list[str]`: Извлекает текст из видео, используя EasyOCR.

- **Суммаризация:** Модуль `api.py` объединяет данные из OCR, ASR и описания видео для создания суммаризации с помощью вызова языковой модели.
  - `summary_modalities(asr, ocr, caption, call_llm, model_path) -> str`: Создает суммаризацию на основе текста из ASR, OCR и описания видео, вызывая языковую модель.
  - `call_vllm_api(text, model, url, max_tokens, temperature) -> str`: Вызывает API языковой модели для генерации текста на основе переданного запроса.

- **Обработка видео:** В модуле `video_capture.py` реализована функциональность для обработки и извлечения кадров из видео.
  - `video_capture(filepath)`: Контекстный менеджер для захвата видео с помощью OpenCV.

**Выборка для обучения:** В решении используется предварительно индексированная база данных из 13 тысяч репрезентативных представителей кластеров.

---

## Описание выбранного стэка технологий

**Технологический стек:**

- **Язык программирования:** Python.
- **Веб-фреймворк:** FastAPI для создания RESTful API.
- **Библиотеки для обработки данных:** `pandas`, `faiss`, `numpy`.
- **Модель для суммаризации:** `SentenceTransformer` для создания векторных представлений.
- **Клиент API:** `requests` для взаимодействия с языковой моделью.
- **Дополнительные:** `uvicorn` для запуска FastAPI, `logging` для логирования, `dataclasses` для упрощения работы с данными, `librosa` для обработки аудио, `scipy` для ресемплинга аудио, `grpc` для взаимодействия с сервером распознавания речи.

**Причины выбора:**

- **Python** - гибкий язык с богатой экосистемой библиотек, подходящий для задач машинного обучения и разработки API.
- **FastAPI** - высокопроизводительный веб-фреймворк с простой интеграцией и поддержкой асинхронного программирования.
- **SentenceTransformer** - мощный инструмент для создания векторных представлений текстов, подходящий для многомодальной обработки и поиска.
- **librosa** и **scipy** - инструменты для обработки аудио, обеспечивающие гибкость и точность в ресемплинге и анализе звуковых данных.

---

## Инструкция по установке и развертыванию Backend
### How to start
```
cd backend
fastapi run --host 0.0.0.0 --port 443 main.py
```

## Инструкция по установке и развертыванию Speechrecognition backend
### How to start
docker build on gpu
```
cd speechrecognition/docker
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml build
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml up -d speechrec
```
docker build on cpu
```
cd speechrecognition/docker
docker compose -p <unique-container-name> -f docker-compose.yml build
docker compose -p <unique-container-name> -f docker-compose.yml up -d speechrec
```
check logs
```
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml logs --follow
```

### Environment variables
In docker-compose.yml:
- use_separator - including/excluding music separation from audio (True or False)
- HOST - host address of docker container
- PORT - port of docker container
in gpu.docker-compose.yml:
- device - inference device (cuda or cuda:0 or cpu)


### How to send requests
```
import sys
sys.path.append("speechrecognition/app")
from inference_pb2 import InferenceRequest, InferenceReply
from inference_pb2_grpc import InferenceServerStub
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
                      server_ : str = "172.21.0.88:5022",
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

## Инструкция по установке и использованию Video captioning module

Модуль для генерации описаний к видео.

Для генерации видео используется модель LLaVA-Next, версия [LLaVA-NeXT-Video-7B-DPO](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO).

### Установка

1. Установите LLaVA-NeXT, следуя инструкции на их [официальном репозитории](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference).
2. Установите зависимости: ```pip install -r requirements.txt```

### Использование

Можно использовать кэпшенинг в коде, импортируйте класс `VideoCaptioner`:
```python
from llava import VideoCaptioner

captioner = VideoCaptioner("cuda")
print(captioner.get_caption("path/to/video.mp4"))
```

Чтобы запустить воркер, который подключиться к серверу и будет выполнять задания, нужно:
1. Настроить конфиг в файле [run_worker.py](../run_worker.py)
2. Запустить: `python run_worker.py`

## Инструкция по установке и использованию vllm serve
Требуемые ресурсы: 
```
LLaMA 3 8B requires around 16GB of disk space and 20GB of VRAM (GPU memory) in FP16. 
```
- Установка зависимостей

```
pip install -r requirements.txt
```

- Зарегистрироваться на HuggingFace и подписать лицензионное соглашние, загрузить модель. [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

```
pip install -U "huggingface_hub[cli]"

huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir <PATH_TO_CKPT>
```

- Запустить vLLM сервер:

```
python -m vllm.entrypoints.openai.api_server --model <PATH_TO_CKPT> --tensor-parallel-size <YOUR_GPUS> --enforce-eager
```

- Проверить работоспособность с помощью:

```
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
		"model": <PATH_TO_CKPT>,
		"messages": [{
		"role": "user",
		"content": "Say hello"}],
		"max_tokens": 5000,
		"temperature": 0
		}

response = requests.post(url, headers=headers, json=data)
text = response.json()["choices"][0]["message"]["content"]
print(text)
```


## Инструкция по установке и использованию OCR module

### Getting started

Install dependencies:

`pip install -r -requirements.txt`

To get recognized text from the video:

```python
from text_extractor import TextExtractor

text_extractor = TextExtractor()
video_link = <link_to_the_video>
texts = text_extractor.extract_text(video_link)

unique_texts = list(dict.fromkeys(texts))
video_text = " ".join(unique_texts)
```

Parameters of `extract_text` method:
- **video_link**: link to the video to get the text from
- **sample_rate**: sampling rate when receiving a frame from a video
- **confidence_threshold**: the threshold for filtering texts in which the model is not confident enough
- **max_image_size**: the maximum image size of the larger side to resize the image
- **max_frames**: the maximum number of frames captured from a video
- **batch_size**: batch size for OCR model
