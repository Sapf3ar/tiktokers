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
  - `extract_text(video_link: str, sample_rate: float, confidence_threshold: float, max_image_size: int, max_frames: int, batch_size: int) -> str`: Извлекает текст из видео, используя EasyOCR.

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

## 1. Инструкция по установке и развертыванию Backend
```
cd backend
fastapi run --host 0.0.0.0 --port 443 main.py
```

## 2. Инструкция по развертыванию Speechrecognition backend
### Сборка докер-контейнера на gpu
```
cd speechrecognition/docker
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml build
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml up -d speechrec
```
### Сборка докер-контейнера на cpu
```
cd speechrecognition/docker
docker compose -p <unique-container-name> -f docker-compose.yml build
docker compose -p <unique-container-name> -f docker-compose.yml up -d speechrec
```
### Проверка логов
```
docker compose -p <unique-container-name> -f docker-compose.yml -f gpu.docker-compose.yml logs --follow
```

### Переменные окружения
В docker-compose.yml:
- use_separator - включение/выключение фильтрации музыки в аудио (True или False)
- HOST - хост адрес докер-контейнера
- PORT - порт докер-контейнера
- device - устройство инференса моделей (cuda или cuda:0 или cpu)


### Инструкция отправки запросов к сервису
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

## 3. Инструкция по установке и использованию Video captioning module

Модуль для генерации описаний к видео.

Для генерации видео используется модель LLaVA-Next, версия [LLaVA-NeXT-Video-7B-DPO](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO).

### Установка

1. Установите LLaVA-NeXT, следуя инструкции на их [официальном репозитории](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference).
2. Установите зависимости: ```pip install -r llava/requirements.txt```

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

## 4. Инструкция по установке и использованию vllm serve
Требуемые ресурсы: 
```
LLaMA 3 8B requires around 16GB of disk space and 20GB of VRAM (GPU memory) in FP16. 
```
- Установка зависимостей

```
pip install -r llm/requirements.txt
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


## 5. Инструкция по установке и использованию OCR module

### Начало работы

Установите зависимости:

`pip install -r backend/requirements.txt`

Для получения текста из видео выполните код ниже:

```python
from text_extractor import TextExtractor

text_extractor = TextExtractor()
video_link = <link_to_the_video>
video_text = text_extractor.extract_text(video_link)
```

Параметры метода `extract_text`:
- **video_link**: путь до ссылки на видео, с которого будет распознан текст
- **sample_rate**: частота дискретизации при получении кадров из видео
- **confidence_threshold**: порог для фильтрации текстов, в предсказании которых модель недостаточно уверена
- **max_image_size**: максимальный размер большей стороны изображения
- **max_frames**: максимальное количество кадров, взятых из видео
- **batch_size**: batch size для модели OCR.

## 6. Серверная часть


Добавьте задачу для обработки видео:
```
curl -X POST "http://localhost:8000/add_task/{link}" -d ""
```

Проверка статуса задачи
Проверьте статус задачи по её идентификатору:

```
curl -X GET "http://localhost:8000/get_task_status?task_id={task_id}"
```

Получение последней задачи
Получите последнюю добавленную задачу:

```
curl -X GET "http://localhost:8000/get_task"
```

Поиск по суммаризациям
Ищите видео по суммаризации:

```
curl -X POST "http://localhost:8000/search" -H "Content-Type: application/json" -d '{"query": "search term"}'
```

Модель данных
Task
id_ (int): Уникальный идентификатор задачи
execution_time (int): Время выполнения задачи
video_link (str): Ссылка на видео
caption (str): Описание видео
error (str): Ошибка, если она произошла


IndexTask
task_id (int): Идентификатор задачи
link (str): Ссылка на видео


Query
query (str): Поисковый запрос


Item
id (str): Идентификатор элемента
name (str): Имя элемента


Swagger API Documentation:


```
openapi: 3.0.2
info:
  title: Video Insight Summarizer API
  version: 1.0.0
paths:
  /push_last_task:
    post:
      summary: Add the last task for processing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Task'
      responses:
        '200':
          description: Task done
          content:
            application/json:
              schema:
                type: string
                example: Task done
        '400':
          description: Task failed with error
          content:
            application/json:
              schema:
                type: string
                example: Task {id} exited with {error}

  /get_task_status:
    get:
      summary: Get status of a task by its ID
      parameters:
        - in: query
          name: task_id
          schema:
            type: integer
      responses:
        '200':
          description: Task status
          content:
            application/json:
              schema:
                type: string
                example: Task finished with {status}
        '400':
          description: Task not found
          content:
            application/json:
              schema:
                type: string
                example: Task not found

  /get_task:
    get:
      summary: Get the last added task
      responses:
        '200':
          description: Last task
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/IndexTask'
        '400':
          description: No tasks found
          content:
            application/json:
              schema:
                type: string
                example: No tasks found

  /add_task/{link}:
    post:
      summary: Add a task to process video from the link
      parameters:
        - in: path
          name: link
          schema:
            type: string
      responses:
        '200':
          description: Task added
          content:
            application/json:
              schema:
                type: string
                example: Task {TASK_ID} added
        '400':
          description: Error adding task
          content:
            application/json:
              schema:
                type: string
                example: Error adding task

  /search:
    post:
      summary: Perform a search over video summaries
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Query'
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: object
                additionalProperties:
                  type: string
        '400':
          description: Search failed
          content:
            application/json:
              schema:
                type: string
                example: Search failed

components:
  schemas:
    Task:
      type: object
      properties:
        id_:
          type: integer
        execution_time:
          type: integer
        video_link:
          type: string
        caption:
          type: string
        error:
          type: string
    IndexTask:
      type: object
      properties:
        task_id:
          type: integer
        link:
          type: string
    Query:
      type: object
      properties:
        query:
          type: string
    Item:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
```