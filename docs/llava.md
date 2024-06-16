# Video captioning module

Модуль для генерации описаний к видео.

Для генерации видео используется модель LLaVA-Next, версия [LLaVA-NeXT-Video-7B-DPO](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO).

## Установка

1. Установите LLaVA-NeXT, следуя инструкции на их [официальном репозитории](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference).
2. Установите зависимости: ```pip install -r requirements.txt```

## Использование

Можно использовать кэпшенинг в коде, импортируйте класс `VideoCaptioner`:
```python
from llava import VideoCaptioner

captioner = VideoCaptioner("cuda")
print(captioner.get_caption("path/to/video.mp4"))
```

Чтобы запустить воркер, который подключиться к серверу и будет выполнять задания, нужно:
1. Настроить конфиг в файле [run_worker.py](../run_worker.py)
2. Запустить: `python run_worker.py`