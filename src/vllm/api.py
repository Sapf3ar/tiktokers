import requests


def summary_modalities(asr: str, ocr: str, caption: str, call_llm: callable) -> str:
    """
    Summarizes information from multiple modalities (speech, text, and video caption).

    Args:
        asr (str): Automatic Speech Recognition (ASR) transcript.
        ocr (str): Optical Character Recognition (OCR) text.
        caption (str): Video caption or description.
        call_llm (callable): Function to call the large language model (LLM).

    Returns:
        str: A concise summary.
    """
    fusion_prompt = '''Контекст и цель:
    Пожалуйста, объедините информацию из нескольких модальностей (речь, текст и описание видео) для создания краткой и точной суммаризации. Цель – получить компактную репрезентацию видео для векторного поиска.

    Модальности:

    Речь (ASR): {asr}
    Текст (OCR): {ocr}
    Описание видео (Captioning): {caption}

    Инструкции:

    - Проанализируйте данные из всех доступных модальностей. Если какая-либо модальность отсутствует (пустая) или содержит значительные ошибки, игнорируйте её при создании суммаризации.
    - Извлеките основные и наиболее значимые аспекты из каждой доступной модальности.
    - Объедините их в одну краткую и содержательную суммаризацию.

    Формат ответа:
    Создайте короткую суммаризацию в одном предложении, которая передает основное содержание видео. Избегайте излишних деталей и фокусируйтесь на ключевых аспектах.
    Данное описание должно послужить оптимальной репрезентацией и позволить представить видео в виде текста таких образом, чтобы пользователь смог найти его при запросе в свободной форме.
    Если ты считаешь, что содержания из ASR и OCR частей достаточно, чтобы отразить суть видео - не добавляй излишних визуальных описаний о видео.

    Не начинайте с фразы "в видео говорится". Начинай сразу с сути. Представь, что ты автор видео и хочешь наиболее информативно его назвать.
    Возвращайте только ответ на русском языке и ничего более:
    "'''
    query = fusion_prompt.format(asr=asr, ocr=ocr, caption=caption)
    return call_llm(query)


def call_vllm_api(
    text: str,
    model: str,
    url: str = "http://localhost:8000/v1/chat/completions",
    max_tokens: int = 1024,
    temperature: float = 0.05,
) -> str:
    """
    Calls a language model API to get a completion for the given text.

    Args:
        text (str): The prompt text for the LLM.
        model (str): The model name or identifier.
        url (str, optional): The API endpoint URL. Defaults to "http://localhost:8000/v1/chat/completions".
        max_tokens (int, optional): Maximum number of tokens in the response. Defaults to 1024.
        temperature (float, optional): Sampling temperature. Defaults to 0.05.

    Returns:
        str: The text generated by the LLM.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, json=data)
    text = response.json()["choices"][0]["message"]["content"]
    return text
