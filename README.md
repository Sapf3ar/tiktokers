1.  Video captioning (Тимур, Игорь)
	1.  llava-next / ...
2. STT (Егор)
	1. vosk 
3. OCR (Артем)
	1. чтобы побыстрее 
4.  деплой (Данил)
	1. cloud.ru
	2. faiss
	3. fast-api
5. реранк (мистер Х) 

discarded:
- микс видео эмбеддингов и текст эмбеддингов
- автокомплит кейвордами
- топик моделинг для сужения области поиска


## vllm serve
Note: 
```
LLaMA 3 8B requires around 16GB of disk space and 20GB of VRAM (GPU memory) in FP16. 
```

- load weights from huggingface. [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

- run vllm server:
```
python -m vllm.entrypoints.openai.api_server --model <PATH_TO_CKPT> --tensor-parallel-size <YOUR_GPUS> --enforce-eager
```

- check with:

```
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
		"model": "PATH_TO_CKPT",
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
