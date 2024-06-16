### vllm serve
Note: 
```
LLaMA 3 8B requires around 16GB of disk space and 20GB of VRAM (GPU memory) in FP16. 
```

- auth and load weights from huggingface, assign license agreement. [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

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
