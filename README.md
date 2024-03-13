# quantkit

A tool for downloading and converting HuggingFace models without drama.

# Install
```
pip3 install llm-quantkit
```

Download a model from HF and don't use HF cache:
```
quantkit mistralai/Mistral-7B-v0.1 --no-cache
```

Download and convert a model to AWQ:
```
quantkit awq mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-AWQ
```


Convert a model to GPTQ (4 bits / group-size 32):
```
quantkit awq Mistral-7B-v0.1 -out Mistral-7B-v0.1-AWQ -b 4 --group-size 32
```


Convert a model to exllamav2:
```
quantkit exl2 mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-exl2-b8-h8 -b 8 -hb 8
```

Still in beta.
