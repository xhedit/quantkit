# quantkit

A tool for downloading and converting HuggingFace models without drama.

# Install
```
pip3 install llm-quantkit
```
<br/>

# Usage

Download a model from HF and don't use HF cache:
```
quantkit teknium/Hermes-Trismegistus-Mistral-7B --no-cache -out 
```
<br/>


Only download the safetensors version of a model (useful for models that have both safetensors and pytorch):
```
quantkit mistralai/Mistral-7B-v0.1 --no-cache --safetensors-only -out mistral7b
```
<br/>


Download and convert a model to safetensors, deleting the original pytorch bins:
```
quantkit safetensor migtissera/Tess-10.7B-v1.5b --delete-original
```
<br/>


Download and convert a model to AWQ:
```
quantkit awq mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-AWQ
```
<br/>


Convert a model to GPTQ (4 bits / group-size 32):
```
quantkit awq Mistral-7B-v0.1 -out Mistral-7B-v0.1-AWQ -b 4 --group-size 32
```
<br/>


Convert a model to exllamav2:
```
quantkit exl2 mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-exl2-b8-h8 -b 8 -hb 8
```
<br/>


Still in beta.
