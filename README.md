# quantkit

A tool for downloading and converting HuggingFace models without drama.

# Install
```
pip3 install llm-quantkit
```
<br/>

# Usage

```
Usage: quantkit [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download    Download model from huggingface.
  safetensor  Download and/or convert a pytorch model to safetensor format.
  awq         Download and/or convert a model to AWQ format.
  exl2        Download and/or convert a model to EXL2 format.
  gptq        Download and/or convert a model to GPTQ format.
```

Download a model from HF and don't use HF cache:
```
quantkit download teknium/Hermes-Trismegistus-Mistral-7B --no-cache
```
<br/>


Only download the safetensors version of a model (useful for models that have both safetensors and pytorch):
```
quantkit download mistralai/Mistral-7B-v0.1 --no-cache --safetensors-only -out mistral7b
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
quantkit gptq mistral7b -out Mistral-7B-v0.1-AWQ -b 4 --group-size 32
```
<br/>


Convert a model to exllamav2:
```
quantkit exl2 mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-exl2-b8-h8 -b 8 -hb 8
```
<br/>


Still in beta.
