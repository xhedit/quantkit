# quantkit

A tool for downloading and converting HuggingFace models without drama.

<br/>

# Install
If you're on a machine with CUDA and want AWQ/GPTQ support:
```
pip3 install llm-quantkit[cuda]
```
<br/>

Otherwise, the default install works. <br/>

```
pip3 install llm-quantkit
```
<br/>

# Requirements

If you need a device specific torch, install it first. <br/>

This project depends on torch, awq, exl2, gptq, and hqq libraries. <br/>
Some of these dependencies do not support Python 3.12 yet. <br/>
Supported Pythons: 3.8, 3.9, 3.10, and 3.11<br/>


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
  gguf        Download and/or convert a model to GGUF format.
  gptq        Download and/or convert a model to GPTQ format.
  hqq         Download and/or convert a model to HQQ format.
```

The first argument after command should be an HF repo id (mistralai/Mistral-7B-v0.1) or a local directory with model files in it already.

The download command defaults to downloading into the HF cache and producing symlinks in the output dir, but there is a --no-cache option which places the model files in the output directory. <br/>


AWQ defaults to 4 bits, group size 128, zero-point True. <br />
GPTQ defaults are 4 bits, group size 128, activation-order False. <br />
EXL2 defaults to 8 head bits but there is no default bitrate. <br />
GGUF defaults to no imatrix but there is no default quant-type. <br />
HQQ defaults to 4 bits, group size 64, zero_point=True. <br />

<br/>

# Examples

Download a model from HF and don't use HF cache:
```
quantkit download teknium/Hermes-Trismegistus-Mistral-7B --no-cache
```

<br/>


Only download the safetensors version of a model (useful for models that have torch and safetensor):
```
quantkit download mistralai/Mistral-7B-v0.1 --no-cache --safetensors-only -out mistral7b
```

<br/>


Download and convert a model to safetensor, deleting the original pytorch bins:
```
quantkit safetensor migtissera/Tess-10.7B-v1.5b --delete-original
```

<br/>


Download and convert a model to GGUF (Q5_K):
```
quantkit gguf TinyLlama/TinyLlama-1.1B-Chat-v1.0 -out TinyLlama-1.1B-Q5_K.gguf Q5_K
```

<br/>


Download and convert a model to GGUF using an imatrix, offloading 200 layers:
```
quantkit gguf TinyLlama/TinyLlama-1.1B-Chat-v1.0 -out TinyLlama-1.1B-IQ4_XS.gguf IQ4_XS --built-in-imatrix -ngl 200
```

<br/>


Download and convert a model to AWQ:
```
quantkit awq mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-AWQ
```

<br/>


Convert a model to GPTQ (4 bits / group-size 32):
```
quantkit gptq mistral7b -out Mistral-7B-v0.1-GPTQ -b 4 --group-size 32
```

<br/>


Convert a model to exllamav2:
```
quantkit exl2 mistralai/Mistral-7B-v0.1 -out Mistral-7B-v0.1-exl2-b8-h8 -b 8 -hb 8
```
<br/>


Convert a model to HQQ:
```
quantkit hqq mistralai/Mistral-7B-v0.1 -out Mistral-7B-HQQ-w4-gs64
```
<br/>

# Hardware Requirements
Here's what has worked for me in testing. Drop a PR or Issue with updates for what is possible on various size cards. <br />
GGUF conversion doesn't need a GPU except for iMatrix and Exllamav2 requires that the largest layer fits on single GPU.

|Model Size|Quant|VRAM|Successful|
|--|--|--|--|
|7B|AWQ|24GB|✅|
|7B|EXL2|24GB|✅|
|7B|GGUF|24GB|✅|
|7B|GPTQ|24GB|✅|
|7B|HQQ|24GB|✅|
|13B|AWQ|24GB|✅|
|13B|EXL2|24GB|✅|
|13B|GGUF|24GB|✅|
|13B|GPTQ|24GB|:x:|
|13B|HQQ|24GB|?|
|34B|AWQ|24GB|:x:|
|34B|EXL2|24GB|✅|
|34B|GGUF|24GB|✅|
|34B|GPTQ|24GB|:x:|
|34B|HQQ|24GB|?|
|70B|AWQ|24GB|:x:|
|70B|EXL2|24GB|✅|
|70B|GGUF|24GB|✅|
|70B|GPTQ|24GB|:x:|
|70B|HQQ|24GB|?|

<br />

# Notes

Still in beta. Llama.cpp offloading is probably not going to work on your platform unless you uninstall llama-cpp-conv and reinstall it with the proper build flags. Look at the llama-cpp-python documentation and follow the revelant command but replace llama-cpp-python with llama-cpp-conv.
