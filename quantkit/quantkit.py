# quantkit can download a hf model, convert a model to safetensors, and quantize
# supports: AWQ, GPTQ, and EXL2

import gc
import os
import site
import time
import datetime

from pathlib import Path
from quantkit.safetensor import convert_multi
from quantkit.convert import do_gguf_conversion

def run_download(model, output, hf_cache, force_download, resume_download, safetensors_only):
    if output is None:
        model_dir = model.split("/")[1]
        path = Path(model_dir)
    else:
        path = Path(output)

    from huggingface_hub import snapshot_download
    snapshot_download(model, local_dir=path, local_dir_use_symlinks=hf_cache, force_download=force_download, resume_download=resume_download, ignore_patterns='pytorch_model*' if safetensors_only else None)

def run_safetensor(model, delete_original):
    path = Path(model)
    if path.is_dir():
        if Path(path / "config.json").is_file():
            # convert
            convert_multi(model, del_pytorch_model=delete_original)
        else:
            click.echo("no config.json found in model dir")
    else:
        model_dir = model.split("/")[1]
        path = Path(model_dir)

        from huggingface_hub import snapshot_download
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=False, resume_download=True)
        convert_multi(model_dir, del_pytorch_model=delete_original)

def run_gguf(model, quant_type, output, keep, f32, built_in_imatrix, imatrix, cal_file, n_gpu_layers):
    if cal_file is not None:
        if not Path(cal_file).is_file():
            print(f"quantkit: could not load {cal_file}")
            return

    path = Path(model)
    if path.is_dir():
        if Path(path / "config.json").is_file():
            model_dir = path
        else:
            click.echo("no config.json found in model dir")
            return
    else:
        model_dir = model.split("/")[1]
        path = Path(model_dir)

        from huggingface_hub import snapshot_download
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=True, resume_download=True)

    do_step_two = False

    #if lower(quant_type) not in ["F32", "F16", "Q8_0"]:
    if quant_type.lower() not in [x.lower() for x in ["F32", "F16", "Q8_0"]]:
        # two step
        if quant_type.lower() not in [x.lower() for x in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "IQ3_XS",
                                                          "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
                                                          "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
                                                          "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ2_M", "IQ3_M"] ]:
            raise ValueError("quant_type must be a valid gguf quant type")
        step_two = quant_type
        do_step_two = True
        quant_type = "F32" if f32 else "F16"

    # fix vocab here

    if output is None:
        output = model_dir + "_" + (step_two.upper() or quant_type.upper()) + ".gguf"

    #def do_gguf_conversion(model: str, output: str, out_type: str, vocab_dir: str, vocab_type: str, context: int, pad_vocab: bool, concurrency: bool, big_endian: bool) -> None:
    if do_step_two:
        do_gguf_conversion(Path(model_dir), "tmp.gguf", quant_type.lower(), None, "spm,hfft", -1, False, False, False)

        if built_in_imatrix or (imatrix is None and cal_file is not None):
            imatrix = run_imatrix(cal_file, n_gpu_layers)

        quantize("tmp.gguf", output, step_two, imatrix)
    else:
        do_gguf_conversion(Path(model_dir), output, quant_type.lower(), None, "spm,hfft", -1, False, False, False)

    if not keep:
        os.remove("tmp.gguf")
        print("deleted tmp.gguf")
    print(f"Finished with {output}")

def run_imatrix(cal_file, n_gpu_layers):
    import site
    import shlex
    import platform
    import subprocess

    binary_ext = ".exe" if platform.system() is "Windows" else ""

    site_dir = site.getusersitepackages()
    imatrix = Path(site_dir) / "bin" / ("imatrix" + binary_ext)

    if not imatrix.is_file():
        for d in site.getsitepackages():
            p = Path(d)
            if(p / "bin" / ("imatrix" + binary_ext)).is_file():
                site_dir = p
                quantize = p / "bin" / ("imatrix" + binary_ext)

    print(f"Attempting to execute {imatrix}")

    if cal_file is None:
        download_wikitrain()
        cal_file = "wiki.train.raw"

    if n_gpu_layers is None or n_gpu_layers < 0:
        n_gpu_layers = 0

    imatrix_args = shlex.split(f"{imatrix} -m tmp.gguf -f {cal_file} -o imatrix.dat -ofreq 128 -ngl {n_gpu_layers}")

    p = subprocess.Popen(imatrix_args, stdin=None, stdout=None)
    p.wait()

def quantize(gguf_file, output, quant_type, imatrix):
    if quant_type.lower() not in [x.lower() for x in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "IQ3_XS",
                                                      "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
                                                      "Q2_K_S", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M",
                                                      "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ2_M", "IQ3_M"] ]:
        raise ValueError("quant_type must be a valid gguf quant type")

    import site
    import shlex
    import platform
    import subprocess

    binary_ext = ".exe" if platform.system() is "Windows" else ""

    site_dir = site.getusersitepackages()
    quantize = Path(site_dir) / "bin" / ("quantize" + binary_ext)

    if not quantize.is_file():
        for d in site.getsitepackages():
            p = Path(d)
            if(p / "bin" / ("quantize" + binary_ext)).is_file():
                site_dir = p
                quantize = p / "bin" / ("quantize" + binary_ext)

    print(f"Attempting to execute {quantize}")

    if imatrix is None:
        quantize_args = shlex.split(f"{quantize} {gguf_file} {output} {quant_type.upper()}")
    else:
        quantize_args = shlex.split(f"{quantize} --imatrix {imatrix} {gguf_file} {output} {quant_type.upper()}")

    p = subprocess.Popen(quantize_args, stdin=None, stdout=None)
    p.wait()

def run_awq(model, output, hf_cache, bits, group_size, zero_point, gemm):
    path = Path(model)
    if path.is_dir():
        if Path(path / "config.json").is_file():
            model_dir = path
        else:
            click.echo("no config.json found in model dir")
            return
    else:
        model_dir = model.split("/")[1]
        path = Path(model_dir)

        from huggingface_hub import snapshot_download
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=True, resume_download=True)

    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = model_dir

    if output is None:
        output = f"{model_path}-awq-w{bits}-gs{group_size}"

    quant_path = output
    quant_config = { "zero_point": zero_point, "q_group_size": group_size, "w_bit": bits, "version": "GEMM" }
    if not gemm:
        quant_config["version"] = "GEMV"

    dt_start = datetime.datetime.now()
    print(f"Starting awq quantization for {quant_path} at {str(dt_start)}")
    start = time.perf_counter()

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    # free GPU mem
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    end = time.perf_counter()
    dt_pq = datetime.datetime.now()
    print(f"Quantization complete ({str(end-start)} seconds) at {str(dt_pq)}")

def run_gptq(model, output, hf_cache, bits, group_size, damp, sym, true_seq, act_order):
    path = Path(model)
    if path.is_dir():
        if Path(path / "config.json").is_file():
            model_dir = path
        else:
            click.echo("no config.json found in model dir")
            return
    else:
        model_dir = model.split("/")[1]
        path = Path(model_dir)

        from huggingface_hub import snapshot_download
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=True, resume_download=True)

    import torch
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    quant_options = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=act_order,
        sym=sym,
        true_sequential=true_seq
    )

    model_path = model_dir

    if output is None:
        output = f"{model_dir}-gptq-w{bits}-gs{group_size}{'-ao' if act_order else ''}"

    quant_path = output

    # gptq needs post-training, use wikitext2
    traindataset, testenc = get_wikitext2(128, 0, 4096, model_path)

    dt_start = datetime.datetime.now()
    print(f"Starting gptq quantization for {quant_path} at {str(dt_start)}")
    start = time.perf_counter()

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quant_options, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    model.quantize(traindataset, use_triton=False, batch_size=1, cache_examples_on_gpu=False)

    model.save_quantized(quant_path, use_safetensors=True)
    tokenizer.save_pretrained(quant_path)

    # free GPU mem
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    end = time.perf_counter()
    dt_pq = datetime.datetime.now()
    print(f"Quantization complete ({str(end-start)} seconds) at {str(dt_pq)}")

def run_exl2(model, output, hf_cache, bits, head_bits, rope_alpha, rope_scale, only_measurement, no_resume):
    path = Path(model)
    if path.is_dir():
        if Path(path / "config.json").is_file():
            model_dir = str(path)
        else:
            click.echo("no config.json found in model dir")
            return
    else:
        model_dir = model.split("/")[1]
        path = Path(model_dir)

        from huggingface_hub import snapshot_download
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=True, resume_download=True)

        if not Path(path / "model.safetensors").is_file() and not Path(path / "model.safetensors.index.json").is_file():
            convert_multi(model_dir, del_pytorch_model=True)

    import torch
    from exl2conv.conversion.qparams import qparams_headoptions
    from exl2conv.conversion.convert import convert_hf_to_exl2

    if output is None:
        compile_full = model_dir + "-exl2"
    else:
        compile_full = output

    cal_dataset = None
    length = 2048
    dataset_rows = 100
    measurement_rows = 16
    measurement_length = 2048
    shard_size = 8192
    measurement = None

    output_measurement = str(Path(model_dir) / "measurement.json")

    if only_measurement:
        no_resume = True

    job = {
        "in_dir": model_dir,
        "out_dir": model_dir + "-exl2.tmp",
        "cal_dataset": cal_dataset,
        "bits": float(bits),
        "dataset_rows": int(dataset_rows),
        "measurement_rows": int(measurement_rows),
        "length": int(length),
        "measurement_length": int(measurement_length),
        "measurement": measurement,
        "head_bits": int(head_bits),
        "shard_size": shard_size if shard_size > 0 else 1024 ** 3,  # 1 PB = unlimited,
        "compile_full": compile_full,
        "no_resume": no_resume,
        "output_measurement": output_measurement,
    }

    if rope_alpha is not None:
        job["rope_alpha"] = float(rope_alpha)
    if rope_scale is not None:
        job["rope_scale"] = float(rope_scale)

    dt_start = datetime.datetime.now()
    print(f"Starting exl2 quantization for {model_dir} at {str(dt_start)}")
    start = time.perf_counter()

    # take measurement if necessary
    if Path(path / "measurement.json").is_file() and not only_measurement:
        job["measurement"] = str(Path(model_dir) / "measurement.json")
    else:
        convert_hf_to_exl2(job)

    if not only_measurement:
        job["output_measurement"] = None
        job["measurement"] = str(Path(path / "measurement.json"))
        job["no_resume"] = True
        convert_hf_to_exl2(job)

    gc.collect()
    torch.cuda.empty_cache()

    end = time.perf_counter()
    dt_pq = datetime.datetime.now()
    print(f"Quantization complete ({str(end-start)} seconds) at {str(dt_pq)}")

def download_wikitrain():
    import requests
    import gzip
    url = "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/wiki.train.raw.gz"

#    try:
    response = requests.get(url)
    with open("wiki.train.raw", "wb") as f:
        f.write(gzip.decompress(response.content))
#    except Exception:
#        raise ValueError(f"Error downloading {url}")

def get_wikitext2(nsamples, seed, seqlen, model):
    import numpy as np
    import torch
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc
