# quantkit can download a hf model, convert a model to safetensors, and quantize
# supports: AWQ, GPTQ, and EXL2

import gc
import time
import datetime

from pathlib import Path
from quantkit.safetensor import convert_multi

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

def run_gguf(model, quant_type, output, hf_cache, cal_file):
    if cal_file is None:
        # no imat
        pass



def run_awq(model, output, hf_cache, bits, group_size, zero_point):
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
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=False, resume_download=True)

    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = model_dir

    if output is None:
        output = f"{model_path}-awq-w{bits}-gs{group_size}"

    quant_path = output
    quant_config = { "zero_point": zero_point, "q_group_size": group_size, "w_bit": bits, "version": "GEMM" }

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

def run_gptq(model, output, hf_cache, bits, group_size, act_order):
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
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=False, resume_download=True)

    import torch
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    quant_options = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=act_order,
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

def run_exl2(model, output, hf_cache, bits, head_bits, new_measurement):
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
        snapshot_download(model, local_dir=path, local_dir_use_symlinks=False, resume_download=True)
        convert_multi(model_dir, del_pytorch_model=delete_original)

    import torch
    from exl2conv.conversion.qparams import qparams_headoptions
    from exl2conv.conversion.convert import convert_hf_to_exl2

    cal_dataset = None
    dataset_rows = 100
    measurement_rows = 16
    length = 2048
    measurement_length = 2048
    shard_size = 8192
    measurement = None
    compile_full = None
    no_resume = None
    no_resume = True
    output_measurement = None

    job = {
        "in_dir": model_dir,
        "out_dir": model_dir + "-exl2",
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
        #"rope_scale": float(rope_scale),
        #"rope_alpha": float(rope_alpha),
        "no_resume": no_resume,
        "output_measurement": output_measurement,
    }

    dt_start = datetime.datetime.now()
    print(f"Starting exl2 quantization for {model_dir} at {str(dt_start)}")
    start = time.perf_counter()

    convert_hf_to_exl2(job)

    gc.collect()
    torch.cuda.empty_cache()

    end = time.perf_counter()
    dt_pq = datetime.datetime.now()
    print(f"Quantization complete ({str(end-start)} seconds) at {str(dt_pq)}")


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
