#!/usr/bin/env python3

import click
from quantkit.quantkit import run_download, run_safetensor, run_gguf, run_awq, run_gptq, run_exl2

# commands: download, safetensor, gguf, awq, gptq, exl2

@click.group()
def run():
    click.echo("quantkit v0.1")

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--force-download/--no-force-download', default=False, help='Download again even if model is in hf cache.')
@click.option('--resume/--no-resume', default=True, help='Resume a previously incomplete download.')
@click.option('--safetensors-only/--everything', default=False, help='Ignore pytorch model-*.bin and pytorch_model.bin.index.json')
def download(model, output, hf_cache, force_download, resume, safetensors_only):
    """Download model from huggingface."""
    click.echo(f"download | model: {model} | use hf cache: {hf_cache} | out: {output} | force_download: {force_download} | resume: {resume} | safetensors_only: {safetensors_only}")
    run_download(model, output, hf_cache, force_download, resume, safetensors_only)

@run.command()
@click.argument('model', required=True)
@click.option('--delete-original/--save-original', default=False, help='Delete pytorch model files after conversion')
#@click.option('--output', '-out')
def safetensor(model, delete_original):
    """Download and/or convert a pytorch model to safetensor format."""
    click.echo(f"safetensor | model: {model} | delete_original: {delete_original}")
    run_safetensor(model, delete_original)

@run.command()
@click.argument('model', required=True)
@click.argument('quant_type', required=True)
@click.option('--output', '-out', help='output name')
@click.option('--keep/--delete', help='keep intermediate conversion GGUF')
@click.option('--f32/--f16', default=False, help='intermediate conversion step uses f32 (requires much more disk space)')
@click.option('--built-in-imatrix/--disable-built-in-imatrix', default=False, help='use built in imatrix')
@click.option('--imatrix', help='Specify pre-generated imatrix')
@click.option('--cal-file', help='Specify calibration dataset')
@click.option('--n-gpu-layers', "-ngl",  default=0, help='how many layers to offload to GPU for imatrix')
def gguf(model, quant_type, output, keep, f32, built_in_imatrix, imatrix, cal_file, n_gpu_layers):
   """Download and/or convert a model to GGUF format."""
   click.echo(f"gguf | model: {model} | quant_type: {quant_type} | out: {output} | keep: {keep} | f32: {f32} | bim: {built_in_imatrix} | imatrix: {imatrix} | cal_file: {cal_file} | ngl: {n_gpu_layers}")
   run_gguf(model, quant_type, output, keep, f32, built_in_imatrix, imatrix, cal_file, int(n_gpu_layers))

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', default=4, help='Bits / bpw')
@click.option('--group-size', default=128, help='Group size')
@click.option('--zero-point/--no-zero-point', default=True, help='Zero point')
@click.option('--gemm/--gemv', default=True, help='GEMM or GEMV')
def awq(model, output, hf_cache, bits, group_size, zero_point, gemm):
    """Download and/or convert a model to AWQ format."""
    click.echo(f"awq | model: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | group_size: {group_size} | zero_point: {zero_point} | gemm: {gemm}")
    run_awq(model, output, hf_cache, bits, group_size, zero_point, gemm)

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', default=4, help='Bits / bpw')
@click.option('--group-size', '-gs', default=128, help='Group size')
@click.option('--damp', default=0.01, help='Dampening percent')
@click.option('--sym/--no-sym', default=True, help='symmetric quantization')
@click.option('--true-seq/--no-true-seq', default=True, help='true sequential quantization')
@click.option('--act-order/--no-act-order', default=False, help='Activation order / desc_act')
def gptq(model, output, hf_cache, bits, group_size, damp, sym, true_seq, act_order):
    """Download and/or convert a model to GPTQ format."""
    click.echo(f"gptq | model: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | group_size: {group_size} | damp: {damp} | sym: {sym} | true_seq: {true_seq} | act_order: {act_order}")

    if not (0 < damp < 1):
        raise ValueError("dampening percent must be between 0 and 1!")

    run_gptq(model, output, hf_cache, bits, group_size, damp, sym, true_seq, act_order)

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', required=True, help='Bits / bpw')
@click.option('--head-bits', '-hb', type=click.Choice(['6', '8']), default='8', help='Bits / bpw for head layer (default: 8)')
@click.option('--rope-alpha', '-ra', help='RoPE alpha value')
@click.option('--rope-scale', '-rs', help='RoPE scale factor, required for some models (deepseek-33b should be 4)')
@click.option('--only-measurement', '-om', default=False, is_flag=True, help='take measurement only')
@click.option('--no-resume/--resume', '-nr/-r', default=False, help='do not attempt to resume job')
def exl2(model, output, hf_cache, bits, head_bits, rope_alpha, rope_scale, only_measurement, no_resume):
    """Download and/or convert a model to EXL2 format."""
    click.echo(f"exl2 | model: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | head_bits: {head_bits} | rope_alpha: {rope_alpha} | rope_scale: {rope_scale} | only_measurement: {only_measurement} | no_resume: {no_resume}")
    run_exl2(model, output, hf_cache, bits, int(head_bits), rope_alpha, rope_scale, only_measurement, no_resume)

run.add_command(download)
run.add_command(safetensor)
#run.add_command(gguf)
run.add_command(awq)
run.add_command(gptq)
run.add_command(exl2)

def main():
    run()

if __name__ == "__main__":
    main()
