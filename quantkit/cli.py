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

#@run.command()
#@click.argument('model', required=True)
#@click.argument('quant_type', required=True)
#@click.option('--output', '-out', help='output name')
#@click.option('--cal-file', help='Specify calibration dataset')
#def gguf(model, quant_type, output, cal_file):
#    """Download and/or convert a model to GGUF format."""
#    click.echo(f"gguf | mnodel: {model} | quant_type: {quant_type} | out: {output} | cal_file: {cal_file}")
#    run_gguf(model, quant_type, output, hf_cache, cal_file)

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', default=4, help='Bits / bpw')
@click.option('--group-size', default=128, help='Group size')
@click.option('--zero-point/--no-zero-point', default=True, help='Zero point')
def awq(model, output, hf_cache, bits, group_size, zero_point):
    """Download and/or convert a model to AWQ format."""
    click.echo(f"awq | mnodel: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | group_size: {group_size} | zero_point: {zero_point}")
    run_awq(model, output, hf_cache, bits, group_size, zero_point)

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', default=4, help='Bits / bpw')
@click.option('--group-size', default=128, help='Group size')
@click.option('--act-order/--no-act-order', default=True, help='Activation order / desc_act')
def gptq(model, output, hf_cache, bits, group_size, act_order):
    """Download and/or convert a model to GPTQ format."""
    click.echo(f"gptq | mnodel: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | group_size: {group_size} | act_order: {act_order}")
    run_gptq(model, output, hf_cache, bits, group_size, act_order)

@run.command()
@click.argument('model', required=True)
@click.option('--output', '-out', help='output directory')
@click.option('--hf-cache/--no-cache', default=True, help='Use huggingface cache dir.')
@click.option('--bits', '-b', required=True, help='Bits / bpw')
@click.option('--head-bits', '-hb', type=click.Choice(['6', '8']), default='8', help='Bits / bpw for head layer (default: 8)')
@click.option('--new-measurement/--no-new-measurement', default=False, help='Take a new measurement')
def exl2(model, output, hf_cache, bits, head_bits, new_measurement):
    """Download and/or convert a model to EXL2 format."""
    click.echo(f"exl2 | mnodel: {model} | out: {output} | use hf cache: {hf_cache} | bits: {bits} | head_bits: {head_bits} | new_measurement: {new_measurement}")
    run_exl2(model, output, hf_cache, bits, int(head_bits), new_measurement)

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
