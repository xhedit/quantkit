from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.architecture import RopeStyle
import argparse, os, shutil
import sys
import json
from exllamav2.conversion.tokenize import tokenize
from exllamav2.conversion.measure import embeddings, measure_quant
from exllamav2.conversion.quantize import quant
from exllamav2.conversion.optimize import optimize
from exllamav2.conversion.compile import compile_model
from exllamav2.conversion.qparams import qparams_headoptions
import torch

def convert_hf_to_exl2(options):

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = options["in_dir"]
    config.qkv_embed = False
    config.prepare()

    # Tokenizer

    tokenizer = ExLlamaV2Tokenizer(config)

    # 
    torch.set_printoptions(precision = 7, sci_mode = False, linewidth = 200)

    # Create job

    def save_job():
        global job_file, job
        with open(job_file, "w", encoding = "utf8") as f:
            f.write(json.dumps(job, indent = 4))


    global job_file
    job_file = os.path.join(options["out_dir"], "job_new.json")

    if options["no_resume"] or not os.path.exists(job_file):
        if not os.path.exists(options["out_dir"]):
            os.makedirs(options["out_dir"])

        print(f" -- Beginning new job")
        if len(os.listdir(options["out_dir"])) != 0:
            print(f" !! Warning: Output directory is not empty: {options['out_dir']}")

            if options["no_resume"]:
                print(f" !! Cleaning output directory: {options['out_dir']}")
                for filename in os.listdir(options["out_dir"]):
                    file_path = os.path.join(options["out_dir"], filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)

    output_measurement = options["output_measurement"]
    if output_measurement is not None:
        if os.path.isdir(output_measurement):
            output_measurement = os.path.join(output_measurement, "measurement.json")

    global job
    job = {
        "in_dir": options['in_dir'],
        "out_dir": options['out_dir'],
        "cal_dataset": options['cal_dataset'],
        "bits": options['bits'],
        "dataset_rows": options['dataset_rows'],
        "measurement_rows": options['measurement_rows'],
        "length": options['length'],
        "measurement_length": options['measurement_length'],
        "head_bits": options['head_bits'],
        "shard_size": options['shard_size'] if options['shard_size'] > 0 else 1024 ** 3,  # 1 PB = unlimited,
        "compile_full": options['compile_full'],
        "status_output": True,
        "hidden_state_offload_layers": options['hidden_state_offload_layers'],
    }

    if "rope_scale" in options:
        job["rope_scale"] = options['rope_scale'],
    if "rope_alpha" in options:
        job["rope_alpha"] = options['rope_alpha'],


    job["output_measurement"] = output_measurement
    job["progress"] = "begin"

    if options["measurement"] is not None:
        with open(options["measurement"], "r", encoding = "utf8") as f:
            imp_measurement = json.load(f)
            job["measurement"] = imp_measurement["measurement"]
            job["last_module_idx"] = imp_measurement["last_module_idx"]
            job["reuse_measurement"] = options["measurement"]

    # Resume existing job

    if options["no_resume"] or not os.path.exists(job_file):
        pass

    else:
        print(f" -- Resuming job")
        print(f" !! Note: Overriding options with settings from existing job")

        with open(job_file, "r", encoding = "utf8") as f:
            resume_job = json.load(f)

        # Override keys in existing job
        del resume_job["out_dir"]

        job.update(resume_job)
        if "invalid" in job:
            print(" ** Error: Corrupted job")
            sys.exit()

        if job["progress"] == "finished":
            print(" !! Job is already finished")
            sys.exit()            

    # Feedback

    print(f" -- Input: {job['in_dir']}")
    print(f" -- Output: {job['out_dir']}")
    if job.get("cal_dataset"):
        print(f" -- Calibration dataset: {job['cal_dataset']}, {job['dataset_rows']} / {job['measurement_rows']} rows, {job['length']} tokens per sample")
    else:
        print(f" -- Using default calibration dataset")
    if job["output_measurement"] is None:
        print(f" -- Target bits per weight: {job['bits']} (decoder), {job['head_bits']} (head)")
        print(f" -- Max shard size: {job['shard_size']} MB")
    else:
        print(f" -- Measurement will be saved to {job['output_measurement']}")
        print(f" !! Conversion script will end after measurement pass")


    if 'rope_scale' in job: print(f" -- RoPE scale: {job['rope_scale']:.2f}")
    if 'rope_alpha' in job: print(f" -- RoPE alpha: {job['rope_alpha']:.2f}")

    # Make sure subfolders exist

    if job.get("compile_full"):
        print(f" -- Full model will be compiled to: {job['compile_full']}")
        if os.path.exists(job["compile_full"]):
            if not os.path.isdir(job["compile_full"]):
                print(f" ## Error: Output path {job['compile_full']} exists but is not a directory")
                sys.exit()
            if len(os.listdir(job["compile_full"])) > 0:
                print(f" !! Warning: Output path {job['compile_full']} exists but is not empty")

    out_tensor_dir = os.path.join(job["out_dir"], "out_tensor")
    if not os.path.exists(out_tensor_dir):
        os.makedirs(out_tensor_dir)

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = job['in_dir']
    config.qkv_embed = False
    config.prepare()

    # Tokenizer

    tokenizer = ExLlamaV2Tokenizer(config)

    # Set scaling for input model

    if "rope_scale" in job: config.scale_pos_emb = job["rope_scale"]
    if "rope_alpha" in job: config.scale_alpha_value = job["rope_alpha"]

    # Create model without loading weights

    model = ExLlamaV2(config)
    model.load(lazy = True)

    # Limit context length if necessary
    
    if model.config.arch.rope_style == RopeStyle.NONE:
        max_ctx = model.config.max_seq_len
        if job["length"] > max_ctx:
            print (f" !! Warning: Reducing calibration length to model max context: {max_ctx}")
            job["length"] = max_ctx
        if job["measurement_length"] > max_ctx:
            print (f" !! Warning: Reducing measurement calibration length to model max context: {max_ctx}")
            job["measurement_length"] = max_ctx

    # Do the things

    save_job()

    while True:

        progress = job["progress"]

        if progress == "begin":

            if "reuse_measurement" in job:

                print(f" -- Reusing measurement: {job['reuse_measurement']}")
                job["progress"] = "optimize"
                save_job()

            else:

                print(f" -- Tokenizing samples (measurement)...")
                tokenize(job, save_job, tokenizer, measure = True)
                job["progress"] = "initial_embeddings"
                save_job()

        if progress == "initial_embeddings":

            print(f" -- Token embeddings (measurement)...")
            embeddings(job, save_job, model)
            job["progress"] = "measure_quant"
            save_job()

        if progress == "measure_quant":
            print(f" -- Measuring quantization impact...")

            model.unload()
            config.max_output_len = 16
            model = ExLlamaV2(config)
            model.load(lazy = True)

            status = measure_quant(job, save_job, model, job["hidden_state_offload_layers"])  # capturing the graceful exits
            if status == "interrupted":
                print("Process interrupted. Exiting gracefully.")
                save_job()
                sys.exit(1)
            if job["output_measurement"] is None:
                job["progress"] = "optimize"
            else:
                job["progress"] = "finished"
            save_job()

            model.unload()
            config.max_output_len = None
            model = ExLlamaV2(config)
            model.load(lazy = True)

        if progress == "optimize":

            print(f" -- Optimizing...")
            optimize(job, save_job, model)
            job["progress"] = "tokens_cal"
            save_job()

        if progress == "tokens_cal":

            print(f" -- Tokenizing samples...")
            tokenize(job, save_job, tokenizer)
            job["progress"] = "embeddings"
            save_job()

        if progress == "embeddings":
            print(f" -- Token embeddings again...")
            embeddings(job, save_job, model)
            job["progress"] = "quant"
            save_job()

        if progress == "quant":

            print(f" -- Quantizing...")
            quant(job, save_job, model)
            job["progress"] = "compile"
            save_job()

        if progress == "compile":

            print(f" -- Compiling output file...")
            compile_model(job, save_job, model)
            job["progress"] = "finished"
            save_job()

        if progress == "finished": break

    print(f" -- Finished")
