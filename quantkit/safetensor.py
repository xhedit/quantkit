# These functions were sourced from https://huggingface.co/spaces/safetensors/convert/blob/main/convert.py
import os

def shared_pointers(tensors):
    from collections import defaultdict
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing

def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

def convert_file(pt_filename: str, sf_filename: str):
    import torch
    from safetensors.torch import load_file, save_file

    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous().half() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")

def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local

def convert_multi(folder: str, del_pytorch_model: bool):
    from tqdm import tqdm
    filename = "pytorch_model.bin.index.json"

    import json
    cwd_dir = os.getcwd()
    #os.chdir(folder)

    if not os.path.exists(os.path.join(folder, filename)):
        if os.path.exists(os.path.join(folder, "pytorch_model.bin")):
            convert_single(folder, del_pytorch_model)

    with open(os.path.join(folder, filename), "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []

    for filename in tqdm(filenames):
        print(f"Converting {filename}")
        pt_filename = filename
        st_filename = rename(pt_filename)

        pt_filename = os.path.join(folder, pt_filename)
        st_filename = os.path.join(folder, st_filename)

        convert_file(pt_filename, st_filename)
        local_filenames.append(st_filename)
        if del_pytorch_model:
            os.remove(pt_filename)

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)
    if del_pytorch_model:
        os.remove(os.path.join(folder, "pytorch_model.bin.index.json"))

    os.chdir(cwd_dir)
    return

def convert_single(folder: str, del_pytorch_model: bool):
    pt_filename = "pytorch_model.bin"
    st_name = "model.safetensors"
    st_filename = os.path.join(folder, st_name)
    convert_file(pt_filename, st_filename)
    if del_pytorch_model:
        os.remove("pytorch_model.bin")
    return
