import os

current_file_path = os.path.abspath(__file__)
print(f"Current file path: {current_file_path}")
desired_path = os.path.dirname(current_file_path)
print(f"Directory path: {desired_path}")
while not desired_path.endswith('/src'):
    desired_path = os.path.dirname(desired_path)

target_path = os.path.join(desired_path,"hifigan")
import json

import torch
import numpy as np


import sys

sys.path.append(target_path)

from models import BigVGAN


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def get_available_checkpoint_keys(model, ckpt):
    print("==> Attemp to reload from %s" % ckpt)
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if("_orig_mod." in key):
            new_state_dict["generator"][key.replace("_orig_mod.","")] = state_dict["generator"][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict

def get_vocoder(config, device, mel_bins):

    if mel_bins == 96:
        with open(f"{target_path}/32k_config.json", "r") as f:
            config = json.load(f)

        config = AttrDict(config)
        vocoder = BigVGAN(config)
        print("Load config for hifigan generator 32k")
        # ckpt = torch.load("/mnt/bn/arnold-yy-audiodata/audioldm/Bigvgan_v2/32khz/v1/g_01500000")
        # print("Load hifigan_generator_22k/gen_02340000")
        # ckpt = torch.load(os.path.join(ROOT, "gen_02340000"))
        # vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    else:
        print("vocoder load failed")
        vocoder = None


    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs
