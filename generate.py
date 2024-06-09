import sys

sys.path.append("Dcase2024-Task7/src")

import os

os.environ["HF_HOME"] = "Dcase2024-Task7/pre_load_models"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACE_HUB_CACHE"] = "Dcase2024-Task7/pre_load_models"
os.environ["TORCH_HOME"] = "Dcase2024-Task7/pre_load_models"


import soundfile as sf
import numpy as np

import argparse
import yaml
import torch
from latent_diffusion.models.ddpm import LatentDiffusion
from latent_diffusion.util import instantiate_from_config
from latent_diffusion.modules.encoders.modules import Clapmicro as CLAP
from tqdm import tqdm
import torchaudio
import json
import ipdb
import shutil


config_root = "Dcase2024-Task7/configs"

config = os.path.join(config_root, "32k_attention.yaml")

config_yaml = yaml.load(open(config, "r"), Loader=yaml.FullLoader)





def get_model():
    latent_diffusion = instantiate_from_config(config_yaml["model"]).to("cuda")
    PATH = "Dcase2024-Task7/checkpoints/test_model.ckpt"
    state_dict = torch.load(PATH)["state_dict"]
    latent_diffusion.load_state_dict(state_dict)
    return latent_diffusion


model = get_model().cuda()
clap = CLAP(model_path = "Dcase2024-Task7/checkpoints/CLAP_weights_2023.pth")
model.clap = clap.cuda().eval()
model = torch.compile(model)
model.eval()




# def process():


def generate_sound(caption_list, model = model ,ngen = 5):
    waveforms = {}
    
    with model.ema_scope("Plotting"):
        with torch.no_grad():
            for caption in tqdm(caption_list):
                batch = {}

                batch["fname"] = [f"{caption}.wav"]
                batch["text"] = [caption]
                batch["waveform"] = torch.rand(1,1,131072).cuda()
                batch["log_mel_spec"] = torch.rand(1,512,96).cuda()
                batch["sampling_rate"] = torch.tensor([32000]).cuda()
                batch["label_vector"] = torch.rand(1,527).cuda()
                batch["stft"] = torch.rand(1,1024,512).cuda()
                waveform = model.generate_sample([batch],unconditional_guidance_scale=2.0,ddim_steps=200,n_gen=ngen)
                waveform = np.squeeze(waveform, axis=0)
                if waveform.shape[0]==1:
                    waveform = waveform[0]
                waveforms[caption] = waveform[:32000*4]

    return waveforms



if __name__ == "__main__":


    text_prompts_list = ["a buzzer is ringing with water in the background",
        "a pig is grunting with water in the background",
        "an alarm of a car door stayin open is ringin with crowd in the background",
        "a small dog is whining with water in the background",
        "a car horn is honking with crowd in the background",]

    # text_prompts_list = ["a buzzer is ringing with water in the background"]

    waveform_list = generate_sound(text_prompts_list)

    os.makedirs('output', exist_ok=True)
    for src_text, src in tqdm(waveform_list.items()):
        _filepath = os.path.join( 'output', f"{src_text}.wav")
        src = src / np.max(np.abs(src)) # normalize the energy of the generation output
        if src.shape[0]==1:
            src = src[0]
        sf.write(_filepath, src, 32000, subtype='PCM_16')

            

