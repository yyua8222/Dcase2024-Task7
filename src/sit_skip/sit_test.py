import os
os.environ["HF_HOME"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
os.environ["WANDB_API_KEY"] = "ec145d92a5f32070c0b6fa4c1db97e478bbb221e"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
os.environ["TORCH_HOME"] = "/mnt/bn/arnold-yy-audiodata/pre_load_models"
# SiT imports:
import torch
from torchvision.utils import save_image
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_XL_2,SiT_XL_2_ldm1
from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

import ipdb


image_size = "256"
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
# Load model:

model = SiT_XL_2_ldm1(input_size=(256,16)).to(device)  # target is 256,16
# state_dict = find_model(f"SiT-XL-2-{image_size}x{image_size}.pt")
# model.load_state_dict(state_dict)
# model.eval() # important!
# vae = AutoencoderKL.from_pretrained(vae_model).to(device)

# test_input = torch.rand(1,3,256,256)

# vae_output = vae.encode(test_input.cuda()).latent_dist.sample().mul_(0.18215)

# ipdb.set_trace()

output = model(x = torch.rand(1,8,256,16).cuda(),t = torch.randint(1,1000,(1,)).cuda(),y = torch.rand(1,1024).cuda())
# output = model(vae_output,torch.rand(1).cuda(),torch.randint(0,10,(1,)).cuda())

ipdb.set_trace()