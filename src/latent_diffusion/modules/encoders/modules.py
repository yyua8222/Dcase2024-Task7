import sys
import os
os.environ["TRANSFORMERS_CACHE"]="pre_load_models"
sys.path.append("src")
import torch
import torch.nn as nn
from clap.open_clip import create_model
from clap.training.data import get_audio_features
from msclap import CLAP
import torchaudio
from transformers import RobertaTokenizer, AutoTokenizer, T5EncoderModel

from transformers import logging

logging.set_verbosity_error()
import torch.nn.functional as F
from latent_diffusion.modules.audiomae.AudioMAE import Vanilla_AudioMAE
from latent_diffusion.modules.phoneme_encoder.encoder import TextEncoder

import numpy as np
import math
import ipdb

"""
The model forward function can return three types of data:
1. tensor: used directly as conditioning signal
2. dict: where there is a main key as condition, there are also other key that you can use to pass loss function and itermediate result. etc.
3. list: the length is 2, in which the first element is tensor, the second element is attntion mask.

The output shape for the cross attention condition should be:
x,x_mask = [bs, seq_len, emb_dim], [bs, seq_len]

All the returned data, in which will be used as diffusion input, will need to be in float type
"""


class FlanT5HiddenState(nn.Module):
    """
    llama = FlanT5HiddenState()
    data = ["","this is not an empty sentence"]
    encoder_hidden_states = llama(data)
    import ipdb;ipdb.set_trace()
    """
    def __init__(self, text_encoder_name="google/flan-t5-large", freeze_text_encoder=True,emb_num=3,return_length = 50,input_caption = False,all_pos = False):
        super().__init__()
        print("the cache dir is",os.getenv('TRANSFORMERS_CACHE'))
        self.emb_num = emb_num
        self.return_length = return_length
        self.freeze_text_encoder = freeze_text_encoder
        self.text_encoder_name = text_encoder_name
        self.all_pos = all_pos
        if self.all_pos:
            self.position_embedding = self.add_position_embedding(self.return_length*self.emb_num,1024)

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name,cache_dir = "pre_load_models")
        self.model = T5EncoderModel.from_pretrained(self.text_encoder_name,cache_dir = "pre_load_models")
        # if tokenizer:
        #     self.tokenizer = tokenizer
        #     self.model = model
        # else:
        #     self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        #     self.model = T5EncoderModel.from_pretrained(text_encoder_name)
        # self.tokenizer = None
        # self.model = None
        self.input_caption = input_caption
        if self.model:
            if(freeze_text_encoder):
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
            else:
                print("=> The text encoder is learnable")

        self.empty_hidden_state_cfg = None
        self.device = None
    def add_position_embedding(self,max_sequence_length, embedding_dim):
    
        position_embeddings = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)
        return position_embeddings
    # Required
    def get_unconditional_condition(self, batchsize):
        param = next(self.model.parameters())
        if(self.freeze_text_encoder):
            assert param.requires_grad == False
        
        # device = param.device
        if(self.empty_hidden_state_cfg is None):
            self.empty_hidden_state_cfg, _ = self([""]*self.emb_num)
        
        hidden_state = torch.cat([self.empty_hidden_state_cfg] * batchsize).float()
        attention_mask = torch.ones((batchsize, hidden_state.size(1))).to(hidden_state.device).float()
        # if self.input_caption:
            # print(f"the hidden_state is {hidden_state.shape} and the attention mask is {attention_mask.shape}")
        return [hidden_state, attention_mask] # Need to return float type

    def forward(self, batch):


        param = next(self.model.parameters())
        if(self.freeze_text_encoder):
            assert param.requires_grad == False

        if (self.device is None):
            self.device = param.device

        return self.encode_text(batch)

    def encode_text(self, prompt):
        # ipdb.set_trace()
        device = self.model.device
        n_gen = len(prompt)
        # print("n_gen is ",n_gen)
        if n_gen==1 and self.input_caption is False:
            ipdb.set_trace()
        batch_size = len(prompt[0])
        
        # for i in range(batch_size):
        #     text_list = []
        encoder_list = []
        attention_list = []
        for i in range(n_gen):
                # text_list.append(prompt[j][i])
            # batch = self.tokenizer(prompt[i], max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
            ##setting the max length into 50
            batch = self.tokenizer(prompt[i], max_length=self.return_length, padding="max_length", truncation=True, return_tensors="pt") ##previous length is 50  mae length is 32

            # ipdb.set_trace()
            input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

            # Get text encoding
            if self.freeze_text_encoder:
                with torch.no_grad():
                    encoder_hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            else:
                encoder_hidden_states = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
            encoder_list.append(encoder_hidden_states)
            attention_list.append(attention_mask)
        # ipdb.set_trace()

        if self.input_caption:
            encoder_return = torch.cat(encoder_list,dim=0)
            attention_return =torch.cat(attention_list,dim=0)

        else:
            encoder_return = torch.cat(encoder_list,dim=1)
            if self.all_pos:
                # ipdb.set_trace()
                encoder_return = encoder_return + self.position_embedding.unsqueeze(0).expand(encoder_return.shape[0], -1, -1).to(self.device)
            attention_return =torch.cat(attention_list,dim=1)
        # if encoder_return.shape[0]==5:
        #     ipdb.set_trace()
        return [encoder_return.detach(), attention_return.float()]
            
            
        # return [encoder_hidden_states.detach(), attention_mask.float()] # Attention mask == 1 means usable token




    
        
class Clapmicro(nn.Module):
    """
        audiomae = AudioMAEConditionCTPool2x2()
        data = torch.randn((4, 1024, 128))
        output = audiomae(data)
        import ipdb;ipdb.set_trace()
        exit(0)
    """
    def __init__(self,model =None, model_path = "checkpoints/CLAP_weights_2023.pth",version = '2023',re_num=1,input_caption = False, encode_type = "text",seq_len = 1024,emb_dim = 1, pos_emb = False, all_pos = False, cross_attention = True):
        super().__init__()
        self.device = None
        self.re_num = re_num
        self.cross_attention = cross_attention
        if not model:
            self.clap_model = CLAP(model_path, version = version, use_cuda=True)
        else:
            self.clap_model = model
        self.input_caption = input_caption
        self.encode_type = encode_type
        self.pos_emb = pos_emb
        self.all_pos = all_pos
        if self.pos_emb:
            self.position_embedding = self.add_position_embedding(seq_len,emb_dim)
            print("using single position embedding")
        if self.all_pos:
            self.position_embedding = self.add_position_embedding(seq_len*re_num,emb_dim)
            print("using all position embedding")
        assert self.pos_emb + self.all_pos < 2


    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        with torch.no_grad():
            audio_emb = self.clap_model.clap.audio_encoder(waveform.cuda())[0]
            text_emb = self.clap_model.get_text_embeddings(text)

            similarity = F.cosine_similarity(audio_emb, text_emb, dim=1)
            # similarity = self.clap_model.compute_similarity(audio_emb, text_emb)
        return similarity.squeeze()

    def add_position_embedding(self,max_sequence_length, embedding_dim):
    
        position_embeddings = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)
    # position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    
        return position_embeddings

    # Required
    def get_unconditional_condition(self, batchsize):
        # param = next(self.clap_model.parameters())
        # assert param.requires_grad == False
        device = self.device
        # time_pool, freq_pool = self.eval_time_pooling, self.eval_freq_pooling
        token_num = int(1024) * self.re_num
        # ipdb.set_trace()
        if not self.cross_attention:
                return torch.zeros((batchsize, token_num)).to(device).float()
        else:
            return [torch.zeros((batchsize, token_num, 1)).to(device).float(), torch.ones((batchsize, token_num)).to(device).float()]

    # Required
    def forward(self, batch):
        # assert batch.size(-2) == 1024 and batch.size(-1) == 128
        batch_size = len(batch)
        # ipdb.set_trace()
        
        with torch.no_grad():
            rep_list = []
            for i in range(batch_size):

                cur_batch = batch[i]

                if isinstance(cur_batch, str):
                    cur_batch = [cur_batch]
                elif isinstance(cur_batch, list):
                    pass
                else:
                    print("input type error")
                if self.encode_type == "audio":
                    representation = self.clap_model.get_audio_embeddings(cur_batch)
                elif self.encode_type == "text":

                    representation = self.clap_model.get_text_embeddings(cur_batch)
                # representation = representation.view(batch_size,representation.shape[1]*representation.shape[2],-1)
                representation = representation.reshape(representation.shape[0],representation.shape[1],1)
                if self.pos_emb:
                    representation = representation + self.position_embedding.unsqueeze(0).expand(representation.shape[0], -1, -1).to(representation.device)

                
                rep_list.append(representation)
            # ipdb.set_trace()
            if self.input_caption:
                representation = torch.cat(rep_list,dim=0)
            else:
                representation = torch.cat(rep_list,dim=1)

            if not self.device:
                self.device = representation.device

            if self.all_pos:
                representation = representation + self.position_embedding.unsqueeze(0).expand(representation.shape[0], -1, -1).to(representation.device)
            if not self.cross_attention:
                return representation.reshape(representation.shape[0], -1)
            else:
                return [representation, torch.ones((representation.size(0), representation.size(1))).to(representation.device).float()]


class Claplaion(nn.Module):
    """
        audiomae = AudioMAEConditionCTPool2x2()
        data = torch.randn((4, 1024, 128))
        output = audiomae(data)
        import ipdb;ipdb.set_trace()
        exit(0)
    """
    def __init__(self,model =None, model_path = "/mnt/bn/arnold-yy-audiodata/pre_load_models/big_clap.pt",version = '2023',re_num=1,input_caption = False, encode_type = "text",seq_len = 512,emb_dim = 1, pos_emb = False, all_pos = False, cross_attention = True):
        super().__init__()
        self.device = None
        self.re_num = re_num
        self.cross_attention = cross_attention
        if not model:
            # self.clap_model = CLAP(model_path, version = version, use_cuda=True)
            self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
            self.clap_model.load_ckpt(model_path)
        else:
            self.clap_model = model
        self.input_caption = input_caption
        self.encode_type = encode_type
        self.pos_emb = pos_emb
        self.all_pos = all_pos
        if self.pos_emb:
            self.position_embedding = self.add_position_embedding(seq_len,emb_dim)
            print("using single position embedding")
        if self.all_pos:
            self.position_embedding = self.add_position_embedding(seq_len*re_num,emb_dim)
            print("using all position embedding")
        assert self.pos_emb + self.all_pos < 2



    def add_position_embedding(self,max_sequence_length, embedding_dim):
    
        position_embeddings = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embedding_dim))
        
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)
    # position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

    
        return position_embeddings

    # Required
    def get_unconditional_condition(self, batchsize):
        # param = next(self.clap_model.parameters())
        # assert param.requires_grad == False
        device = self.device
        # time_pool, freq_pool = self.eval_time_pooling, self.eval_freq_pooling
        token_num = int(512) * self.re_num
        # ipdb.set_trace()
        if not self.cross_attention:
                return torch.zeros((batchsize, token_num)).to(device).float()
        else:
            return [torch.zeros((batchsize, token_num, 1)).to(device).float(), torch.ones((batchsize, token_num)).to(device).float()]

    # Required
    def forward(self, batch):
        # assert batch.size(-2) == 1024 and batch.size(-1) == 128
        

        batch_size = len(batch)

        
        # ipdb.set_trace()
        
        with torch.no_grad():
            rep_list = []
            for i in range(batch_size):

                cur_batch = batch[i]

                if isinstance(cur_batch, str):
                    cur_batch = [cur_batch]
                elif isinstance(cur_batch, list):
                    pass
                else:
                    print("input type error")
                if self.encode_type == "audio":
                    representation = self.clap_model.get_audio_embedding_from_filelist(cur_batch,use_tensor = True)
                elif self.encode_type == "text":
                    # ipdb.set_trace()
                    if ".wav" in cur_batch[0]: 

                        print(f"error found in claplaion with batch text {cur_batch[0]}")
                        # ipdb.set_trace()
                    if len(cur_batch) == 1:
                        representation = self.clap_model.get_text_embedding([cur_batch[0],cur_batch[0]],use_tensor = True)[:1]
                    else:
                        representation = self.clap_model.get_text_embedding(cur_batch,use_tensor = True)
                # representation = representation.view(batch_size,representation.shape[1]*representation.shape[2],-1)
                representation = representation.reshape(representation.shape[0],representation.shape[1],1)
                if self.pos_emb:
                    representation = representation + self.position_embedding.unsqueeze(0).expand(representation.shape[0], -1, -1).to(representation.device)
                rep_list.append(representation)
            # ipdb.set_trace()
            if self.input_caption:
                representation = torch.cat(rep_list,dim=0)
            else:
                representation = torch.cat(rep_list,dim=1)

            if not self.device:
                self.device = representation.device

            if self.all_pos:
                representation = representation + self.position_embedding.unsqueeze(0).expand(representation.shape[0], -1, -1).to(representation.device)
            if not self.cross_attention:
                return representation.reshape(representation.shape[0], -1)
            else:
                return [representation, torch.ones((representation.size(0), representation.size(1))).to(representation.device).float()]



class CLAPAudioEmbeddingClassifierFreev2(nn.Module):
    def __init__(
        self,
        pretrained_path,
        sampling_rate=16000,
        embed_mode="audio",
        amodel="HTSAT-base",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=False,
    ):
        super().__init__()
        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )
        audio_cfg = self.model_cfg["audio_cfg"]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            win_length=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.unconditional_token = None
        if self.training_mode:
            self.model.train()
        else:
            self.model.eval()

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]
        return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.size(-1)
        for i in range(waveform.size(0)):
            mute_size = int(
                self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion))
            )
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()
    
    def wav_similarity(self, waveform1, waveform2):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform1.cuda())
            self.embed_mode = "audio"
            text_emb = self(waveform2.cuda())
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()

    def build_unconditional_emb(self):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]

    def forward(self, batch):
        # If you want this conditioner to be unconditional, set self.unconditional_prob = 1.0
        # If you want this conditioner to be fully conditional, set self.unconditional_prob = 0.0
        if self.model.training == True and not self.training_mode:
            print(
                "The pretrained CLAP model is under training mode !!"
            )
            self.model, self.model_cfg = create_model(
                self.amodel,
                self.tmodel,
                self.pretrained,
                precision=self.precision,
                device="cuda",
                enable_fusion=self.enable_fusion,
                fusion_type=self.fusion_type,
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        
        if(self.unconditional_token is None):
            self.build_unconditional_emb()

        # if(self.training_mode):
        #     assert self.model.training == True
        # else:
        #     assert self.model.training == False

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        if self.embed_mode == "audio":
            with torch.no_grad():
                # assert (
                #     self.sampling_rate == 16000
                # ), "We only support 16000 sampling rate"
                
                # if self.random_mute:
                #     batch = self._random_mute(batch)
                # batch: [bs, 1, t-samples]
                if(self.sampling_rate != 48000):
                    batch = torchaudio.functional.resample(
                        batch, orig_freq=self.sampling_rate, new_freq=48000
                    )

                audio_data = batch.squeeze(1)
                mel = self.mel_transform(audio_data)
                audio_dict = get_audio_features(
                    audio_data,
                    mel, 
                    int(32000*4.096),
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict)
        elif self.embed_mode == "text":
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                
                if(isinstance(batch, str) or (isinstance(batch, list) and len(batch) == 1)):
                    for key in text_data.keys():
                        text_data[key] = text_data[key].unsqueeze(0)
                        
                embed = self.model.get_text_embedding(text_data)

        embed = embed.unsqueeze(1)
        for i in range(embed.size(0)):
            if self.make_decision(self.unconditional_prob):
                embed[i] = self.unconditional_token
        # embed = torch.randn((batch.size(0), 1, 512)).type_as(batch)
        return embed.detach()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}

if __name__ == "__main__":
    model = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path="/mnt/bn/lqhaoheliu/exps/checkpoints/audioldm/ckpt/CLAP.pt", 
        embed_mode = "text",
        amodel="HTSAT-tiny")
    # data = torch.randn((6, 1, int(16000*10.24)))
    data = ["text", "text"]
    res = model(data)
    import ipdb;ipdb.set_trace()
