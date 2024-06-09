import sys
from xml.etree.ElementTree import TreeBuilder
sys.path.append("/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/src")
import os
import pandas as pd
import yaml
import utilities.audio as Audio
from utilities.tools import load_json
from utilities.data.add_on import *
import ipdb

import random
from torch.utils.data import Dataset
import torch.nn.functional
import torch
import numpy as np
import torchaudio
import json
from utilities.data.big_vgan_mel import wav2mel





def load_audio(wav):
    

    esc_path = "/mnt/bn/arnold-yy-audiodata/audio_data/esc50/ESC-50-master/"

    ub8k_path = "/mnt/bn/arnold-yy-audiodata/audio_data/urbansound8K/UrbanSound8K/audio/"
    if "audio" in wav:
        wav = os.path.join(esc_path,wav)
    else:
        wav = os.path.join(ub8k_path,wav)

    waveform,sr = torchaudio.load(wav)


    if sr != 44100:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)

        waveform = resampler(waveform)
    # ipdb.set_trace()

    return waveform[0:1],round(waveform.shape[1]/44100,2)

def wav_longer(wav,length):
    target_length = 44100*length


    padded_tensor = torch.zeros(target_length)

    padded_tensor[:wav.size(1)] = wav[0][:target_length]

    return padded_tensor.reshape(1,-1)


def find_caption(label):
    label_list = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock', 'can_opening', 'crow', 'clapping', 

    'fireworks', 'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops', 'church_bells', 'clock_alarm', 

    'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow', 'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 

    'rain', 'insects', 'laughing', 'hen', 'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing', 'glass_breaking', 'snoring', 

    'toilet_flush', 'pig', 'washing_machine', 'clock_tick', 'sneezing', 'rooster', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets',
    'dog_bark', 'children_playing', 'air_conditioner', 'street_music', 'gun_shot', 'engine_idling', 'jackhammer', 'drilling']


    cpation_list = [["a dog is barking", "some dog barks", "dog barking", "some sounds of dog bark"],
    ["birds chirping", "some birds chirping","sounds of birds chriping","birds playing"],
    ["some sounds of vacuum cleaner","someone is using vacuum cleaner","someone cleaning the floor","vacuum cleaner running in the back"],
    ["some sound of thunderstorm","sound of thunderstorm","raining heavily outside","sound of thunder"],
    ["someone knocking","people knocking","someone knock on the door","knocking on the door"],
    ["sound of opening can","someone open the can","people opening the can","opening the can"],
    ["crow flying","some crow flying","sound of crow","some crow flying in the back"],
    ["people clapping","someone clapping","cheering and clapping","audience clapping"],
    ["sound of fireworks","some fireworks","someone is setting off fireworks","some firworks in the back"],
    ["the sound of chainsaw","a chainsaw is running","someone is using chainsaw","running chainsaw"],
    ["an airplane flying by","an airplane in the back","sound of airplane engine","airplane is flying"],
    ["clicking the mouse","someone click the mouse","sound of clicking mouse","a man is clicking the mouse"],
    ["pouring the water","someone pours the water","sound of pouring water","a man is pouring the water"],
    ["a train passed by","a train is passing by","a train passing in the back","a train coming through"],
    ["a sheep in the back","a sheep is braying","sound of sheep braying","sheep are braying"],
    ["water is dropping","dropping the water","sound of water dropping","water dropping continuously"],
    ["bells ringing","someone ring the bell","a man rang the bell","the sound of church bells"],
    ["the clock is alarming","sound of alarm","alarm sound","the alarm of the clock"],
    ["typing the keyboard","someone is typing the keyboard","a man is typing the keyboard","sound of keyboard typing"],
    ["wind","sound of wind","wind blowing outside","wind blowing in the back"],
    ["someone walking","sound of footsteps","a man is walking","someone is walking"],
    ["sound of a frog","a frog in the back","frog is croaking"],
    ["sound of cows","some cows in the back","cows are braying"],
    ["sound of brushing teeth","a man is brushing his teeth","someone is brushing teeth","a boy is brushing teeth in the end"],
    ["a car is honking","sound of car horn","someone use the car horn","car horn in the back"],
    ["sound of crackling fire","some fire cracking","fire cracking in the back"],
    ["sound of helicopter","a helicopter flying by","a man is flying a helicopter","a helicopter passing by"],
    ["someone is drinking","a man is drinking","sound of sipping"],
    ["raining outside","sound of raining","raining heavily","some rain"],
    ["insects flying","sound of insects","some insects"],
    ["someone is laughing","people laughing","audience is laughing","some laugh"],
    ["sound of a hen","hen crows","some hen passing by","some hen in the back"],
    ["engine running","a running engine","sound of an engine"],
    ["sound of breathing","some breathing","people breathing hard"],
    ["a crying baby","a baby crying","some baby is crying"],
    ["someone using hand saw","a man is using hand saw","sound of handsaw"],
    ["coughing","someone is coughing","a man is coughing","some cough"],
    ["a glass is broken","someone break the glass","a man broke the glass","sound of breaking the glass"],
    ["someone snoring","sound of snoring","some snoring","a man snoring"], 
    ["toilet flush","someone flush the toilet","a man flush the toilet","some flush toilet"],
    ["sound of pig","some pig in the back","pig bleating","some pig bleating"],
    ["someone is using washing machine","washing machine is running","a man is using washing machine","washing machine is operating"],
    ["clock ticking","the clock ticking","sound of ticking","sound of clock ticking"],
    ["sneezing","someone sneezing","sound of sneezing","someone sneezes nose"],
    ["some roosters in the back",'some roosters',"sound of roosters","rooster crows"],
    ["waves of sea","sound of waves","sea waves","some waves"],
    ["siren","sound of siren","siren in the back"],
    ["some cats","cats in the back","sound of cats","some cats meow"],
    ["door creaks","wood creaks","old door opens","sound of door creaks"],
    ["crickets chirping","some crickets","some crickets in the back","sound of crickets"],
    ["a dog is barking", "some dog barks", "dog barking", "some sounds of dog bark"],
    ["children playing", "some children are playing","children playing in the back", "children are having fun"],
    ["someone opens the air conditioner","some air conditioner starts to work","air conditioner is working","some machine running"],
    ["some street music","some music in the background","some music in the back","some music"],
    ["gun shots","some gun shots","someone takes the shot"],
    ["some engine running","sound of engine","sound of engine running","engine idling"],
    ["someone uses the hammer","people using the jackhammer","a hammer is hitting","some hammer hitting"],
    ["the sound of drilling","someone is drilling","some drilling","people are drilling"]
    ]



    cap_index = label_list.index(label)
    cap_list = cpation_list[cap_index]
    return random.choice(cap_list)


def get_ub8k(label):

    label_list = ["dog_bark", "children_playing", "car_horn",  "gun_shot", "siren",  "jackhammer", "air_conditioner", "street_music", "engine_idling", "drilling"]

    caption_list1 = ["a dog is barking", "a children is playing", "a car is horning",  "a man is using a gun", "siren",  "a man is using jackhammer", "air conditioner", "street music", "engines", "drilling"]

    caption_list2 = ["a big dog is barking", "a children is playing", "a car is horning",  "a man is shooting", "siren",  "a man is using jackhammer", "air conditioner", "street music", "engines", "drilling"]

    caption_list3 = ["a small dog is barking", "a children is playing", "a car is horning",  "a gun is shooting", "siren",  "a person is using jackhammer", "air conditioner", "street music", "engines", "drilling"]

    caption_list4 = ["a dog is whinning", "a children is playing", "a car is horning",  "a person is shooting", "siren",  "a boy is using jackhammer", "air conditioner", "street music", "engines", "drilling"]

    caption_all = [caption_list1,caption_list2,caption_list3,caption_list4]

    label_index = label_list.index(label)

    label_list = random.choice([0, 1, 2, 3])

    caption = caption_all[label_list][label_index]

    return caption


def get_esc(label):


    label_list = ['dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock', 'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane', 'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops', 'church_bells', 'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow', 'brushing_teeth', 'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects', 'laughing', 'hen', 'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing', 'glass_breaking', 'snoring', 'toilet_flush', 'pig', 'washing_machine', 'clock_tick', 'sneezing', 'rooster', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets']

    caption_list1 = ['a dog is barking', 'birds', 'vacuum cleaner', 'thunderstorm', 'a person is koncking on the wood door', 'a person is opening the can', 'a crow is grunting', 'a person is clapping', 'fireworks', 'a person is using the chainsaw', 'airplane', 'a man clicks the mouse', 'water', 'a train is passing through', 'a sheep is baas', 'water drops', 'church bells', 'clock alarm is ringing ', 'a person is typing the keyboard', 'winds', 'a person is walking', 'frog is croaking', 'a cow is mooing', 'a person is brushing teeth', 'the car is horning', 'fire crackling', 'helicopters', 'water', 'rains', 'a insect is flying', 'a person is laughing', 'a hen is crowing', 'engines', 'a person is breathing', 'a baby is crying', 'a person is using hand saw', 'a person is is coughing', 'a man is breaking the glass', 'snoring', 'a person is flush the toliet', 'a pig is oinking', ' washing machine', 'a clock is ticking', 'a person is sneezing', 'a rooster is crowing', 'sea waves', 'siren', 'cat is meowing', 'wood door is creaking', 'crickets']

    caption_list2 = ['a big dog is barking', 'birds', 'vacuum cleaner', 'thunderstorm', 'a man is koncking on the door', 'the can is opened', 'a crow is mooing ', 'a man is clapping', 'fireworks', 'a man is using the chainsaw', 'airplane', 'a person is clicks the mouse', 'water', 'a train is running', 'a sheep is baaing', 'water drops', 'church bell the belling', 'clock alarm is alarming ', 'a man is typing the keyboard', 'winds', 'a man is walking', 'frog is croaking', 'a cow is mooing', 'a man is brushing teeth', 'the car is horning', 'fire crackling', 'helicopters', 'water', 'rains', 'a is insect is flying', 'a person is laughing', 'a hen is crowing', 'engines', 'a person is breathing', 'a baby is crying', 'a person is using hand saw', 'a man is coughing', 'a person is breaking the glass', 'snoring', 'a person is flush the toliet', 'a pig is oinking', ' washing machine', 'a clock is ticking', 'a person is sneezing', 'a rooster is crowing', 'sea waves', 'siren', 'a cat is meowing', 'a wood door is creaking', 'crickets']

    caption_list3 = ['a small dog is barking', 'birds', 'vacuum cleaner', 'thunderstorm', 'a person is the knocking on the wood door', 'a person is opening the can', 'a crow is mooing', 'a man is clapping hand', 'fireworks', 'a chainsaw is running', 'airplane', 'a man is clicking the mouse', 'water', 'a train is operating', 'a sheep is baaing', 'water drops', 'church bell the ringing', 'clock alarm is ringing ', 'a person is typing the keyboard', 'winds', 'a boy is walking', 'frog is ribbitting', 'a cow the mooing', 'a person is brushing teeth', 'the car the horning', 'fire crackling', 'helicopters', 'water', 'rains', 'a insect is flying', 'a person is laughing', 'a hen is crowing', 'engines', 'a person is breathing', 'a baby is crying', 'a person is using hand saw', 'a man is coughing', 'a man is breaking the glass', 'snoring', 'a man is flush the toliet', 'a pig is gurnting', ' washing machine', 'a clock is ticking', 'a person is sneezing', 'a rooster is crowing', 'sea waves', 'siren', 'a cat is meowing', 'a wood door is opening', 'crickets']

    caption_list4 = ['a dog is whinning', 'birds', 'vacuum cleaner', 'thunderstorm', 'a person is knocking', 'a person is opening the can', 'a crow is mooing', 'a man is clapping', 'fireworks', 'a chainsaw is starting', 'airplane', 'a person is clicking the mouse', 'water', 'a train is coming through', 'a sheep is baaing', 'water drops', 'church bell the ringing', 'clock alarm is alarming ', 'a man is typing the keyboard', 'winds', 'a girl is walking', 'frog is ribbitting', 'a cow the mooing', 'a boy is brushing teeth', 'the car is horning', 'fire crackling', 'helicopters', 'water', 'rains', 'a insect is flying', 'a person is laughing', 'a hen is crowing', 'engines', 'a person is breathing', 'a baby crying', 'a person is using hand saw', 'a person is is coughing', 'a person is breaking the glass', 'snoring', 'a boy is flush the toliet', 'a pig is gurnting', ' washing machine', 'a clock is ticking', 'a person is sneezing', 'a rooster is crowing', 'sea waves', 'siren', 'a cat is meowing', 'a wood door is opening creaks', 'crickets']

    caption_all = [caption_list1,caption_list2,caption_list3,caption_list4]

    label_index = label_list.index(label)

    label_list = random.choice([0, 1, 2, 3])

    caption = caption_all[label_list][label_index]

    return caption

def mix_audio(waveform1,waveform2):
    # 获取waveform的长度（以样本为单位）

    # ipdb.set_trace()
    len1 = waveform1.shape[1]
    len2 = waveform2.shape[1]

    # 如果waveform1较短，则交换两个waveform，确保waveform1是较长的那个
    if len1 < len2:
        waveform1, waveform2 = waveform2, waveform1
        len1, len2 = len2, len1

    # 计算随机插入的开始位置
    start = np.random.randint(0, len1 - len2 + 1)
    
    # 创建一个与waveform1长度相同的空波形
    mixed_waveform = torch.zeros_like(waveform1)
    
    # 将waveform1复制到mixed_waveform
    mixed_waveform[:, :len1] = waveform1
    
    # 在随机选定的开始位置插入waveform2
    mixed_waveform[:, start:start+len2] += waveform2

    # 确保混合后的波形不超过原始波形的动态范围
    mixed_waveform = torch.clamp(mixed_waveform, min=-1.0, max=1.0)
    return mixed_waveform


def get_mixed(save_waveform,cur_length,cur_label, data):

    select = torch.rand(1)

    same_list = [" and ", " while "," at the same time "," and ", " while "]

    # print("selecting the mixed")

    if select >=0.8:


        next_data = random.choice(data)

        # while next_data["length"] > cur_length or next_data["label"] == cur_label:
        while next_data["length"] > cur_length:
            next_data = random.choice(data)

        next_wav = next_data["wav"]
        next_caption = next_data["caption"]
        next_length = next_data["length"]
        next_label = next_data["label"]
        next_waveform, next_length  = load_audio(next_wav)

        new_waveform = mix_audio(save_waveform,next_waveform)

        # print("generating mixed caption and waveform")

        return new_waveform,next_label
    else:
        # print("selected passed")
        return save_waveform,None

def get_mix_retrival(wav_list,label_list,length_list,data):

    caption_list,waveform_list = [],[]

    connect_list = [" and "," followed by "," then "," follow by "," and then "," followed by "," then "," and then followed by "]

    same_list = [" and ", " while "," at the same time "," and ", " while "]


    my_list = [j for j in range(len(wav_list))]

    my_before = [j for j in range(len(wav_list))]

    if len(my_list) ==2 :
        choice_list = [1]
    elif len(my_list) == 3 :
        choice_list = [1,2,2,2]
    else:
        choice_list = [2,2,2,3]

    all_remains = []

    for se in range(2):

        remain_list = random.sample(my_list, random.choice(choice_list))

        my_before = [x for x in my_before if x not in remain_list]

        all_remains.append(remain_list)
    
    remain_list = random.sample(my_list, random.choice(choice_list))

    my_before = [x for x in my_before if x not in remain_list]

    if len(my_before) > 0:
        if len(my_before) > 1:
            remain_list = my_before
        else:
            remain_list[0] = my_before[0]


    all_remains.append(remain_list)

    for re in range(len(all_remains)):

        all_wav = []
        all_label = []
        sh_list = []
        all_length = []
        total_length = 0

        cur_remain = all_remains[re]

        # ipdb.set_trace()

        for each in cur_remain:
            all_wav.append(wav_list[each])
            sh_list.append(wav_list[each])
            all_label.append(label_list[each])
            all_length.append(length_list[each])
            total_length += length_list[each]


        while total_length <10:
            if total_length > 9: 
                break
            next_data = random.choice(data)

            while next_data["label"] in all_label:
                next_data = random.choice(data)

            while next_data["length"] + total_length > 10:
                next_data = random.choice(data)

            next_length = next_data["length"]
            
            next_label = next_data["label"]
            next_wav = next_data["wav"]
            next_caption = next_data["caption"]
            # next_waveform = load_audio(next_wav)

            # if next_length < 1:
            #     next_length = 1
            #     try:
            #         next_waveform = wav_longer(next_waveform,1)
            #     except:
            #         ipdb.set_trace()
            total_length += next_length
            all_label.append(next_label)
            all_wav.append(os.path.join(next_wav.split("/")[-2],next_wav.split("/")[-1]))
            sh_list.append(os.path.join(next_wav.split("/")[-2],next_wav.split("/")[-1]))
            all_length.append(next_length)

        

        random.shuffle(sh_list)

        
        temp_label = []
        save_caption = None
        save_waveform = None
        for each_wav in sh_list:
            each_label = all_label[all_wav.index(each_wav)]
            temp_label.append(each_label)

            temp_waveform,temp_length= load_audio(each_wav)
            extra_label = None
            if temp_length >= 4:
                temp_waveform,extra_label = get_mixed(temp_waveform,temp_length,each_label,data)


            # if temp_length < 1:
            #     temp_length = 1
            #     try:
            #         temp_waveform = wav_longer(temp_waveform,1)
            #     except:
            #         ipdb.set_trace()
            if save_waveform is None:
                save_waveform = temp_waveform
            else:
                save_waveform  = torch.cat([save_waveform ,temp_waveform],dim = 1)

            if save_caption is None:
                save_caption = find_caption(each_label)
            else:
                save_caption = save_caption + random.choice(connect_list) + find_caption(each_label)

            if extra_label is not None:
                save_caption = save_caption + random.choice(same_list) + find_caption(extra_label)

        # print("all labels is",all_label)
        # print("the caption is",save_caption)
        # ipdb.set_trace()


        save_waveform = wav_longer(save_waveform,10)

        caption_list.append(save_caption)
        waveform_list.append(save_waveform)

    return waveform_list,caption_list


class AudioDataset(Dataset):
    def __init__(
        self,
        config=None,
        split="train",
        waveform_only=False,
        add_ons = [],
        new_json = True,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.config = config
        if "drum" in config["data"]["test"]:
            self.bigvgan = True
            print("using bigvgan code for mel")
        else:
            self.bigvgan = False
        self.split = split
        self.pad_wav_start_sample = 0 # If none, random choose
        self.trim_wav = False
        self.waveform_only = waveform_only
        self.metadata_root = load_json(self.config["metadata_root"])
        self.dataset_name = self.config["data"][self.split]
        self.add_ons = [eval(x) for x in add_ons]
        self.new_json = new_json
        self.fbank_shift = None
        self.data_mean = None
        self.data_std = None
        self.mix_data = None
        self.data_root = "/mnt/fast/datasets/audio"
        print("Add-ons:", self.add_ons)
        assert split in self.config["data"].keys(), (
            "The dataset split %s you specified is not present in the config. You can choose from %s"
            % (split, self.config["data"].keys())
        )

        self.build_setting_parameters()
        self.build_dataset()
        self.build_dsp()
        self.build_id_to_label()
        self.label_num = len(self.index_dict)
        if self.config["data"]["retrival"]:
            self.retrival = True
            self.re_num = self.config["data"]["re_num"]
            if self.config["data"]["data_root"]:
                self.data_root = self.config["data"]["data_root"]
            
        else:
            self.retrival=False
            self.re_num=0

        if self.config["data"]["clap_trainable"]:
            print("geting trainalbe data for clap")
            self.get_list = True
            self.get_num = self.config["data"]["ge_num"]
        else:
            self.get_list = False
            self.get_num = 0

        try:
            self.fbank_shift = self.config["data"]["fbank_shift"]
            print(f"mel-spectrogram value shifting for {self.fbank_shift}")
        except:
            pass
        
        try:
            self.mix_audio = self.config["data"]["mix_audio"]
            if "train" in self.config["data"]["mix_train"]:
                self.mix_data = self.data
            else:
                self.mix_data = [json.loads(line) for line in open(self.get_dataset_metadata_path(self.config["data"]["mix_train"], key=self.split), 'r')]
            # ipdb.set_trace()
        except:
            self.mix_audio = False
            self.mix_data  = []
        # ipdb.set_trace()

        try:
            self.dcase = self.config["data"]["dcase"]
            self.dcase_data = []
            for each_back in self.config["data"]["dcase_back"]:
                self.dcase_data += [json.loads(line) for line in open(each_back, 'r')]

            self.dcase_length = len(self.dcase_data)
            print("using background data for dacse, data length is",self.dcase_length)
        except:
            self.dcase = False
            self.dcase_data = []
        self.mix_length = len(self.mix_data)

        try:
            self.data_mean = self.config["data"]["data_mean"]
            self.data_std = self.config["data"]["data_std"]
            print(f"data get norm with mean {self.data_mean} and std {self.data_std}")
        except:
            pass
        print("Dataset initialize finished")

    def __getitem__(self, index):
        (
            fname,
            waveform,
            stft,
            log_mel_spec,
            label_vector,  # the one-hot representation of the audio class
            # the metadata of the sampled audio file and the mixup audio file (if exist)
            (datum, mix_datum),
            random_start,
            ret_list,
            cur_ret,
            name_list,
            cap_list,
            vae_list,
            fcaption,
            cur_label,
        ) = self.feature_extraction(index,self.retrival,self.re_num)

        if self.fbank_shift:
            log_mel_spec = log_mel_spec+self.fbank_shift

        if self.data_mean:
            log_mel_spec = (log_mel_spec - self.data_mean) / self.data_std


        # text = self.get_sample_text_caption(datum, mix_datum, label_vector)
        text = datum["caption"]

        data = {
            "fname": fname,  # list
            "text": text,  # list
            # tensor, [batchsize, class_num]
            "label_vector": label_vector.float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if(stft is None) else stft.float(),  
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if(log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
            "retrival_fbank":ret_list,
            "cur_fbank":cur_ret,
            "vae_list":vae_list
        }

        if self.dcase:

            mixed_waveform, mixed_mel,mixed_text  = self.get_dcase_mixed(fname,waveform.numpy()[0],text)

            # torchaudio.save("temp.wav", mixed_waveform, 32000)
            # ipdb.set_trace()
            data = {
            "fname": fname,  # list
            "text": mixed_text,  # list
            # tensor, [batchsize, class_num]
            "label_vector": label_vector.float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": mixed_waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if(stft is None) else stft.float(),  
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": mixed_mel.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
            "retrival_fbank":ret_list,
            "cur_fbank":cur_ret,
            "vae_list":vae_list
        }

        if self.mix_audio:

            mixed_waveform, mixed_mel  = self.get_audio_mixed(fname,waveform.numpy()[0],cur_label)
            data = {
            "fname": fname,  # list
            "text": text,  # list
            # tensor, [batchsize, class_num]
            "label_vector": label_vector.float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if(stft is None) else stft.float(),  
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if(log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
            "retrival_fbank":ret_list,
            "cur_fbank":cur_ret,
            "vae_list":vae_list,
            "mixed_waveform":mixed_waveform.float(),
            "mixed_mel":mixed_mel.float(),
            "caption":fcaption,
        }


        if self.get_list:
            
            data = {
            "fname": fname,  # list
            "text": text,  # list
            # tensor, [batchsize, class_num]
            "label_vector": label_vector.float(),
            # tensor, [batchsize, 1, samples_num]
            "waveform": waveform.float(),
            # tensor, [batchsize, t-steps, f-bins]
            "stft": "" if(stft is None) else stft.float(),  
            # tensor, [batchsize, t-steps, mel-bins]
            "log_mel_spec": "" if(log_mel_spec is None) else log_mel_spec.float(),
            "duration": self.duration,
            "sampling_rate": self.sampling_rate,
            "random_start_sample_in_original_audio_file": random_start,
            "retrival_fbank":ret_list,
            "cur_fbank":cur_ret,
            "tname_list":name_list,
            "tcaps_list":cap_list,
            "vae_list":vae_list,
            "caption":fcaption,
        }
            
        # print("the tname_list is",name_list)
        # print("the tcaps_list is",cap_list)
        # print(f"all_name is {name_list} and all_cap is {cap_list}")
        

        for add_on in self.add_ons:
            data.update(add_on(self.config, data, self.data[index]))
        return data

    def get_dataset_root_path(self, dataset):
        assert dataset in self.metadata_root.keys()
        return self.metadata_root[dataset]
    
    def get_dataset_metadata_path(self, dataset, key):
        # key: train, test, val, class_label_indices
        # pdb.set_trace()
        try:
            if dataset in self.metadata_root["metadata"]["path"].keys():
                return self.metadata_root["metadata"]["path"][dataset][key]
            # pdb.set_trace()
        except:
            raise ValueError("Dataset %s does not metadata \"%s\" specified" % (dataset, key))
        
        
            # return None

    def __len__(self):
        return len(self.data)

    def get_audio_mixed(self,fname,waveform,cur_label):

        select = torch.randint(0, self.mix_length, (1,)).item()

        selected_mix = self.mix_data[select]["wav"]

        try:
            noise_waveform, random_start = self.read_wav_file(selected_mix)
            # log_mel_spec, stft, mix_lambda, noise_waveform, random_start = self.read_audio_file(selected_mix)
        except:
            select = torch.randint(0, self.mix_length, (1,)).item()
            selected_mix = self.mix_data[select]["wav"]
            noise_waveform, random_start = self.read_wav_file(selected_mix)
        mixed_error = True

        while mixed_error:


            label_match = False
            mixed_label = self.mix_data[select]["label"]
            for each in mixed_label.split(","):
                if each in cur_label:
                    label_match = True
                    break

            

            while (len(noise_waveform[0]) < 1) + (os.path.basename(selected_mix) == os.path.basename(fname)) + (sum(noise_waveform[0])==0)+ label_match >0:
                select = torch.randint(0, self.mix_length, (1,)).item()
                selected_mix = self.mix_data[select]["wav"]

                try:
                    noise_waveform, random_start = self.read_wav_file(selected_mix)
                except:
                    select = torch.randint(0, self.mix_length, (1,)).item()
                    selected_mix = self.mix_data[select]["wav"]
                    noise_waveform, random_start = self.read_wav_file(selected_mix)

                label_match = False
                mixed_label = self.mix_data[select]["label"]
                for each in mixed_label.split(","):
                    if each in cur_label:
                        label_match = True
                        break

            noise_waveform = noise_waveform[0][:len(waveform)]

            # create audio mixture with a specific SNR level
            source_power = np.mean(waveform ** 2)
            noise_power = np.mean(noise_waveform ** 2)

            snr = torch.randint(-15, 15, (1,)).item()
            desired_noise_power = source_power / (10 ** (snr / 10))
            scaling_factor = np.sqrt(desired_noise_power / noise_power)
            noise_waveform = noise_waveform * scaling_factor

            mixed_waveform = waveform + noise_waveform

            try:
                mixed_mel, stft = self.wav_feature_extraction(mixed_waveform.reshape(1,-1))
                mixed_error = False
            except:
                print("error on mixing with file",selected_mix)
                mixed_error = True

        return torch.from_numpy(mixed_waveform.reshape(1,-1)),mixed_mel 


    def get_dcase_mixed(self,fname,waveform,text):

        select = torch.randint(0, self.dcase_length, (1,)).item()

        back_data = self.dcase_data[select]

        back_wav = back_data["wav"]

        mixed_error = True

        while mixed_error:
            try:
                back_waveform, random_start = self.read_wav_file(back_wav)
                # log_mel_spec, stft, mix_lambda, noise_waveform, random_start = self.read_audio_file(selected_mix)
            except:
                select = torch.randint(0, self.dcase_length, (1,)).item()
                back_data = self.dcase_data[select]
                back_wav = back_data["wav"]
                back_waveform, random_start = self.read_wav_file(back_wav)


            back_label = back_data["label"]


            if "ub8k" in back_label:
                back_caption = get_ub8k(back_data["caption"])
            else:
                back_caption = get_esc(back_data["caption"])

            if text[-1] == ".":
                text = text[:-1]

            new_caption = text.lower() + " with " + back_caption + " in the background. "

            

            noise_waveform = back_waveform[0][:len(waveform)]

                # create audio mixture with a specific SNR level
            source_power = np.mean(waveform ** 2)
            noise_power = np.mean(noise_waveform ** 2)

            snr = torch.randint(-15, 15, (1,)).item()
            desired_noise_power = source_power / (10 ** (snr / 10))
            scaling_factor = np.sqrt(desired_noise_power / noise_power)
            noise_reduction_factor = 0.4
            noise_waveform = noise_waveform * scaling_factor * noise_reduction_factor

            mixed_waveform = waveform + noise_waveform

            try:
                mixed_mel, stft = self.wav_feature_extraction(mixed_waveform.reshape(1,-1))
                mixed_error = False
            except:
                print("error on mixing with file",os.path.basename(back_wav))
                mixed_error = True

        if new_caption[-1] == ".":
            new_caption = new_caption[:-1]

        return torch.from_numpy(mixed_waveform.reshape(1,-1)),mixed_mel,new_caption


    def feature_extraction(self, index,retrival=False,re_num=0):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                label_indices = np.zeros(self.label_num, dtype=np.float32)
                # if random.random() < self.mixup:

                datum = self.data[index]
                log_mel_spec, stft, mix_lambda, waveform, random_start = self.read_audio_file(
                    datum["wav"]
                )
                mix_datum = None
                if(self.label_num > 0 and "labels" in datum.keys()):
                    for label_str in datum["labels"].split(","):
                        try:
                            label_indices[int(self.index_dict[label_str])] = 1.0
                        except:
                            pass
                
                # If the key "label" is not in the metadata, return all zero vector
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                print("Error encounter during audio feature extraction: ", e, datum["wav"])
                continue

        # The filename of the wav file
        fname = datum["wav"]
        fcaption = datum["caption"]
        # t_step = log_mel_spec.size(0)
        # waveform = torch.FloatTensor(waveform[..., : int(self.hopsize * t_step)])
        waveform = torch.FloatTensor(waveform)

        retrival_list = []

        vae_list = []

        # if not retrival:
        cur_retrival = []

        if retrival:
            try:
                retrival_list = datum["score_list"]
                retrival_list = retrival_list[:re_num+1]
                dir_path = os.path.dirname(fname)
                waves = []
                vaes = []
                norm_mean= -4.2677393
                norm_std= 4.5689974
                for retrival_name in retrival_list:
                    if self.get_list:
                        if len(retrival_name[0].split("/"))>4:
                            retrival_file = retrival_name[0]
                        else :
                            retrival_file = os.path.join(self.data_root,retrival_name[0])
                            if retrival_file[-3:]!= "wav":
                                retrival_file = retrival_file+".wav"
                            if os.path.exists(retrival_file):
                                pass
                            else:
                                new_path = "/mnt/bn/lqhaoheliu/datasets/audiocaps/audios/test"
                                retrival_file  = os.path.join(new_path,retrival_name[0])
                                if os.path.exists(retrival_file):
                                    pass
                                else:
                                    new_path = "/mnt/bn/lqhaoheliu/datasets/audiocaps/audios/val"
                                    retrival_file  = os.path.join(new_path,retrival_name[0])
                                    if os.path.exists(retrival_file):
                                        pass
                                    else:
                                        ipdb.set_trace()
                    else:
                        retrival_file = os.path.join(dir_path,retrival_name)
                    try:
                        vae_fbank = self.read_audio_file(retrival_file)[0]
                    except:
                        ipdb.set_trace()

                    vaes.append(vae_fbank.reshape(1,-1,64))
                    retrival_wav = self.read_wav_file(retrival_file)

                    retrival_wav = retrival_wav[0].reshape(1,-1)

                    fbank = torchaudio.compliance.kaldi.fbank(torch.FloatTensor(retrival_wav), htk_compat=True, sample_frequency=16000, use_energy=False,
                                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
                    log_mel_spec_hifigan = log_mel_spec

                    TARGET_LEN = log_mel_spec_hifigan.size(0)
                    
                    n_frames = fbank.shape[0]
                    p = TARGET_LEN - n_frames
                    if p > 0:
                        m = torch.nn.ZeroPad2d((0, 0, 0, p))
                        fbank = m(fbank)
                    elif p < 0:
                        fbank = fbank[:TARGET_LEN, :]

                    fbank = (fbank - norm_mean) / (norm_std * 2)


                    waves.append(fbank.reshape(1,-1,128))


                vae_list = torch.cat(vaes[1:])

                retrival_list = torch.cat(waves)
                cur_retrival = retrival_list[0:1]
                if len(retrival_list==3):
                    retrival_list = retrival_list[1:]
                else:
                    retrival_list = retrival_list[1:]
            except: 
                try:
                    wav_list = datum["wav_list"]
                except:
                    ipdb.set_trace()


                retrival_list = torch.zeros(re_num,1024,128)

                cur_retrival = torch.zeros(1,1024,128)

                

                if self.mix_data is None:
                    data_path_2 = "/mnt/bn/arnold-yy-audiodata/audio_data/urbansound8K/UrbanSound8K/urbansound_data_wcap.json"
                    data_path_1 = "/mnt/bn/arnold-yy-audiodata/audio_data/esc50/esc_50_2k_caps.json"
                    data_1 = [json.loads(line) for line in open(data_path_1, 'r')]
                    data_2 = [json.loads(line) for line in open(data_path_2, 'r')]
                    self.mix_data = data_1+data_2
                    random.shuffle(self.mix_data)

                label_list = datum["label"]

                length_list = datum["list"]

                mix_retrival_wavs,mix_retrival_captions = get_mix_retrival(wav_list,label_list,length_list,self.mix_data)

                vaes = []
                for mix_wav in mix_retrival_wavs:
                    vae_fbank = self.wav_feature_extraction(mix_wav)[0]
                    vaes.append(vae_fbank.reshape(1,-1,64))
                vae_list = torch.cat(vaes)

                # ipdb.set_trace()

                    


        all_name=[]
        all_cap=[]
        if self.get_list:
            try:
                temp_list = datum["score_list"]
                
                temp_length = len(temp_list)
                # print(f"the temp_list length is {temp_length} and list is {temp_list}")
                if temp_length< self.get_num:
                    print("num is bigger than the retrival dataset")
                    self.get_num = temp_length
                if temp_length <=3:
                    for i in range(self.get_num):
                        audio_name = temp_list[i][0]
                        if len(audio_name.split("/"))>9:
                            retrival_file = audio_name[0]
                        elif 6>len(audio_name.split("/"))>2:
                            retrival_file = os.path.join(self.data_root,audio_name[0])
                            retrival_file = retrival_file+".wav"
                        else:
                            retrival_file = os.path.join(dir_path,audio_name[0])
                        all_name.append(retrival_file)
                        all_cap.append(temp_list[i][1])
                else:

                    for i in range(1,self.get_num+1):
                        audio_name = temp_list[i][0]
                        if len(audio_name.split("/"))>9:
                            retrival_file = audio_name
                        elif 6>len(audio_name.split("/"))>2:
                            retrival_file = os.path.join(self.data_root,audio_name)
                            retrival_file = retrival_file+".wav"
                        else:
                            retrival_file = os.path.join(dir_path,audio_name)
                        all_name.append(retrival_file)
                        all_cap.append(temp_list[i][1])
            except:
                try:
                    all_cap = mix_retrival_captions

                    for i in range(len(mix_retrival_captions)):

                        all_name.append("temp_file.wav")
                except:
                    ipdb.set_trace()



        return (fname, waveform, stft, log_mel_spec, label_indices, (datum, mix_datum), random_start,retrival_list,cur_retrival,all_name,all_cap,vae_list,fcaption,datum["label"])


            # return (fname, waveform, stft, log_mel_spec, label_indices, (datum, mix_datum), random_start)

    # def augmentation(self, log_mel_spec):
    #     assert torch.min(log_mel_spec) < 0
    #     log_mel_spec = log_mel_spec.exp()

    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     # this is just to satisfy new torchaudio version.
    #     log_mel_spec = log_mel_spec.unsqueeze(0)
    #     if self.freqm != 0:
    #         log_mel_spec = self.frequency_masking(log_mel_spec, self.freqm)
    #     if self.timem != 0:
    #         log_mel_spec = self.time_masking(
    #             log_mel_spec, self.timem)  # self.timem=0

    #     log_mel_spec = (log_mel_spec + 1e-7).log()
    #     # squeeze back
    #     log_mel_spec = log_mel_spec.squeeze(0)
    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     return log_mel_spec

    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        # self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        # self.timem = self.config["preprocessing"]["mel"]["timem"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]

        # Calculate parameter derivations
        # self.waveform_sample_length = int(self.target_length * self.hopsize)

        # if (self.config["balance_sampling_weight"]):
        #     self.samples_weight = np.loadtxt(
        #         self.config["balance_sampling_weight"], delimiter=","
        #     )

        if "train" not in self.split:
            self.mixup = 0.0
            # self.freqm = 0
            # self.timem = 0

    def _relative_path_to_absolute_path(self, metadata, dataset_name):
        root_path = self.get_dataset_root_path(dataset_name)
        for i in range(len(metadata["data"])):
            assert metadata["data"][i]["wav"][0] != "/", "The dataset metadata should only contain relative path to the audio file: %s" % metadata["data"][i]["wav"][0] 
            metadata["data"][i]["wav"] = os.path.join(
                root_path, metadata["data"][i]["wav"]
            )
        return metadata

    def build_dataset(self):
        self.data = []
        print("Build dataset split %s from %s" % (self.split, self.dataset_name))
        
        if type(self.dataset_name) is str:
            
            if "retrival" in self.dataset_name or self.new_json:
                data_json =[json.loads(line) for line in open(self.get_dataset_metadata_path(self.dataset_name, key=self.split), 'r')]
                self.data = data_json
            else:
                data_json = load_json(self.get_dataset_metadata_path(self.dataset_name, key=self.split))
                data_json = self._relative_path_to_absolute_path(data_json, self.dataset_name)
                self.data = data_json["data"]
        elif type(self.dataset_name) is list:
            for dataset_name in self.dataset_name:
                print("dataset_name is ",dataset_name)


                if "dcase_2024" in dataset_name:
                    data_json = [json.loads(line) for line in open(dataset_name, 'r')]
                    self.data+=data_json
                
                elif "retrival" in self.dataset_name or self.new_json:
                    data_json =[json.loads(line) for line in open(self.get_dataset_metadata_path(dataset_name, key=self.split), 'r')]
                    self.data +=data_json

                else:
                    data_json = load_json(self.get_dataset_metadata_path(dataset_name, key=self.split))
                    data_json = self._relative_path_to_absolute_path(data_json, dataset_name)
                    self.data += data_json["data"]
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

    def build_dsp(self):
        self.STFT = Audio.stft.TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )
        # self.stft_transform = torchaudio.transforms.Spectrogram(
        #     n_fft=1024, hop_length=160
        # )
        # self.melscale_transform = torchaudio.transforms.MelScale(
        #     sample_rate=16000, n_stft=1024 // 2 + 1, n_mels=64
        # )

    def build_id_to_label(self):
        id2label = {}
        id2num = {}
        num2label = {}
        class_label_indices_path = self.get_dataset_metadata_path(dataset = self.config["data"]["class_label_indices"], key = "class_label_indices")
        if(class_label_indices_path is not None):
            df = pd.read_csv(class_label_indices_path)
            for _, row in df.iterrows():
                index, mid, display_name = row["index"], row["mid"], row["display_name"]
                id2label[mid] = display_name
                id2num[mid] = index
                num2label[index] = display_name
            self.id2label, self.index_dict, self.num2label = id2label, id2num, num2label
        else:
            self.id2label, self.index_dict, self.num2label = {}, {}, {}

    def resample(self, waveform, sr):
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        # waveform = librosa.resample(waveform, sr, self.sampling_rate)
        return waveform

        # if sr == 16000:
        #     return waveform
        # if sr == 32000 and self.sampling_rate == 16000:
        #     waveform = waveform[::2]
        #     return waveform
        # if sr == 48000 and self.sampling_rate == 16000:
        #     waveform = waveform[::3]
        #     return waveform
        # else:
        #     raise ValueError(
        #         "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
        #         % (sr, self.sampling_rate)
        #     )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        random_start = int(
            self.random_uniform(0, waveform_length - target_length)
        )
        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if(self.pad_wav_start_sample is None):
            rand_start = int(
                self.random_uniform(0, target_length - waveform_length)
            )
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        waveform, random_start = self.random_segment_wav(waveform, target_length = int(sr * self.duration))

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)
        
        if(self.trim_wav):
            waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(waveform, target_length = int(self.sampling_rate * self.duration))
        return waveform, random_start

    def mix_two_waveforms(self, waveform1, waveform2):
        mix_lambda = np.random.beta(5, 5)
        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        return self.normalize_wav(mix_waveform), mix_lambda

    def read_audio_file(self, filename, filename2=None):

        waveform, random_start = self.read_wav_file(filename)
        mix_lambda = 0.0
        if(not self.waveform_only):
            log_mel_spec, stft = self.wav_feature_extraction(waveform)
        else:
            log_mel_spec, stft = None, None

        if self.bigvgan:
            log_mel_spec = wav2mel(filename,hop_length=self.config["preprocessing"]["stft"]["hop_length"],mel_channel=self.config["preprocessing"]["mel"]["n_mel_channels"])
        # ipdb.set_trace()


        return log_mel_spec, stft, mix_lambda, waveform, random_start

    def get_sample_text_caption(self, datum, mix_datum, label_indices):
        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += " " + self.label_indices_to_text(mix_datum, label_indices)
        return text

    # This one is significantly slower than "wav_feature_extraction_torchaudio" if num_worker > 1
    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # print("waveform shape is",waveform.shape)

        log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)
        # print("mel shape is",log_mel_spec.shape)
        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        # print("mel shape is",log_mel_spec.shape)
        return log_mel_spec, stft

    # @profile
    # def wav_feature_extraction_torchaudio(self, waveform):
    #     waveform = waveform[0, ...]
    #     waveform = torch.FloatTensor(waveform)

    #     stft = self.stft_transform(waveform)
    #     mel_spec = self.melscale_transform(stft)
    #     log_mel_spec = torch.log(mel_spec + 1e-7)

    #     log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    #     stft = torch.FloatTensor(stft.T)

    #     log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
    #     return log_mel_spec, stft

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def _read_datum_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        random_index = torch.randint(0,len(caption_keys),(1,))[0].item()
        return datum[caption_keys[random_index]]

    def _is_contain_caption(self, datum):
        caption_keys = [x for x in datum.keys() if ("caption" in x)]
        return len(caption_keys) > 0 

    def label_indices_to_text(self, datum, label_indices):
        if self._is_contain_caption(datum):
            return self._read_datum_caption(datum)
        elif "label" in datum.keys():
            name_indices = torch.where(label_indices > 0.1)[0]
            # description_header = "This audio contains the sound of "
            description_header = ""
            labels = ""
            for id, each in enumerate(name_indices):
                if id == len(name_indices) - 1:
                    labels += "%s." % self.num2label[int(each)]
                else:
                    labels += "%s, " % self.num2label[int(each)]
            return description_header + labels
        else:
            return "" # TODO, if both label and caption are not provided, return empty string

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec

if __name__ == "__main__":
    import torch
    from tqdm import tqdm

    from torch.utils.data import DataLoader

    def write_json(my_dict, fname):
        # print("Save json file at "+fname)
        json_str = json.dumps(my_dict)
        with open(fname, 'w') as json_file:
            json_file.write(json_str)

    def load_json(fname):
        with open(fname,'r') as f:
            data = json.load(f)
            return data

    config = yaml.load(
        open(
            "/mnt/bn/arnold-yy-audiodata/audioldm/audioldm2/new_clap_configs/retrival_audiocaps_witht5_vaeclap_pos1_3_0205_scale0.1_esc.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    add_ons = config["data"]["dataloader_add_ons"]

    # load_json(data)
    dataset = AudioDataset(config=config, split="train", waveform_only=False, add_ons=add_ons)

    loader = DataLoader(dataset, batch_size=4,shuffle=True, num_workers=16)

    loading = iter(loader)

    one  = next(loading)

    # for each in one.keys(): 
    #     try:
    #         print(f"{each} is {one[each].shape}")
            
    #     except:
    #         print(f"{each} is {len(one[each])}")
    print(one["fname"])
            

    ipdb.set_trace()

    for i in tqdm(range(10000)):
        two = next(loading)
    

    for cnt, each in tqdm(enumerate(loader)):
        # print(each["waveform"].size(), each["log_mel_spec"].size())
        # print(each['freq_energy_percentile'])
        import ipdb;ipdb.set_trace()
        # pass