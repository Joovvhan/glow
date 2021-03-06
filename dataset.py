import random
from glob import glob
import librosa
from tqdm import tqdm
import numpy as np
import json

import torch

from functools import partial

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

def concat_variable_length_files(speech_files, anchor=6.05):
    speech_file_lengths = [(f, librosa.core.get_duration(filename=f)) for f in tqdm(speech_files)]
    speech_file_lengths.sort(key=lambda x: x[1]) 
    
    idx = np.argmin([abs(l[1] - anchor) for l in speech_file_lengths])
    
    if idx % 2 != 1:
        idx +=1 

    folding_files = speech_file_lengths[:idx]
    left_files = [(s[0], s[1]) for s in speech_file_lengths[idx:]]

    def fold(input_list):
        center = int((len(input_list) - 1) / 2)
        x = input_list[:center]
        y = input_list[center:][::-1]

        return [((a[0], b[0]), (a[1], b[1])) for a, b in zip(x, y)]

    folded_files = fold(folding_files)

    merge_files = folded_files + left_files
    
    # return merge_files
    return [f[0] for f in merge_files]


def load_wavs(wavs):
    if isinstance(wavs, str):
        wavs = [wavs]
    wavs = list(wavs) # change tuple to list to support shuffling
    random.shuffle(wavs)
        
    for i, wav in enumerate(wavs):
        _y, sr = librosa.core.load(wav, sr=22050, mono=True)
        if i == 0:
            y = _y
        else:
            y = np.concatenate([y, _y], axis=0)

    return y


def collate_function(files, h):

    mels = list()

    for file in files:
        y = torch.FloatTensor(load_wavs(file)).unsqueeze(0)
        mel = mel_spectrogram(y, n_fft=h['n_fft'], num_mels=h['num_mels'], 
                            sampling_rate=h['sampling_rate'], hop_size=h['hop_size'], 
                            win_size=h['win_size'], fmin=h['fmin'], fmax=h['fmax'])
        mel = mel.squeeze(0).T # [1, 80, 587] => [587, 80]

        mels.append(mel)

    mel_tensor = pad_sequence(mels, batch_first=True, padding_value=-10)

    # return mel_tensor # [B, T, 80]
    return torch.transpose(mel_tensor, 1, 2) # [B, T, 80]


def get_data_loader(speech_files, h, num_workers=4):

    dataloader = DataLoader(speech_files, batch_size=4, 
                            shuffle=True, num_workers=num_workers,
                            collate_fn=partial(collate_function, h=h))
    
    return dataloader


if __name__ == "__main__":
    speech_files = sorted(glob('./data/kss/*/*.wav'))
    before_folding = len(speech_files)
    speech_files = concat_variable_length_files(speech_files)
    # speech_files = [f[0] for f in speech_files]
    print(f'{before_folding:5} => {len(speech_files):5}')

    # dataloader = get_data_loader()

    # load_wavs()

    h = json.load(open('config.json', 'r'))
    # print(h)

    '''

    y = torch.FloatTensor(load_wavs(speech_files[0])).unsqueeze(0)
    mel = mel_spectrogram(y, n_fft=h['n_fft'], num_mels=h['num_mels'], 
                          sampling_rate=h['sampling_rate'], hop_size=h['hop_size'], 
                          win_size=h['win_size'], fmin=h['fmin'], fmax=h['fmax'])
    mel = mel.squeeze(0).T # [1, 80, 587] => [587, 80]

    '''

    # print(mel.shape)
    

    # plt.figure()
    # plt.imshow(mel[0, :, :].numpy())
    # plt.show()

    data_loader = get_data_loader(speech_files, h)

    for mels in data_loader:
        print(mels.shape)
        break

    plt.figure()
    plt.imshow(mels[0, :, :].numpy())
    plt.show()

# https://github.com/openai/glow