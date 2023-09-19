import numpy as np
import numpy
import math
import soundfile as sf
import scipy.signal as sps
import librosa
import random
import os

import spaudiopy
import torch
import torch as th

import torch.utils.data as tud
from scipy import signal
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

eps = np.finfo(np.float32).eps
ft_len = 512
ft_overlap = 256
channel = 16


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def parse_scp(scp, path_list):
    with open(scp) as fid:
        for line in fid:
            tmp = line.strip()
            path_list.append(tmp)


class FixDataset(Dataset):

    def __init__(self,
                 wav_scp,
                 mix_dir,
                 ref_dir,
                 mic_dir,
                 repeat=1,
                 chunk=4,
                 sample_rate=16000,
                 sph_order=4,
                 ):
        super(FixDataset, self).__init__()

        self.wav_list = list()
        parse_scp(wav_scp, self.wav_list)
        self.mix_dir = mix_dir
        self.ref_dir = ref_dir
        self.mic_dir = mic_dir
        self.segment_length = chunk * sample_rate
        self.wav_list *= repeat
        self.sph_order = sph_order

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        utt_id = self.wav_list[index]
        mix_path = utt_id
        ref_path = os.path.join(self.ref_dir, utt_id.split('/')[-1])

        # L x C
        mix = audioread(mix_path)  # <class 'tuple'>: (64000, 16)
        ref = audioread(ref_path)

        # 计算球谐系数：
        # 增加麦克风位置,读取麦克风位置信息
        mic_path = os.path.join(self.mic_dir, 'mic_array_pos_' + utt_id.split('#')[2].split('_')[1] + '_' +
                                utt_id.split('#')[2].split('_')[2])
        mic_data = np.load(mic_path)  # <class 'tuple'>: (C, 3)

        # Compute the center of the array
        center = np.mean(mic_data, axis=0)
        # Subtract the center from each coordinate to get the new coordinates
        transformed_coords = mic_data - center

        mic_positions_spherical = self.microphone_positions_spherical(transformed_coords)
        colat = mic_positions_spherical[:, 2]  # 极角
        azi = mic_positions_spherical[:, 1]  # 方位角
        sh_type = 'real'

        coeffs_input = spaudiopy.sph.src_to_sh(mix.T, azi, colat, self.sph_order, sh_type)
        coeffs_target = spaudiopy.sph.src_to_sh(ref.T, azi, colat, self.sph_order, sh_type)

        input_1 = np.float32(coeffs_input[0:4])
        input_2 = np.float32(coeffs_input[4:9])
        input_3 = np.float32(coeffs_input[9:16])
        input_4 = np.float32(coeffs_input[16:25])

        target_1 = np.float32(coeffs_target[0:4])
        target_2 = np.float32(coeffs_target[0:9])
        target_3 = np.float32(coeffs_target[0:16])
        target_4 = np.float32(coeffs_target[0:25])

        egs = {
            "input": np.float32(mix.T),
            "input_1": input_1,
            "input_2": input_2,
            "input_3": input_3,
            "input_4": input_4,
            "target": np.float32(ref.T),
            "target_1": target_1,
            "target_2": target_2,
            "target_3": target_3,
            "target_4": target_4,

        }
        return egs

    def cart2sph(self, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return r, theta, phi

    def microphone_positions_spherical(self, cartesian_positions):
        positions_spherical = np.zeros(cartesian_positions.shape)
        for i, (x, y, z) in enumerate(cartesian_positions):
            positions_spherical[i] = self.cart2sph(x, y, z)
        return positions_spherical


def make_fix_loader(wav_scp, mix_dir, ref_dir, mic_dir, batch_size=8, repeat=1, num_workers=16,
                    chunk=4, sample_rate=16000, sph_order=4):
    dataset = FixDataset(
        wav_scp=wav_scp,
        mix_dir=mix_dir,
        ref_dir=ref_dir,
        mic_dir=mic_dir,
        repeat=repeat,
        chunk=chunk,
        sample_rate=sample_rate,
        sph_order=sph_order,
    )

    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
    )
    return loader
