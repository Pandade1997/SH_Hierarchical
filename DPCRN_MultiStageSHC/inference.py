# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:18 2020

@author: admin
"""
import shutil
import librosa
import spaudiopy
import torch
import os, fnmatch
import numpy as np
import soundfile as sf
from scipy import io
from tqdm import tqdm
from evaluation_fixed import NB_PESQ, STOI

from networks.DPCRN import DPCRN_NET, DPCRN_multi
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

device = 'cpu'


def audioread(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, sr, fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    return wave_data


def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def microphone_positions_spherical(cartesian_positions):
    positions_spherical = np.zeros(cartesian_positions.shape)
    for i, (x, y, z) in enumerate(cartesian_positions):
        positions_spherical[i] = cart2sph(x, y, z)
    return positions_spherical


def calculate_metrics(params):
    ref, mix, est = params
    return NB_PESQ(ref, mix), NB_PESQ(ref, est), STOI(ref, mix), STOI(ref, est)


def wav_generator(mix_path, ref_path, mic_path, sph_order):
    mix = audioread(mix_path)
    ref = audioread(ref_path)

    # 计算球谐系数：
    # 增加麦克风位置,读取麦克风位置信息
    mic_data = np.load(mic_path)  # <class 'tuple'>: (C, 3)

    # Compute the center of the array
    center = np.mean(mic_data, axis=0)
    # Subtract the center from each coordinate to get the new coordinates
    transformed_coords = mic_data - center

    mic_positions_spherical = microphone_positions_spherical(transformed_coords)
    colat = mic_positions_spherical[:, 2]  # 极角
    azi = mic_positions_spherical[:, 1]  # 方位角
    sh_type = 'real'

    coeffs_input = spaudiopy.sph.src_to_sh(mix.T, azi, colat, sph_order, sh_type)
    coeffs_target = spaudiopy.sph.src_to_sh(ref.T, azi, colat, sph_order, sh_type)

    input_1 = torch.tensor(np.float32(coeffs_input[0:4])).unsqueeze(0).to(device)
    input_2 = torch.tensor(np.float32(coeffs_input[4:9])).unsqueeze(0).to(device)
    input_3 = torch.tensor(np.float32(coeffs_input[9:16])).unsqueeze(0).to(device)
    input_4 = torch.tensor(np.float32(coeffs_input[16:25])).unsqueeze(0).to(device)

    target_1 = torch.tensor(np.float32(coeffs_target[0:4])).unsqueeze(0).to(device)
    target_2 = torch.tensor(np.float32(coeffs_target[0:9])).unsqueeze(0).to(device)
    target_3 = torch.tensor(np.float32(coeffs_target[0:16])).unsqueeze(0).to(device)
    target_4 = torch.tensor(np.float32(coeffs_target[0:25])).unsqueeze(0).to(device)

    load_network.eval()
    with torch.no_grad():
        _, _, _, outputs = load_network(input_1, input_2, input_3, input_4)
    est = outputs.squeeze().T.cpu().detach().numpy()

    est_single = spaudiopy.sph.inverse_sht(est.T, azi, colat, sh_type, N=sph_order, Y_nm=None)

    params = [(ref.T[k], mix.T[k], est_single[k]) for k in range(len(mix.T))]

    with Pool(processes=16) as pool:  # 8 is just an example, adjust according to your hardware
        results = pool.map(calculate_metrics, params)

    pesq_mix, pesq_est, stoi_mix, stoi_est = zip(*results)
    pesq_mix = np.mean(np.array(pesq_mix))
    pesq_est = np.mean(np.array(pesq_est))
    stoi_mix = np.mean(np.array(stoi_mix))
    stoi_est = np.mean(np.array(stoi_est))
    return pesq_mix, pesq_est, stoi_mix, stoi_est


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    # modelpath = 'model_dpcrn_multi/'
    modelpath = 'model_dpcrn_multi_position/'
    test_list = ['test_0.2_snr-10', 'test_0.2_snr-5', 'test_0.2_snr0', 'test_0.2_snr5', 'test_0.2_snr10',
                 'test_0.6_snr-10', 'test_0.6_snr-5', 'test_0.6_snr0', 'test_0.6_snr5', 'test_0.6_snr10',
                 'test_1.0_snr-10', 'test_1.0_snr-5', 'test_1.0_snr0', 'test_1.0_snr5', 'test_1.0_snr10',
                 ]
    sph_order = 4

    for test_name in test_list:
        test_wav_scp = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt/wav_scp_' + test_name + '.txt'
        wav_path = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/test/' + test_name + '/mix/'
        ref_dir = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/test/' + test_name + '/noreverb_ref/'
        mic_dir = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/RIR/cir_uniform_16/test_rir/MIC_' + test_name.split('_')[
            1]
        modelname = modelpath + 'model_best.pth'

        print("Processing test ..." + str(test_name))

        load_network = DPCRN_multi()

        new_state_dict = {}
        for k, v in torch.load(str(modelname), map_location=device).items():
            new_state_dict[k[7:]] = v  # 键值包含‘module.’ 则删除
        load_network.load_state_dict(new_state_dict, strict=False)
        load_network = load_network.to(device)

        # load_network = torch.nn.DataParallel(load_network)
        # load_network.load_state_dict(torch.load(modelname))
        # load_network = load_network.cuda()

        pesq_mix_list = []
        pesq_est_list = []
        stoi_mix_list = []
        stoi_est_list = []
        with open(test_wav_scp, 'r', encoding='utf-8') as infile:
            data = infile.readlines()
            for i in tqdm(range(len(data))):
                utt_id = data[i].strip("\n").split('/')[-1]
                mix_path = os.path.join(wav_path, utt_id)
                ref_path = os.path.join(ref_dir, utt_id)
                mic_path = os.path.join(mic_dir, 'mic_array_pos_' + utt_id.split('#')[2].split('_')[1] + '_' +
                                        utt_id.split('#')[2].split('_')[2])

                pesq_mix, pesq_est, stoi_mix, stoi_est = wav_generator(mix_path, ref_path, mic_path, sph_order)

                pesq_mix_list.append(pesq_mix)
                pesq_est_list.append(pesq_est)
                stoi_mix_list.append(stoi_mix)
                stoi_est_list.append(stoi_est)

        pesq_mix = np.mean(np.array(pesq_mix_list))
        pesq_est = np.mean(np.array(pesq_est_list))
        stoi_mix = np.mean(np.array(stoi_mix_list))
        stoi_est = np.mean(np.array(stoi_est_list))

        # print("model_best_pesq_result:")
        print(test_name + "_result:")
        print('pesq_mix:' + str(pesq_mix) + '   ' + 'pesq_est:' + str(pesq_est) + '   ' + 'stoi_mix:' + str(
            stoi_mix) + '   ' + 'stoi_est:' + str(stoi_est))

        res1path = modelpath + '/result_model_best/' + test_name
        if not os.path.isdir(res1path):
            os.makedirs(res1path)
        io.savemat(res1path + '_metrics.mat',
                   {'pesq_mix': pesq_mix, 'pesq_est': pesq_est, 'stoi_mix': stoi_mix, 'stoi_est': stoi_est})
