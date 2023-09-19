import os
import shutil
import random
import scipy.io


def move(dir_A, dir_B, percent):
    # 如果目录 B 不存在，则创建目录 B
    if not os.path.exists(dir_B):
        os.makedirs(dir_B)

    # 获取目录 A 中所有 .wav 文件的路径
    wav_files = [f for f in os.listdir(dir_A) if f.endswith('.WAV')]

    # 计算要提取的文件数
    num_files_to_extract = int(len(wav_files) * percent)

    # 随机选择要提取的文件
    files_to_extract = random.sample(wav_files, num_files_to_extract)

    # 将选定的文件移动到目录 B
    for file_name in files_to_extract:
        file_path = os.path.join(dir_A, file_name)
        dest_path = os.path.join(dir_B, file_name)
        shutil.move(file_path, dest_path)
    return True


def split_txt(input_path, part):
    with open(input_path, 'r') as input_file:
        lines = input_file.readlines()
        num_lines = len(lines)
        chunk_size = num_lines / part
        name = -15
        for i in range(part):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size
            if i == part - 1:
                end_index = num_lines
            chunk = lines[start_index:end_index]
            name += 5
            with open(f'/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt/test_clean_snr{name}_list.txt',
                      'w') as output_file:
                output_file.writelines(chunk)


def gen_wav_scp(file_dir, txt_path):
    file_paths = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    with open(txt_path, "w") as f:
        for file_path in file_paths:
            f.write(file_path + "\n")


def compare_folders(folder1, folder2, folder3):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))

    # 找到不一致的文件
    diff_files = (files1 ^ files2) | (files1 ^ files3) | (files2 ^ files3)

    for file in diff_files:
        file_path1 = os.path.join(folder1, file)
        file_path2 = os.path.join(folder2, file)
        file_path3 = os.path.join(folder3, file)

        # 删除不一致的文件
        if os.path.exists(file_path1):
            print(file_path1)
            os.remove(file_path1)
        if os.path.exists(file_path2):
            print(file_path2)
            os.remove(file_path2)
        if os.path.exists(file_path3):
            print(file_path3)
            os.remove(file_path3)
    return True


def merge_files(folder_path, output_file):
    rir_list = ['1.0', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    snr_list = ['-10', '-5', '0', '5', '10']

    merged_content = ""
    for snr in snr_list:
        for rir in rir_list:
            txt_path = folder_path + 'wav_scp_test_' + rir + '_snr' + snr + '.txt'
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as file:
                    file_content = file.read()
                    merged_content += file_content
    with open(output_file, "w") as merged_file:
        merged_file.write(merged_content)
    return True


if __name__ == '__main__':
    # 定义目录 A 和目录 B 的路径
    # dir_A = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/Data/train_val_clean/'
    # dir_B = '/ddnstor/imu_panjiahui/dataset/SHdomain/TIMIT/Data/val_clean/'

    # 1、按照比例移动A的内容到B中
    # 定义要提取的文件比例（这里是 10%）
    # percent = 0.1
    # move(dir_A, dir_B, percent)

    # 2、移动目录A 到目录B中
    # 使用 shutil.move() 函数将目录 A 移动到目录 B 中
    # dir_A = '/home/imu_panjiahui/SHdomain/SHC_baseline/data_prepare/MIC_data/'
    # dir_B = '/ddnstor/imu_panjiahui/dataset/SHdomain/'
    # shutil.move(dir_A, dir_B)

    # 3、A.txt里的内容分为5部分 分别写入五个.txt文件中
    # input_path = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt/test_clean_list.txt'
    # part = 5
    # split_txt(input_path, part)

    # # 4、把生成的数据路径写入wav_scp
    # file_dir = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/train/mix'
    # txt_path = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/loader_txt/wav_scp_train.txt'
    # gen_wav_scp(file_dir, txt_path)

    # 5、比较并删除不一致的文件
    # folderA = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/train/mix'
    # folderB = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/train/noreverb_ref'
    # folderC = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/train/reverb_ref'
    # compare_folders(folderA, folderB, folderC)

    # rir_list = ['1.0', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    # snr_list = ['-10', '-5', '0', '5', '10']
    # for snr in snr_list:
    #     for rir in rir_list:
    #         folderA = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/test/test_' + rir + '_snr' + snr + '/mix'
    #         folderB = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/test/test_' + rir + '_snr' + snr + '/noreverb_ref'
    #         folderC = '/ddnstor/imu_panjiahui/dataset/SHdomainNew/TIMIT/generated_data_cir_uniform_16/test/test_' + rir + '_snr' + snr + '/reverb_ref'
    #         compare_folders(folderA, folderB, folderC)
    #
    #         txt_path = '/data01/data_pjh/SHdomainNew/loader_txt/wav_scp_test_' + rir + '_snr' + snr + '.txt'
    #         gen_wav_scp(folderA, txt_path)

    # 6、将文件夹内的 几个txt 文件的内容合并到一个新的 merged.txt 文件中：
    # folder_path = '/data01/data_pjh/SHdomainNew/loader_txt/'
    # output_file = '/data01/data_pjh/SHdomainNew/loader_txt/wav_scp_test.txt'
    # merge_files(folder_path, output_file)

    # 7、加载.mat文件并输出
    test_list = ['test_0.2_snr-10', 'test_0.2_snr-5', 'test_0.2_snr0', 'test_0.2_snr5', 'test_0.2_snr10',
                 'test_0.6_snr-10', 'test_0.6_snr-5', 'test_0.6_snr0', 'test_0.6_snr5', 'test_0.6_snr10',
                 'test_1.0_snr-10', 'test_1.0_snr-5', 'test_1.0_snr0', 'test_1.0_snr5', 'test_1.0_snr10'
                 ]
    snr_list = [-10, -5, 0, 5, 10]
    t60_list = [1.0]

    for t60 in t60_list:
        mix_avg = 0
        DPCRN_avg = 0
        Pro_avg = 0
        GCRN_avg = 0
        for snr in snr_list:
            test_name = 'test_' + str(t60) + '_snr' + str(snr)

            respath = '/home/imu_panjiahui/MultiStageSHC/DPCRN_baseline/model_dpcrn/result_model_best/' + test_name
            res1path = '/home/imu_panjiahui/MultiStageSHC/DPCRN_MultiStageSHC/model_dpcrn_multi_position/result_model_best/' + test_name
            res2path = '/home/imu_panjiahui/MultiStageSHC/GCRN_baseline/model_gcrn/result_model_best//' + test_name
            # res1path = '/home/imu_panjiahui/MultiStageSHC/DPCRN_MultiStageSHC/model_dpcrn_multi_position/result_model_best/' + test_name

            dpcrn_base = scipy.io.loadmat(respath + '_metrics.mat')
            multiStage_base = scipy.io.loadmat(res1path + '_metrics.mat')
            gcrn_base = scipy.io.loadmat(res2path + '_metrics.mat')

            # PESQ
            # mix_avg += float("{:.2f}".format(float(dpcrn_base['pesq_mix'])))
            # DPCRN_avg += float("{:.2f}".format(float(dpcrn_base['pesq_est'])))
            # Pro_avg += float("{:.2f}".format(float(multiStage_base['pesq_est'])))
            # GCRN_avg += float("{:.2f}".format(float(multiStage_base['pesq_est'])))
            #
            # print(test_name + " PESQ:")
            # print(
            #     'pesq_mix:' + "{:.2f}".format(float(dpcrn_base['pesq_mix'])) + ' '
            #     + 'pesq_dpcrn:' + "{:.2f}".format(float(dpcrn_base['pesq_est'])) + ' '
            #     + 'pesq_multiStage:' + "{:.2f}".format(float(multiStage_base['pesq_est'])) + ' '
            #     + 'pesq_gcrn:' + "{:.2f}".format(float(gcrn_base['pesq_est']))
            # )

            # STOI
            mix_avg += float("{:.2f}".format(float(dpcrn_base['stoi_mix']) * 100))
            DPCRN_avg += float("{:.2f}".format(float(dpcrn_base['stoi_est']) * 100))
            Pro_avg += float("{:.2f}".format(float(multiStage_base['stoi_est']) * 100))
            GCRN_avg += float("{:.2f}".format(float(gcrn_base['stoi_est']) * 100))

            print(test_name + " STOI:")
            print(
                'stoi_mix:' + "{:.2f}".format(float(dpcrn_base['stoi_mix']) * 100) + ' '
                + 'stoi_dpcrn:' + "{:.2f}".format(float(dpcrn_base['stoi_est']) * 100) + ' '
                + 'stoi_multiStage:' + "{:.2f}".format(float(multiStage_base['stoi_est']) * 100)+' '
                + 'stoi_gcrn:' + "{:.2f}".format(float(gcrn_base['stoi_est']) * 100) + ' '
            )

        print('SNR = ' + str(t60))
        print('mix_avg : ' + str("{:.2f}".format(mix_avg / 5)))
        print('DPCRN_avg : ' + str("{:.2f}".format(DPCRN_avg / 5)))
        print('Pro_avg : ' + str("{:.2f}".format(Pro_avg / 5)))
        print('GCRN_avg : ' + str("{:.2f}".format(GCRN_avg / 5)))