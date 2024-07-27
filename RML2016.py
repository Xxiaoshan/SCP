import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils import data
import argparse
import math
from random import *
import random
from torch.utils import data
from torch.utils.data import TensorDataset
from test1 import dataselect


def Data_Augmentation(inputs, random_seed, theta_list, std_list):

    if random_seed == None:
        random_seed = randint(14, 15)
    if theta_list == None:
        theta_list = [0, math.pi/2, math.pi, math.pi/2*3]
    if std_list == None:
        std_list = [0, 0.05, 0.1, 0.15]

    if random_seed == 1:
        flip_v = torch.zeros_like(inputs)
        flip_v[:, 0, :] = inputs[:, 0, :]
        flip_v[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v
    elif random_seed == 2:
        flip_h = torch.zeros_like(inputs)
        flip_h[:, 0, :] = -inputs[:, 0, :]
        flip_h[:, 1, :] = inputs[:, 1, :]
        Data_Aug = flip_h
    elif random_seed == 3:
        flip_v_h = torch.zeros_like(inputs)
        flip_v_h[:, 0, :] = -inputs[:, 0, :]
        flip_v_h[:, 1, :] = -inputs[:, 1, :]
        Data_Aug = flip_v_h

    elif random_seed == 0:
        std_temp = random.choice(std_list)
        noise = torch.normal(mean=0, std=std_temp, size=inputs.shape)
        Data_Aug = inputs + noise

    elif random_seed > 3:
        theta = random.choice(theta_list)
        i_data = inputs[:, 0, :]
        q_data = inputs[:, 1, :]
        i_data_gen = math.cos(theta) * i_data - math.sin(theta) * q_data
        q_data_gen = math.sin(theta) * i_data + math.cos(theta) * q_data
        Data_Aug = torch.cat([i_data_gen.unsqueeze(1), q_data_gen.unsqueeze(1)], dim=1)
    elif random_seed == 14:
        Data_Aug = torch.flip(inputs, dims=[2])
    elif random_seed == 15:
        Data_Aug = inputs
    elif random_seed == 12:
        I_random_mask_num = randint(1, 110)
        Q_random_mask_num = randint(1, 110)
        inputs[:, 0, I_random_mask_num: I_random_mask_num + 10] = 0
        inputs[:, 1, Q_random_mask_num: Q_random_mask_num + 10] = 0
        Data_Aug = inputs

    return Data_Aug

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data

def load_RML(args):
    """Load RML2016A, RML2016B, RML2018 dataset.
    The data is split and normalized between train and test sets.
    """
    args.cudaTF = torch.cuda.is_available()
    kwargs = {'num_workers': args.threads,
              'pin_memory': True} if args.cudaTF else {}

    if args.ab_choose == 'RML201610A':
        if args.snr_tat == "ALL":
            filename_train_sne = args.RML2016a_path + "train_ALL_SNR_MV_dataset"
            filename_test_sne = args.RML2016a_path + "test_ALL_SNR_MV_dataset"

            IQ_train = load_pickle(filename_train_sne)
            IQ_test = load_pickle(filename_test_sne)
            train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs)
        else:
            filename_train_sne = args.RML2016a_path + str(args.snr_tat) + "_train_MV_dataset"
            filename_test_sne = args.RML2016a_path + str(args.snr_tat) + "_test_MV_dataset"
            IQ_train = load_pickle(filename_train_sne)
            IQ_test = load_pickle(filename_test_sne)
            # 采用随机的index
            indics = torch.randperm(len(IQ_train.tensors[0]))
            # 加载保存的index
            # indics = torch.load("73.7_6dBbest_indice.pt")

            a0 = IQ_train.tensors[0]
            a1 = IQ_train.tensors[1]
            a3 = torch.arange(0, 8800)
            shuffled_data = a0[indics]
            shuffled_label = a1[indics]
            #给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号


            a1_selected = []
            a0_selected = []
            a3_selected = []

            count_num = args.N_shot   # 50 shot
            # 从0到10遍历，每个数字选取10个

            for i in range(11):  # 0到10共11个数字
                count = 0
                cout_idx = -1
                for num in a1:
                    cout_idx = cout_idx + 1
                    if num == i:
                        a1_selected.append(shuffled_label[cout_idx].unsqueeze(dim=0))
                        a0_selected.append(shuffled_data[cout_idx, :, :].unsqueeze(dim=0))
                        a3_selected.append(a3[cout_idx].unsqueeze(dim=0))
                        count += 1
                        if count == count_num:
                            break
            # 输出选中数字列表
            print(len(a1_selected))
            a1_selected = torch.cat(a1_selected)
            a0_selected = torch.cat(a0_selected)
            a3_selected = torch.cat(a3_selected)
            new_dataset = TensorDataset(a0_selected, a1_selected, a3_selected)
            # new_dataset = TensorDataset(a0[10:120, :, :], a1[10:120], a3[10:120])   #控制选择的输入数据集大小

            # #part train sample
            # data_temp = IQ_train.tensors[0]
            # label_temp = IQ_train.tensors[1]
            # IQ_train = data.TensorDataset(data_temp[0:8800,:,:], label_temp[0:8800])

            # train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
            # test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True,  **kwargs)
            train_sampler = None
            train_loader_IQ = data.DataLoader(IQ_train, batch_size=args.batch_size, shuffle=True,  **kwargs, sampler=train_sampler)
            test_loader_IQ = data.DataLoader(IQ_test, batch_size=args.batch_size, shuffle=True,  **kwargs, sampler=train_sampler)
            fintune_set = data.DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True,  **kwargs, sampler=train_sampler)
    elif args.ab_choose == 'RML201610B':
        print('choosed dataset:RML2016B')
        #选择2016b的数据集
        filename_train_sne = args.RML2016b_path + str(args.snr_tat) + "_MT4_train_dataset"
        filename_test_sne = args.RML2016b_path + str(args.snr_tat) + "_MT4_test_dataset"
        IQ_train = load_pickle(filename_train_sne)
        IQ_test = load_pickle(filename_test_sne)
        # 采用随机的index
        indics = torch.randperm(len(IQ_train.tensors[0]))
        # 加载保存的index
        # indics = torch.load("73.7_6dBbest_indice.pt")

        a0 = IQ_train.tensors[0]
        a1 = IQ_train.tensors[1]
        a3 = torch.arange(0, len(IQ_train.tensors[0]))

        if args.N_shot == 0:
            new_dataset = TensorDataset(a0, a1, a3)

        else:
            shuffled_data = a0[indics]
            shuffled_label = a1[indics]
            # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号

            a1_selected = []
            a0_selected = []
            a3_selected = []

            count_num = args.N_shot  # 50 shot
            # 从0到10遍历，每个数字选取10个

            for i in range(10):  # 0到10共11个数字
                count = 0
                cout_idx = -1
                for num in a1:
                    cout_idx = cout_idx + 1
                    if num == i:
                        a1_selected.append(shuffled_label[cout_idx].unsqueeze(dim=0))
                        a0_selected.append(shuffled_data[cout_idx, :, :].unsqueeze(dim=0))
                        a3_selected.append(a3[cout_idx].unsqueeze(dim=0))
                        count += 1
                        if count == count_num:
                            break
            # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号
            a1_selected = torch.cat(a1_selected)
            a0_selected = torch.cat(a0_selected)
            a3_selected = torch.cat(a3_selected)
            new_dataset = TensorDataset(a0_selected, a1_selected, a3_selected)
        train_sampler = None
        train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        fintune_set = data.DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, **kwargs,
                                  sampler=train_sampler)
        print('choosed dataset:RML2016B')
    else:
        assert args.ab_choose == 'RML2018'
        print('choosed dataset:RML2018')
        #选择2018的数据集

        filename_train_sne = args.RML2018_path +"_MV4_snr_"+ str(args.snr_tat) + "_train_dataset"
        filename_test_sne = args.RML2018_path +"_MV4_snr_"+ str(args.snr_tat) + "_test_dataset"
        IQ_train = load_pickle(filename_train_sne)
        IQ_test = load_pickle(filename_test_sne)
        # 采用随机的index
        indics = torch.randperm(len(IQ_train.tensors[0]))
        # 加载保存的index
        # indics = torch.load("73.7_6dBbest_indice.pt")

        a0 = IQ_train.tensors[0]
        a1 = IQ_train.tensors[1]
        a3 = torch.arange(0, len(IQ_train.tensors[0]))

        if args.N_shot == 0:
            new_dataset = TensorDataset(a0, a1, a3)

        else:
            shuffled_data = a0[indics]
            shuffled_label = a1[indics]
            # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号

            a1_selected = []
            a0_selected = []
            a3_selected = []

            count_num = args.N_shot  # 50 shot
            # 从0到10遍历，每个数字选取10个

            for i in range(24):  # 0到10共11个数字
                count = 0
                cout_idx = -1
                for num in a1:
                    cout_idx = cout_idx + 1
                    if num == i:
                        a1_selected.append(shuffled_label[cout_idx].unsqueeze(dim=0))
                        a0_selected.append(shuffled_data[cout_idx, :, :].unsqueeze(dim=0))
                        a3_selected.append(a3[cout_idx].unsqueeze(dim=0))
                        count += 1
                        if count == count_num:
                            break
        # 给予数据集读入时三个参数，数据、标签、以及每个样本在当前数据集中的标号
            a1_selected = torch.cat(a1_selected)
            a0_selected = torch.cat(a0_selected)
            a3_selected = torch.cat(a3_selected)
            new_dataset = TensorDataset(a0_selected, a1_selected, a3_selected)
            # a0 = IQ_train.tensors[0]
            # a1 = IQ_train.tensors[1]
            # a3 = torch.arange(0, len(IQ_train))
            # new_dataset = TensorDataset(a0, a1, a3)  # 控制选择的输入数据集大小
        train_sampler = None
        train_loader_IQ = data.DataLoader(dataset=IQ_train, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        test_loader_IQ = data.DataLoader(dataset=IQ_test, batch_size=args.batch_size, shuffle=True, **kwargs, sampler=train_sampler)
        fintune_set = data.DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, **kwargs,
                                  sampler=train_sampler)

    # num of samples

    print('choosed dataset: ' + str(args.ab_choose))
    print('filename_train_dataset: ' + str(filename_train_sne))
    n_data = len(new_dataset)
    print('number of train samples: {}'.format(n_data))

    return train_loader_IQ, test_loader_IQ, n_data, fintune_set, new_dataset, IQ_test




def load_RML2016_old(args):
    """Load RML2016 dataset.
    The data is split and normalized between train and test sets.
    """
    kwargs = {'num_workers': args.threads, 'pin_memory': True} if args.cudaTF else {}


    if args.snr_tat == None: # 设置为0就取所有dB
        # A_ang_train = load_pickle(args.A_ang_train_path) # 我直接的IQ两路
        # A_ang_test = load_pickle(args.A_ang_test_path)
        IQ_train = load_pickle(args.IQ_train_path)
        IQ_test = load_pickle(args.IQ_test_path)


        # label_t1 = IQ_test[1]
        # for i in range(len(label_t1)):
        #     for j in range(11):
        #         if label_t1[i, j] == 1:
        #             label_t1[i, 0] = j
        #         else:
        #             j = j + 1
        # IQ_test[1] = label_t1[:,:1]


        #使用torch.tensor 对数据进行打包
        IQ_label_train_tensor = torch.tensor(IQ_train[1])
        IQ_data_train_tensor = torch.tensor(IQ_train[0])
        dataset_train = data.TensorDataset(IQ_data_train_tensor, IQ_label_train_tensor)
        IQ_label_test_tensor = torch.tensor(IQ_test[1])
        IQ_data_test_tensor = torch.tensor(IQ_test[0])
        dataset_test = data.TensorDataset(IQ_data_test_tensor, IQ_label_test_tensor)

        train_loader_IQ = data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
        test_loader_IQ = data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True,  **kwargs)
        # train_loader_A_ang = data.DataLoader(dataset=A_ang_train, batch_size=args.batch_size, shuffle=True, **kwargs)
        # test_loader_A_ang = data.DataLoader(dataset=A_ang_test, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:

        if args.v3_TF != None: # 这个不用
            filename_train_sne = args.savedata_path + str(args.snr_tat) + "_train_IQ_Aa1_dataset"
            filename_test_sne = args.savedata_path + str(args.snr_tat) + "_test_IQ_Aa1" \
                                                                         "" \
                                                                         "_dataset"
            IQ_A_train = load_pickle(filename_train_sne)
            IQ_A_test = load_pickle(filename_test_sne)
            train_loader_IQ = data.DataLoader(dataset=IQ_A_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
            test_loader_IQ = data.DataLoader(dataset=IQ_A_test, batch_size=args.batch_size, shuffle=True,  **kwargs)
        else:
            # filename_train_sne = args.single_data_path + str(args.snr_tat) + "_train_AMV_dataset"
            # filename_test_sne = args.single_data_path + str(args.snr_tat) + "_test_AMV_dataset"

            # 取出单一信噪比数据（只有I\Q两路信号）
            filename_train_sne = args.single_data_path + str(args.snr_tat) + "_train_IQ_Aa1_dataset"
            filename_test_sne = args.single_data_path + str(args.snr_tat) + "_test_IQ_Aa1_dataset"
            IQ_A_train = load_pickle(filename_train_sne)
            IQ_A_test = load_pickle(filename_test_sne)

            #处理分类任务数据标签
            # label_t = IQ_A_train[1]
            # for i in range(len(label_t)):
            #     for j in range(11):
            #         if label_t[i, j] == 1:
            #             label_t[i, 0] = j
            #         else:
            #             j = j + 1
            # IQ_A_train[1] = label_t[:,:1]

            IQ_label_train_tensor = torch.tensor(IQ_A_train[1])
            IQ_data_train_tensor = torch.tensor(IQ_A_train[0])
            dataset_train = data.TensorDataset(IQ_data_train_tensor, IQ_label_train_tensor)
            IQ_label_test_tensor = torch.tensor(IQ_A_test[1])
            IQ_data_test_tensor = torch.tensor(IQ_A_test[0])
            dataset_test = data.TensorDataset(IQ_data_test_tensor, IQ_label_test_tensor)

            train_loader_IQ = data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
            test_loader_IQ = data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True,  **kwargs)

    return train_loader_IQ, test_loader_IQ
