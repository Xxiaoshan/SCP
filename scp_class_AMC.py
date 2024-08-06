# Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from RML2016 import load_RML

## Custrom Imports
from src.logger_v1 import setup_logs
from src.training_v1 import train_scp, snapshot
from src.validation_v1 import validation_scp, plot_confusion_matrix
from src.prediction_v1 import prediction_scp
from src.model.model import SCP_pre_training, ScpClassifier

from utils import *

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

############ Control Center and Hyperparameter ###############
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

class ScheduledOptim(object):  #class类
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128     #类属性参数
        self.n_warmup_steps = n_warmup_steps  #实例属性
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2    #2*delta

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


# def main(steplen):
def main(labelsum1,k):
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')    #创建解析器
    # load dataset
    parser.add_argument('--v3_TF', type=int,                                #添加参数
                        default=None, help='是否通过小波变换，True则经过小波,None则不通过')
    parser.add_argument('--snr_tat', type=int,
                        default=12, help='训练测试的信噪比，如果为None则为0dB以上全训练测试')
    parser.add_argument('--threads', type=int, default=0,  # 我电脑只用能num_work=0
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--N_shot',type=int,
                        default=labelsum1, help='微调时每个信噪比每种调制类型的标签数量')
    parser.add_argument('--RML2016a_path', type=str, default="D:\\PYALL\\RML2016\\datasetsave\\2016MV_NEW\\",
                        help="保存数据集合的位置")
    parser.add_argument('--ab_choose', type=str, default="RML2018",
                        help="可选择RML201610A, RML201610B, RML2018")
    parser.add_argument('--RML2016b_path', type=str, default="D:\\PYALL\\RML2016\\RML2016_10B\\pre_snr_train_test\\",
                        help="2016bMV训练测试数据集路径1")
    parser.add_argument('--RML2018_path', type=str, default="D:\\PYALL\\RML2018\\dataset2018_py_now\\",
                        help="2018MV训练测试数据集路径1")
    parser.add_argument('--cudaTF', type=bool, default=True)
    parser.add_argument('--logging_dir', default='spk_class_new',
                        help='model save directory')
    # parser.add_argument('--model_path', default='pretrain_IQAV')
    parser.add_argument('--model_path', default='pre2018_channl256_step_16_only_rot/pre_snr_12/best_-model.pth')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',   #训练轮次
                        help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256,  #128
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=1024,  #观测窗口
                        help='window length to sample from each utterance')
    parser.add_argument('--frame_window', type=int, default=1)
    parser.add_argument('--spk_num', type=int, default=24)
    parser.add_argument('--timestep', type=int, default=32)  #12
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()                              #解析参数
    args.logging_dir = args.logging_dir +'//2018_Finetune_ALL_AMC1d_pre_rot_spk_flp_win1024_step16IQ_128_256_classno_dp_spk_snr10times_{}'.format(args.snr_tat)+'//_{}_shot_'.format(args.N_shot)+'{}'.format(k)
    ensure_path(args.logging_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    logger = setup_logs(args.logging_dir, run_name)  # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    pretrain_model = SCP_pre_training(args.timestep, args.batch_size, args.audio_window).cuda()
    # checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)  # load everything onto CPU
    # pretrain_model.load_state_dict(checkpoint['state_dict'])
    if args.model_path:
        logger.info('===> loading pretrain parameters from {}'.format(args.model_path))
        pretrained_parameters = torch.load(args.model_path)['state_dict']
        pretrain_model = update_param(pretrain_model, pretrained_parameters)
    else:
        print('===> Does not loading pretrain parameters')
    for param in pretrain_model.parameters():
        param.requires_grad = True  # True为微调pretrain_model F不微调
    class_model = ScpClassifier(args.spk_num).cuda()
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}   #?

    logger.info('===> loading train, validation and eval dataset')
    # train_loader, validation_loader, n_data = load_RML(args)
    train_loader, validation_loader, n_data, fintune_set, new_dataset, IQ_test = load_RML(args)

    # nanxin optimizer
    optimizer = ScheduledOptim(
        
        #仅微调分类器
        # optim.Adam(
        #     filter(lambda p: p.requires_grad, class_model.parameters()),
        #     betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        # args.n_warmup_steps)
        # 微调模型以及分类器 finetuning
        optim.Adam(
            [{"params":filter(lambda p: p.requires_grad, pretrain_model.parameters()),'lr':0.0005},
             {"params":filter(lambda y: y.requires_grad, class_model.parameters()),'lr':0.001}],
             betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in class_model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(class_model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train_scp(args, pretrain_model, class_model, device, fintune_set, optimizer, epoch, args.batch_size,
                  args.frame_window)
        if epoch == 1 or epoch % 3 == 0:
            test_acc, test_loss, all_pred, all_target = validation_scp(args, pretrain_model, class_model, device, validation_loader, args.batch_size,
                                           args.frame_window)

        # Save
        if test_acc > best_acc:
            best_acc = max(test_acc, best_acc)
            # 每次最优画混淆矩阵
            plot_confusion_matrix(args.logging_dir, epoch, all_pred, all_target)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'test_acc': test_acc,
                'state_dict': class_model.state_dict(),
                'test_loss': test_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))


    # last epoch 也画混淆矩阵
    plot_confusion_matrix(args.logging_dir, epoch, all_pred, all_target)
    # spk training save last epoch model
    snapshot(args.logging_dir, run_name, {
        'last_epoch': epoch + 1,
        'state_dict': class_model.state_dict(),
        'optimizer': optimizer.state_dict(),
    })
    ## prediction
    logger.info('===> loading best model for prediction')
    checkpoint = torch.load(os.path.join(args.logging_dir, run_name + '-model_last.pth'))
    class_model.load_state_dict(checkpoint['state_dict'])

    prediction_scp(args, pretrain_model, class_model, device, validation_loader, args.batch_size, args.frame_window)
    ## end
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
    logger.handlers.clear()
    return best_acc, args.snr_tat

if __name__ == '__main__':
    # for snr in range(-20,20,2):
    # snr = 12
    for labelsum1 in [2, 5, 10, 20, 50, 100]:
    # for labelsum1 in [100]:
        acc_list = []
        for k in range(0, 10):  #寻优次数
            # labelsum = labelsum1*20
            best_acc, SNR = main(labelsum1, k)
            acc_list.append(best_acc)
        min_acc = round(min(acc_list), 4)
        max_acc = round(max(acc_list), 4)
        mean_acc = round(sum(acc_list)/len(acc_list),4)
        with open('2018_Finetune_IQ_128_256_classno_dp_10timesSNR_{}_acclist{}min_{}max-{}mean{}.txt'.format(SNR,labelsum1,min_acc,max_acc,mean_acc),'w') as f:
            f.write(str(acc_list))
