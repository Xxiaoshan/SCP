"""
pre-training for SCP
"""
## Utilities
from __future__ import print_function
import argparse
import time
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch
import torch.optim as optim

## src Imports
from src.logger_v1 import setup_logs
from src.training_v1 import train, snapshot
from src.validation_v1 import validation
from src.model.model import SCP_pre_training
from RML2016 import load_RML
from utils import *

############ Control Center and Hyperparameter ###############
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        #预热学习率动态调整
        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def main():

    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # load dataset
    parser.add_argument('--v3_TF', type=int,
                        default=None, help='是否通过小波变换，True则经过小波,None则不通过')
    parser.add_argument('--snr_tat', type=int,
                        default=6, help='训练测试的信噪比，如果为None则为0dB以上全训练测试')
    parser.add_argument('--threads', type=int, default=0,  # 我电脑只用能num_work=0
                        help='number of threads for data loader to use. default=4')
    parser.add_argument("--N_shot", default=0, type=int, help='每个信噪比下每个类别的有标签样本个数')
    parser.add_argument('--RML2016a_path', type=str, default="D:\\PYALL\\RML2016\\datasetsave\\2016MV_NEW\\",
                        help="保存数据集合的位置")
    parser.add_argument('--ab_choose', type=str, default="RML2018",
                        help="可选择RML201610A, RML201610B, RML2018")
    parser.add_argument('--RML2016b_path', type=str, default="D:\\PYALL\\RML2016\\RML2016_10B\\pre_snr_train_test\\",
                        help="2016bMV训练测试数据集路径1")
    parser.add_argument('--RML2018_path', type=str, default="D:\\PYALL\\RML2018\\dataset2018_py_now\\",
                        help="2018MV训练测试数据集路径1")
    parser.add_argument('--model_path', type=str,
                        default='savedata_path',
                        help="模型保存的路径")
    parser.add_argument('--cudaTF', type=bool, default=True)

    parser.add_argument('--logging-dir', default='pre2018_ALLSNR_channl_256_step_16/pre',
                        help='model save directory')
    parser.add_argument('--epochs', type=int, default=240, metavar='N',
                        help='number of epochs to train')  #训练的轮次
    parser.add_argument('--n-warmup-steps', type=int, default=50)   #
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--audio-window', type=int, default=902,
                        help='window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default =16)   #预测未来步长8.12.16.20.24
    parser.add_argument('--masked-frames', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    args.logging_dir = args.logging_dir + '_snr_{}'.format(args.snr_tat)
    ensure_path(args.logging_dir)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    logger = setup_logs(args.logging_dir, run_name)  # setup logs
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SCP_pre_training(args.timestep, args.batch_size, args.audio_window).to(device)   #观测窗口，model里的输入长度
    ## Loading the dataset
    params = {'num_workers': 0,
              'pin_memory': False} if use_cuda else {}

    logger.info('===> loading train, validation and eval dataset')
    train_loader, validation_loader, n_data, _, _, _ = load_RML(args)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))

    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()
        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)

        # Save
        run_name1 = run_name + '_best_'
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            snapshot(args.logging_dir, run_name1, {
                'epoch': epoch + 1,
                'validation_acc': val_acc,
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
    run_name2 = run_name + '_last_'
    snapshot(args.logging_dir, run_name2, {
        'last_epoch': epoch + 1,
        # 'validation_acc': val_acc,
        'state_dict': model.state_dict(),
        # 'validation_loss': val_loss,
        'optimizer': optimizer.state_dict(),
    })
    ## end
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
    #logger.handlers.clear()  #添加解决重复打印问题

if __name__ == '__main__':
    main()
