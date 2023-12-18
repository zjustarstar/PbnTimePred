import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torchvision import transforms
from dataset_stru import TimePredictionDataSet_Stru
from torch.utils.data import DataLoader
import networks
import argparse
import logging
import os
from pytorchtools import EarlyStopping

# parameters setting
parser = argparse.ArgumentParser(description='TimePrediction')
# Datasets
parser.add_argument('--dataset_train',
                    default='./file/train_stru.json', type=str)
parser.add_argument('--dataset_test',
                    default='./file/test_stru.json', type=str)
# Optimization options
parser.add_argument('--epochs', default=1000000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=4, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--checkpoints_dir', default='./checkpoints/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--test_interval', default=1, type=int, metavar='N',
                    help='test every X interval')
parser.add_argument('--save_interval', default=30, type=int, metavar='N',
                    help='save every X interval ')

# Architecture
parser.add_argument('--model_name', default='mlp', type=str)

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()



trainset = TimePredictionDataSet_Stru(args.dataset_train)
trainloader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0, drop_last=True)
testset = TimePredictionDataSet_Stru(args.dataset_test)
testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=0, drop_last=False)
dataloaders = {'train': trainloader, 'val': testloader}


def init_logger(checkpoints_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    handler = logging.FileHandler(f"./{checkpoints_dir}/log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def write_log_train_test(logger, loss, epoch, test_accuracy_60=0, test_accuracy_120=0, test_accuracy_180=0, mode='TRAIN'):
    if mode == "TRAIN":
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=60:%.3f  mae<=120:%.3f  mae<=180:%.3f)' % \
              (epoch, loss, test_accuracy_60, test_accuracy_120, test_accuracy_180)
        logger.info(log)
    else:
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=60:%.3f  mae<=120:%.3f  mae<=180:%.3f)' % \
              (epoch, loss, test_accuracy_60, test_accuracy_120, test_accuracy_180)
        logger.info(log)


def write_log(logger, loss, epoch, test_accuracy_5=0, test_accuracy_10=0, test_accuracy_20=0, mode='TRAIN'):
    if mode == "TRAIN":
        log = f'[{mode}] epoch: %3d  loss: %.3f' % (epoch, loss)
        logger.info(log)
    else:
        log = f'[{mode}]  epoch: %3d  MAE loss: %.3f  accuracy (mae<=50:%.3f  mae<=100:%.3f  mae<=200:%.3f)' % \
              (epoch, loss, test_accuracy_5, test_accuracy_10, test_accuracy_20)
        logger.info(log)


def train_model(model, model_type, criterion, optimizer, exp_lr_scheduler, early_stopping, save_interval, test_interval, checkpoints_dir, num_epochs):

    logger = init_logger(checkpoints_dir)

    for epoch in range(num_epochs):

        train_loss = 0.0
        train_img_num = 0

        true_num_60 = 0
        true_num_120 = 0
        true_num_180 = 0

        for img, color_num, blocks_num, blk_per_color, area_per_color, hint, labels in dataloaders['train']:
            img = img.cuda() # 占位，不用管
            # 色号，色块，面积分布特征，位置分布特征，色号对应的色块数量分布特征
            color_num = color_num.cuda().unsqueeze(-1)
            blocks_num = blocks_num.cuda().unsqueeze(-1)
            blk_per_color = blk_per_color.cuda().unsqueeze(-1).float()
            area_per_color = area_per_color.cuda().unsqueeze(-1).float()
            hint = hint.cuda().unsqueeze(-1).float()


            labels = labels.cuda().unsqueeze(-1)
            results = model([color_num, blocks_num, blk_per_color, area_per_color, hint], model_type=model_type)
            # results = model(img, model_type=model_type)
            # print(sehao.size(), sekuai.size(), labels.size())
            # results = model([sehao, sekuai], model_type=model_type)
            # print(results.size())
            # raise
            # results = results.squeeze(-1).float()

            results = results.float()
            loss = criterion(results, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            train_loss += loss.data * color_num.size(0)
            train_img_num += color_num.size(0)

            results = results.squeeze(-1).detach().cpu().numpy()
            labels = labels.detach().squeeze().cpu().numpy()


            temp = np.abs(results - labels)
            num_60 = np.sum(temp <= 60)
            true_num_60 += num_60
            num_120 = np.sum(temp <= 120)
            true_num_120 += num_120
            num_180 = np.sum(temp <= 180)
            true_num_180 += num_180

        # 训练准确率
        average_train_loss = train_loss / train_img_num
        test_accuracy_60 = true_num_60 / train_img_num
        test_accuracy_120 = true_num_120 / train_img_num
        test_accuracy_180 = true_num_180 / train_img_num

        write_log_train_test(logger, average_train_loss, epoch+1, test_accuracy_60, test_accuracy_120, test_accuracy_180, mode='TRAIN')

        # print("learning rate: ", exp_lr_scheduler.get_last_lr()[0])

        # if (epoch+1) % save_interval == 0:
        #     torch.save(model.state_dict(), f"{args.checkpoints_dir}/model_{epoch+1}.pt")

        if (epoch+1) % test_interval == 0:
            test_loss = 0
            test_img_num = 0
            true_num_60 = 0
            true_num_120 = 0
            true_num_180 = 0
            for img, color_num, blocks_num, blk_per_color, area_per_color, hint, labels in dataloaders['val']:
                img = img.cuda()
                color_num = color_num.cuda().unsqueeze(-1)
                blocks_num = blocks_num.cuda().unsqueeze(-1)
                blk_per_color = blk_per_color.cuda().unsqueeze(-1).float()
                area_per_color = area_per_color.cuda().unsqueeze(-1).float()
                hint = hint.cuda().unsqueeze(-1).float()
                labels = labels.cuda().unsqueeze(-1)

                with torch.no_grad():
                    # results = model(img, model_type=model_type)
                    results = model([color_num, blocks_num, blk_per_color, area_per_color, hint], model_type=model_type)
                    results = results.float()
                    # print(results)
                    # results = results.squeeze(-1)
                    # print(results.size(), labels.size())
                    test_loss += nn.L1Loss()(results, labels).data * color_num.size(0)
                    test_img_num += color_num.size(0)
                    results = results.squeeze(-1).detach().cpu().numpy()
                    labels = labels.detach().squeeze().cpu().numpy()
                    # print("-------------------",type(results), type(labels))
                    # print("-------------------", results.shape, labels.shape)
                    temp = np.abs(results - labels)


                    num_60 = np.sum(temp <= 60)
                    true_num_60 += num_60
                    num_120 = np.sum(temp <= 120)
                    true_num_120 += num_120
                    num_180 = np.sum(temp <= 180)
                    true_num_180 += num_180

            # 测试准确率
            average_test_loss = test_loss / test_img_num
            test_accuracy_60 = true_num_60 / test_img_num
            test_accuracy_120 = true_num_120 / test_img_num
            test_accuracy_180 = true_num_180 / test_img_num
            write_log_train_test(logger, average_test_loss, epoch+1, test_accuracy_60, test_accuracy_120, test_accuracy_180, mode='TEST')

            early_stopping(average_test_loss, model)
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break

        # print("-" * 150)


if __name__ == '__main__':
    print("Model :", args.model_name)
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.model_name)
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)
    model = networks.TimeModel(model_type=args.model_name)
    model.cuda()
    model.train()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=210, gamma=1)
    early_stopping = EarlyStopping(patience=100, verbose=False, path=os.path.join(args.checkpoints_dir, "best_checkpoint.pth"))
    train_model(model, args.model_name, criterion, optimizer, exp_lr_scheduler, early_stopping, args.save_interval, args.test_interval, args.checkpoints_dir, args.epochs)

