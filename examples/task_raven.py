"""
Copy from https://github.com/WellyZhang/RAVEN/blob/master/src/model/main.py
"""
import os
import sys
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import RAVENdataset as dataset, RAVENToTensor as ToTensor
from models import *


parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default='/home/xiaojian/workspace/eb_lang_learner/dataset/raven')
parser.add_argument('--save', type=str, default='./experiments/checkpoint/')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--meta_alpha', type=float, default=0.0)
parser.add_argument('--meta_beta', type=float, default=0.0)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.cuda.set_device(args.device)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

trainset = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]))
validset = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
testset = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=16)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)

print ('Train/Validation/Test:{0}/{1}/{2}'.format(len(trainset), len(validset), len(testset)))
print ('Image size:', args.img_size)

if args.model == "resnet":
    model = RAVENResnet18_MLP(args)
elif args.model == 'trans':
    model = RAVENTrans(args)
elif args.model == 'ntm':
    model = RAVENNTM(args)
    
if args.resume:
    model.load_model(args.save, 0)
    print('Loaded model')
if args.cuda:
    model = model.cuda()

def train(epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(trainloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        loss, acc = model.train_(image, target, meta_target, meta_structure, embedding, indicator)
        print('Train: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc))
        loss_all += loss
        acc_all += acc
    if counter > 0:
        print("Avg Training Loss: {:.6f}".format(loss_all/float(counter)))

def validate(epoch):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(validloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        loss, acc = model.validate_(image, target, meta_target, meta_structure, embedding, indicator)
        # print('Validate: Epoch:{}, Batch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, batch_idx, loss, acc)) 
        loss_all += loss
        acc_all += acc
    if counter > 0:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)

def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target, meta_target, meta_structure, embedding, indicator) in enumerate(testloader):
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        acc = model.test_(image, target, meta_target, meta_structure, embedding, indicator)
        # print('Test: Epoch:{}, Batch:{}, Acc:{:.4f}.'.format(epoch, batch_idx, acc))  
        acc_all += acc
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(acc_all / float(counter)))
    return acc_all/float(counter)

def main():
    for epoch in range(0, args.epochs):
        train(epoch)
        avg_loss, avg_acc = validate(epoch)
        test(epoch)
        model.save_model(args.save, epoch, avg_acc, avg_loss)


if __name__ == '__main__':
    main()