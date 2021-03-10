"""
Copy from https://github.com/WellyZhang/RAVEN/blob/master/src/model/main.py
"""
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import RAVENdataset as dataset, RAVENToTensor as ToTensor
from models import *

GPUID = '2'
device_ids = [2]
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

parser = argparse.ArgumentParser(description='our_model')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--tag', type=str, default='resnet')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--mgpu', type=bool, default=False)
parser.add_argument('--load_workers', type=int, default=16)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default='/home/robot/workspace/eb_lang_learner/dataset/raven')
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
if args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.autograd.set_detect_anomaly(True)

if not os.path.exists(args.save):
    os.makedirs(args.save)

trainset = dataset(args.path, "train", args.img_size, transform=transforms.Compose([ToTensor()]))
validset = dataset(args.path, "val", args.img_size, transform=transforms.Compose([ToTensor()]))
testset = dataset(args.path, "test", args.img_size, transform=transforms.Compose([ToTensor()]))

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.load_workers)
validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.load_workers)

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
if args.mgpu:
    torch.cuda.set_device(device_ids[0])
    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)
    torch.backends.cudnn.benchmark = True    
if args.cuda and not args.mgpu:
    torch.cuda.set_device(args.device)
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)


def compute_loss(args, output, target, meta_target, meta_structure):
    pred, meta_target_pred, meta_struct_pred = output[0], output[1], output[2]

    target_loss = F.cross_entropy(pred, target)
    meta_target_pred = torch.chunk(meta_target_pred, chunks=9, dim=1)
    meta_target = torch.chunk(meta_target, chunks=9, dim=1)
    meta_target_loss = 0.
    for idx in range(0, 9):
        meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_target_pred[idx]), meta_target[idx])

    meta_struct_pred = torch.chunk(meta_struct_pred, chunks=21, dim=1)
    meta_structure = torch.chunk(meta_structure, chunks=21, dim=1)
    meta_struct_loss = 0.
    for idx in range(0, 21):
        meta_struct_loss += F.binary_cross_entropy(torch.sigmoid(meta_struct_pred[idx]), meta_structure[idx])
    loss = target_loss + args.meta_alpha*meta_struct_loss/21. + args.meta_beta*meta_target_loss/9.
    return loss

def train(args, epoch):
    model.train()
    train_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    total_correct = 0
    total_samples = 0
    t = tqdm(trainloader, desc='-')
    for image, target, meta_target, meta_structure, embedding, indicator in t:
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()

        optimizer.zero_grad()
        output = model(image, embedding, indicator)
        loss = compute_loss(args, output, target, meta_target, meta_structure)
        loss.backward()
        optimizer.step()
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        acc = correct * 100.0 / target.size()[0]
        loss = loss.item()

        t.set_description('Train: Epoch:{}, Loss:{:.6f}, Acc:{:.4f}.'.format(epoch, loss, acc))
        t.refresh()
        loss_all += loss
        acc_all += acc
        total_correct += correct
        total_samples += target.size(0)
    if counter > 0:
        print("Avg Training Loss: {:.6f}".format(loss_all/float(counter)))
    return 100 * total_correct/float(total_samples)

def validate(args, epoch):
    model.eval()
    val_loss = 0
    accuracy = 0

    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    total_correct = 0
    total_samples = 0
    t = tqdm(validloader, desc='valid')
    for image, target, meta_target, meta_structure, embedding, indicator in t:
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        with torch.no_grad():
            output = model(image, embedding, indicator)
        loss = compute_loss(args, output, target, meta_target, meta_structure)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        acc = correct * 100.0 / target.size()[0]
        loss = loss.item()
        loss_all += loss
        acc_all += acc
        total_correct += correct
        total_samples += target.size(0)
    if counter > 0:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), 100 * total_correct/float(total_samples)))
    return loss_all/float(counter), 100 * (total_correct/float(total_samples))

def test(epoch):
    model.eval()
    accuracy = 0

    acc_all = 0.0
    counter = 0
    total_correct = 0
    total_samples = 0
    t = tqdm(testloader, desc='test')
    for image, target, meta_target, meta_structure, embedding, indicator in t:
        counter += 1
        if args.cuda:
            image = image.cuda()
            target = target.cuda()
            meta_target = meta_target.cuda()
            meta_structure = meta_structure.cuda()
            embedding = embedding.cuda()
            indicator = indicator.cuda()
        with torch.no_grad():
            output = model(image, embedding, indicator)
        pred = output[0].data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        acc = correct * 100.0 / target.size()[0]
        acc_all += acc
        total_correct += correct
        total_samples += target.size(0)
    if counter > 0:
        print("Total Testing Acc: {:.4f}".format(100 * total_correct / float(total_samples)))
    return 100 * total_correct/float(total_samples)

def main():
    best_train_acc = 0
    best_val_acc= 0
    best_test_acc = 0
    for epoch in range(0, args.epochs):
        print('{}--{}--{}--bs{}'.format(args.tag, args.model, args.path, args.batch_size))
        train_acc = train(args, epoch)
        avg_loss, val_acc = validate(args, epoch)
        test_acc = test(epoch)
        best_train_acc = train_acc if train_acc > best_train_acc else best_train_acc
        best_val_acc = val_acc if val_acc > best_val_acc else best_val_acc
        best_test_acc = test_acc if test_acc > best_test_acc else best_test_acc
        print('{}--{}--{}--bs{}'.format(args.tag, args.model, args.path, args.batch_size))
        print("""
        best train acc:{}
        best val acc:{}
        best test acc:{}
                """.format(best_train_acc, best_val_acc, best_test_acc))
        # model.save_model(args.save, epoch, avg_acc, avg_loss)


if __name__ == '__main__':
    main()
