# -*- coding: utf-8 -*-
import os
import wandb
import torch
import random
import pickle
import argparse
from utils.nc_metric import analysis_feat
from utils.utils import set_log_path, log, print_args, get_scheduler, get_logits_labels_feats, AverageMeter
from utils.utils import CrossEntropyLabelSmooth, CrossEntropyHinge, FocalLoss, SymmetricCrossEntropy

from model import ResNet, MLP
from dataset.data import get_dataloader
from dataset.noisy_data import DatasetGenerator

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


def classwise_acc(targets, preds):
    eps = np.finfo(np.float64).eps
    cf = confusion_matrix(targets, preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / (cls_cnt + eps)
    return cls_acc


def train_one_epoch(model, criterion, train_loader, optimizer, args,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = AverageMeter('Loss', ':.4e')
    train_acc = AverageMeter('Train_acc', ':.4e')
    
    for batch_idx, (data, target, target_clean) in enumerate(train_loader):
        if data.shape[0] != args.batch_size:
            continue

        data, target = data.to(device), target.to(device)
        out, feat = model(data, ret_feat=True)

        optimizer.zero_grad()
        loss = criterion(out, target) 
        loss.backward()
        optimizer.step()
        
        pred = out.argmax(dim=-1)
        train_loss.update(loss.item(), target.size(0))
        train_acc.update((pred == target).float().mean().item(), target.size(0))

    return train_loss, train_acc


def main(args):
    MAX_TEST_ACC, MIN_TEST_LOSS = 0.0, 100.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==================== data loader ====================
    dataset = DatasetGenerator(
        batchSize=args.batch_size,
        dataPath=args.data_path,
        noise_rate=args.noise_rate,
        asym=args.asym,
        dataset_type=args.dset,
        seed=args.seed,
        numOfWorkers=args.num_workers
    )
    dataLoader = dataset.getDataLoader()
    train_loader, test_loader = dataLoader['train_dataset'], dataLoader['test_dataset']

    # ====================  define model ====================
    if args.model.lower() == 'mlp':
        model = MLP(hidden = args.width, depth = args.depth, fc_bias=args.bias, num_classes=args.num_classes)
    else:
        model = ResNet(pretrained=False, num_classes=args.num_classes, backbone=args.model, args=args)
    model = model.to(device)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'cse':
        criterion = SymmetricCrossEntropy(alpha=args.alpha, beta=args.beta)
    elif args.loss == 'ls':
        criterion = CrossEntropyLabelSmooth(args.num_classes, epsilon=args.eps)
    elif args.loss == 'fl':
        criterion = FocalLoss(gamma=args.eps)
    elif args.loss == 'ceh':
        criterion = CrossEntropyHinge(args.num_classes, epsilon=0.05)
    elif args.loss == 'hinge':
        criterion = nn.MultiMarginLoss(p=1, margin=args.margin, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = get_scheduler(args, optimizer)

    # ====================  start training ====================
    for epoch in range(args.max_epochs):
        
        # ==================== train         
        train_loss, train_acc = train_one_epoch(model, criterion, train_loader, optimizer, args)
        lr_scheduler.step()
        
        # ================= check NC
        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            logits, feats, labels, labels_clean = get_logits_labels_feats(train_loader, model)  # on cuda
            clean_mask = labels == labels_clean
            
            # NC analysis on training set
            nc_train, centroid = analysis_feat(labels, feats, args, W=model.classifier.weight.detach(),centroid=None)
            train_loss_nc = F.cross_entropy(logits, labels, reduction='mean').item() 
            train_acc_nc = (logits.argmax(dim=-1) == labels).sum().item()/len(labels)
            
            # NC analysis on clean training set
            if clean_mask.sum() > 0:
                clean_feats, clean_labels = feats[clean_mask], labels[clean_mask]
                nc_train_clean, _ = analysis_feat(clean_labels, clean_feats, args, W=model.classifier.weight.detach(), centroid=None)
            else:
                nc_train_clean = nc_train
            
            # NC analysis on test set
            logits_test, feats_test, labels_test, labels_clean_test = get_logits_labels_feats(test_loader, model)
            val_loss = F.cross_entropy(logits_test, labels_test, reduction='mean').item()
            val_acc = (logits_test.argmax(dim=-1) == labels_test).sum().item() / len(labels_test)
            nc_val, _ = analysis_feat(labels_test, feats_test, args, W=model.classifier.weight.detach(), centroid=None)
            
            wandb.log({
                'overall/lr': optimizer.param_groups[0]['lr'],
                'overall/val_loss': val_loss,
                'overall/val_acc': val_acc,
                'overall/train_loss': train_loss.avg,
                'overall/train_acc': train_acc.avg,
                'overall/train_acc1':train_acc_nc,
                'overall/train_loss1':train_loss_nc, 

                'train_nc/nc1': nc_train['nc1'],  'train_nc/nc2': nc_train['nc2'],
                'train_nc/nc3': nc_train['nc3'],  'train_nc/nc2h': nc_train['nc2h'],
                'train_nc/nc2w': nc_train['nc2w'],
                
                'norm/w_norm': nc_train['w_norm'], 'other_nc/h_norm': nc_train['h_norm'],
                
                'val_nc/nc1': nc_val['nc1'], 'val_nc/nc2': nc_val['nc2'],
                'val_nc/nc3': nc_val['nc3'], 'val_nc/nc2h': nc_val['nc2h'],
                
                'train_clean_nc/nc1': nc_train_clean['nc1'],
                'train_clean_nc/nc2': nc_train_clean['nc2'],
                'train_clean_nc/nc3': nc_train_clean['nc3'],
            }, step=epoch)
        
        if (args.save_ckpt > 0) and ((epoch+1) % args.save_ckpt ==0 or epoch == 0 or epoch == args.max_epochs-1):
            checkpoint_path = os.path.join(args.output_dir, f'ep{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc.avg,
                'val_acc': val_acc,
                'args': args
            }, checkpoint_path)
            log(f'Saved checkpoint to {checkpoint_path}')
       
        # ================= store the model
        # if (val_acc > MAX_TEST_ACC and epoch >= 100) and args.save_ckpt > 0:
        #         MAX_TEST_ACC = val_acc
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_acc_net.pt"))
        #         log('EP{} Store model (best TEST ACC) to {}'.format(epoch, os.path.join(args.output_dir, "best_acc_net.pt")))
        # if (val_loss < MIN_TEST_LOSS and epoch >= 100) and args.save_ckpt > 0:
        #         MIN_TEST_LOSS = val_loss
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_loss_net.pt"))
        #         log('EP{} Store model (best TEST LOSS) to {}'.format(epoch, os.path.join(args.output_dir, "best_loss_net.pt")))
        # if (val_ece20 < MIN_TEST_ECE and epoch >= 100) and args.save_ckpt > 0:
        #         MIN_TEST_ECE = val_ece20
        #         BEST_NET = model.state_dict()
        #         torch.save(BEST_NET, os.path.join(args.output_dir, "best_ece_net.pt"))
        #         log('EP{} Store model (best TEST ECE) to {}'.format(epoch, os.path.join(args.output_dir, "best_ece_net.pt")))


def set_seed(SEED=666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='neural collapse')
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument('--dset', type=str, default='cifar10')
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--num_workers', type=int, default=2)
    
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--ETF_fc', action='store_true', default=False)
    parser.add_argument('--noise_rate', type=float, default=0)
    parser.add_argument('--asym', action='store_true', default=False)

    # dataset parameters
    parser.add_argument('--aug', type=str, default='null')
    parser.add_argument('--num_classes', type=int, default=10)
   
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--no-bias', dest='bias', default=True, action='store_false')

    # optimization params
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--scheduler', type=str, default='ms')  # step|ms/multi_step/cosine
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=600)
    parser.add_argument('--wd', type=float, default=5e-4)  # '54'|'01_54' | '01_54_54'
    parser.add_argument('--loss', type=str, default='ce')  # ce|ls|ceh|hinge
    parser.add_argument('--eps', type=float, default=0.05)  # for ls loss

    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--save_ckpt', type=int, default=-1)
    parser.add_argument('--log_freq', type=int, default=2)

    args = parser.parse_args()
    args.output_dir = os.path.join('/scratch/lg154/sseg/neural_collapse/result/{}/{}/'.format(args.dset, args.model), args.exp_name)

    if args.dset == 'cifar100':
        args.num_classes=100
    elif args.dset == 'tinyi':
        args.num_classes=200
    elif args.dset == 'cifar10':
        args.num_classes = 10
    elif args.dset == 'stl10':
        args.num_classes = 10

    set_seed(SEED=args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    os.environ["WANDB_API_KEY"] = "0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee"
    os.environ["WANDB_MODE"] = "online"  # "dryrun"
    os.environ["WANDB_CACHE_DIR"] = "/scratch/lg154/sseg/.cache/wandb"
    os.environ["WANDB_CONFIG_DIR"] = "/scratch/lg154/sseg/.config/wandb"
    wandb.login(key='0c0abb4e8b5ce4ee1b1a4ef799edece5f15386ee')
    wandb.init(project='noisy_nc',
               name=args.exp_name
               )
    wandb.config.update(args)

    main(args)