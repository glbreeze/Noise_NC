import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse.linalg import svds


def _get_polynomial_decay(lr, end_lr, decay_epochs, from_epoch=0, power=1.0):
  # Note: epochs are zero indexed by pytorch
  end_epoch = float(from_epoch + decay_epochs)

  def lr_lambda(epoch):
    if epoch < from_epoch:
      return 1.0
    epoch = min(epoch, end_epoch)
    new_lr = ((lr - end_lr) * (1. - epoch / end_epoch) ** power + end_lr)
    return new_lr / lr  # LambdaLR expects returning a factor

  return lr_lambda


def get_scheduler(args, optimizer):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    if args.scheduler in ['ms', 'multi_step']:
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.max_epochs//3, args.max_epochs*2/3], gamma=0.1)
    elif args.scheduler in ['cos', 'cosine']:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)

        if torch.cuda.is_available(): targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction: return loss.mean()
        else: return loss.sum()


class CrossEntropyHinge(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:  Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.05, reduction=True):
        super(CrossEntropyHinge, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(device)

        loss = (- targets * log_probs).sum(dim=1)

        mask = loss.detach() >= -torch.log(torch.tensor(1-self.epsilon))
        loss = loss * mask

        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        if y_true.ndim == 1:
            y_true = torch.zeros_like(y_pred).scatter_(1, y_true.unsqueeze(1), 1)

        # Standard CE: -y_true * log(y_pred)
        y_pred_clamp = torch.clamp(y_pred, min=1e-7, max=1.0)
        ce_loss = torch.sum(-y_true * torch.log(y_pred_clamp), dim=-1)

        # Reverse CE: -y_pred * log(y_true)
        y_true_clamp = torch.clamp(y_true, min=1e-4, max=1.0)
        rce_loss = torch.sum(-y_pred * torch.log(y_true_clamp), dim=-1)

        loss = self.alpha * ce_loss.mean() + self.beta * rce_loss.mean()
        return loss


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        one_hot = F.one_hot(targets, num_classes=num_classes).float()

        probs = F.softmax(logits, dim=1)
        loss = torch.mean(torch.sum(torch.abs(probs - one_hot), dim=1))
        return loss


class GCELoss(nn.Module):
    def __init__(self, q=0.5, reduction='mean'):
        super(GCELoss, self).__init__()
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        true_probs = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)  # [B]

        if self.q == 1.0:
            loss = -torch.log(true_probs)
        else:
            loss = (1.0 - true_probs.pow(self.q)) / self.q

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: [B]
        

def feature_to_weight_loss(feat, target_w, metric="l2", feat_norm=False, hb_delta=1.0, eps=1e-8):
    """
    feat: [B, D]
    target_w: [B, D]  (already indexed from classifier weights)
    """
    if feat_norm: 
        feat = F.normalize(feat, dim=1, eps=eps)
        
    if metric == "l2":
        loss = ((feat - target_w) ** 2).sum(dim=1).mean()
    elif metric == 'l1':
        loss = (feat - target_w).abs().sum(dim=1).mean()
    elif metric == "cos": # 1 - cosine similarity
        f = F.normalize(feat, dim=1, eps=eps)
        w = F.normalize(target_w, dim=1, eps=eps)
        loss = 1.0 - (f * w).sum(dim=1).mean()
    elif metric == "huber":
        loss = F.smooth_l1_loss(feat, target_w, beta=hb_delta, reduction="mean")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return loss



def set_optimizer(model, args, momentum, log, conv_wd=None, bn_wd=None, cls_wd=None):
    conv_params, bn_params, cls_params = [], [], []

    for name, param in model.named_parameters():
        if 'conv' in name or "downsample.0" in name or "features.0" in name:
            conv_params.append(param)
        elif 'bn' in name or 'downsample.1' in name or "features.1" in name:
            bn_params.append(param)
        elif 'classifier' in name or 'fc' in name:
            cls_params.append(param)

    params_to_optimize = [
        {"params": conv_params, "weight_decay": conv_wd if conv_wd is not None else args.conv_wd},
        {"params": bn_params, "weight_decay": bn_wd if bn_wd is not None else args.bn_wd},
        {"params": cls_params, "weight_decay": cls_wd if cls_wd is not None else args.cls_wd},
    ]

    optimizer = optim.SGD(params_to_optimize, lr=args.lr, momentum=momentum)
    log('>>>>>Set Optimizer conv_wd:{}, bn_wd:{}, cls_wd:{}'.format(
        conv_wd if conv_wd is not None else args.conv_wd,
        bn_wd if bn_wd is not None else args.bn_wd,
        cls_wd if cls_wd is not None else args.cls_wd))
    return optimizer


def get_logits_labels_feats(data_loader, net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = []
    labels = []
    labels_clean = []
    feats = []
    net.eval()
    with torch.no_grad():
        for data, label, label_clean in data_loader:
            data, label, label_clean = data.to(device), label.to(device), label_clean.to(device)
            logit, feat = net(data, ret_feat=True)
            logits.append(logit)
            feats.append(feat)
            labels.append(label)
            labels_clean.append(label_clean)
        labels = torch.cat(labels)
        labels_clean = torch.cat(labels_clean)
        logits = torch.cat(logits, dim=0)
        feats = torch.cat(feats, dim=0) # [N, 512]
    return logits, feats, labels, labels_clean


def get_logits_labels(data_loader, net):
    logits_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for data, label in data_loader:
            data = data.cuda()
            logits = net(data)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
    return logits, labels

# =================== for logging  =================== 

def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)