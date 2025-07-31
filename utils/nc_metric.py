import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
import scipy


def entropy(net_output):
    p = F.softmax(net_output, dim=1)
    logp = F.log_softmax(net_output, dim=1)
    plogp = p * logp
    entropy = - torch.sum(plogp, dim=1)
    return entropy


def compute_ETF(W, device):  # W [K, 512]
    K = W.shape[0]
    # W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)            # [K, 512] [512, K] -> [K, K]
    WWT /= torch.norm(WWT, p='fro')   # [K, K]

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):  # W:[K, 512] H:[512, K]
    """ H is already normalized"""
    K = W.shape[0]

    # W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))   # [K, 512] [512, K]
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def analysis_feat(labels, feats, args, W,):
    # analysis without extracting features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_cls = [0 for _ in range(args.num_classes)]  # within class sample size
    mean_cls = [0 for _ in range(args.num_classes)]
    sse_cls = [0 for _ in range(args.num_classes)]
    cov_cls = [0 for _ in range(args.num_classes)]

    # ====== compute mean and var for each class
    for c in range(args.num_classes):
        feats_c = feats[labels == c]   # [N, 512]
        num_cls[c] = len(feats_c)
        mean_cls[c] = torch.mean(feats_c, dim=0)

        # update within-class cov
        X = feats_c - mean_cls[c].unsqueeze(0)   # [N, 512]
        cov_cls[c] = X.T @ X / num_cls[c]        # [512, 512]
        sse_cls[c] = X.T @ X                     # [512, 512]

    # global mean
    M = torch.stack(mean_cls)        # [K, 512]
    mean_all = torch.mean(M, dim=0)  # [512]
    h_norm = torch.norm(M - mean_all.unsqueeze(0), dim=-1).mean().item()
    w_norm = torch.norm(W, dim=-1).mean().item()

    # =========== NC1
    Sigma_b = (M - mean_all.unsqueeze(0)).T @ (M - mean_all.unsqueeze(0)) / args.num_classes
    Sigma_w = torch.stack([cov * num for cov, num in zip(cov_cls, num_cls)]).sum(dim=0) / sum(num_cls)
    Sigma_t = (feats - mean_all.unsqueeze(0)).T @ (feats - mean_all.unsqueeze(0)) / len(feats)

    Sigma_b = Sigma_b.cpu().numpy()
    Sigma_w = Sigma_w.cpu().numpy()
    nc1 = np.trace(Sigma_w @ scipy.linalg.pinv(Sigma_b))
    nc1_cls = [np.trace(cov.cpu().numpy() @ scipy.linalg.pinv(Sigma_b)) for cov in cov_cls]
    var_cls = [np.trace(cov.cpu().numpy() / h_norm**2) for cov in cov_cls]

    # =========== NC2
    nc2h = compute_ETF(M - mean_all.unsqueeze(0), device)
    nc2w = compute_ETF(W, device)
    nc2 = compute_W_H_relation(W, (M - mean_all.unsqueeze(0)).T, device)

    # =========== NC3
    normalized_M = (M - mean_all.unsqueeze(0)) / torch.norm(M - mean_all.unsqueeze(0), 'fro')
    normalized_W = W / torch.norm(W, 'fro')
    nc3d = (torch.norm(normalized_W - normalized_M) ** 2).item()

    nc_dt = {
        'nc1': nc1,
        'nc2h': nc2h,
        'nc2w': nc2w,
        'nc2': nc2,
        'nc3': nc3d,
        'h_norm': h_norm,
        'w_norm': w_norm,
        'nc1_cls': nc1_cls,
        'var_cls': var_cls
    }

    return nc_dt, M