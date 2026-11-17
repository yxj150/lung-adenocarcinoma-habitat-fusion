"""
=============================================================================
DL_V2 : re-run of the 3D deep-learning branch with alternative backbones
=============================================================================
Motivation
  The deployed branch (EfficientNet-B0-3D, trained from scratch, no
  pretraining) contributed little in the ablation.  This script re-trains the
  branch with several *common* 3D architectures, including Kinetics-400
  pretrained video CNNs / transformers, under the SAME protocol as the
  original pipeline so results are directly comparable.

Isolation
  Reads (read-only) : F:/yu/dl_cache/*.npy , F:/yu/DL_V2/00_split/*.csv
  Writes            : F:/yu/DL_V2/** only.  Nothing in RERUN_FULL / LAST is
                      touched.

Protocol (identical to step3b_dl_upgrade.py unless stated)
  - Train 2591 only; early stopping on Internal_Val macro AUROC, patience 10
  - class-weighted CE + label smoothing 0.1 + dropout 0.3
  - MONAI aug: flip / axial rot +-15deg / zoom 0.9-1.1 / gaussian noise
  - AdamW, cosine annealing, 60 epochs, 8-flip TTA, 3 seeds
  - LR: 1e-3 from scratch; pretrained backbones 3e-4 (head 1e-3)
  - input normalisation (x-0.45)/0.225 for every backbone (single channel)
Usage
  python 03d_dl_v2_train.py --archs r3d18 --seeds 42 --screen
  python 03d_dl_v2_train.py --archs r3d18,mc3_18 --seeds 42,123,2024
=============================================================================
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, auc, accuracy_score,
                             balanced_accuracy_score, confusion_matrix)

from monai.transforms import (Compose, RandFlip, RandRotate, RandZoom,
                              RandGaussianNoise)

ROOT = 'F:/yu/DL_V2'
CACHE = 'F:/yu/dl_cache'
OUT = f'{ROOT}/02_train'
os.makedirs(OUT, exist_ok=True)

SPLIT = pd.read_csv(f'{ROOT}/00_split/00_dataset_assignments.csv')
SPLIT = SPLIT.sort_values('ID').reset_index(drop=True)
DS_ORDER = ['Train', 'Internal_Val', 'External_Val_1', 'External_Val_2']
MEAN, STD = 0.45, 0.225

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
FLIP_COMBOS = [None, (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


# --------------------------------------------------------------- backbones
def patch_first_conv(net, conv_name):
    """Mean-reduce a 3-channel (Kinetics/ImageNet) stem to 1 channel."""
    parts = conv_name.split('.')
    parent = net
    for p in parts[:-1]:
        parent = getattr(parent, p)
    conv = getattr(parent, parts[-1])
    w = conv.weight.data.mean(dim=1, keepdim=True)
    new = nn.Conv3d(1, w.shape[0], kernel_size=conv.kernel_size,
                    stride=conv.stride, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups,
                    bias=conv.bias is not None,
                    padding_mode=conv.padding_mode)
    new.weight.data = w
    if conv.bias is not None:
        new.bias.data = conv.bias.data.clone()
    setattr(parent, parts[-1], new)


def build_backbone(arch):
    """Return (feature extractor, feature dim, batch size, pretrained)."""
    if arch in ('r3d18', 'r2p1d18', 'mc3_18', 'swin3d_t'):
        from torchvision.models.video import (r3d_18, r2plus1d_18, mc3_18,
                                              swin3d_t, s3d, R3D_18_Weights,
                                              R2Plus1D_18_Weights,
                                              MC3_18_Weights,
                                              Swin3D_T_Weights, S3D_Weights)
        if arch == 'r3d18':
            m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            patch_first_conv(m, 'stem.0'); m.fc = nn.Identity(); dim = 512
            bs = 64
        elif arch == 'r2p1d18':
            m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
            patch_first_conv(m, 'stem.0'); m.fc = nn.Identity(); dim = 512
            bs = 48
        elif arch == 'mc3_18':
            m = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            patch_first_conv(m, 'stem.0'); m.fc = nn.Identity(); dim = 512
            bs = 64
        else:
            m = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
            patch_first_conv(m, 'patch_embed.proj')
            m.head = nn.Identity(); dim = 768
            bs = 12
        return m, dim, bs, True

    if arch == 's3d':
        from torchvision.models.video import s3d, S3D_Weights
        m = s3d(weights=S3D_Weights.KINETICS400_V1)
        conv = m.features[0][0][0]          # Conv3d(3, 64, (1,7,7))
        w = conv.weight.data.sum(1, keepdim=True)   # fold RGB -> 1 channel
        conv.in_channels = 1
        conv.weight = nn.Parameter(w)
        m.avgpool = nn.AdaptiveAvgPool3d(1)  # fixed (2,7,7) kernel overflows at 64^3
        m.classifier = nn.Identity()        # feature dim 1024
        return m, 1024, 32, True

    if arch in ('slow_r50', 'csn_r101'):
        import sys
        if r'F:\yu\DL_V2\_pkgs' not in sys.path:
            sys.path.insert(0, r'F:\yu\DL_V2\_pkgs')
        from pytorchvideo.models.hub import slow_r50, csn_r101
        m = (slow_r50 if arch == 'slow_r50' else csn_r101)(pretrained=True)
        old = m.blocks[0].conv
        newc = nn.Conv3d(1, old.out_channels, old.kernel_size, old.stride,
                         old.padding, bias=old.bias is not None)
        with torch.no_grad():
            newc.weight[:] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                newc.bias[:] = old.bias
        m.blocks[0].conv = newc
        m.blocks[-1] = nn.Identity()       # drop ResNetBasicHead
        return m, 2048, 24 if arch == 'slow_r50' else 32, True

    if arch in ('effb0', 'effb3'):
        from monai.networks.nets.efficientnet import EfficientNetBN
        name = 'efficientnet-b0' if arch == 'effb0' else 'efficientnet-b3'
        net = EfficientNetBN(name, pretrained=False, spatial_dims=3,
                             in_channels=1, num_classes=3)
        head = net._fc
        dim = head.in_features
        net._fc = nn.Identity()
        return net, dim, 32 if arch == 'effb3' else 96, False

    if arch in ('dn121', 'dn169'):
        from monai.networks.nets import densenet121, densenet169
        net = (densenet121 if arch == 'dn121' else densenet169)(
            spatial_dims=3, in_channels=1, out_channels=3)
        dim = net.class_layers.out.in_features
        net.class_layers.out = nn.Identity()
        return net, dim, 32, False

    raise ValueError(f'unknown arch {arch}')


class Net(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.backbone, self.dim, self.bs, self.pretrained = \
            build_backbone(arch)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.dim, 3)

    def forward(self, x, features=False):
        h = self.backbone(x)
        if h.ndim > 2:
            h = h.mean(dim=tuple(range(2, h.ndim)))
        if features:
            return h
        return self.fc(self.drop(h))


# ------------------------------------------------------------------- data
class VolDataset(Dataset):
    def __init__(self, ids, y, train, seed, aug='base'):
        self.ids, self.y, self.train = ids, y, train
        if train:
            if aug == 'strong':
                from monai.transforms import (RandAffine, RandGaussianSmooth,
                                              RandAdjustContrast)
                self.aug = Compose([
                    RandFlip(prob=0.5, spatial_axis=None),
                    RandRotate(range_x=0.2, range_y=0.2, range_z=0.3,
                               prob=0.6, padding_mode='constant',
                               mode='nearest'),
                    RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.5,
                             mode='nearest', padding_mode='constant'),
                    RandAffine(prob=0.4, rotate_range=(0.1, 0.1, 0.1),
                               shear_range=(0.05, 0.05, 0.05),
                               translate_range=(3, 3, 3),
                               padding_mode='constant', mode='nearest'),
                    RandGaussianSmooth(prob=0.3, sigma_x=(0.5, 1.0),
                                       sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
                    RandAdjustContrast(prob=0.3, gamma=(0.8, 1.25)),
                    RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
                ])
            else:
                self.aug = Compose([
                    RandFlip(prob=0.5, spatial_axis=None),
                    RandRotate(range_x=0.0, range_y=0.0, range_z=(-0.25, 0.25),
                               prob=0.5, padding_mode='constant',
                               mode='nearest'),
                    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.4,
                             mode='nearest', padding_mode='constant'),
                    RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
                ])
            self.aug.set_random_state(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        v = np.load(f'{CACHE}/{self.ids[i]}.npy').astype(np.float32)
        v = (v - MEAN) / STD
        x = v[None]
        if self.train:
            x = self.aug(x)
        return (torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i]),
                int(self.ids[i]))


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, xb, tta=True):
    model.eval()
    combos = FLIP_COMBOS if tta else [None]
    ps = []
    for ax in combos:
        xx = torch.flip(xb, dims=[a + 2 for a in ax]) if ax else xb
        with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            out = model(xx)
        ps.append(F.softmax(out.float(), 1).cpu().numpy())
    return np.mean(ps, 0)


def evaluate(model, loader, tta=True, feats=False):
    model.eval()
    ys, ps, ids, fs = [], [], [], []
    for xb, yb, idb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        ps.append(predict(model, xb, tta=tta))
        ys.append(yb.numpy()); ids.append(idb.numpy())
        if feats:
            with torch.no_grad(), torch.autocast('cuda',
                                                 enabled=DEVICE.type == 'cuda'):
                fs.append(model(xb, features=True).float().cpu().numpy())
    out = [np.concatenate(ids), np.concatenate(ys), np.concatenate(ps)]
    if feats:
        out.append(np.concatenate(fs))
    return out


# ------------------------------------------------------------------ train
def train_one(arch, seed, max_epochs, patience, hp):
    print(f'\n=== {arch} | seed {seed} | hp {hp} ===', flush=True)
    t0 = time.time()
    set_seed(seed)

    tr = SPLIT[SPLIT.Dataset == 'Train'].sort_values('ID').reset_index(drop=True)
    iv = SPLIT[SPLIT.Dataset == 'Internal_Val'].sort_values('ID').reset_index(drop=True)
    ytr = tr.Output.values - 1
    cw = len(ytr) / (3 * np.bincount(ytr))

    model = Net(arch).to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())
    bs, pretrained = model.bs, model.pretrained
    print(f'  params {nparam/1e6:.2f}M | pretrained={pretrained} | bs={bs} | '
          f'class weights {cw.round(3)}', flush=True)

    nw = 6 if DEVICE.type == 'cuda' else 0
    dl_tr = DataLoader(VolDataset(tr.ID.values, ytr, True, seed, hp['aug']),
                       batch_size=bs, shuffle=True, num_workers=nw,
                       pin_memory=True, persistent_workers=nw > 0,
                       drop_last=False)
    dl_iv = DataLoader(VolDataset(iv.ID.values, iv.Output.values - 1, False,
                                  seed),
                       batch_size=bs, num_workers=0, pin_memory=True)

    lossf = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE),
        label_smoothing=0.1)

    bb_lr = hp['bb_lr'] if pretrained else hp['bb_lr_scratch']
    hd_lr = hp['head_lr']
    opt = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': bb_lr},
        {'params': list(model.fc.parameters()) +
         list(model.drop.parameters()), 'lr': hd_lr},
    ], weight_decay=hp['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    freeze_ep = hp['freeze_epochs'] if pretrained else 0

    best, best_state, best_ep, bad = -1.0, None, -1, 0
    hist = {'epoch': [], 'loss': [], 'ival_auc': []}
    for ep in range(max_epochs):
        if freeze_ep and ep < freeze_ep:
            for p in model.backbone.parameters():
                p.requires_grad = False
        elif freeze_ep and ep == freeze_ep:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f'  backbone unfrozen at ep {ep+1}', flush=True)
        model.train()
        tot, nb = 0.0, 0
        for xb, yb, _ in dl_tr:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE)
            lam = 1.0
            if hp['mixup'] > 0:
                lam = float(np.random.beta(hp['mixup'], hp['mixup']))
                perm = torch.randperm(xb.size(0), device=xb.device)
                xb = lam * xb + (1 - lam) * xb[perm]
            with torch.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                out = model(xb)
                if lam < 1.0:
                    loss = (lam * lossf(out, yb) +
                            (1 - lam) * lossf(out, yb[perm]))
                else:
                    loss = lossf(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        _, yv, pv = evaluate(model, dl_iv, tta=False)
        vauc = roc_auc_score(yv, pv, multi_class='ovr', average='macro')
        hist['epoch'].append(ep + 1); hist['loss'].append(tot / max(nb, 1))
        hist['ival_auc'].append(vauc)
        print(f'  ep {ep+1:3d}  loss {tot/max(nb,1):.4f}  invalAUC {vauc:.4f}'
              f'  ({time.time()-t0:.0f}s)', flush=True)
        if vauc > best:
            best, best_ep, bad = vauc, ep + 1, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop at ep {ep+1} (best {best_ep})', flush=True)
                break

    model.load_state_dict(best_state)
    dt = (time.time() - t0) / 60
    print(f'  done {dt:.1f} min | best ep {best_ep} | inval {best:.4f}',
          flush=True)
    return model, {'seed': seed, 'params': nparam, 'best_ep': best_ep,
                   'inval_auc': best, 'hist': hist, 'bs': bs,
                   'pretrained': pretrained, 'minutes': dt,
                   'hp': hp}


def metrics_row(ds, y, p):
    row = {'Dataset': ds, 'N': len(y),
           'AUC_macro': roc_auc_score(y, p, multi_class='ovr',
                                      average='macro')}
    for c in range(3):
        row[f'AUC_class{c+1}'] = roc_auc_score((y == c).astype(int), p[:, c])
    row['Accuracy'] = accuracy_score(y, p.argmax(1))
    row['Balanced_acc'] = balanced_accuracy_score(y, p.argmax(1))
    return row


def plot_curves(hists, arch, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=250)
    n = min(len(h['epoch']) for h in hists)
    ax.plot(range(1, n + 1), np.mean([h['loss'][:n] for h in hists], 0),
            color='#1f77b4', label='Train loss')
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), np.mean([h['ival_auc'][:n] for h in hists], 0),
             color='#d62728', label='IntVal macro AUROC')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss', color='#1f77b4')
    ax2.set_ylabel('IntVal macro AUROC', color='#d62728')
    ax.grid(alpha=0.3)
    ax.set_title(f'{arch} training curves (mean over {len(hists)} seeds)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------- main
def run(arch, seeds, max_epochs, patience, screen, hp):
    run_id = arch if not hp['tag'] else f'{arch}_{hp["tag"]}'
    ar = f'{OUT}/{run_id}'
    os.makedirs(ar, exist_ok=True)
    runs, models, hists = [], [], []
    for s in seeds:
        model, info = train_one(arch, s, max_epochs, patience, hp)
        models.append(model)
        # TTA predictions for every split (screen mode: Internal_Val only)
        targets = ['Internal_Val'] if screen else DS_ORDER
        preds, feats = {}, {}
        for ds in targets:
            sub = SPLIT[SPLIT.Dataset == ds].sort_values('ID').reset_index(drop=True)
            dl = DataLoader(VolDataset(sub.ID.values, sub.Output.values - 1,
                                       False, s),
                            batch_size=model.bs, num_workers=0,
                            pin_memory=True)
            ids, ys, ps, ff = evaluate(model, dl, tta=True, feats=True)
            preds[ds] = pd.DataFrame({
                'ID': ids, 'True_class': ys + 1, 'Pred_class': ps.argmax(1) + 1,
                'p_class1': ps[:, 0], 'p_class2': ps[:, 1],
                'p_class3': ps[:, 2], 'Dataset': ds})
            feats[ds] = pd.DataFrame(ff, columns=[f'f{i}' for i in range(ff.shape[1])])
            feats[ds].insert(0, 'ID', ids); feats[ds].insert(1, 'Dataset', ds)
        runs.append({**info, 'preds': preds, 'feats': feats})
        hists.append(info['hist'])
        torch.save({'arch': arch, 'seed': s, 'inval_auc': info['inval_auc'],
                    'hp': hp, 'state_dict': model.state_dict()},
                   f'{ar}/model_seed{s}.pth')
        del model
        torch.cuda.empty_cache()

    met = pd.DataFrame([{'Seed': r['seed'], 'Params': r['params'],
                         'Best_Epoch': r['best_ep'], 'Minutes': r['minutes'],
                         **metrics_row(ds, r['preds'][ds].True_class.values - 1,
                                       r['preds'][ds][['p_class1', 'p_class2',
                                                       'p_class3']].values)}
                        for r in runs for ds in r['preds']])
    met.to_csv(f'{ar}/metrics_{run_id}.csv', index=False)
    print(f'\n===== {arch} per-seed =====')
    print(met.groupby('Dataset')['AUC_macro'].agg(['mean', 'std']).round(4)
          .to_string(), flush=True)

    if not screen:
        ens = {}
        for ds in DS_ORDER:
            dfs = [r['preds'][ds].set_index('ID') for r in runs]
            pid = dfs[0].index
            ps = np.mean([d[['p_class1', 'p_class2', 'p_class3']].values
                          for d in dfs], 0)
            ens[ds] = pd.DataFrame({
                'ID': pid.values, 'True_class': dfs[0].True_class.values,
                'Pred_class': ps.argmax(1) + 1, 'p_class1': ps[:, 0],
                'p_class2': ps[:, 1], 'p_class3': ps[:, 2], 'Dataset': ds})
        ens_df = pd.concat(ens.values(), ignore_index=True)
        ens_df.to_csv(f'{ar}/predictions_ens_{run_id}.csv', index=False)
        emet = pd.DataFrame([metrics_row(ds, ens[ds].True_class.values - 1,
                                         ens[ds][['p_class1', 'p_class2',
                                                  'p_class3']].values)
                             for ds in DS_ORDER])
        emet.to_csv(f'{ar}/metrics_ens_{run_id}.csv', index=False)
        print(f'\n===== {arch} {len(runs)}-seed ensemble =====')
        print(emet.round(4).to_string(index=False), flush=True)

        best = max(runs, key=lambda r: r['inval_auc'])
        pd.concat(best['preds'].values(), ignore_index=True).to_csv(
            f'{ar}/predictions_bestseed_{run_id}.csv', index=False)
        pd.concat(best['feats'].values(), ignore_index=True).to_csv(
            f'{ar}/embeddings_bestseed_{run_id}.csv', index=False)

        plot_curves(hists, run_id, f'{ar}/training_curves_{run_id}.png')
        json.dump({'arch': arch, 'seeds': seeds,
                   'params_M': round(runs[0]['params'] / 1e6, 2),
                   'pretrained': runs[0]['pretrained'],
                   'batch': runs[0]['bs'],
                   'best_seed': best['seed'],
                   'inval_auc_by_seed': {r['seed']: round(r['inval_auc'], 4)
                                         for r in runs},
                   'minutes_per_seed': [round(r['minutes'], 1) for r in runs],
                   'protocol': 'AdamW cosine, CE+LS0.1, dropout0.3, '
                               'flip/rot/zoom/noise aug, 8-flip TTA'},
                  open(f'{ar}/info_{run_id}.json', 'w'), indent=2)
    return met


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', default='r3d18')
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--screen', action='store_true')
    ap.add_argument('--bb-lr', type=float, default=3e-4,
                    help='LR for pretrained backbone')
    ap.add_argument('--scratch-lr', type=float, default=1e-3,
                    help='LR for from-scratch backbone')
    ap.add_argument('--head-lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--freeze-epochs', type=int, default=0)
    ap.add_argument('--aug', default='base', choices=['base', 'strong'])
    ap.add_argument('--mixup', type=float, default=0.0)
    ap.add_argument('--tag', default='', help='suffix for the run folder')
    a = ap.parse_args()
    hp = {'bb_lr': a.bb_lr, 'bb_lr_scratch': a.scratch_lr,
          'head_lr': a.head_lr, 'wd': a.wd,
          'freeze_epochs': a.freeze_epochs, 'aug': a.aug,
          'mixup': a.mixup, 'tag': a.tag}
    archs = a.archs.split(',')
    seeds = [int(s) for s in a.seeds.split(',')]
    print(f'device={DEVICE} archs={archs} seeds={seeds} screen={a.screen} '
          f'hp={hp}', flush=True)
    allm = []
    for arch in archs:
        m = run(arch, seeds, a.epochs, a.patience, a.screen, hp)
        m.insert(0, 'Arch', arch)
        allm.append(m)
    pd.concat(allm, ignore_index=True).to_csv(
        f'{OUT}/metrics_all_{a.tag or "_".join(archs)}.csv', index=False)
    print('\nALL DONE', flush=True)
