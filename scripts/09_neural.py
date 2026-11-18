"""Stage 3c: neural models for sensor-assisted reconstruction (scenario A), station-blocked.

Models
  gcn           GRU per node (inputs + weather) -> 2 graph-convolution layers (Gaussian kernel)
  attn          GRU per node -> 2 spatial self-attention layers with learned distance bias
  srn           proposed scale-recursive network: one shared operator F applied recursively over
                dyadic neighbourhood scales (90, 180, 360, 720 m) with scale decay
                alpha_j = exp(-kappa * j); kappa initialised from the training-only scaling
                exponent of station anomalies and then freely updated.
All models share the output: level head (crop group x depth, no network input), trained on
station-year means, plus dynamics head, trained on within-year anomalies; prediction = sum.
  srn_unshared  same as srn but a separate operator per scale (weight-sharing ablation)

Protocol (fold k, from config/folds.csv)
  test stations   never visible: their readings are zeroed and masked at every step
  calib stations  visible as inputs, never used as training targets or for early stopping
  development     training targets (random node masking), except one inner fold held out
                  for early stopping (its inputs are masked when it is scored)
  normalisation   statistics from development stations only
Outputs results/neural/{model}_f{k}_s{seed}.parquet with columns date, station, depth, fold, obs, pred.
Usage: python scripts/09_neural.py --model srn --fold 0 --seed 0 [--leak-test] [--epochs 30]
"""
import argparse, os, time, json, math
import numpy as np, pandas as pd, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True, choices=['gcn', 'attn', 'srn', 'srn_unshared'])
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--patience', type=int, default=6)
ap.add_argument('--window', type=int, default=14)
ap.add_argument('--hidden', type=int, default=48)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--kappa-mode', default='init', choices=['free', 'init', 'constrained'])
ap.add_argument('--data', default='data/processed')
ap.add_argument('--folds', default='config/folds.csv')
ap.add_argument('--out', default='results/neural')
ap.add_argument('--leak-test', action='store_true')
ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.use_deterministic_algorithms(True, warn_only=True)
dev_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPTHS = [30, 60, 90, 120, 150]
ND = len(DEPTHS)

# ---------------- data ----------------
panel = pd.read_parquet(f'{a.data}/panel.parquet')
folds = pd.read_csv(a.folds)
st = pd.read_csv(f'{a.data}/stations.csv').set_index('station')
sd = pd.read_csv(f'{a.data}/station_depth.csv')
crops = pd.read_csv(f'{a.data}/crops.csv')
wx = pd.read_csv(f'{a.data}/weather.csv', parse_dates=['date']).set_index('date')
stations = list(folds.station)
S = len(stations)
dates = pd.date_range(panel.date.min(), panel.date.max())
T = len(dates)
V = np.full((T, S, ND), np.nan, dtype=np.float32)
V[(panel.date - dates[0]).dt.days.values, panel.station.map({s: i for i, s in enumerate(stations)}).values,
  panel.depth.map({d: i for i, d in enumerate(DEPTHS)}).values] = panel.vw.values
XY = st.loc[stations, ['x', 'y']].values
DIST = np.hypot(*(XY[:, None, :] - XY[None, :, :]).transpose(2, 0, 1)).astype(np.float32)

role = np.array(folds.set_index('station')[f'role_fold{a.fold}'].reindex(stations).tolist(), dtype=str)
test = role == 'test'
calib = role == 'calib'
inner = [r for r in sorted(set(role)) if r.startswith('inner')]
val_name = inner[0]
val = role == val_name
train_t = np.char.startswith(role, 'inner') & ~val  # training targets
devm = np.char.startswith(role, 'inner')            # all development stations
visible = ~test                                                  # may serve as inputs

if a.leak_test:  # corrupt test-station readings; predictions must not change
    rng = np.random.default_rng(123)
    V[:, test, :] = rng.uniform(0, 0.6, V[:, test, :].shape).astype(np.float32)

# normalisation from development stations only
mu_d = np.nanmean(V[:, devm, :], axis=(0, 1)).astype(np.float32)
sd_d = np.nanstd(V[:, devm, :], axis=(0, 1)).astype(np.float32)
Z = (V - mu_d) / sd_d
M = ~np.isnan(Z)
Zf = np.where(M, Z, 0).astype(np.float32)

# station x depth x year mean (level target) and within-year anomaly (dynamics target)
OFF = np.zeros_like(Zf)
yrs = dates.year.values
for y in np.unique(yrs):
    m_ = np.nanmean(np.where(M[yrs == y], Z[yrs == y], np.nan), axis=0)  # [S,ND]
    OFF[yrs == y] = np.nan_to_num(m_, nan=0.0)
ANOM = np.where(M, Zf - OFF, 0).astype(np.float32)

# static features (standardised with development stations)
scols = ['x', 'y', 'elev', 'slope_deg', 'tpi70m', 'curvature', 'aspect_sin', 'aspect_cos', 'eca_spring', 'eca_fall']
Sx = st.loc[stations, scols].values.astype(np.float32)
bt = sd.pivot(index='station', columns='depth', values='bt_prob').loc[stations].values.astype(np.float32)
Sx = np.c_[Sx, bt]
Sx = (Sx - np.nanmean(Sx[devm], 0)) / (np.nanstd(Sx[devm], 0) + 1e-6)
Sx = np.nan_to_num(Sx, nan=0.0).astype(np.float32)  # missing terrain derivatives (DEM edge) -> development mean
CROP_GROUP = {'WW': 0, 'WT': 0, 'WB': 0, 'SW': 1, 'SB': 1, 'SC': 2, 'WC': 2,
              'SL': 3, 'SP': 3, 'WL': 3, 'WP': 3, 'Fallow': 4}
cm = crops.set_index(['field', 'strip', 'year']).crop.to_dict()
CR = np.zeros((T, S, 5), dtype=np.float32)
for j, s in enumerate(stations):
    for y in np.unique(dates.year):
        c = cm.get((st.loc[s, 'field'], st.loc[s, 'strip'], y), 'Fallow')
        if s == 'CAF217' and y == 2016:
            c = 'SC'
        CR[dates.year == y, j, CROP_GROUP.get(c, 4)] = 1
W = wx.loc[dates].values.astype(np.float32)
W = (W - W.mean(0)) / (W.std(0) + 1e-6)

# training-only scaling exponent of station anomalies (for kappa initialisation)
def scaling_exponent():
    A = Z[:, devm, :] - np.nanmean(Z[:, devm, :], axis=0, keepdims=True)
    D = DIST[np.ix_(devm, devm)]
    iu = np.triu_indices(devm.sum(), 1)
    g = np.nanmean(0.5 * (A[:, iu[0], :] - A[:, iu[1], :]) ** 2, axis=(0, 2))
    h = D[iu]
    bins = [60, 90, 180, 360, 720]
    hb, gb = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (h >= lo) & (h < hi)
        if m.sum() > 3:
            hb.append(h[m].mean()); gb.append(g[m].mean())
    slope = np.polyfit(np.log(hb), np.log(gb), 1)[0]  # gamma ~ h^(2H)
    return float(max(slope / 2, 0.0)), [float(x) for x in hb], [float(x) for x in gb]

H_emp, hb, gb = scaling_exponent()
kappa0 = H_emp * math.log(2)

# ---------------- model ----------------
Fw, Fs = W.shape[1], Sx.shape[1]
Dt = torch.tensor(DIST, device=dev_)
RADII = [90.0, 180.0, 360.0, 720.0]


class Encoder(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.gru = nn.GRU(2 * ND + Fw + 5, h, batch_first=True)
        s.st = nn.Sequential(nn.Linear(Fs, h), nn.ReLU())
        s.mix = nn.Linear(2 * h, h)

    def forward(s, obs, msk, w, cr, stat):  # obs/msk [B,L,S,ND], w [B,L,Fw], cr [B,L,S,5]
        B, L, S_, _ = obs.shape
        x = torch.cat([obs, msk, w[:, :, None, :].expand(B, L, S_, Fw), cr], -1)
        x = x.permute(0, 2, 1, 3).reshape(B * S_, L, -1)
        _, hN = s.gru(x)
        h = hN[-1].view(B, S_, -1)
        return torch.relu(s.mix(torch.cat([h, s.st(stat)[None].expand(B, -1, -1)], -1)))


class Head(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.dyn = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ND))
        # level: crop group x depth only (no network input). Static covariates are excluded
        # because station levels are not predictable from them (Stage 2) and a level head on
        # 16 static features memorises the ~27 training stations (validation loss rises).
        s.off = nn.Linear(5, ND)

    def forward(s, h, stat, cr_now):
        return s.off(cr_now), s.dyn(h)


class GCN(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.enc, s.head = Encoder(h), Head(h)
        A = torch.exp(-(Dt / 150.0) ** 2); A.fill_diagonal_(0)
        s.register_buffer('A', A / A.sum(1, keepdim=True))
        s.l = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(2)])

    def forward(s, obs, msk, w, cr, stat):
        h = s.enc(obs, msk, w, cr, stat)
        for l in s.l:
            h = h + torch.relu(l(torch.cat([h, s.A @ h], -1)))
        return s.head(h, stat, cr[:, -1])


class ATTN(nn.Module):
    def __init__(s, h, heads=4):
        super().__init__()
        s.enc, s.head = Encoder(h), Head(h)
        s.att = nn.ModuleList([nn.MultiheadAttention(h, heads, batch_first=True) for _ in range(2)])
        s.ff = nn.ModuleList([nn.Sequential(nn.Linear(h, h), nn.ReLU()) for _ in range(2)])
        s.lam = nn.Parameter(torch.full((heads,), 0.01))
        s.heads = heads

    def forward(s, obs, msk, w, cr, stat):
        h = s.enc(obs, msk, w, cr, stat)
        B, S_, _ = h.shape
        bias = -(torch.relu(s.lam)[:, None, None] * Dt[None])          # [heads,S,S]
        bias = bias.repeat(B, 1, 1)
        for att, ff in zip(s.att, s.ff):
            z, _ = att(h, h, h, attn_mask=bias)
            h = h + z
            h = h + ff(h)
        return s.head(h, stat, cr[:, -1])


class SRN(nn.Module):
    def __init__(s, h, shared=True, kappa_mode='init'):
        super().__init__()
        s.enc, s.head = Encoder(h), Head(h)
        n = len(RADII)
        mk = lambda: nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))
        s.F = nn.ModuleList([mk()] if shared else [mk() for _ in range(n)])
        s.shared = shared
        Ns = []
        for r in RADII:
            Nm = ((Dt <= r) & (Dt > 0)).float()
            Ns.append(Nm / Nm.sum(1, keepdim=True).clamp(min=1))
        s.register_buffer('N', torch.stack(Ns))                        # [J,S,S]
        k0 = kappa0 if kappa_mode in ('init', 'constrained') else 0.5
        s.kappa = nn.Parameter(torch.tensor(float(k0)), requires_grad=kappa_mode != 'constrained')
        s.norm = nn.LayerNorm(h)

    def forward(s, obs, msk, w, cr, stat):
        h = s.enc(obs, msk, w, cr, stat)
        prev = h
        for j in range(len(RADII)):
            agg = s.N[j] @ h                                           # neighbourhood mean at scale j
            detail = agg - (s.N[j - 1] @ h if j > 0 else h)            # band-pass (graph-wavelet-like)
            F = s.F[0] if s.shared else s.F[j]
            h = s.norm(h + torch.exp(-s.kappa * j) * F(torch.cat([h, agg, detail], -1)))
        return s.head(h, stat, cr[:, -1])


net = {'gcn': lambda: GCN(a.hidden), 'attn': lambda: ATTN(a.hidden),
       'srn': lambda: SRN(a.hidden, True, a.kappa_mode),
       'srn_unshared': lambda: SRN(a.hidden, False, a.kappa_mode)}[a.model]().to(dev_)
nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)

# ---------------- batching ----------------
L = a.window
Zp = np.concatenate([np.zeros((L - 1, S, ND), np.float32), Zf])
Mp = np.concatenate([np.zeros((L - 1, S, ND), bool), M])
Wp = np.concatenate([np.repeat(W[:1], L - 1, 0), W])
CRp = np.concatenate([np.repeat(CR[:1], L - 1, 0), CR])
statT = torch.tensor(Sx, device=dev_)


def batch(ts, hide):
    """ts: day indices; hide: bool[B,S] stations whose inputs are hidden (targets / test)."""
    idx = ts[:, None] + np.arange(L)[None]                              # window in padded arrays
    obs = Zp[idx].copy(); msk = Mp[idx].copy()
    hid = hide | test[None, :]
    obs[np.broadcast_to(hid[:, None, :, None], obs.shape)] = 0
    msk[np.broadcast_to(hid[:, None, :, None], msk.shape)] = False
    t = lambda x: torch.tensor(x, device=dev_)
    return t(obs), t(msk.astype(np.float32)), t(Wp[idx]), t(CRp[idx])


def loss_on(pred, ts, tgt, parts=False):
    off, dyn = pred
    t = lambda x: torch.tensor(x[ts], device=dev_)
    m = t(M) & torch.tensor(tgt, device=dev_)[:, :, None]
    n = m.sum().clamp(min=1)
    ld = ((dyn - t(ANOM)) ** 2 * m).sum() / n
    lo = ((off - t(OFF)) ** 2 * m).sum() / n
    return (ld, lo) if parts else ld + lo


off_params = [p for n_, p in net.named_parameters() if 'head.off' in n_]
other = [p for n_, p in net.named_parameters() if 'head.off' not in n_]
opt = torch.optim.AdamW([{'params': other, 'weight_decay': 1e-4},
                         {'params': off_params, 'weight_decay': 5e-2}], lr=a.lr)
rng = np.random.default_rng(a.seed)
train_idx_st = np.where(train_t)[0]
best, bad, hist = 1e9, 0, []
t0 = time.time()
for ep in range(a.epochs):
    net.train()
    order = rng.permutation(T)
    tl = []
    for b in range(0, T, a.batch):
        ts = order[b:b + a.batch]
        tgt = np.zeros((len(ts), S), bool)
        for i in range(len(ts)):  # mask a random ~25% of training stations as targets
            tgt[i, rng.choice(train_idx_st, max(1, len(train_idx_st) // 4), replace=False)] = True
        hide = tgt | val[None, :]  # validation stations never visible during training
        pred = net(*batch(ts, hide), statT)
        loss = loss_on(pred, ts, tgt)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        tl.append(loss.item())
    net.eval()
    with torch.no_grad():
        vd, vo = [], []
        for b in range(0, T, 64):
            ts = np.arange(b, min(b + 64, T))
            hide = np.broadcast_to(val[None, :], (len(ts), S))
            ld, lo = loss_on(net(*batch(ts, hide), statT), ts, hide, parts=True)
            vd.append(ld.item()); vo.append(lo.item())
    vl = float(np.mean(vd) + np.mean(vo))
    hist.append({'epoch': ep, 'train': float(np.mean(tl)), 'val': vl, 'val_dyn': float(np.mean(vd)),
                 'val_level': float(np.mean(vo)), 'sec': round(time.time() - t0)})
    print(hist[-1], flush=True)
    if vl < best - 1e-4:
        best, bad = vl, 0
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience:
            break
net.load_state_dict(state)

# ---------------- predict test stations ----------------
net.eval()
P = np.zeros((T, S, ND), np.float32)
with torch.no_grad():
    for b in range(0, T, 64):
        ts = np.arange(b, min(b + 64, T))
        off, dyn = net(*batch(ts, np.zeros((len(ts), S), bool)), statT)
        P[ts] = (off + dyn).cpu().numpy()
P = P * sd_d + mu_d
ti = np.where(test)[0]
tt, jj, dd = np.meshgrid(np.arange(T), ti, np.arange(ND), indexing='ij')
Vtrue = np.full((T, S, ND), np.nan, np.float32)
Vtrue[(panel.date - dates[0]).dt.days.values, panel.station.map({s: i for i, s in enumerate(stations)}).values,
      panel.depth.map({d: i for i, d in enumerate(DEPTHS)}).values] = panel.vw.values
out = pd.DataFrame({'date': dates[tt.ravel()], 'station': np.array(stations)[jj.ravel()],
                    'depth': np.array(DEPTHS)[dd.ravel()], 'fold': a.fold,
                    'obs': Vtrue[tt.ravel(), jj.ravel(), dd.ravel()],
                    'pred': P[tt.ravel(), jj.ravel(), dd.ravel()]})
out = out[out.obs.notna()]
tag = f'{a.model}_f{a.fold}_s{a.seed}' + ('_leak' if a.leak_test else '')
out.to_parquet(f'{a.out}/{tag}.parquet')
meta = {'model': a.model, 'fold': a.fold, 'seed': a.seed, 'params': nparams, 'epochs_run': len(hist),
        'best_val_mse_z': best, 'seconds': round(time.time() - t0), 'device': str(dev_),
        'H_emp': H_emp, 'kappa0': kappa0, 'variogram_bins_m': hb, 'variogram_gamma_z': gb,
        'kappa_final': float(net.kappa.item()) if hasattr(net, 'kappa') else None,
        'val_inner_fold': val_name, 'args': vars(a), 'history': hist}
json.dump(meta, open(f'{a.out}/{tag}.json', 'w'), indent=1)
print('done', tag, 'params', nparams, 'sec', meta['seconds'])
