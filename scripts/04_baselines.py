"""Stage 2c: baseline pilot under the station-blocked protocol.

For each outer fold k:
  test stations   T  : never used for inputs, fitting, scaling or features
  calibration     C  : reserved; readings may serve as network inputs at prediction time,
                       but not used to fit or tune any model
  development     Dv : fit models (inner folds kept for later tuning; no tuning in the pilot)
  input network   R = C u Dv  (all functioning non-test stations on that day)

Models: mean, climatology, nearest neighbour, IDW, ordinary kriging,
        LightGBM and random forest in scenario A (sensor-assisted) and B (covariates only).
Usage: python baselines.py DATA_DIR OUT_DIR [--leak-test]
"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

D, OUT = sys.argv[1], sys.argv[2]
LEAK = '--leak-test' in sys.argv
os.makedirs(OUT, exist_ok=True)
DEPTHS = [30, 60, 90, 120, 150]
rng = np.random.default_rng(0)

panel = pd.read_parquet(f'{D}/panel.parquet')
folds = pd.read_csv(os.environ.get('FOLDS', f'{D}/folds.csv'))
st = pd.read_csv(f'{D}/stations.csv').set_index('station')
sd = pd.read_csv(f'{D}/station_depth.csv')
crops = pd.read_csv(f'{D}/crops.csv')
wx = pd.read_csv(f'{D}/weather.csv', parse_dates=['date']).set_index('date')

stations = list(folds.station)
S, dates = len(stations), pd.date_range(panel.date.min(), panel.date.max())
T = len(dates)
sidx = {s: i for i, s in enumerate(stations)}
V = np.full((T, S, len(DEPTHS)), np.nan)
ti = (panel.date - dates[0]).dt.days.values
si = panel.station.map(sidx).values
di = panel.depth.map({d: i for i, d in enumerate(DEPTHS)}).values
V[ti, si, di] = panel.vw.values
XY = st.loc[stations, ['x', 'y']].values
DIST = np.hypot(*(XY[:, None, :] - XY[None, :, :]).transpose(2, 0, 1))

# ---------- static + dynamic feature tables ----------
stat_cols = ['x', 'y', 'elev', 'slope_deg', 'tpi70m', 'curvature', 'aspect_sin', 'aspect_cos',
             'eca_spring', 'eca_fall']
bt = sd.pivot(index='station', columns='depth', values='bt_prob').loc[stations].values
crop_map = crops.set_index(['field', 'strip', 'year']).crop.to_dict()
CROP_GROUP = {'WW': 'winter_cereal', 'WT': 'winter_cereal', 'WB': 'winter_cereal',
              'SW': 'spring_cereal', 'SB': 'spring_cereal', 'SC': 'spring_oilseed',
              'WC': 'winter_oilseed', 'SL': 'spring_legume', 'SP': 'spring_legume',
              'WL': 'winter_legume', 'WP': 'winter_legume', 'Fallow': 'fallow'}
groups = sorted(set(CROP_GROUP.values()))
years = dates.year.values
CROP = np.zeros((T, S), dtype=int)  # crop group code per station-day
for j, s in enumerate(stations):
    f, k = st.loc[s, 'field'], st.loc[s, 'strip']
    for y in np.unique(years):
        c = crop_map.get((f, k, y), 'Fallow')
        if s == 'CAF217' and y == 2016:
            c = 'SC'  # footnote in CAF_CropID.txt
        CROP[years == y, j] = groups.index(CROP_GROUP.get(c, 'fallow'))
wcols = [c for c in wx.columns]
W = wx.loc[dates, wcols].values


def network_features(Vin, Rmask):
    """Network features at every station from input stations Rmask (bool[S]); self excluded."""
    M = (~np.isnan(Vin)) & Rmask[None, :, None]
    Vz = np.where(M, Vin, 0.0)
    w = 1.0 / np.maximum(DIST, 1.0) ** 2
    np.fill_diagonal(w, 0.0)
    num = np.einsum('ij,tjd->tid', w, Vz)
    den = np.einsum('ij,tjd->tid', w, M.astype(float))
    idw = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    # leave-self-out network mean and sd
    n = M.sum(1, keepdims=True) - M
    s1 = Vz.sum(1, keepdims=True) - Vz
    s2 = (Vz ** 2).sum(1, keepdims=True) - Vz ** 2
    mean = np.where(n > 0, s1 / np.maximum(n, 1), np.nan)
    sdv = np.sqrt(np.maximum(np.where(n > 1, s2 / np.maximum(n, 1) - mean ** 2, np.nan), 0))
    # nearest available neighbour (excluding self)
    nn = np.full_like(idw, np.nan)
    nnd = np.full_like(idw, np.nan)
    order = np.argsort(DIST, axis=1)[:, 1:]
    for i in range(S):
        for j in order[i]:
            fill = np.isnan(nn[:, i, :]) & M[:, j, :]
            nn[:, i, :] = np.where(fill, Vin[:, j, :], nn[:, i, :])
            nnd[:, i, :] = np.where(fill, DIST[i, j], nnd[:, i, :])
    return idw, mean, sdv, nn, nnd


def build_rows(sel_st, net, scenario):
    """Long feature matrix for stations sel_st (indices) over all dates/depths."""
    idw, mean, sdv, nn, nnd = net
    tt, ss, dd = np.meshgrid(np.arange(T), sel_st, np.arange(len(DEPTHS)), indexing='ij')
    tt, ss, dd = tt.ravel(), ss.ravel(), dd.ravel()
    cols = {'depth': np.array(DEPTHS)[dd]}
    for c in stat_cols:
        cols[c] = st.loc[stations, c].values[ss]
    cols['bt_prob'] = bt[ss, dd]
    cols['crop_group'] = CROP[tt, ss]
    for j, c in enumerate(wcols):
        cols[c] = W[tt, j]
    if scenario == 'A':
        cols['idw'] = idw[tt, ss, dd]
        cols['net_mean'] = mean[tt, ss, dd]
        cols['net_sd'] = sdv[tt, ss, dd]
        cols['nn'] = nn[tt, ss, dd]
        cols['nn_dist'] = nnd[tt, ss, dd]
        for k, dep in enumerate(DEPTHS):
            cols[f'idw_d{dep}'] = idw[tt, ss, k]
    X = pd.DataFrame(cols)
    y = V[tt, ss, dd]
    key = pd.DataFrame({'t': tt, 's': ss, 'd': dd})
    return X, y, key


# ---------- ordinary kriging ----------
def expo(h, nug, psill, rng_):
    return nug + psill * (1 - np.exp(-h / rng_))


def fit_variogram(dev_idx, d):
    sub = V[:, dev_idx, d]
    Dsub = DIST[np.ix_(dev_idx, dev_idx)]
    iu = np.triu_indices(len(dev_idx), 1)
    h = Dsub[iu]
    g = 0.5 * (sub[:, iu[0]] - sub[:, iu[1]]) ** 2
    g = np.nanmean(g, axis=0)
    bins = np.array([0, 90, 140, 200, 280, 380, 520, 1000])
    hb, gb = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (h >= a) & (h < b) & ~np.isnan(g)
        if m.sum() >= 3:
            hb.append(h[m].mean()); gb.append(g[m].mean())
    hb, gb = np.array(hb), np.array(gb)
    try:
        p, _ = curve_fit(expo, hb, gb, p0=[gb.min() * 0.5, gb.max(), 200],
                         bounds=([0, 1e-8, 20], [np.inf, np.inf, 5000]), maxfev=20000)
    except Exception:
        p = np.array([gb.min() * 0.5, gb.max(), 200.0])
    return p


def ok_predict(R_idx, test_idx, d, p):
    out = np.full((T, len(test_idx)), np.nan)
    gam = lambda h: np.where(h > 0, expo(h, *p), 0.0)
    for t in range(T):
        avail = R_idx[~np.isnan(V[t, R_idx, d])]
        n = len(avail)
        if n < 3:
            continue
        A = np.ones((n + 1, n + 1)); A[-1, -1] = 0
        A[:n, :n] = gam(DIST[np.ix_(avail, avail)])
        b = np.ones((n + 1, len(test_idx)))
        b[:n] = gam(DIST[np.ix_(avail, test_idx)])
        lam = np.linalg.solve(A, b)[:n]
        out[t] = lam.T @ V[t, avail, d]
    return out


# ---------- main loop ----------
preds = []
meta = {}
t0 = time.time()
doy = dates.dayofyear.values
fold_ids = [0] if LEAK else range(5)
for k in fold_ids:
    role = np.array(folds.set_index('station')[f'role_fold{k}'].reindex(stations).tolist(), dtype=str)
    test_idx = np.where(role == 'test')[0]
    dev_idx = np.where(np.char.startswith(role, 'inner'))[0]
    R_idx = np.where(role != 'test')[0]
    Rmask = role != 'test'
    Vin = V.copy()
    if LEAK:  # corrupt test-station readings: predictions must not change
        Vin[:, test_idx, :] = rng.uniform(0, 0.6, Vin[:, test_idx, :].shape)
    net = network_features(Vin, Rmask)
    idw, mean, sdv, nn, nnd = net
    yT = V[:, test_idx, :]
    P = {}
    # simple references (fit on development stations only)
    P['mean'] = np.broadcast_to(np.nanmean(V[:, dev_idx, :], axis=(0, 1)), yT.shape)
    clim = np.full((367, len(DEPTHS)), np.nan)
    dv = V[:, dev_idx, :]
    for dd_ in range(1, 367):
        win = np.abs(((doy - dd_ + 183) % 366) - 183) <= 15
        clim[dd_] = np.nanmean(dv[win], axis=(0, 1))
    P['climatology'] = np.broadcast_to(clim[doy][:, None, :], yT.shape)
    P['nearest_neighbour'] = nn[:, test_idx, :]
    P['idw'] = idw[:, test_idx, :]
    # ordinary kriging
    okp = np.full(yT.shape, np.nan)
    vg = {}
    for d in range(len(DEPTHS)):
        p = fit_variogram(dev_idx, d)
        vg[DEPTHS[d]] = [float(x) for x in p]
        if not LEAK:
            okp[:, :, d] = ok_predict(R_idx, test_idx, d, p)
    P['ordinary_kriging'] = okp
    # tree models
    for scen in ['A', 'B']:
        Xtr, ytr, _ = build_rows(dev_idx, net, scen)
        ok_ = ~np.isnan(ytr)
        Xte, _, kte = build_rows(test_idx, net, scen)
        m1 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                               min_child_samples=50, subsample=0.7, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=2,
                               deterministic=True, force_col_wise=True)
        m1.fit(Xtr[ok_], ytr[ok_])
        out = np.full(yT.shape, np.nan)
        out[kte.t.values, np.searchsorted(test_idx, kte.s.values), kte.d.values] = m1.predict(Xte)
        P[f'lgbm_{scen}'] = out
        if not LEAK:
            m2 = RandomForestRegressor(n_estimators=150, max_samples=0.2, min_samples_leaf=5,
                                       max_features=0.4, n_jobs=2, random_state=0)
            m2.fit(Xtr[ok_].fillna(-999), ytr[ok_])
            out = np.full(yT.shape, np.nan)
            out[kte.t.values, np.searchsorted(test_idx, kte.s.values), kte.d.values] = \
                m2.predict(Xte.fillna(-999))
            P[f'rf_{scen}'] = out
    meta[k] = {'n_test': int(len(test_idx)), 'n_dev': int(len(dev_idx)),
               'n_calib': int((role == 'calib').sum()), 'variogram_nug_psill_range': vg}
    # collect
    tt, jj, dd = np.meshgrid(np.arange(T), np.arange(len(test_idx)), np.arange(len(DEPTHS)), indexing='ij')
    base = pd.DataFrame({'date': dates[tt.ravel()], 'station': np.array(stations)[test_idx][jj.ravel()],
                         'depth': np.array(DEPTHS)[dd.ravel()], 'fold': k, 'obs': yT.ravel()})
    for name, arr in P.items():
        base[name] = np.asarray(arr).ravel()
    base['clim_ref'] = np.asarray(P['climatology']).ravel()
    preds.append(base[base.obs.notna()])
    print(f'fold {k} done {time.time() - t0:.0f}s', flush=True)

pr = pd.concat(preds, ignore_index=True)
if LEAK:
    # rerun fold 0 without corruption and compare
    pr.to_parquet(f'{OUT}/leak_fold0.parquet')
else:
    pr.to_parquet(f'{OUT}/predictions.parquet')
    json.dump(meta, open(f'{OUT}/fold_meta.json', 'w'), indent=1)
print('rows', len(pr))
