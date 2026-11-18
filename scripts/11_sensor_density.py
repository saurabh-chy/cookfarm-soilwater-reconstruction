"""Stage 4a: sensor-density and placement experiment (both targets).

For each outer fold, the input network (all non-test stations) is reduced to a fraction
f in {1.0, 0.75, 0.5, 0.3, 0.15} of its stations:
  random   3 independent random draws per level
  spread   1 draw by farthest-point sampling (spatially even placement)
Test stations are fixed. Models: IDW, ordinary kriging, LightGBM A (network features computed
from the reduced network; training targets are all development stations, features exclude self).
Random forest B uses no network inputs, so its error is constant across densities (reference).
Metrics per held-out station x depth: RMSE (level) and ubRMSE (dynamics).
"""
import sys, time
src = open('scripts/04_baselines.py').read().split('# ---------- main loop ----------')[0]
sys.argv = ['x', 'data/processed', 'results']
exec(src)
import os
os.makedirs('results/sensor_density', exist_ok=True)
FRACS = [1.0, 0.75, 0.5, 0.3, 0.15]
rng = np.random.default_rng(11)


def farthest_point(cands, n):
    c = XY[cands].mean(0)
    chosen = [cands[np.argmin(np.hypot(*(XY[cands] - c).T))]]
    while len(chosen) < n:
        d = DIST[np.ix_(cands, chosen)].min(1)
        chosen.append(cands[np.argmax(d)])
    return np.array(chosen)


def series_metrics(pred, obs):
    rows = []
    for j in range(pred.shape[1]):
        for d in range(pred.shape[2]):
            o, p = obs[:, j, d], pred[:, j, d]
            ok = ~np.isnan(o) & ~np.isnan(p)
            if ok.sum() < 30:
                continue
            e = p[ok] - o[ok]
            rows.append((j, d, np.sqrt((e ** 2).mean()), np.sqrt(((e - e.mean()) ** 2).mean())))
    return rows


out, t0 = [], time.time()
for k in range(5):
    role = np.array(folds.set_index('station')[f'role_fold{k}'].reindex(stations).tolist(), dtype=str)
    test_idx = np.where(role == 'test')[0]
    dev_idx = np.where(np.char.startswith(role, 'inner'))[0]
    R_all = np.where(role != 'test')[0]
    yT = V[:, test_idx, :]
    draws = []
    for f in FRACS:
        n = max(3, int(round(f * len(R_all))))
        if f == 1.0:
            draws.append((f, 'all', 0, R_all))
            continue
        for r in range(3):
            draws.append((f, 'random', r, np.sort(rng.choice(R_all, n, replace=False))))
        draws.append((f, 'spread', 0, np.sort(farthest_point(R_all, n))))
    for f, kind, r, R_idx in draws:
        Rmask = np.zeros(S, bool); Rmask[R_idx] = True
        net = network_features(V, Rmask)
        preds = {'idw': net[0][:, test_idx, :]}
        okp = np.full(yT.shape, np.nan)
        for d in range(len(DEPTHS)):
            vg = fit_variogram(dev_idx, d)  # variogram from development stations (unchanged)
            okp[:, :, d] = ok_predict(R_idx, test_idx, d, vg)
        preds['ordinary_kriging'] = okp
        Xtr, ytr, _ = build_rows(dev_idx, net, 'A')
        ok_ = ~np.isnan(ytr)
        Xte, _, kte = build_rows(test_idx, net, 'A')
        m1 = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                               subsample=0.7, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                               random_state=0, n_jobs=2, deterministic=True, force_col_wise=True)
        m1.fit(Xtr[ok_], ytr[ok_])
        pl = np.full(yT.shape, np.nan)
        pl[kte.t.values, np.searchsorted(test_idx, kte.s.values), kte.d.values] = m1.predict(Xte)
        preds['lgbm_A'] = pl
        for m, P in preds.items():
            for j, d, rmse, ub in series_metrics(P, yT):
                out.append({'fold': k, 'frac': f, 'placement': kind, 'draw': r, 'n_inputs': len(R_idx),
                            'model': m, 'station': stations[test_idx[j]], 'depth': DEPTHS[d],
                            'rmse': rmse, 'ubrmse': ub})
        print(f'fold {k} frac {f} {kind}{r} n={len(R_idx)} {time.time() - t0:.0f}s', flush=True)

res = pd.DataFrame(out)
res.to_csv('results/sensor_density/station_metrics.csv', index=False)
# station-level: mean over draws and depths per station, then across stations
g = res.groupby(['model', 'frac', 'placement', 'station'])[['rmse', 'ubrmse']].mean().reset_index()
summ = g.groupby(['model', 'frac', 'placement']).agg(
    rmse=('rmse', 'mean'), ubrmse=('ubrmse', 'mean'), n_stations=('station', 'nunique')).reset_index()
summ['n_inputs_mean'] = summ.apply(lambda r: res[(res.frac == r.frac) & (res.placement == r.placement)]
                                   .groupby('fold').n_inputs.first().mean(), axis=1)
summ.to_csv('results/sensor_density/summary.csv', index=False)
pd.set_option('display.width', 200)
print(summ.round(4).to_string(index=False))
