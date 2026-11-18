"""Stage 4b: prediction intervals from the reserved calibration stations (split conformal).

For each outer fold:
  models are fit on development stations only (as in Stage 2) and predict both the reserved
  calibration stations and the test stations; IDW for a calibration station uses the other
  non-test stations (self excluded).
  For each depth, the 80% and 95% quantiles of |observed - predicted| over the calibration
  stations' days give symmetric intervals pred +/- q.
Coverage and width are scored at the test stations, pooled and per station. Calibration
stations were never used for fitting or tuning, so the intervals are honest out-of-station.
Models: random forest B, LightGBM A, IDW.
"""
import sys, os
SUMMARY_ONLY = '--summary-only' in sys.argv
src = open('scripts/04_baselines.py').read().split('# ---------- main loop ----------')[0]
sys.argv = ['x', 'data/processed', 'results']
exec(src)
os.makedirs('results/intervals', exist_ok=True)
LEVELS = [0.8, 0.95]
rows, qrows = [], []
for k in ([] if SUMMARY_ONLY else range(5)):
    role = np.array(folds.set_index('station')[f'role_fold{k}'].reindex(stations).tolist(), dtype=str)
    test_idx = np.where(role == 'test')[0]
    cal_idx = np.where(role == 'calib')[0]
    dev_idx = np.where(np.char.startswith(role, 'inner'))[0]
    Rmask = role != 'test'
    net = network_features(V, Rmask)
    preds = {}
    preds['idw'] = (net[0][:, cal_idx, :], net[0][:, test_idx, :])
    for name, scen in [('rf_B', 'B'), ('lgbm_A', 'A')]:
        Xtr, ytr, _ = build_rows(dev_idx, net, scen)
        ok_ = ~np.isnan(ytr)
        if name == 'rf_B':
            m = RandomForestRegressor(n_estimators=150, max_samples=0.2, min_samples_leaf=5,
                                      max_features=0.4, n_jobs=1, random_state=0)
            m.fit(Xtr[ok_].fillna(-999), ytr[ok_]); f = lambda X: m.predict(X.fillna(-999))
        else:
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
                                  subsample=0.7, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                  random_state=0, n_jobs=1, deterministic=True, force_col_wise=True)
            m.fit(Xtr[ok_], ytr[ok_]); f = m.predict
        pair = []
        for idx in (cal_idx, test_idx):
            X, _, key = build_rows(idx, net, scen)
            P = np.full((T, len(idx), len(DEPTHS)), np.nan)
            P[key.t.values, np.searchsorted(idx, key.s.values), key.d.values] = f(X)
            pair.append(P)
        preds[name] = tuple(pair)
    for name, (Pc, Pt) in preds.items():
        for d in range(len(DEPTHS)):
            rc = np.abs(V[:, cal_idx, d] - Pc[:, :, d]).ravel()
            rc = rc[~np.isnan(rc)]
            for lev in LEVELS:
                q = float(np.quantile(rc, min(1.0, lev * (1 + 1 / len(rc)))))
                qrows.append({'fold': k, 'model': name, 'depth': DEPTHS[d], 'level': lev, 'q': q, 'n_cal': len(rc)})
                for j, s in enumerate(test_idx):
                    o, p = V[:, s, d], Pt[:, j, d]
                    ok = ~np.isnan(o) & ~np.isnan(p)
                    if ok.sum() < 30:
                        continue
                    rows.append({'fold': k, 'model': name, 'depth': DEPTHS[d], 'level': lev,
                                 'station': stations[s], 'coverage': float((np.abs(o[ok] - p[ok]) <= q).mean()),
                                 'width': 2 * q, 'n': int(ok.sum())})
    print('fold', k, 'done', flush=True)

if not SUMMARY_ONLY:
    cov = pd.DataFrame(rows); qs = pd.DataFrame(qrows)
    cov.to_csv('results/intervals/station_coverage.csv', index=False)
    qs.to_csv('results/intervals/calibration_quantiles.csv', index=False)
cov = pd.read_csv('results/intervals/station_coverage.csv')
# ---------- summary (also runnable alone: python scripts/12_intervals.py --summary-only) ----------
w = lambda g: np.average(g.coverage, weights=g.n)
summ = []
for (m, lev), g in cov.groupby(['model', 'level']):
    stc = g.groupby('station').coverage.mean()
    summ.append({'model': m, 'level': lev, 'pooled_coverage': w(g), 'mean_width': g.width.mean(),
                 'station_coverage_p10': stc.quantile(0.1), 'station_coverage_median': stc.median(),
                 'share_stations_below_nominal_minus_10pts': (stc < lev - 0.10).mean()})
summ = pd.DataFrame(summ)
summ.to_csv('results/intervals/summary.csv', index=False)
bd = cov.groupby(['model', 'level', 'depth']).apply(lambda g: pd.Series({'coverage': w(g), 'width': g.width.mean()}),
                                                    include_groups=False).reset_index()
bd.to_csv('results/intervals/by_depth.csv', index=False)
pd.set_option('display.width', 200)
print(summ.round(3).to_string(index=False))
print(bd[bd.model == 'rf_B'].round(3).to_string(index=False))
