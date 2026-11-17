"""Stage 2d: metrics for the baseline pilot.

Station is the independent unit: metrics are computed per held-out station x depth, then
averaged across stations; 95% intervals come from resampling stations (2000 draws).
Also reports pooled outer-prediction metrics by depth and the leakage test result.
"""
import sys, json
import numpy as np, pandas as pd

R = sys.argv[1]
pr = pd.read_parquet(f'{R}/predictions.parquet')
MODELS = ['mean', 'climatology', 'nearest_neighbour', 'idw', 'ordinary_kriging',
          'rf_B', 'lgbm_B', 'rf_A', 'lgbm_A']
rng = np.random.default_rng(1)


def station_metrics(g, m):
    e = g[m] - g.obs
    ok = e.notna()
    e, o, p, c = e[ok], g.obs[ok], g[m][ok], g.clim_ref[ok]
    if len(e) < 30:
        return None
    rmse = np.sqrt((e ** 2).mean())
    return {'rmse': rmse, 'mae': e.abs().mean(), 'bias': e.mean(),
            'ubrmse': np.sqrt(max(rmse ** 2 - e.mean() ** 2, 0)),
            # temporal anomaly skill at the station: error vs training-fold climatology
            'skill_vs_clim': 1 - (e ** 2).mean() / ((c - o) ** 2).mean(),
            'r_time': np.corrcoef(o, p)[0, 1] if o.std() > 0 and p.std() > 0 else np.nan,
            'n': len(e)}


rows = []
for (s, d, f), g in pr.groupby(['station', 'depth', 'fold']):
    for m in MODELS:
        r = station_metrics(g, m)
        if r:
            rows.append({'station': s, 'depth': d, 'fold': f, 'model': m, **r})
sm = pd.DataFrame(rows)
sm.to_csv(f'{R}/station_metrics.csv', index=False)


def boot_ci(x):
    x = np.asarray(x)
    b = rng.choice(x, (2000, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


summ = []
for (m, d), g in sm.groupby(['model', 'depth']):
    lo, hi = boot_ci(g.rmse)
    summ.append({'model': m, 'depth': d, 'n_stations': len(g), 'rmse': g.rmse.mean(),
                 'rmse_lo': lo, 'rmse_hi': hi, 'mae': g.mae.mean(), 'bias': g.bias.mean(),
                 'ubrmse': g.ubrmse.mean(), 'skill_vs_clim': g.skill_vs_clim.median(),
                 'r_time': g.r_time.median()})
for m, g in sm.groupby('model'):
    lo, hi = boot_ci(g.groupby('station').rmse.mean())
    summ.append({'model': m, 'depth': 'all', 'n_stations': g.station.nunique(),
                 'rmse': g.groupby('station').rmse.mean().mean(), 'rmse_lo': lo, 'rmse_hi': hi,
                 'mae': g.mae.mean(), 'bias': g.bias.mean(), 'ubrmse': g.ubrmse.mean(),
                 'skill_vs_clim': g.skill_vs_clim.median(), 'r_time': g.r_time.median()})
summ = pd.DataFrame(summ)
summ.to_csv(f'{R}/summary_by_model_depth.csv', index=False)

# pooled outer predictions (R2 across stations and dates) by depth
pooled = []
for d, g in pr.groupby('depth'):
    for m in MODELS:
        ok = g[m].notna()
        e = g[m][ok] - g.obs[ok]
        pooled.append({'model': m, 'depth': d, 'coverage_of_targets': ok.mean(),
                       'rmse_pooled': np.sqrt((e ** 2).mean()),
                       'r2_pooled': 1 - (e ** 2).sum() / ((g.obs[ok] - g.obs[ok].mean()) ** 2).sum()})
pooled = pd.DataFrame(pooled)
pooled.to_csv(f'{R}/pooled_by_depth.csv', index=False)

# paired difference vs IDW (station-level, all depths)
pair = []
w = sm.pivot_table(index=['station', 'depth'], columns='model', values='rmse')
for m in MODELS:
    if m == 'idw':
        continue
    diff = (w[m] - w['idw']).groupby('station').mean().dropna()
    lo, hi = boot_ci(diff)
    pair.append({'model': m, 'mean_rmse_minus_idw': diff.mean(), 'lo': lo, 'hi': hi,
                 'stations_better_than_idw': int((diff < 0).sum()), 'n': len(diff)})
pair = pd.DataFrame(pair)
pair.to_csv(f'{R}/paired_vs_idw.csv', index=False)

# leakage test: fold-0 predictions with corrupted test-station readings must be identical
leak = {}
try:
    lk = pd.read_parquet(f'{R}/leak_fold0.parquet')
    base = pr[pr.fold == 0].set_index(['date', 'station', 'depth'])
    lk = lk.set_index(['date', 'station', 'depth']).loc[base.index]
    for m in ['nearest_neighbour', 'idw', 'lgbm_A', 'lgbm_B', 'climatology', 'mean']:
        leak[m] = float(np.nanmax(np.abs(lk[m].values - base[m].values)))
except Exception as ex:
    leak['error'] = str(ex)
json.dump(leak, open(f'{R}/leak_test.json', 'w'), indent=1)

pd.set_option('display.width', 200)
print(summ[summ.depth == 'all'].sort_values('rmse').round(4).to_string(index=False))
print(summ[summ.depth != 'all'].pivot(index='model', columns='depth', values='rmse').loc[MODELS].round(4))
print(pooled.pivot(index='model', columns='depth', values='r2_pooled').loc[MODELS].round(3))
print(pair.round(4).to_string(index=False))
print('leak test max abs diff:', leak)
