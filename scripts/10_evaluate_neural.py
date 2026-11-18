"""Stage 3c: evaluate neural models against the Stage 2 baselines for both targets.

Target 1 (absolute level): RMSE, MAE, bias.
Target 2 (dynamics): ubRMSE (bias removed per station x depth), temporal correlation, and
skill against the training-fold climatology after removing each series' own bias.
Station is the unit: metrics per held-out station x depth, averaged per station, then across
stations; 95% CIs by resampling stations. Paired differences against IDW and random forest B.
With several seeds, metrics are computed per seed and then averaged (seed spread reported).
Usage: python scripts/10_evaluate_neural.py [--seeds 0 1 2 ...]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[0])
ap.add_argument('--neural-dir', default='results/neural')
ap.add_argument('--out', default='results/neural_eval')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(1)
KEY = ['date', 'station', 'depth']
BASE = ['climatology', 'idw', 'ordinary_kriging', 'rf_B', 'lgbm_A', 'rf_A']
NEURAL = ['gcn', 'attn', 'srn', 'srn_unshared']

base = pd.read_parquet('results/predictions.parquet')[KEY + ['fold', 'obs', 'clim_ref'] + BASE]


def load(model, seed):
    fs = sorted(glob.glob(f'{a.neural_dir}/{model}_f[0-4]_s{seed}.parquet'))
    if len(fs) < 5:
        return None
    d = pd.concat([pd.read_parquet(f) for f in fs])[KEY + ['pred']]
    return d.rename(columns={'pred': model})


def station_metrics(df, models):
    rows = []
    for (s, d), g in df.groupby(['station', 'depth']):
        for m in models:
            ok = g[m].notna()
            e = (g[m] - g.obs)[ok]
            if len(e) < 30:
                continue
            o, p, c = g.obs[ok], g[m][ok], g.clim_ref[ok]
            ub = e - e.mean()
            cb = (c - o) - (c - o).mean()
            rows.append({'station': s, 'depth': d, 'model': m,
                         'rmse': np.sqrt((e ** 2).mean()), 'bias': e.mean(),
                         'ubrmse': np.sqrt((ub ** 2).mean()),
                         'r_time': np.corrcoef(o, p)[0, 1] if p.std() > 0 else np.nan,
                         'dyn_skill_vs_clim': 1 - (ub ** 2).mean() / (cb ** 2).mean(),
                         'bias_share': e.mean() ** 2 / (e ** 2).mean()})
    return pd.DataFrame(rows)


def boot(x):
    x = np.asarray(x); b = rng.choice(x, (2000, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


per_seed = []
for seed in a.seeds:
    df = base.copy()
    avail = []
    for m in NEURAL:
        d = load(m, seed)
        if d is not None:
            df = df.merge(d, on=KEY, how='left'); avail.append(m)
    sm = station_metrics(df, BASE + avail)
    sm['seed'] = seed
    per_seed.append(sm)
sm = pd.concat(per_seed)
sm.to_csv(f'{a.out}/station_metrics.csv', index=False)

# summary: average over seeds within station x depth, then station, then across stations
stl = sm.groupby(['model', 'station', 'depth'])[['rmse', 'bias', 'ubrmse', 'r_time', 'dyn_skill_vs_clim', 'bias_share']].mean().reset_index()
rows = []
for m, g in stl.groupby('model'):
    per_st = g.groupby('station')[['rmse', 'ubrmse']].mean()
    lo, hi = boot(per_st.rmse); ulo, uhi = boot(per_st.ubrmse)
    seed_rmse = sm[sm.model == m].groupby('seed').rmse.mean()
    rows.append({'model': m, 'rmse': per_st.rmse.mean(), 'rmse_lo': lo, 'rmse_hi': hi,
                 'ubrmse': per_st.ubrmse.mean(), 'ubrmse_lo': ulo, 'ubrmse_hi': uhi,
                 'r_time_median': g.r_time.median(), 'dyn_skill_median': g.dyn_skill_vs_clim.median(),
                 'bias_share_mean': g.bias_share.mean(), 'seed_sd_rmse': seed_rmse.std() if len(seed_rmse) > 1 else np.nan})
summ = pd.DataFrame(rows).sort_values('rmse')
summ.to_csv(f'{a.out}/summary.csv', index=False)
bydepth = stl.pivot_table(index='model', columns='depth', values=['rmse', 'ubrmse']).round(4)
bydepth.to_csv(f'{a.out}/by_depth.csv')

pairs = []
w = stl.pivot_table(index=['station', 'depth'], columns='model', values=['rmse', 'ubrmse'])
for ref in ['idw', 'rf_B']:
    for m in stl.model.unique():
        if m == ref:
            continue
        for met in ['rmse', 'ubrmse']:
            dif = (w[met][m] - w[met][ref]).groupby('station').mean().dropna()
            lo, hi = boot(dif)
            pairs.append({'model': m, 'reference': ref, 'metric': met, 'mean_diff': dif.mean(),
                          'lo': lo, 'hi': hi, 'stations_better': int((dif < 0).sum()), 'n': len(dif)})
pairs = pd.DataFrame(pairs)
pairs.to_csv(f'{a.out}/paired.csv', index=False)

meta = {}
for f in sorted(glob.glob(f'{a.neural_dir}/*_s[0-9]*.json')):
    if '_leak' in f:
        continue
    j = json.load(open(f))
    meta[os.path.basename(f)[:-5]] = {k: j[k] for k in ['params', 'epochs_run', 'seconds', 'H_emp', 'kappa0', 'kappa_final']}
json.dump(meta, open(f'{a.out}/run_meta.json', 'w'), indent=1)

pd.set_option('display.width', 220)
print(summ.round(4).to_string(index=False))
print(bydepth)
print(pairs[pairs.model.isin(NEURAL)].round(4).to_string(index=False))
