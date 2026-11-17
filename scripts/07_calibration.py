"""Stage 3a: short-calibration pilot.

For each held-out station x depth and each model's spatial reconstruction:
  * continuous window of N days (7, 14, 28, 56) at the target -> correction
  * k spot readings (1, 3, 6, 12) on random days within a 180-day campaign, with
    measurement noise sd 0 or 0.02 m3/m3 -> correction
Corrections: constant offset; offset + gain (only if the record's predicted range sd >= 0.01).
Evaluation: the 365 days after the record ends (forward use), never the record itself.
Windows start every 30 days from 2011-01-01 so that a full 365-day evaluation year follows.
"""
import numpy as np, pandas as pd, json

pr = pd.read_parquet('results/predictions.parquet')
MODELS = ['climatology', 'idw', 'ordinary_kriging', 'rf_B', 'lgbm_A']
dates = pd.date_range('2011-01-01', '2016-06-16')
T = len(dates)
starts = np.arange(0, T, 30)
rng = np.random.default_rng(42)
SEASON = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
          6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}


def correct(obs_r, pred_r, pred_e, kind):
    off = np.mean(obs_r - pred_r)
    if kind == 'offset' or np.std(pred_r) < 0.01 or len(obs_r) < 3:
        return pred_e + off
    b, a = np.polyfit(pred_r, obs_r, 1)
    b = np.clip(b, 0.25, 4.0)
    a = np.mean(obs_r) - b * np.mean(pred_r)
    return a + b * pred_e


rows = []
for (s, d), g in pr.groupby(['station', 'depth']):
    idx = (g.date - dates[0]).dt.days.values
    obs = np.full(T, np.nan); obs[idx] = g.obs.values
    P = {}
    for m in MODELS:
        a = np.full(T, np.nan); a[idx] = g[m].values; P[m] = a
    for st0 in starts:
        season = SEASON[dates[st0].month]
        records = []
        for N in [7, 14, 28, 56]:
            r = np.arange(st0, min(st0 + N, T))
            records.append((f'window_{N}d', r, 0.0))
        for k in [1, 3, 6, 12]:
            camp = np.arange(st0, min(st0 + 180, T))
            for noise in [0.0, 0.02]:
                valid = camp[~np.isnan(obs[camp])]
                if len(valid) >= k:
                    r = np.sort(rng.choice(valid, k, replace=False))
                    records.append((f'spot_{k}', r, noise, camp[-1]))
        for rec in records:
            name, r, noise = rec[0], rec[1], rec[2]
            end = rec[3] if len(rec) > 3 else r[-1]
            ev = np.arange(end + 1, min(end + 366, T))
            if len(ev) < 300:
                continue
            ok_r = ~np.isnan(obs[r])
            if name.startswith('window') and ok_r.mean() < 0.5:
                continue
            ok_e = ~np.isnan(obs[ev])
            if ok_e.sum() < 100:
                continue
            o_r = obs[r][ok_r] + (rng.normal(0, noise, ok_r.sum()) if noise else 0)
            for m in MODELS:
                p_r, p_e = P[m][r][ok_r], P[m][ev][ok_e]
                if np.isnan(p_r).any() or np.isnan(p_e).any():
                    continue
                o_e = obs[ev][ok_e]
                base = p_e - o_e
                res = {'station': s, 'depth': d, 'model': m, 'record': name, 'noise': noise,
                       'start': str(dates[st0].date()), 'season': season,
                       'rmse_none': np.sqrt(np.mean(base ** 2)),
                       'ubrmse_eval': np.sqrt(max(np.mean(base ** 2) - np.mean(base) ** 2, 0))}
                for kind in ['offset', 'offset_gain']:
                    e = correct(o_r, p_r, p_e, kind) - o_e
                    res[f'rmse_{kind}'] = np.sqrt(np.mean(e ** 2))
                rows.append(res)

cal = pd.DataFrame(rows)
cal.to_parquet('results/calibration_runs.parquet')

# station-level means first (station = unit), then across stations
key = ['model', 'record', 'noise']
st_lvl = cal.groupby(key + ['station'])[['rmse_none', 'rmse_offset', 'rmse_offset_gain', 'ubrmse_eval']].mean()
summ = st_lvl.groupby(key).mean().round(4)
summ.to_csv('results/calibration_summary.csv')
by_season = cal[(cal.noise == 0)].groupby(['model', 'record', 'season', 'station'])[['rmse_none', 'rmse_offset']].mean() \
    .groupby(['model', 'record', 'season']).mean().round(4)
by_season.to_csv('results/calibration_by_season.csv')
by_depth = cal[(cal.noise == 0)].groupby(['model', 'record', 'depth', 'station'])[['rmse_none', 'rmse_offset', 'ubrmse_eval']].mean() \
    .groupby(['model', 'record', 'depth']).mean().round(4)
by_depth.to_csv('results/calibration_by_depth.csv')
pd.set_option('display.width', 200)
order = ['spot_1', 'spot_3', 'spot_6', 'spot_12', 'window_7d', 'window_14d', 'window_28d', 'window_56d']
for m in ['rf_B', 'idw', 'climatology']:
    print(m); print(summ.loc[m].reindex(order, level=0).to_string())
print(by_season.loc[('rf_B', 'window_14d')].to_string())
print(by_season.loc[('rf_B', 'spot_3')].to_string())
print(by_depth.loc[('rf_B', 'window_14d')].to_string())
