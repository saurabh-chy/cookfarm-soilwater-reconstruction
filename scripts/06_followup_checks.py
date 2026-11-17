"""Stage 2e: follow-up checks on the pilot.

1. Scenario A2: add neighbour *anomaly* features (each input station's reading minus that
   station's own mean at that depth, IDW-weighted), so the model can transfer day-to-day
   dynamics separately from the level.
2. Surveyed-covariate scenario S: A2 + texture and bulk density measured at the target.
3. Combined spatial + temporal test: train on development stations 2011-2014,
   predict test stations in 2015.
"""
import sys
src = open('scripts/04_baselines.py').read().split('# ---------- main loop ----------')[0]
sys.argv = ['x', 'data/processed', 'results']
exec(src)
import time

tex = sd.pivot(index='station', columns='depth', values=['Sand', 'Clay', 'BulkDensity']).loc[stations]
TEX = {c: tex[c].values for c in ['Sand', 'Clay', 'BulkDensity']}


def anomaly_idw(Vin, Rmask):
    mu = np.nanmean(Vin, axis=0, keepdims=True)  # each station's own mean by depth
    A = Vin - mu
    M = (~np.isnan(A)) & Rmask[None, :, None]
    w = 1.0 / np.maximum(DIST, 1.0) ** 2
    np.fill_diagonal(w, 0.0)
    num = np.einsum('ij,tjd->tid', w, np.where(M, A, 0))
    den = np.einsum('ij,tjd->tid', w, M.astype(float))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


rows, t0 = [], time.time()
for k in range(5):
    role = np.array(folds.set_index('station')[f'role_fold{k}'].reindex(stations).tolist(), dtype=str)
    test_idx = np.where(role == 'test')[0]
    dev_idx = np.where(np.char.startswith(role, 'inner'))[0]
    Rmask = role != 'test'
    net = network_features(V, Rmask)
    an = anomaly_idw(V, Rmask)
    for scen in ['A2', 'S']:
        def rows_for(idx):
            X, y, key = build_rows(idx, net, 'A')
            X['anom_idw'] = an[key.t, key.s, key.d]
            for dep_i, dep in enumerate(DEPTHS):
                X[f'anom_idw_d{dep}'] = an[key.t, key.s, dep_i]
            if scen == 'S':
                for c in TEX:
                    X[c] = TEX[c][key.s, key.d]
            return X, y, key
        Xtr, ytr, ktr = rows_for(dev_idx)
        Xte, yte, kte = rows_for(test_idx)
        for split in ['spatial', 'spatial_2015']:
            if split == 'spatial':
                tr = ~np.isnan(ytr); te = ~np.isnan(yte)
            else:
                tr = ~np.isnan(ytr) & (dates[ktr.t].year <= 2014)
                te = ~np.isnan(yte) & (dates[kte.t].year == 2015)
            for name, mdl in [('lgbm', lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                min_child_samples=50, subsample=0.7, subsample_freq=1,
                                colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=2,
                               deterministic=True, force_col_wise=True)),
                              ('rf', RandomForestRegressor(n_estimators=150, max_samples=0.2,
                                min_samples_leaf=5, max_features=0.4, n_jobs=2, random_state=0))]:
                mdl.fit(Xtr[tr].fillna(-999), ytr[tr])
                p = mdl.predict(Xte[te].fillna(-999))
                rows.append(pd.DataFrame({'fold': k, 'scenario': scen, 'split': split, 'model': name,
                                          'station': np.array(stations)[kte.s[te]],
                                          'depth': np.array(DEPTHS)[kte.d[te]],
                                          'date': dates[kte.t[te]], 'obs': yte[te], 'pred': p}))
    print('fold', k, f'{time.time()-t0:.0f}s', flush=True)

ex = pd.concat(rows, ignore_index=True)
ex.to_parquet('results/extra_predictions.parquet')


def summarise(df):
    g = df.groupby(['scenario', 'split', 'model', 'station', 'depth'])
    st_ = g.apply(lambda x: pd.Series({'rmse': np.sqrt(((x.pred - x.obs) ** 2).mean()),
                                       'bias': (x.pred - x.obs).mean()}), include_groups=False).reset_index()
    st_['ubrmse'] = np.sqrt(np.maximum(st_.rmse ** 2 - st_.bias ** 2, 0))
    return st_.groupby(['scenario', 'split', 'model'])[['rmse', 'ubrmse']].mean().round(4), st_

s, st_ = summarise(ex)
st_.to_csv('results/extra_station_metrics.csv', index=False)
print(s)
print(st_[st_.split == 'spatial'].pivot_table(index=['scenario', 'model'], columns='depth', values='rmse').round(4))
