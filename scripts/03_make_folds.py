"""Stage 2b: freeze five spatially contiguous outer folds, calibration subsets and inner folds.

Folds are chosen from many k-means runs on station coordinates: sizes must be 8-9 and the
choice minimises imbalance in elevation, 30 cm clay, spring ECa and field (A/B/C) mix.
Within each outer-training set, a spatially contiguous calibration subset of 6-7 stations is
reserved before any tuning; the remaining training stations keep their outer-fold labels as
inner folds (grouped inner CV).
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

D = sys.argv[1]
st = pd.read_csv(f'{D}/stations.csv')
sd = pd.read_csv(f'{D}/station_depth.csv')
st = st.merge(sd[sd.depth == 30][['station', 'Clay']], on='station', how='left')
st['Clay'] = st['Clay'].fillna(st['Clay'].median())
XY = st[['x', 'y']].values
z = lambda c: (st[c] - st[c].mean()) / st[c].std()
covs = np.c_[z('elev'), z('Clay'), z('eca_spring')]
fields = pd.get_dummies(st.field).values.astype(float)

best = None
for seed in range(2000):
    lab = KMeans(5, n_init=1, random_state=seed).fit_predict(XY)
    sizes = np.bincount(lab, minlength=5)
    if sizes.min() < 8 or sizes.max() > 9:
        continue
    means = np.array([covs[lab == k].mean(0) for k in range(5)])
    fmix = np.array([fields[lab == k].mean(0) for k in range(5)])
    score = np.abs(means).sum() + np.abs(fmix - fields.mean(0)).sum()
    if best is None or score < best[0]:
        best = (score, lab, seed)
score, lab, seed = best
# order fold ids west -> east for readability
order = np.argsort([XY[lab == k, 0].mean() for k in range(5)])
remap = {old: new for new, old in enumerate(order)}
st['outer_fold'] = [remap[l] for l in lab]

roles = {}
for k in range(5):
    tr = st[st.outer_fold != k].copy()
    best_c = None
    for s2 in range(200):
        l2 = KMeans(5, n_init=1, random_state=s2).fit_predict(tr[['x', 'y']].values)
        for c in range(5):
            n = (l2 == c).sum()
            if 6 <= n <= 7:
                # prefer a calibration cluster that is not adjacent to the test fold
                cen = tr[['x', 'y']].values[l2 == c].mean(0)
                dmin = np.min(np.hypot(*(st[st.outer_fold == k][['x', 'y']].values - cen).T))
                if best_c is None or dmin > best_c[0]:
                    best_c = (dmin, tr.station.values[l2 == c])
    calib = set(best_c[1])
    roles[k] = {s: ('test' if f == k else 'calib' if s in calib else f'inner{f}')
                for s, f in zip(st.station, st.outer_fold)}
    st[f'role_fold{k}'] = st.station.map(roles[k])

st[['station', 'x', 'y', 'field', 'strip', 'elev', 'Clay', 'eca_spring', 'outer_fold'] +
   [f'role_fold{k}' for k in range(5)]].to_csv(f'{D}/folds.csv', index=False)
summ = st.groupby('outer_fold').agg(n=('station', 'size'), elev=('elev', 'mean'),
                                     clay30=('Clay', 'mean'), eca=('eca_spring', 'mean'),
                                     fields=('field', lambda s: ''.join(sorted(s))))
st[['station', 'outer_fold'] + [f'role_fold{k}' for k in range(5)]].to_csv(f'{D}/folds_frozen_format.csv', index=False)
print('seed', seed, 'score', round(score, 2))
print(summ.round(2))
print({k: sum(v == 'calib' for v in roles[k].values()) for k in roles})
json.dump({'kmeans_seed': int(seed), 'balance_score': float(score)}, open(f'{D}/folds_meta.json', 'w'))
