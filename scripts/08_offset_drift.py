"""Stage 3b: what drives the year-to-year drift in station offsets?

Response (observation-only, no model involved):
  a[s,d,y]   = mean over year y of (vw[s,d,t] - network mean at depth d on day t)
               (network mean = mean of all other stations reporting that day; removes
               weather/season common to the field)
  delta[s,d,y] = a[s,d,y] - mean_y a[s,d,y]   (year-to-year drift around the station's own level)
Station-depth-years need >= 50% valid days; station-depths need >= 3 such years.
Calendar years 2011-2015 (2016 is partial).

Candidate explanations, scored by leave-one-year-out cross-validated R2 on delta:
  crop_now     crop group of the station's strip in year y (x depth)
  crop_prev    crop group in year y-1 (x depth)
  crop_both    both
  trend        station-depth linear trend in year (sensor drift)
  wet_year     station-depth slope on annual precipitation anomaly (location responds
               differently in wet vs dry years)
  sensor_year  sensor type (ECH2O vs 5TE) x year
Crop effects use both crop sources (CAF_CropID.txt and CAF_strips.shp).
Permutation test for crop: crop sequences are shuffled across strips (whole sequences, so
rotation structure is kept), 1000 permutations.
"""
import sys, json
import numpy as np, pandas as pd, shapefile

PROC, RAW, OUT = 'data/processed', 'data/raw/CAF', 'results/offset_drift'
import os; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
YEARS = [2011, 2012, 2013, 2014, 2015]

p = pd.read_parquet(f'{PROC}/panel.parquet')
st = pd.read_csv(f'{PROC}/stations.csv')
sd = pd.read_csv(f'{PROC}/station_depth.csv')
wx = pd.read_csv(f'{PROC}/weather.csv', parse_dates=['date'])

# ---- response ----
p['n'] = p.groupby(['date', 'depth']).vw.transform('count')
p['s'] = p.groupby(['date', 'depth']).vw.transform('sum')
p['net_other'] = (p.s - p.vw) / (p.n - 1)
p['rel'] = p.vw - p.net_other
p['year'] = p.date.dt.year
p = p[p.year.isin(YEARS)]
ann = p.groupby(['station', 'depth', 'year']).agg(a=('rel', 'mean'), nvalid=('rel', 'count')).reset_index()
ann = ann[ann.nvalid >= 183]
ann = ann[ann.groupby(['station', 'depth']).year.transform('size') >= 3].copy()
ann['delta'] = ann.a - ann.groupby(['station', 'depth']).a.transform('mean')

# ---- covariates ----
CROP_GROUP = {'WW': 'winter_cereal', 'WT': 'winter_cereal', 'WB': 'winter_cereal',
              'SW': 'spring_cereal', 'SB': 'spring_cereal', 'SC': 'oilseed', 'WC': 'oilseed',
              'SL': 'legume', 'SP': 'legume', 'WL': 'legume', 'WP': 'legume', 'Fallow': 'fallow'}
crop_txt = pd.read_csv(f'{PROC}/crops.csv')
sf = shapefile.Reader(f'{RAW}/CAF_strips.shp', encoding='cp1252')
shp = pd.DataFrame(sf.records(), columns=[f[0] for f in sf.fields[1:]])
crop_shp = shp.melt(['Field', 'Strip'], [f'Y{y}' for y in range(2007, 2017)], 'yr', 'crop')
crop_shp['year'] = crop_shp.yr.str[1:].astype(int)
crop_shp = crop_shp.rename(columns={'Field': 'field', 'Strip': 'strip'})[['field', 'strip', 'year', 'crop']]
crop_shp['strip'] = crop_shp.strip.astype(int)
crop_shp['crop'] = crop_shp.crop.astype(str).str.strip()
crop_sources = {'CropID_txt': crop_txt, 'strips_shp': crop_shp}

ann = ann.merge(st[['station', 'field', 'strip']], on='station')
ann = ann.merge(sd[['station', 'depth', 'sensor']], on=['station', 'depth'], how='left')
ann['echo'] = (ann.sensor.str.strip() == 'ECH2O').astype(int)
P = wx.groupby(wx.date.dt.year).prcp.sum()
ann['p_anom'] = ann.year.map(P - P.loc[YEARS].mean()) / 100.0  # per 100 mm
ann['yr_c'] = ann.year - 2013

disagree = crop_txt.merge(crop_shp, on=['field', 'strip', 'year'], suffixes=('_txt', '_shp'))
disagree = disagree[disagree.year.isin(range(2010, 2016)) & (disagree.crop_txt != disagree.crop_shp)]


def add_crop(df, src, strip_perm=None):
    c = crop_sources[src].copy()
    if strip_perm is not None:  # permute whole crop sequences across strips
        keys = c[['field', 'strip']].drop_duplicates().reset_index(drop=True)
        perm = keys.iloc[strip_perm].reset_index(drop=True)
        mp = {(a, b): (pa, pb) for (a, b), (pa, pb) in zip(keys.values, perm.values)}
        df = df.copy()
        df[['cf', 'cs']] = [mp[(a, b)] for a, b in zip(df.field, df.strip)]
    else:
        df = df.assign(cf=df.field, cs=df.strip)
    cm = c.set_index(['field', 'strip', 'year']).crop.to_dict()
    g = lambda f, s, y: CROP_GROUP.get(cm.get((f, s, y), 'Fallow'), 'fallow')
    df['crop_now'] = [g(f, s, y) for f, s, y in zip(df.cf, df.cs, df.year)]
    df['crop_prev'] = [g(f, s, y - 1) for f, s, y in zip(df.cf, df.cs, df.year)]
    return df


def loyo_r2(df, kind):
    """Leave-one-year-out CV R2 for predicting delta."""
    sse = sst = 0.0
    pred_all = []
    for y in YEARS:
        tr, te = df[df.year != y], df[df.year == y]
        if kind in ('crop_now', 'crop_prev'):
            m = tr.groupby([kind, 'depth']).delta.mean()
            pr = np.array([m.get((c, d), 0.0) for c, d in zip(te[kind], te.depth)])
        elif kind == 'crop_both':
            m1 = tr.groupby(['crop_now', 'depth']).delta.mean()
            r = tr.delta - np.array([m1.get((c, d), 0.0) for c, d in zip(tr.crop_now, tr.depth)])
            m2 = tr.assign(r=r).groupby(['crop_prev', 'depth']).r.mean()
            pr = np.array([m1.get((c, d), 0.0) + m2.get((c2, d), 0.0)
                           for c, c2, d in zip(te.crop_now, te.crop_prev, te.depth)])
        elif kind in ('trend', 'wet_year'):
            x = 'yr_c' if kind == 'trend' else 'p_anom'
            coefs = {}
            for key, g in tr.groupby(['station', 'depth']):
                if g[x].std() > 0 and len(g) >= 2:
                    coefs[key] = np.polyfit(g[x], g.delta, 1)
            pr = np.array([np.polyval(coefs[(s, d)], v) if (s, d) in coefs else 0.0
                           for s, d, v in zip(te.station, te.depth, te[x])])
        elif kind == 'sensor_year':
            m = tr.groupby(['echo', 'depth']).apply(
                lambda g: np.polyfit(g.yr_c, g.delta, 1) if g.yr_c.nunique() > 1 else np.array([0, 0]),
                include_groups=False)
            pr = np.array([np.polyval(m.get((e, d), np.array([0, 0])), v)
                           for e, d, v in zip(te.echo, te.depth, te.yr_c)])
        sse += ((te.delta - pr) ** 2).sum()
        sst += (te.delta ** 2).sum()  # delta has mean zero per station-depth by construction
    return 1 - sse / sst


res = {'n_station_depth_years': int(len(ann)), 'n_station_depths': int(ann.groupby(['station', 'depth']).ngroups),
       'sd_delta': float(ann.delta.std()), 'sd_station_level': float(ann.groupby(['station', 'depth']).a.mean().std()),
       'annual_precip_mm': {int(k): round(float(v)) for k, v in P.loc[YEARS].items()},
       'crop_source_disagreements_2010_2015': disagree.to_dict('records')}
res['sd_delta_by_depth'] = ann.groupby('depth').delta.std().round(4).to_dict()

cv = {}
for src in crop_sources:
    df = add_crop(ann, src)
    for kind in ['crop_now', 'crop_prev', 'crop_both']:
        cv[f'{kind}[{src}]'] = loyo_r2(df, kind)
df = add_crop(ann, 'CropID_txt')
for kind in ['trend', 'wet_year', 'sensor_year']:
    cv[kind] = loyo_r2(df, kind)
res['loyo_cv_r2'] = {k: round(float(v), 3) for k, v in cv.items()}

# permutation test for crop (CropID_txt), whole-sequence shuffles across strips
nstrips = crop_txt[['field', 'strip']].drop_duplicates().shape[0]
null = {'crop_now': [], 'crop_both': []}
for _ in range(1000):
    perm = rng.permutation(nstrips)
    dfp = add_crop(ann, 'CropID_txt', perm)
    for kind in null:
        null[kind].append(loyo_r2(dfp, kind))
res['permutation_p'] = {k: float((np.array(v) >= cv[f'{k}[CropID_txt]']).mean()) for k, v in null.items()}
res['permutation_null_95pct'] = {k: round(float(np.percentile(v, 95)), 3) for k, v in null.items()}

# effect sizes: mean delta by current crop and depth (in-sample, descriptive)
eff = df.groupby(['crop_now', 'depth']).delta.agg(['mean', 'count']).round(4)
eff.to_csv(f'{OUT}/delta_by_crop_now_depth.csv')
effp = df.groupby(['crop_prev', 'depth']).delta.agg(['mean', 'count']).round(4)
effp.to_csv(f'{OUT}/delta_by_crop_prev_depth.csv')
ann.to_csv(f'{OUT}/annual_relative_offsets.csv', index=False)
json.dump(res, open(f'{OUT}/offset_drift_summary.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k != 'crop_source_disagreements_2010_2015'}, indent=1))
print('disagreements:', len(disagree)); print(disagree.to_string(index=False))
print(eff.unstack('depth')['mean'])
print(effp.unstack('depth')['mean'])
