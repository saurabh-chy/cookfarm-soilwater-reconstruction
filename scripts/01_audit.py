"""Stage 1 data audit for the Cook Agronomy Farm (CAF) sensor dataset."""
import sys, glob, os, io, json
import numpy as np, pandas as pd
import shapefile

RAW, DAYMET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
DEPTHS = [30, 60, 90, 120, 150]

def read_txt(p):
    t = open(p, encoding='latin-1').read().replace('\r\n', '\n').replace('\r', '\n')
    return pd.read_csv(io.StringIO(t), sep='\t', quotechar='"', na_values=['NA'])

# ---------- sensor readings (daily) ----------
frames = [read_txt(p) for p in sorted(glob.glob(f'{RAW}/caf_sensors/Daily/*.txt'))]
d = pd.concat(frames, ignore_index=True)
d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%Y')
long = d.melt(id_vars=['Location', 'Date'], value_vars=[f'VW_{k}cm' for k in DEPTHS],
              var_name='var', value_name='vw')
long['depth'] = long['var'].str.extract(r'(\d+)').astype(int)
long = long.drop(columns='var')

# ---------- QC flags ----------
flags = []
for kind in ['range', 'flats', 'spikes', 'breaks']:
    for k in DEPTHS:
        f = read_txt(f'{RAW}/QC/Flags/{kind}VW_{k}.txt')
        if len(f) == 0:
            continue
        f['Date'] = pd.to_datetime(f['Date'], format='%m/%d/%Y')
        if 'Time' in f.columns:  # hourly flags -> count flagged hours per day
            g = f.groupby(['Location', 'Date']).size().rename('hrs').reset_index()
        else:
            g = f[['Location', 'Date']].drop_duplicates().assign(hrs=24)
        g['depth'] = k
        g['kind'] = kind
        flags.append(g)
flags = pd.concat(flags, ignore_index=True)
fw = flags.pivot_table(index=['Location', 'Date', 'depth'], columns='kind', values='hrs',
                       aggfunc='sum', fill_value=0).reset_index()
long = long.merge(fw, on=['Location', 'Date', 'depth'], how='left').fillna(
    {c: 0 for c in ['range', 'flats', 'spikes', 'breaks']})
for c in ['range', 'flats', 'spikes', 'breaks']:
    if c not in long:
        long[c] = 0

# Rule A (strict): drop a day if any hour carries a range/spike/break flag, or >=12 h flat.
# Rule B (lenient): ignore 'flats' (deep layers can be legitimately stable); keep range/spike/break.
present = long['vw'].notna()
bad_core = (long['range'] > 0) | (long['spikes'] > 0) | (long['breaks'] > 0)
long['valid_B'] = present & ~bad_core & long['vw'].between(0, 0.6)
long['valid_A'] = long['valid_B'] & (long['flats'] < 12)
long['year'] = long['Date'].dt.year

# ---------- coordinates ----------
sf = shapefile.Reader(f'{RAW}/CAF_sensors.shp')
fields = [f[0] for f in sf.fields[1:]]
coords = pd.DataFrame([dict(zip(fields, r.record), x=r.shape.points[0][0], y=r.shape.points[0][1])
                       for r in sf.shapeRecords()])
loccol = [c for c in coords.columns if c.lower().startswith('loc')][0]
coords = coords.rename(columns={loccol: 'Location'})[['Location', 'x', 'y']]
coords['Location'] = coords['Location'].astype(str)
coords.loc[~coords['Location'].str.startswith('CAF'), 'Location'] = \
    'CAF' + coords['Location'].str.zfill(3)

# ---------- summaries ----------
res = {}
res['n_station_files'] = int(d['Location'].nunique())
res['date_range'] = [str(d['Date'].min().date()), str(d['Date'].max().date())]
res['raw_station_depth_days'] = int(len(long))
res['present'] = int(present.sum())
res['valid_lenient_B'] = int(long['valid_B'].sum())
res['valid_strict_A'] = int(long['valid_A'].sum())
res['days_flagged'] = {c: int((long[c] > 0).sum()) for c in ['range', 'flats', 'spikes', 'breaks']}

first = long[long['valid_B']].groupby('Location')['Date'].min()
last = long[long['valid_B']].groupby('Location')['Date'].max()
res['first_valid_date_by_station'] = {k: str(v.date()) for k, v in first.sort_values().items()}
res['stations_valid_before_2009'] = int((first < '2009-01-01').sum())

# valid fraction by station-year (any depth) and active-station counts per year
sy = long.groupby(['Location', 'year', 'depth'])['valid_B'].sum().reset_index()
days_in_year = long.groupby(['year'])['Date'].nunique()
sy['frac'] = sy['valid_B'] / sy['year'].map(days_in_year)
st_year = sy.groupby(['Location', 'year'])['frac'].mean().unstack()
st_year.round(2).to_csv(f'{OUT}/valid_fraction_station_by_year.csv')
active = (st_year >= 0.5).sum()
res['stations_with_>=50pct_valid_by_year'] = {int(k): int(v) for k, v in active.items()}
dep_year = long.groupby(['depth', 'year'])['valid_B'].mean().unstack().round(2)
dep_year.to_csv(f'{OUT}/valid_fraction_depth_by_year.csv')

# daily count of stations with a valid 30 cm reading
cnt = long[long['valid_B']].groupby(['Date', 'depth'])['Location'].nunique().unstack()
cnt.to_csv(f'{OUT}/daily_valid_station_count_by_depth.csv')
w = cnt.loc['2009-07-01':]
res['median_daily_valid_stations_from_2009-07_by_depth'] = {int(k): float(v) for k, v in w.median().items()}
res['days_from_2009-07_with_>=30_stations_all_depths'] = int((w.min(axis=1) >= 30).sum())
res['total_days_from_2009-07'] = int(len(w))

# ---------- fold feasibility ----------
from sklearn.cluster import KMeans
c2 = coords.set_index('Location').loc[sorted(long['Location'].unique())]
best = None
for seed in range(50):
    km = KMeans(5, n_init=1, random_state=seed).fit(c2[['x', 'y']])
    sizes = np.bincount(km.labels_)
    score = sizes.max() - sizes.min()
    if best is None or score < best[0]:
        best = (score, km.labels_, sizes)
c2['fold'] = best[1]
c2.reset_index().to_csv(f'{OUT}/station_coords_and_draft_folds.csv', index=False)
res['draft_fold_sizes'] = [int(s) for s in best[2]]
dd = np.sqrt(((c2[['x', 'y']].values[:, None] - c2[['x', 'y']].values[None]) ** 2).sum(-1))
np.fill_diagonal(dd, np.inf)
nn = dd.min(1)
res['nearest_neighbour_spacing_m'] = {'min': round(float(nn.min()), 1), 'median': round(float(np.median(nn)), 1),
                                      'max': round(float(nn.max()), 1)}
res['field_extent_m'] = [round(float(c2.x.max() - c2.x.min())), round(float(c2.y.max() - c2.y.min()))]

# ---------- weather ----------
met = pd.read_csv(DAYMET, skiprows=6)
res['daymet_rows'] = int(len(met))
res['daymet_years'] = [int(met.year.min()), int(met.year.max())]
res['daymet_missing_values'] = int(met.isna().sum().sum())
res['daymet_annual_precip_mm'] = {int(k): round(float(v)) for k, v in met.groupby('year')['prcp (mm/day)'].sum().items()}

# ---------- static covariates ----------
ps = read_txt(f'{RAW}/CAF_ParticleSize.txt')
bd = read_txt(f'{RAW}/CAF_BulkDensity.txt')
res['particle_size_rows'] = int(len(ps)); res['bulk_density_rows'] = int(len(bd))
res['particle_size_missing'] = int(ps[['Sand', 'Silt', 'Clay']].isna().any(axis=1).sum())
res['bulk_density_missing'] = int(bd['BulkDensity'].isna().sum())

long.to_parquet(f'{OUT}/caf_daily_long_qc.parquet')
json.dump(res, open(f'{OUT}/audit_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
print(dep_year)
