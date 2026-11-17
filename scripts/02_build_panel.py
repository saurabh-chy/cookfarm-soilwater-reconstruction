"""Stage 2a: clean daily panel for CAF, 2011-01-01 to 2016-06-16.

Outputs (in OUT):
  panel.parquet     one row per station x date x depth with target vw (NaN if invalid)
  stations.csv      station static covariates (coordinates, terrain, ECa, sensor type, strip)
  station_depth.csv texture, bulk density, Bt probability, sensor type by station x depth
  weather.csv       Daymet daily weather + lagged features
  qc_report.json    counts removed by each QC rule
"""
import sys, io, glob, json, os
import numpy as np, pandas as pd, shapefile, rasterio
from shapely.geometry import shape, Point

RAW, DAYMET, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
DEPTHS = [30, 60, 90, 120, 150]
START, END = pd.Timestamp('2011-01-01'), pd.Timestamp('2016-06-16')
PLATEAU_DAYS = int(os.environ.get("PLATEAU_DAYS", 60))


def read_txt(p):
    t = open(p, encoding='latin-1').read().replace('\r\n', '\n').replace('\r', '\n')
    return pd.read_csv(io.StringIO(t), sep='\t', quotechar='"', na_values=['NA'])


# ---------------- readings ----------------
d = pd.concat([read_txt(p) for p in sorted(glob.glob(f'{RAW}/caf_sensors/Daily/*.txt'))])
d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%Y')
d = d[(d.Date >= START) & (d.Date <= END)]
p = d.melt(['Location', 'Date'], [f'VW_{k}cm' for k in DEPTHS], 'var', 'vw')
p['depth'] = p['var'].str.extract(r'(\d+)').astype(int)
p = p.drop(columns='var').rename(columns={'Location': 'station', 'Date': 'date'})
qc = {'window': [str(START.date()), str(END.date())], 'rows': int(len(p)),
      'present': int(p.vw.notna().sum())}

# published flags: range, spikes, breaks (any hour flagged -> drop the day)
bad = []
for kind in ['range', 'spikes', 'breaks']:
    for k in DEPTHS:
        f = read_txt(f'{RAW}/QC/Flags/{kind}VW_{k}.txt')
        if len(f):
            f['date'] = pd.to_datetime(f['Date'], format='%m/%d/%Y')
            bad.append(f[['Location', 'date']].drop_duplicates().rename(
                columns={'Location': 'station'}).assign(depth=k, kind=kind))
bad = pd.concat(bad)
p = p.merge(bad.groupby(['station', 'date', 'depth'])['kind'].first().reset_index(),
            on=['station', 'date', 'depth'], how='left')
for kind in ['range', 'spikes', 'breaks']:
    qc[f'removed_{kind}'] = int(((p.kind == kind) & p.vw.notna()).sum())
p.loc[p.kind.notna(), 'vw'] = np.nan
oor = p.vw.notna() & ~p.vw.between(0, 0.6)
qc['removed_outside_0_0.6'] = int(oor.sum())
p.loc[oor, 'vw'] = np.nan

# stricter plateau test: identical daily value at sensor resolution (0.001) for >= PLATEAU_DAYS consecutive days
p = p.sort_values(['station', 'depth', 'date']).reset_index(drop=True)
r = p.vw.round(3)
grp = ((r != r.shift()) | (p.station != p.station.shift()) | (p.depth != p.depth.shift())).cumsum()
runlen = r.groupby(grp).transform('size')
plateau = r.notna() & (runlen >= PLATEAU_DAYS)
qc['removed_plateau'] = int(plateau.sum())
p.loc[plateau, 'vw'] = np.nan
p = p.drop(columns='kind')
qc['valid_after_qc'] = int(p.vw.notna().sum())

# ---------------- station coordinates & rasters ----------------
s = shapefile.Reader(f'{RAW}/CAF_sensors.shp', encoding='cp1252')
st = pd.DataFrame([r_ for r_ in s.records()], columns=['station', 'x', 'y'])


def sample(fname, xs, ys):
    with rasterio.open(f'{RAW}/{fname}') as r_:
        a = r_.read(1).astype(float)
        if r_.nodata is not None:
            a[a == r_.nodata] = np.nan
        rows, cols = rasterio.transform.rowcol(r_.transform, xs, ys)
        return a, r_.transform, a[np.array(rows), np.array(cols)]


dem, tr, st['elev'] = sample('CAF_DEM.tif', st.x, st.y)
# terrain derivatives from the 10 m DEM
gy, gx = np.gradient(dem, 10.0)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
from scipy.ndimage import uniform_filter, generic_filter
tpi = dem - uniform_filter(np.nan_to_num(dem, nan=np.nanmean(dem)), size=7)  # ~70 m window
curv = np.gradient(gx, 10.0, axis=1) + np.gradient(gy, 10.0, axis=0)
rows, cols = rasterio.transform.rowcol(tr, st.x, st.y)
rows, cols = np.array(rows), np.array(cols)
st['slope_deg'] = slope[rows, cols]
st['tpi70m'] = tpi[rows, cols]
st['curvature'] = curv[rows, cols]
st['aspect_sin'] = np.sin(np.arctan2(-gx, gy))[rows, cols]
st['aspect_cos'] = np.cos(np.arctan2(-gx, gy))[rows, cols]
_, _, st['eca_spring'] = sample('CAF_Spring_ECa.tif', st.x, st.y)
_, _, st['eca_fall'] = sample('CAF_Fall_ECa.tif', st.x, st.y)

# crop strip by point-in-polygon; crops from CAF_CropID.txt
sf = shapefile.Reader(f'{RAW}/CAF_strips.shp', encoding='cp1252')
polys = [(rec['Field'], int(rec['Strip']), shape(shp.__geo_interface__))
         for shp, rec in zip(sf.shapes(), sf.records())]
fs = []
for _, row in st.iterrows():
    hit = [(f_, s_) for f_, s_, g in polys if g.contains(Point(row.x, row.y))]
    if not hit:  # nearest polygon if on a boundary
        dists = [(g.distance(Point(row.x, row.y)), f_, s_) for f_, s_, g in polys]
        hit = [min(dists)[1:]]
    fs.append(hit[0])
st['field'] = [h[0] for h in fs]
st['strip'] = [h[1] for h in fs]
st.to_csv(f'{OUT}/stations.csv', index=False)

crop = read_txt(f'{RAW}/CAF_CropID.txt')
crop.columns = [c.strip() for c in crop.columns]
crop = crop.dropna(subset=['Strip'])
crop = crop[crop['Field'].isin(['A', 'B', 'C'])]
crop['Strip'] = crop['Strip'].astype(int)
cl = crop.melt(['Field', 'Strip'], [f'Y{y}' for y in range(2007, 2017)], 'yr', 'crop')
cl['year'] = cl.yr.str[1:].astype(int)
cl['crop'] = cl.crop.astype(str).str.strip().str.rstrip('*')
cl = cl.rename(columns={'Field': 'field', 'Strip': 'strip'})[['field', 'strip', 'year', 'crop']]
# footnote in CropID: CAF217 was spring canola in 2016
cl.to_csv(f'{OUT}/crops.csv', index=False)

# ---------------- station x depth covariates ----------------
ps = read_txt(f'{RAW}/CAF_ParticleSize.txt').rename(columns={'Location': 'station', 'Depth': 'depth'})
bd = read_txt(f'{RAW}/CAF_BulkDensity.txt').rename(columns={'Location': 'station', 'Depth': 'depth'})
sv = read_txt(f'{RAW}/QC/CAF_sensor_version.txt')
sv.columns = ['station', 'depth', 'sensor']
sd = ps.merge(bd, on=['station', 'depth'], how='outer').merge(sv, on=['station', 'depth'], how='left')
bt = {}
for k in DEPTHS:
    _, _, v = sample(f'CAF_Bt_{k}cm.tif', st.x, st.y)
    bt.update({(s_, k): vv for s_, vv in zip(st.station, v)})
sd['bt_prob'] = [bt.get((a, b)) for a, b in zip(sd.station, sd.depth)]
sd['sensor_echo'] = (sd.sensor.str.strip() == 'ECH2O').astype(int)
sd.to_csv(f'{OUT}/station_depth.csv', index=False)

# ---------------- weather ----------------
m = pd.read_csv(DAYMET, skiprows=6)
m.columns = ['year', 'yday', 'dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
m['date'] = pd.to_datetime(m.year.astype(str), format='%Y') + pd.to_timedelta(m.yday - 1, 'D')
m = m.set_index('date').drop(columns=['year', 'yday'])
# Daymet uses 365-day years (drops 31 Dec in leap years): reindex and fill that day
m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq='D'))
m['prcp'] = m['prcp'].fillna(0)
m = m.ffill()
w = pd.DataFrame(index=m.index)
w['tmean'] = (m.tmax + m.tmin) / 2
for c in ['prcp', 'tmax', 'tmin', 'srad', 'vp', 'swe']:
    w[c] = m[c]
for k in [3, 7, 14, 30, 90]:
    w[f'prcp_{k}d'] = m.prcp.rolling(k, min_periods=1).sum()
for k in [7, 30]:
    w[f'tmean_{k}d'] = w.tmean.rolling(k, min_periods=1).mean()
    w[f'srad_{k}d'] = m.srad.rolling(k, min_periods=1).mean()
w['doy_sin'] = np.sin(2 * np.pi * w.index.dayofyear / 365.25)
w['doy_cos'] = np.cos(2 * np.pi * w.index.dayofyear / 365.25)
w.index.name = 'date'
w = w.loc[START:END]
w.to_csv(f'{OUT}/weather.csv')

p.to_parquet(f'{OUT}/panel.parquet')
qc['valid_by_depth'] = {int(k): int(v) for k, v in p.groupby('depth').vw.count().items()}
qc['valid_by_year'] = {int(k): int(v) for k, v in p.groupby(p.date.dt.year).vw.count().items()}
qc['stations_with_any_valid'] = int(p.dropna().station.nunique())
qc['sensor_types'] = sd.sensor.str.strip().value_counts().to_dict()
qc['station_strip_assignment'] = st[['station', 'field', 'strip']].values.tolist()
json.dump(qc, open(f'{OUT}/qc_report.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in qc.items() if k != 'station_strip_assignment'}, indent=1))
