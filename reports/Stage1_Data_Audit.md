# Stage 1 data audit: Cook Agronomy Farm (CAF)

Run date: 17 Nov 2026. Inputs: `CAF_Sensor_Dataset_2.zip` (readme updated 04/01/2017) and the Daymet v4 single-pixel CSV for 46.7815 N, −117.082 W, 2007–2016.

**Decision: GO**, with a narrower analysis window and a change to how the quality-control flags are used.

## What the data contain
- 42 station files, daily and hourly, 20 Apr 2007 – 16 Jun 2016. Five depths: 30, 60, 90, 120 and 150 cm.
- 702,660 station-depth-days in total. 378,590 (54%) have a water-content value.
- Static covariates are complete: particle size and bulk density at all 42 × 5 depths, except one particle-size row that is missing.
- Raster and vector files: DEM, spring and fall ECa, Bt probability at five depths, strips and crop history 2007–2016.
- Daymet weather: 3,650 days, no missing values. Annual precipitation was 493–774 mm.

## Network history (measured, not assumed)
- 12 stations have readings from May 2007. Most of the other 30 start on 18–21 May 2009. CAF031 starts in May 2011 and CAF009 in July 2011.
- 2009–2010 coverage is poor despite the 2009 expansion. The median number of stations reporting fell to about 1 in late 2009 – early 2010.
- Stations with at least 50% valid days: 37 in 2011, 37 in 2012, 40 in 2013, 40 in 2014, 33 in 2015 and 28 in 2016 (January–June only).
- From July 2009 onward, a median of 32–33 stations report per day at each depth.

## Quality-control finding
- **Used:** the range, spike and break flags. These remove only about 700 station-depth-days.
- **Not used as an exclusion:** the "flats" flag. It marks a day as a plateau when the rolling 24 h variance is ≤ 0.0005 (m³/m³)², which means a standard deviation of about 0.022 m³/m³. Almost all real daily soil-water series fall below that, so the flag covers about 99% of days with data. Excluding on it would leave only about 10,000 values.
- **Replacement:** Stage 2 will add a narrower plateau test (identical readings repeated for several days) and check sensor changes using `CAF_sensor_version.txt`. Some positions use ECH2O sensors instead of 5TE.

## Spatial design
- Draft five spatially contiguous folds (k-means on station coordinates): 8, 9, 9, 8 and 8 stations.
- Spacing to the nearest neighbouring station: median 64.5 m (range 60–129 m). Field extent: about 870 × 519 m.
- Dyadic neighbourhood scales of about 60, 120, 240 and 480 m fit inside the field. Up to four scales are plausible, so the three-scale requirement for the scaling test is attainable.

## Recommended changes to the concept note
1. **Primary analysis window:** 1 Jan 2011 – 16 Jun 2016, when the network is dense.
   - 2007–2010 is too sparse for the station-blocked design. Report it only descriptively.
2. **Temporal split:**
   - development: 2011–2013
   - model selection and calibration: 2014
   - untouched test: 2015
   - Jan–Jun 2016 as a partial-year check
3. **QC rule:** state that the published "flats" flag is not used for exclusion, give the reason above, and describe the replacement plateau test.

## Files produced (in the working session)
- `out/audit_summary.json`
- `out/valid_fraction_station_by_year.csv`
- `out/valid_fraction_depth_by_year.csv`
- `out/daily_valid_station_count_by_depth.csv`
- `out/station_coords_and_draft_folds.csv`
- `out/caf_daily_long_qc.parquet`
- `audit.py`
