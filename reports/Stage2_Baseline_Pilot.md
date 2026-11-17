# Stage 2: clean panel, frozen folds and baseline pilot (CAF)

Run date: 17 Nov 2026. Builds on `Stage1_Data_Audit.md`. Scripts: `build_panel.py`, `make_folds.py`, `baselines.py`, `extra.py` and `evaluate.py`.

**Outcome: the protocol works and the leakage test passed.** The baselines also show something important about the research question. Most of the remaining error is a fixed, station-specific offset in soil-water level that no method or covariate predicts. More nearby sensors do not reduce it.

## 1. Clean daily panel
- **Window:** 1 Jan 2011 – 16 Jun 2016, covering 42 stations × 5 depths × 1,994 days (418,740 rows). 324,457 of these have a reading.
- **QC rule:** the published range flags (364 values) and spike flags (151 values) are applied. The break flags remove nothing.
- **Plateau test:**
  - Sensors report to 0.001 m³/m³, so deep soil can legitimately hold one value for weeks. A 7-day identical-value rule would remove 27% of the data, including normal deep-soil behaviour.
  - The frozen rule removes runs of 60 or more identical days: 720 values, mostly at 150 cm. A 30-day rule is kept as a sensitivity check.
- **Result:** 323,222 valid values, evenly spread across depths (63,500–66,300 each).
- **Covariates assembled:**
  - Terrain from the DEM: elevation, slope, aspect, curvature and topographic position over about 70 m.
  - Spring and fall apparent electrical conductivity, and Bt probability at each depth.
  - Crop group by strip and year.
  - Daymet weather with 3–90-day rainfall sums and 7- and 30-day temperature and radiation means.
  - For the surveyed scenario only: texture and bulk density at each depth.
  - Sensor type: 18 of the 210 positions are ECH2O sensors and the rest are 5TE.
- **Data issue found:** crop codes in `CAF_strips.shp` disagree with `CAF_CropID.txt` for some strips in 2015–2016. The pilot uses `CAF_CropID.txt`. This needs resolving or reporting.

## 2. Frozen folds
- **Outer folds:** five spatially contiguous groups of 8, 9, 9, 8 and 8 stations. They were chosen from 2,000 k-means runs to balance elevation, 30 cm clay, spring ECa and field mix (seed 179).
- **Imbalance that remains:** because blocking is spatial, the folds differ in elevation (783–803 m) and in which fields they cover. This is expected.
- **Calibration subsets:** in each outer-training set, a contiguous subset of 6 stations is reserved before any tuning, placed as far from the test fold as possible. The remaining 27–28 development stations keep their outer-fold labels as inner folds.

## 3. Baseline results: station-blocked, all depths
Station is the unit of analysis. Metrics are means over the 42 held-out stations, with 95% intervals from resampling stations. "Skill vs climatology" is the median per-station reduction in error relative to the training-fold seasonal climatology.

| Model | RMSE (m³/m³) | 95% CI | ubRMSE | Skill vs climatology | Correlation over time |
|---|---|---|---|---|---|
| Training mean | 0.088 | 0.081–0.097 | 0.066 | −0.53 | 0.00 |
| Seasonal climatology | 0.073 | 0.065–0.082 | 0.049 | 0 | 0.77 |
| Nearest neighbour | 0.089 | 0.080–0.097 | 0.064 | −0.43 | 0.67 |
| Inverse distance weighting (IDW) | 0.070 | 0.063–0.079 | 0.046 | 0.08 | 0.81 |
| Ordinary kriging | 0.070 | 0.063–0.078 | 0.046 | 0.10 | 0.81 |
| Random forest, covariates only (B) | **0.066** | 0.059–0.074 | 0.045 | 0.22 | 0.82 |
| LightGBM, covariates only (B) | 0.067 | 0.061–0.075 | 0.044 | 0.18 | 0.83 |
| Random forest, sensor-assisted (A) | 0.067 | 0.060–0.074 | 0.045 | 0.16 | 0.82 |
| LightGBM, sensor-assisted (A) | 0.068 | 0.060–0.077 | 0.045 | 0.20 | 0.83 |

- **Paired comparison against IDW:** random forest B is better by 0.0044 m³/m³ (95% CI 0.0002–0.0091), and better at 24 of 42 stations. The A models and kriging are not distinguishable from IDW.
- **By depth:** error rises from 0.045–0.048 at 30 cm to 0.077–0.083 at 120–150 cm. Pooled R² falls from about 0.55 at 30 cm to about 0.25 at 150 cm.
- **Sanity check:** this agrees with published work. The OpenGeoHub station-blocked analysis of this farm reports RMSE ≈ 0.063 and R² ≈ 0.39.

## 4. Follow-up checks
| Scenario | Random forest RMSE | Random forest ubRMSE |
|---|---|---|
| A2: A + neighbour anomaly features | 0.066 | 0.045 |
| S: A2 + measured texture and bulk density at the target | 0.066 | 0.045 |
| A2, trained 2011–2014 and tested on 2015 test stations | 0.063 | 0.031 |
| IDW, 2015 test stations | 0.066 | 0.029 |
| Climatology, 2015 test stations | 0.068 | 0.031 |

## 5. What the pilot shows
1. **Station offsets dominate.** 54–59% of every model's squared error is a constant bias at the held-out station. Average soil-water levels differ between stations by 0.04 m³/m³ at 30 cm and 0.07 m³/m³ at 120–150 cm, but no model predicts these differences: R² of station mean level ≤ 0.19, often below 0. That holds even with texture and bulk density measured at the target.
2. **Neighbouring sensors add almost nothing for absolute level.** The sensor-assisted models do not beat the covariates-only ones. At 60–130 m spacing, a neighbour's level does not tell you the target's level.
3. **Dynamics are predictable.** Day-to-day changes track well: correlation over time of about 0.82 and ubRMSE of 0.045 overall (about 0.03 within a year). The best trained models still do not clearly beat IDW on dynamics in the 2015 test.
4. **There is little spatial structure at the station scale.** Fold-0 variograms are almost all nugget at 30 and 90 cm, and have long ranges with large nuggets at 120–150 cm. Little spatial correlation exists between 60 m and about 500 m, which threatens the fractal scaling test (Section 7.5 of the concept note).
5. **The offsets are probably not soil texture.** They may reflect installation, local soil structure or residual sensor calibration. Gasch et al. (2017) calibrated sensors after installation, so some offset may remain.

## 6. Implications for the paper (decisions needed)
- **Report two targets.**
  - Absolute volumetric water content, where an error floor of about 0.06 m³/m³ is a finding in itself.
  - Dynamics: anomalies relative to each station's own mean, scored by ubRMSE and correlation.
  - The sensor-density curves should be reported for both.
- **Consider a "short calibration" scenario.** Give the target location a brief temporary record, for example 2–8 weeks of a portable sensor or a few gravimetric samples, to fix its offset. Then measure error against calibration length. This is practical, and from what we know so far it would be new.
- **Scaling test:** run it on anomalies as well as levels. Expect the fractal claim on levels to be untestable or unsupported.
- **Stage 3 neural models (graph network, attention model, proposed model):** these should target dynamics plus an offset term. A neural model is unlikely to fix the offset problem.

## 7. Files
- `data/panel.parquet` and the station, depth, crop, weather and fold CSVs
- `results/predictions.parquet`, `station_metrics.csv`, `summary_by_model_depth.csv`, `pooled_by_depth.csv`, `paired_vs_idw.csv`, `leak_test.json` and `extra_*`
- **Leakage test:** fold 0 was rerun with the test stations' readings replaced by random values. The maximum change in predictions was 0 for mean, climatology, nearest neighbour, IDW and both LightGBM scenarios.
- **Pilot limits:** fixed hyperparameters, no tuning yet, no prediction intervals yet, and no neural baselines yet (planned for Stage 3).
