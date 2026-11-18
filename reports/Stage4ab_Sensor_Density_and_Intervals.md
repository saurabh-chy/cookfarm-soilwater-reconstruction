# Stage 4a–b: sensor-density curves and prediction intervals (CAF)

Run date: 18 Nov 2026. Scripts: `scripts/11_sensor_density.py` and `scripts/12_intervals.py`. Outputs: `results/sensor_density/` and `results/intervals/`.

## 4a. Sensor density and placement
**Design.**
- The test stations are fixed in each outer fold.
- The input network (all non-test stations, 33–34 per fold) is cut to 100%, 75%, 50%, 30% or 15% of its stations: on average 33.6, 25.6, 16.6, 10 and 5 stations.
- Two placement rules: three random draws per level, and one spatially even draw by farthest-point sampling.
- Models:
  - IDW and ordinary kriging, with the variogram fitted from development stations.
  - LightGBM A, retrained at each level with network features from the reduced network.
  - Random forest B uses no network inputs, so its error stays at 0.066 / 0.045 at every density.
- Scoring: RMSE for level and ubRMSE for dynamics, averaged per held-out station and then across the 42 stations.

| Input stations | IDW, level | IDW, dynamics | Kriging, level | Kriging, dynamics | LightGBM A, level | LightGBM A, dynamics |
|---|---|---|---|---|---|---|
| 33.6 (all) | 0.070 | 0.046 | 0.070 | 0.046 | 0.068 | 0.045 |
| 25.6 | 0.071 | 0.047 | 0.070 | 0.046 | 0.068 | 0.045 |
| 16.6 | 0.072 | 0.048 | 0.071 | 0.047 | 0.068 | 0.045 |
| 10 | 0.074 | 0.050 | 0.073 | 0.048 | 0.068 | 0.045 |
| 5 | 0.079 | 0.056 | 0.077 | 0.053 | 0.067 | 0.045 |

(Random placement. Random forest B, which uses no network, is 0.066 / 0.045 throughout.)

**Findings.**
1. **Interpolation needs many sensors; the learned models do not.** IDW and kriging lose accuracy steadily as stations are removed. From 34 to 5 stations, level error rises by about 0.006–0.008 m³/m³ and dynamics error by about 0.007–0.010. LightGBM A stays flat (0.067–0.068 level, 0.045 dynamics) all the way down to 5 input stations, and random forest B, with no network at all, is as good or better.
2. **What the network contributes, the weather already supplies.** At this site, day-to-day dynamics at a new location are driven by weather, season and crop. A model given those, plus a handful of sensors or none, matches the full network. Extra sensors do not help reconstruct other locations. Their value lies at their own location: they fix that location's level (Stage 3a).
3. **Even spacing does not beat random placement.** Farthest-point placement is slightly worse for IDW and kriging at 5–17 stations (for example IDW 0.078 against 0.074 at 10 stations), because it pushes sensors to the field edges. For LightGBM A, placement makes no difference.

## 4b. Prediction intervals (split conformal, reserved calibration stations)
**Design.**
- Models are fitted on development stations only.
- For each depth, the 80% and 95% quantiles of |error| at the 6 reserved calibration stations give symmetric intervals around the prediction.
- Coverage and width are scored at the test stations.

| Model | Nominal | Pooled coverage | Mean width (m³/m³) | Station coverage, 10th percentile | Share of stations >10 points below nominal |
|---|---|---|---|---|---|
| Random forest B | 80% | 0.81 | 0.19 | 0.61 | 24% |
| Random forest B | 95% | 0.94 | 0.29 | 0.80 | 14% |
| LightGBM A | 80% | 0.80 | 0.20 | 0.62 | 26% |
| LightGBM A | 95% | 0.93 | 0.31 | 0.80 | 17% |
| IDW | 80% | 0.76 | 0.18 | 0.55 | 36% |
| IDW | 95% | 0.91 | 0.29 | 0.78 | 26% |

**By depth** (random forest B):
- 80% coverage: 0.66 at 30 cm, 0.84–0.85 at 60–150 cm.
- 95% coverage: 0.87 at 30 cm, 0.95–0.97 below.
- 80% width: 0.09 at 30 cm, rising to 0.24 at 150 cm.

**Findings.**
1. **Pooled coverage is about right for the tree models,** so calibrating on held-out stations works on average.
2. **The intervals are too wide for most decisions.** They span 0.19–0.31 m³/m³, compared with a field range of soil-water content of about 0.1–0.45. They are wide because they must absorb unknown station offsets.
3. **Coverage is uneven across locations.** A quarter of stations are badly under-covered at 80%, the ones whose offset is unusual. At 30 cm the intervals are too narrow, because the calibration stations happen to have smaller shallow-depth offsets than the test stations.
4. **Implication:** honest uncertainty at an unsensored location is dominated by its unknown level. Short calibration (Stage 3a) is the practical way to narrow it; another model is not.

## Summary for the paper
- Adding sensors improves interpolation but not the best learned models.
- Placement rules matter little.
- Honest, calibrated intervals are wide because of station offsets.

Together with Stages 2–3c, the message is consistent: at field scale, reconstruction at a new location is limited by location-specific level, not by network density or model capacity. Dynamics are largely weather- and crop-driven.
