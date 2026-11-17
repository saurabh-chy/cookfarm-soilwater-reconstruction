# Stage 3a: short-calibration pilot (CAF)

Run date: 17 Nov 2026. Builds on `Stage2_Baseline_Pilot.md` and concept note v2.4, Section 7.7. Script: `calibration.py`.

## Design
- **Starting point:** the Stage 2 spatial reconstructions at each held-out station (5 folds, 42 stations, 5 depths) for five models: climatology, IDW, ordinary kriging, random forest B and LightGBM A.
- **Calibration records taken at the target station:**
  - Continuous windows of 7, 14, 28 or 56 days.
  - 1, 3, 6 or 12 spot readings on random days within a 180-day campaign, with measurement noise of SD 0 or 0.02 m³/m³.
- **Corrections:** a constant offset, or an offset plus gain. The gain is fitted only when the record's predicted values vary enough.
- **Evaluation:** the 365 days after the record ends. The record itself is never scored.
- **Starts:** records begin every 30 days from January 2011.
- **Unit of analysis:** station-depth results are averaged per station, then across stations.

## Main results: random forest B, all depths

| Calibration record | RMSE, no correction | RMSE with offset correction | Lowest possible (ubRMSE) |
|---|---|---|---|
| 1 spot reading | 0.065 | 0.057 | 0.037 |
| 3 spot readings | 0.065 | 0.052 | 0.037 |
| 6 spot readings | 0.065 | 0.051 | 0.037 |
| 12 spot readings | 0.065 | 0.050 | 0.037 |
| 7-day window | 0.064 | 0.054 | 0.037 |
| 14-day window | 0.064 | 0.053 | 0.037 |
| 28-day window | 0.064 | 0.053 | 0.037 |
| 56-day window | 0.064 | 0.052 | 0.037 |
| 91-day window | 0.064 | 0.051 | 0.037 |
| 182-day window | 0.064 | 0.049 | 0.037 |
| 365-day window | 0.064 | 0.046 | 0.037 |

- **Other models:** IDW, kriging and climatology follow the same pattern. Offset-corrected errors are 0.052–0.059, 0.051–0.058 and 0.055–0.062 respectively.
- **Measurement noise:** 0.02 m³/m³ of noise costs little once there are 3 or more spot readings.
- **Offset plus gain:** always worse than a constant offset. It overfits short records.
- **Season of the record:** makes little difference, with offset-corrected RMSE of 0.052–0.055 for a 14-day window.
- **By depth** (14-day window, random forest B):

| Depth | No correction | Offset correction | ubRMSE |
|---|---|---|---|
| 30 cm | 0.043 | 0.037 | 0.026 |
| 60 cm | 0.060 | 0.054 | 0.038 |
| 90 cm | 0.071 | 0.059 | 0.042 |
| 120 cm | 0.075 | 0.057 | 0.039 |
| 150 cm | 0.074 | 0.060 | 0.040 |

## Interpretation
1. **A short record removes about 20% of the absolute error.** Three to six spot readings spread over a season do about as well as a continuous 1–2 month sensor record. Most of the benefit comes from the first few readings.
2. **There is a plateau well above the floor.** Even a full year of calibration leaves 0.046, against a floor of 0.037. The reason is that each station's offset is not constant over the years:
   - The SD of a station's annual mean offset across years is 0.025 m³/m³.
   - The SD of offsets between stations is 0.055 m³/m³.
   - So about two-thirds of the offset is a persistent property of the location, and the rest changes from year to year.
3. **Candidate causes of the year-to-year part:**
   - Crop rotation by strip changes water use at depth each year.
   - Sensor drift or recalibration.
   - Real differences in how locations respond between wet and dry years.

   This can be tested in Stage 3 by relating annual offsets to crop sequence, sensor type and annual rainfall.
4. **Practical message for the paper:** at a new location, a handful of spot measurements in the first season improves the absolute level noticeably. Reaching dynamics-level accuracy needs either repeated recalibration each season or year, or a model of what drives the year-to-year offset.

## Next steps (Stage 3)
- **Year-to-year offset:** explain it using crop sequence, rainfall year and sensor type.
- **Neural baselines and the proposed model:** the graph network and attention model, plus the proposed model with its separate dynamic component and location offset, all under the frozen folds.
- **Prediction intervals:** add them for both targets.
