# Stage 3b: what drives the year-to-year drift in station offsets? (CAF)

Run date: 17 Nov 2026. Script: `scripts/08_offset_drift.py`. Outputs: `results/offset_drift/`.

## Question
Stage 3a found that a station's offset (how much wetter or drier it runs than the model predicts) is not fixed: it shifts between years by about 0.025 m³/m³. This stage tests what explains that shift.

## Method
- **Offset, measured from the data alone** (no model involved): each station's daily reading minus the mean of all other stations at the same depth that day, averaged over each calendar year from 2011 to 2015. Subtracting the network mean removes weather and season shared across the field.
- **Drift** = each year's offset minus the station's long-run offset.
- **Data used:** 890 station-depth-years across 201 station-depth series. A year counts only if at least half its days are valid, and a series only if it has at least 3 such years.
- **Scale:** the drift SD is 0.026 m³/m³. For comparison, the SD of long-run levels between stations is 0.057 m³/m³.
- **Candidate explanations,** each scored by leave-one-year-out cross-validated R²: crop this year, crop last year, a steady sensor-drift trend at each station, each station's response to wet versus dry years, and sensor type × year.
- **Crop significance:** tested by shuffling whole crop sequences between strips (1,000 permutations), which keeps the rotation structure.

## Results
| Explanation | Cross-validated R² of drift |
|---|---|
| Crop this year (crop table) | **0.12** (permutation p < 0.001; 95th percentile of the null −0.003) |
| Crop this year (strip map) | 0.08 |
| Crop last year | 0.02 |
| This year's and last year's crop | 0.10 |
| Steady trend at each station (sensor drift) | < 0 (no predictive value) |
| Station response to wet or dry year | < 0 (no predictive value) |
| Sensor type × year | < 0 |

**The crop effect makes agronomic sense and grows with depth.** The table gives the mean drift by current crop, in m³/m³:

| Current crop | 30 cm | 60 cm | 90 cm | 120 cm | 150 cm |
|---|---|---|---|---|---|
| Legume (pea, lentil) | +0.009 | +0.011 | +0.013 | +0.016 | +0.023 |
| Oilseed (canola) | +0.008 | +0.005 | +0.006 | +0.000 | −0.002 |
| Spring cereal | −0.002 | +0.002 | +0.004 | +0.004 | +0.001 |
| Winter cereal | −0.005 | −0.009 | −0.012 | −0.013 | −0.014 |

- **Depth:** cross-validated R² by depth is 0.08, 0.08, 0.08, 0.12 and 0.22 from 30 to 150 cm.
- **Mechanism:** a station under winter wheat runs drier than usual, especially at depth. Winter wheat roots deeply and grows for most of the year. Under shallow-rooted, short-season legumes the deep soil stays wetter. In the year after a legume, the pattern reverses slightly.

## Crop-record discrepancy resolved in scope
- **Where they differ:** the two crop records disagree only in 2015, for 12 of 17 strips. For 2011–2014 they agree, and give the same R² of 0.10.
- **Which fits better:** for 2015, the crop table explains the drift better (R² 0.12 against 0.08 for all years), which is weak evidence that it is the correct record.
- **Decision:** use the crop table as primary and the strip map as a sensitivity check. Ask WSU or USDA to confirm if possible.

## Interpretation
1. **Crop rotation is a real, significant cause** of year-to-year offset drift. It explains about 12% of it, rising to 22% at 150 cm.
2. **Not sensor drift:** there is no steady trend. Drift is only weakly persistent from one year to the next, with a lag-1 correlation of −0.07 where independent years would give about −0.25.
3. **Not wet-year sensitivity:** stations do not respond differently to wet and dry years in a way that predicts out of sample.
4. **Most drift (about 88%) is unexplained** by the available records. Likely contributors are management and residue differences within strips, local differences in root depth and yield, and measurement noise. The crop group already appears as an input in the Stage 2 models, but only as a weak main effect.

## Implications
- **Proposed model:** let the offset term depend on the current crop, by depth (location offset + crop-by-depth adjustment).
- **Short calibration:** when a calibration record and its use span different crop years, adjust the transferred offset by the crop-effect difference, estimated only from development stations. This can be tested in Stage 4 against the 365-day plateau of 0.046.
- **Paper:** report crop rotation as a measurable driver of reconstruction error at depth. This is a hydrological result that does not depend on any neural model.
