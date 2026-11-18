# Stage 3c: neural models, one-seed pilot (CAF)

Run date: 18 Nov 2026. Scripts: `scripts/09_neural.py` (models and training) and `scripts/10_evaluate_neural.py` (evaluation for both targets). Full multi-seed sweeps: `notebooks/run_neural_sweep.ipynb`.

**Headline:** under the frozen station-blocked protocol, none of the neural models beats the tree-based baselines. The proposed scale-recursive model is not better than a standard graph network. The training data also show no measurable scale structure in station anomalies (H ≈ 0–0.08), so the fractal premise is not supported at this site. These are pilot results (one seed, no tuning), but they agree with Stages 2–3b.

## Models (sensor-assisted scenario A)
All four models share an input encoder and an output head:
- **Encoder:** a GRU per station over a 14-day window of readings (with a missingness mask), weather and crop, combined with the station's static covariates.
- **Output head:** two parts, whose sum is the prediction:
  - a **level head** (crop group × depth, with no network input), trained on station-year means;
  - a **dynamics head**, trained on within-year anomalies.

| Model | Spatial module | Parameters |
|---|---|---|
| `gcn` | 2 graph-convolution layers, Gaussian distance kernel (150 m) | 29,315 |
| `attn` | 2 spatial self-attention layers, learned distance bias | 43,527 |
| `srn` (proposed) | one shared operator applied recursively over 90, 180, 360 and 720 m neighbourhoods using band-pass (detail) aggregates; scale decay exp(−κj), κ initialised from the training-only scaling exponent and then freely updated | 29,412 |
| `srn_unshared` | the same model with a separate operator per scale (weight-sharing ablation) | 57,348 |

**Training protocol:**
- **Test stations:** inputs zeroed and masked at every step.
- **Calibration stations:** visible as inputs only.
- **Training targets:** a random 25% of the development stations are masked as targets in each batch.
- **Early stopping:** one inner fold of development stations, never visible during training.
- **Normalisation:** statistics from development stations only.
- **Run time:** about 30 s per run on 2 CPU cores.

**Leakage test:** fold 0 was rerun with the test stations' readings replaced by random values. The maximum change in predictions was 0.0 for all four models.

## Two design findings during development
1. **Level heads overfit.** The first design gave the level head the 16 static covariates, as an MLP and then as a linear model. Validation error on levels rose from the first epoch, because the head memorised the levels of the roughly 27 training stations. Restricting the level head to crop group × depth fixed this. That agrees with Stage 2 (station levels cannot be predicted from covariates) and Stage 3b (crop is the one measurable driver).
2. **No scale structure.** The training-only semivariance of station anomalies is flat from about 70 to 440 m. In fold 0 the binned values are 0.24, 0.29, 0.31 and 0.25 in standardised units. The fitted scaling exponent H is 0.02, 0.08, 0.04, 0.03 and 0.00 across the five folds. The learned decay κ stayed small (0.06–0.13), so the model weights all scales almost equally.

## Results (5 folds, seed 0; station-level means with 95% CIs from resampling stations)
| Model | RMSE, level (m³/m³) | 95% CI | ubRMSE, dynamics | Median temporal r | Dynamics skill vs climatology |
|---|---|---|---|---|---|
| Random forest, covariates only (B) | **0.066** | 0.059–0.074 | **0.045** | 0.82 | 0.21 |
| Random forest, sensor-assisted (A) | 0.067 | 0.060–0.074 | 0.045 | 0.82 | 0.21 |
| LightGBM, sensor-assisted (A) | 0.068 | 0.061–0.077 | 0.045 | 0.83 | 0.20 |
| Inverse distance weighting | 0.070 | 0.063–0.079 | 0.046 | 0.81 | 0.14 |
| Ordinary kriging | 0.070 | 0.063–0.079 | 0.046 | 0.81 | 0.14 |
| Graph network `gcn` | 0.071 | 0.063–0.079 | 0.048 | 0.81 | 0.09 |
| Attention `attn` | 0.072 | 0.064–0.080 | 0.048 | 0.80 | 0.11 |
| **Proposed `srn`** | 0.072 | 0.064–0.081 | 0.049 | 0.79 | 0.07 |
| Seasonal climatology | 0.073 | 0.065–0.081 | 0.049 | 0.77 | 0 |
| `srn_unshared` (ablation) | 0.073 | 0.065–0.082 | 0.049 | 0.79 | 0.04 |

**Paired, station-level differences:**
- **Against random forest B:** every neural model is worse on both targets.
  - `srn`: +0.0058 RMSE (CI +0.0013 to +0.0106) and +0.0041 ubRMSE (CI +0.0024 to +0.0061).
  - `gcn`: +0.0045 RMSE and +0.0028 ubRMSE.
- **Against IDW:**
  - `gcn` is equal on level (+0.0001, CI −0.0019 to +0.0022).
  - `srn` is equal on level (+0.0014, CI −0.0014 to +0.0044) and worse on dynamics (+0.0028, CI +0.0004 to +0.0053).
- **Weight sharing:** `srn` is slightly better than `srn_unshared` (0.072 against 0.073) with half the parameters. The difference is small and not tested across seeds.

**Sensitivity check:** a smaller, slower configuration (hidden size 24, 30-day window, learning rate 1e-3) made `gcn` and `srn` slightly worse (0.072 and 0.073), so the pilot conclusion does not depend on that choice.

## Interpretation
- **The bottleneck is not model capacity.** It is information: station levels are mostly unpredictable, and anomalies decorrelate within about one station spacing. Flexible spatial models have little structure to learn, and with about 27 training stations they lose to random forest, which averages more conservatively.
- **The scaling premise behind the proposed architecture is absent at this site.** Under concept note Section 7.5, the fractal/scaling claim is marked unsupported for this dataset. The scale-recursive model can be reported only as a parameter-efficient graph model: about equal to `gcn` with the same parameter count, and better than its unshared ablation with half the parameters.

## What remains (to firm up these conclusions)
- **Seeds and tuning:** run 5 seeds with inner-fold tuning, using the notebook (about 2–3 hours on a CPU, much faster on a GPU), to confirm the pilot ranking and the shared versus unshared comparison.
- **Scale-decay variants:** free, initialised and constrained κ, using `--kappa-mode`.
- **Neural models in scenario B** (covariates only), for completeness.

## Recommendation for the paper
Lead with the hydrology and benchmark results:
- the station-blocked benchmark;
- the error floor caused by station offsets;
- short calibration;
- crop rotation as a driver at depth;
- the absence of scale structure.

Report the neural models, including the proposed one, as a negative but informative comparison, not as the main contribution.
