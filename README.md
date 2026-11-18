# cookfarm-soilwater-reconstruction

Station-blocked benchmark for reconstructing multi-depth soil water content at unsensored locations from a sparse sensor network at the R. J. Cook Agronomy Farm (CAF), Pullman, Washington, USA. The study covers sensor density, short calibration at new locations, and scale-recursive neural models. Every table in the reports can be regenerated from the scripts here.

> Status: work in progress. Stages 1 (data audit), 2 (clean panel, frozen folds, baseline pilot), 3a (short-calibration pilot), 3b (offset-drift analysis) the 3c neural pilot (one seed), and 4a–b (sensor-density curves and prediction intervals) are complete. Remaining: multi-seed neural sweeps (`notebooks/run_neural_sweep.ipynb`), the literature check and the manuscript.

## Study design in brief
- **Data:** 42 stations, five depths (30–150 cm), daily volumetric water content, 1 Jan 2011 – 16 Jun 2016, plus terrain, soil, crop and Daymet weather covariates.
- **Validation:** five spatially contiguous outer folds of 8–9 stations (`config/folds.csv`, frozen). A calibration subset of 6 stations is reserved in each outer-training set. Held-out stations never contribute to inputs, fitting, scaling or tuning. A leakage test confirms this.
- **Two targets:**
  - absolute soil-water level
  - dynamics: deviation from each location's own mean
- **Short calibration:** how much a brief record at a new location (spot readings or a short sensor window) improves reconstruction.

The concept note is in `docs/concept_note_v2.5.docx`. Stage reports are in `reports/`.

## Reproducing the results

### 1. Environment
```bash
conda env create -f environment.yml
conda activate cookfarm-soilwater
```
Or install with pip from `requirements.txt`, and from `requirements-neural.txt` for the neural models (PyTorch; preinstalled on Colab). The results were produced with Python 3.13 and the pinned versions listed there.

### 2. Data (not redistributed here)
Download the two input files listed in [`data/README.md`](data/README.md) and place them as follows:
```
data/raw/CAF/                         # contents of CAF_Sensor_Dataset_2.zip
data/raw/daymet_caf_2007_2016.csv     # Daymet single-pixel CSV
```
Check the SHA-256 checksums given in `data/README.md`.

### 3. Run
```bash
make all        # audit -> panel -> folds check -> baselines (+ leakage test) -> evaluate -> follow-up checks -> calibration -> offset drift
```
Individual steps: `make audit`, `make panel`, `make folds-check`, `make baselines`, `make evaluate`, `make followup`, `make calibration`, `make drift`, `make neural-pilot`, `make density`, `make intervals`. Runtime is about 30 minutes on 2 CPU cores.

All randomness uses fixed seeds. `make folds-check` regenerates the folds and confirms they match the frozen `config/folds.csv`.

## Repository layout
```
config/folds.csv            frozen outer folds and per-fold roles (test / calib / inner k)
scripts/01_audit.py         Stage 1 data audit
scripts/02_build_panel.py   clean daily panel, QC rule, covariates, weather features
scripts/03_make_folds.py    fold construction (used to create config/folds.csv)
scripts/04_baselines.py     mean, climatology, nearest neighbour, IDW, ordinary kriging,
                            random forest and LightGBM (scenarios A and B); --leak-test
scripts/05_evaluate.py      station-level metrics, bootstrap CIs, paired tests, leakage check
scripts/06_followup_checks.py  neighbour-anomaly and surveyed-covariate scenarios; 2015 test
scripts/07_calibration.py   short-calibration experiment
scripts/08_offset_drift.py  drivers of year-to-year station offset drift (crop, trend, wet year, sensor)
scripts/09_neural.py        neural models: gcn, attn, proposed scale-recursive srn, srn_unshared ablation; --leak-test
scripts/10_evaluate_neural.py  neural vs baselines for both targets, multi-seed
notebooks/run_neural_sweep.ipynb  full models x folds x seeds sweep (Jupyter / Colab, GPU-aware)
scripts/11_sensor_density.py   input network cut to 100-15% (random and spread placement); level and dynamics
scripts/12_intervals.py        split-conformal 80/95% intervals from the reserved calibration stations
results/                    summary tables (CSV/JSON); large prediction files are not committed
reports/                    stage reports
docs/                       concept note
```

## Key results so far (pilot, not confirmatory)
| Model (station-blocked) | RMSE, m³/m³ (95% CI) |
|---|---|
| Seasonal climatology | 0.073 (0.065–0.082) |
| Inverse distance weighting | 0.070 (0.063–0.079) |
| Ordinary kriging | 0.070 (0.063–0.078) |
| Random forest, covariates only | 0.066 (0.059–0.074) |

- **Station offsets dominate:** 54–59% of each model's squared error is a constant station offset, and station mean levels are essentially unpredictable from covariates or neighbouring sensors.
- **Dynamics are predictable:** temporal correlation is about 0.82.
- **Short calibration:** a few spot readings at the target cut error by about 20%. Station offsets also drift from year to year by about 0.025 m³/m³.
- **Neural models (pilot):** graph, attention and the proposed scale-recursive model reach RMSE 0.071–0.073, at the level of IDW and below random forest (0.066), on both targets. Station anomalies show no scale structure (scaling exponent H ≈ 0–0.08), so the fractal premise is not supported at this site.
- **Sensor density:** cutting the input network from about 34 to 5 stations raises IDW and kriging error by 0.006–0.008 m³/m³, but leaves gradient boosting flat (0.067–0.068). Even spatial placement is no better than random.
- **Intervals:** calibration-station intervals reach about nominal pooled coverage (0.81 / 0.94) but are 0.19–0.29 m³/m³ wide, and they under-cover about a quarter of stations.
- **Crop rotation:** the current crop explains about 12% of that drift (22% at 150 cm; permutation p < 0.001). Winter cereals run drier and legumes wetter, mainly at depth. Sensor-drift trends and wet-year sensitivity show no predictive value.

## Data sources and citation
- Gasch, C. K., et al. (2017). A field-scale sensor network data set for monitoring and modeling the spatial and temporal variation of soil water content in a dryland agricultural field. *Water Resources Research*, 53, 10878–10887. https://doi.org/10.1002/2017WR021307 Data: https://doi.org/10.15482/USDA.ADC/1349683
- Thornton, M. M., et al. (2022). Daymet: Daily surface weather data on a 1-km grid for North America, Version 4 R1. ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2129

To cite this repository, see `CITATION.cff`. A Zenodo DOI will be added on release.

## License
Code: MIT (see `LICENSE`). The input data remain under their original terms and are not included.
