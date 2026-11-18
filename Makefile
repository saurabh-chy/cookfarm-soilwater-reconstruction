PY      ?= python3 -I
RAW     := data/raw/CAF
DAYMET  := data/raw/daymet_caf_2007_2016.csv
PROC    := data/processed
RES     := results
export FOLDS := config/folds.csv

.PHONY: all audit panel folds-check baselines evaluate followup calibration drift neural-pilot density intervals clean

all: audit panel folds-check baselines evaluate followup calibration drift

audit:
	$(PY) scripts/01_audit.py $(RAW) $(DAYMET) $(RES)/audit

panel:
	$(PY) scripts/02_build_panel.py $(RAW) $(DAYMET) $(PROC)

# regenerate folds and confirm they match the frozen config/folds.csv
folds-check: panel
	$(PY) scripts/03_make_folds.py $(PROC)
	cmp $(PROC)/folds_frozen_format.csv config/folds.csv && echo "folds match config/folds.csv"

baselines: panel
	$(PY) scripts/04_baselines.py $(PROC) $(RES) --leak-test
	$(PY) scripts/04_baselines.py $(PROC) $(RES)

evaluate:
	$(PY) scripts/05_evaluate.py $(RES)

followup: panel
	$(PY) scripts/06_followup_checks.py

calibration:
	$(PY) scripts/07_calibration.py

drift: panel
	$(PY) scripts/08_offset_drift.py

# one-seed neural pilot (about 10 min on 2 CPU cores); full sweeps: notebooks/run_neural_sweep.ipynb
neural-pilot: panel
	for f in 0 1 2 3 4; do for m in gcn attn srn srn_unshared; do \
	  $(PY) scripts/09_neural.py --model $$m --fold $$f --seed 0 || exit 1; done; done
	$(PY) scripts/10_evaluate_neural.py --seeds 0

density: panel
	$(PY) scripts/11_sensor_density.py

intervals: panel
	$(PY) scripts/12_intervals.py

clean:
	rm -rf $(PROC) $(RES)/*.parquet $(RES)/audit/*.parquet
