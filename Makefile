PY      ?= python3 -I
RAW     := data/raw/CAF
DAYMET  := data/raw/daymet_caf_2007_2016.csv
PROC    := data/processed
RES     := results
export FOLDS := config/folds.csv

.PHONY: all audit panel folds-check baselines evaluate followup calibration drift clean

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

clean:
	rm -rf $(PROC) $(RES)/*.parquet $(RES)/audit/*.parquet
