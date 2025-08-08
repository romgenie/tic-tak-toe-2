PYTHON ?= python

.PHONY: reproduce-small
reproduce-small:
	@echo "Exporting minimal dataset (canonical-only, no augmentation, CSV)"
	$(PYTHON) -m tictactoe.cli datasets export --out data_raw/small --canonical-only --no-augmentation --epsilons 0.1 --format csv
	@echo "Verifying export integrity"
	$(PYTHON) scripts/verify_export.py data_raw/small
	@echo "OK"

.PHONY: reproduce-all
reproduce-all:
	@echo "Seeding and running full deterministic pipeline"
	PYTHONHASHSEED=0 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
	$(PYTHON) -m tictactoe.cli --deterministic --seed 0 datasets export --out data_raw/full --format both --epsilons 0.05,0.1 --canonical-only
	@echo "Running multi-seed benchmarks (no randomness, but for CI timing)"
	$(PYTHON) scripts/run_benchmarks.py
	@echo "Building docs"
	$(PYTHON) -m pip install .[dev]
	mkdocs build --strict
