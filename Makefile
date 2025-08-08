PYTHON ?= python

.PHONY: setup
setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

.PHONY: lint
lint:
	ruff check .
	black --check .

.PHONY: type
type:
	mypy src

.PHONY: test
test:
	pytest -q --cov=tictactoe --cov-branch --cov-report=term-missing --cov-fail-under=90

.PHONY: docs
docs:
	mkdocs build --strict

.PHONY: docs-serve
docs-serve:
	@echo "Serving docs at http://127.0.0.1:8000 (Ctrl+C to stop)"
	mkdocs serve -a 127.0.0.1:8000

.PHONY: reproduce-min
reproduce-min:
	@echo "Running minimal smoke reproduction (<1 min)"
	PYTHONHASHSEED=0 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
	$(PYTHON) scripts/run_experiments.py min --seed 0

.PHONY: reproduce
reproduce:
	@echo "Running full reproducible pipeline"
	PYTHONHASHSEED=0 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
	$(PYTHON) scripts/run_experiments.py full --seed 0

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
	mkdocs build --strict

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup            Install dev extras"
	@echo "  lint             Ruff + black --check"
	@echo "  type             mypy"
	@echo "  test             pytest with coverage (>=90%)"
	@echo "  docs             Build MkDocs"
	@echo "  reproduce-min    Smoke: run_experiments min"
	@echo "  reproduce        Full: run_experiments full"
	@echo "  reproduce-small  Export small canonical dataset and verify"
	@echo "  reproduce-all    Full deterministic pipeline + docs build"

