PYTHON ?= python

.PHONY: reproduce-small
reproduce-small:
	@echo "Exporting minimal dataset (canonical-only, no augmentation, CSV)"
	$(PYTHON) -m tictactoe.cli datasets export --out data_raw/small --canonical-only --no-augmentation --epsilons 0.1 --format csv
	@echo "Verifying export integrity"
	$(PYTHON) scripts/verify_export.py data_raw/small
	@echo "OK"
