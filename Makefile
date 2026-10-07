.PHONY: help install test lint demo audit corpus clean

help:
	@echo "install  install the package and dev tools"
	@echo "test     run the test suite"
	@echo "lint     run ruff"
	@echo "demo     synthetic end-to-end run, no data needed"
	@echo "audit    Part I audit (set METADATA=path/to.csv)"
	@echo "clean    remove caches and generated outputs"

install:
	pip install -e ".[dev]"

test:
	pytest tests -q

lint:
	ruff check src scripts tests

demo:
	python scripts/00_demo_end_to_end.py --out outputs/demo

audit:
	python scripts/01_run_audit.py --metadata $(METADATA) --out outputs/audit

clean:
	rm -rf .pytest_cache .ruff_cache outputs
	find . -name __pycache__ -type d -exec rm -rf {} +
