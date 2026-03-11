.PHONY: setup install install-dev format lint test streamlit notebook clean

# One-time setup: venv + install
setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install -e . 2>/dev/null || .venv/bin/python -m pip install -e .
	@echo "Done. Activate: .venv\\Scripts\\activate (Win) or source .venv/bin/activate (Unix)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	black research apps scripts
	isort research apps scripts

lint:
	ruff check research apps scripts
	mypy research apps --ignore-missing-imports

test:
	pytest tests -v

streamlit:
	streamlit run apps/streamlit/Home.py --server.port $${STREAMLIT_SERVER_PORT:-8501}

notebook:
	jupyter notebook notebooks/

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
