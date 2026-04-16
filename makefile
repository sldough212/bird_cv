# Makefile for your project

.PHONY: init clean

init:
	uv sync --group dev
	uv pip install -e ../sam2
	uv run pre-commit install

# Remove the virtual environment
clean:
	rm -rf .venv