# Makefile for your project

VENV := .venv
PYTHON := python3

.PHONY: init

init:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "source $(VENV)/bin/activate"
	poetry env use $(VENV)/bin/python
	poetry install
	poetry run pip install -e ../sam2
	poetry run pre-commit install


# Remove the virtual environment
clean:
	rm -rf $(VENV)