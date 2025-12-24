# Makefile for your project

VENV := .venv
PYTHON := python3

.PHONY: init

init:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"
	@echo "source $(VENV)/bin/activate"
	poetry install

# Remove the virtual environment
clean:
	rm -rf $(VENV)