VENV := .venv

.PHONY: init clean

init:
	poetry config virtualenvs.in-project true --local
	poetry install
	poetry run pre-commit install
	poetry env info

clean:
	rm -rf $(VENV)