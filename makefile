# Makefile for your project

.PHONY: init clean

init:
	uv sync --group dev
	# Installed as an isolated tool (not a project dependency) since
	# label-studio's own pinned deps conflict with this project's.
	uv tool install label-studio
	uv run pre-commit install

# Remove the virtual environment
clean:
	rm -rf .venv