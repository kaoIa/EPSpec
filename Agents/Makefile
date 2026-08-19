.PHONY: install test coverage lint typecheck doctor demo api mcp build

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

coverage:
	python -m pytest --cov=epspec_agents --cov-branch --cov-report=term-missing

lint:
	python -m ruff check .

typecheck:
	python -m mypy epspec_agents

doctor:
	python -m epspec_agents doctor

demo:
	python -m epspec_agents demo

api:
	python -m epspec_agents serve

mcp:
	python -m epspec_agents mcp

build:
	python -m build
