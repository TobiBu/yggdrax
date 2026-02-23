# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Local quality checks

Run these before opening a pull request:

```bash
black --check .
isort --check-only .
pydoclint .
pytest
```

## Pre-commit

Install and enable hooks once:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks on demand:

```bash
pre-commit run --all-files
```

## Testing and coverage

CI enforces test coverage through `pytest-cov`.

```bash
pytest --cov=yggdrax --cov-report=term-missing
```

## Docstrings

Public APIs should include concise docstrings that describe behavior and key
inputs/outputs. Keep implementation details in private modules and keep wrapper
module docstrings focused on user-facing contracts.

## Pull requests

- Keep PRs focused and scoped.
- Include tests for behavior changes.
- Update README and examples when public APIs change.
