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

## Testing and coverage

CI enforces test coverage through `pytest-cov`.

```bash
pytest --cov=yggdrasil --cov-report=term-missing
```

## Docstrings

Public APIs should include concise docstrings that describe behavior and key
inputs/outputs. Keep implementation details in private modules and keep wrapper
module docstrings focused on user-facing contracts.

## Pull requests

- Keep PRs focused and scoped.
- Include tests for behavior changes.
- Update README and examples when public APIs change.
