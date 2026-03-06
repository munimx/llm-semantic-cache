# Contributing

Thanks for contributing to Recallm. Keep changes small, focused, and limited to one behavior at a time.

## Development setup

```bash
git clone https://github.com/munimx/recallm
cd recallm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,redis]"
```

Use `fakeredis` for Redis backend tests; you should not need a real Redis server for the test suite.

## Run the checks

```bash
pytest                    # all tests
pytest tests/storage/     # storage tests only
mypy src/                 # type checking
ruff check src/ tests/    # linting
```

## Pull request requirements

- Include tests for new behavior.
- Run `pytest`, `mypy src/`, and `ruff check src/ tests/` before opening the PR.
- For bug fixes, include a regression test that fails before your fix and passes after.
- Keep PR scope tight: one problem, one focused solution.

## Scope policy

Features not in AGENTS.md §1 are out of scope and will be declined unless maintainers explicitly expand scope first.

## What makes a good PR

A good PR is small, easy to review, and does one thing well. Prefer clear names, straightforward control flow, and explicit tests over broad refactors.
