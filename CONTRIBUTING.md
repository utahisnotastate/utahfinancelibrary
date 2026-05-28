# Contributing to Utah Finance Library

Thank you for helping improve the [Utah Finance Library](https://github.com/utahisnotastate/utahfinancelibrary).

## Development setup

```bash
git clone https://github.com/utahisnotastate/utahfinancelibrary.git
cd utahfinancelibrary
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

Optional JAX stack:

```bash
pip install -e ".[jax,dev]"
```

## Pull request guidelines

1. Keep changes focused; one feature or fix per PR.
2. Add or update tests in `tests/`.
3. Do not change `SOVEREIGN_PROTOCOL_TITHE` (0.023) without an explicit governance note in the PR description.
4. Update relevant docs under `docs/` when behavior changes.
5. Run `pytest` and `python -m src.app.ignite --manifest Utahfile --dry-run` before submitting.

## Code style

- Python 3.10+
- Prefer `dataclasses`, type hints, and explicit constants in `src/core/constants.py`
- Log audit and settlement events at INFO

## Related ecosystem

- [utahisnotastate](https://github.com/utahisnotastate) — related engines (`utahcontainerengine`, `wavetheory`, etc.)

## Questions

Open a [GitHub Issue](https://github.com/utahisnotastate/utahfinancelibrary/issues) for bugs or feature requests.
