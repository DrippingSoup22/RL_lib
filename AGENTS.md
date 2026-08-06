# Project guidance

This is a reinforcement-learning study project.

## Development environment

- The project runs in WSL using the virtual environment at
  `/home/daniele/.venvs/rl-lib`.
- Zed's Codex extension does not inherit virtual-environment activation from
  the integrated terminal. Run Python tools through the virtual environment's
  interpreter explicitly.
- Run tests with
  `/home/daniele/.venvs/rl-lib/bin/python -m pytest`.
- Run linting with
  `/home/daniele/.venvs/rl-lib/bin/python -m ruff check .`.
- Check formatting with
  `/home/daniele/.venvs/rl-lib/bin/python -m ruff format --check .`.
- The repository is opened from
  `/mnt/c/Users/danie/SharedFolder/RL_lib`.
- Prefer the equivalent Makefile shortcuts (`make test`, `make lint`,
  `make format-check`, and `make check`) for routine work. The Makefile uses
  the same explicit WSL virtual-environment interpreter.

When working on algorithms:

- Explain the mathematical idea before proposing code.
- Do not implement an entire algorithm unless explicitly requested.
- Let the user write the central update rule when the goal is learning.
- Review mathematical correctness separately from code quality.
- Prefer small deterministic tests before Gymnasium experiments.
- Distinguish implementation bugs from hyperparameter problems.
- Keep reusable code in `src/rl_lib` and experiment definitions outside it.
