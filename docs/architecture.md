# Architecture

The repository has four boundaries:

1. `src/rl_lib` contains reusable, installable code.
2. `experiments` selects algorithms, environments, and parameters for a run.
3. `runs` contains raw generated output and is not version controlled.
4. `results` contains selected summaries and figures worth keeping.

Dependencies point in one direction: experiments may import `rl_lib`, while
`rl_lib` never imports from experiments, runs, or results.

Start library code in the smallest relevant package. Add new packages such as
`envs` or `multi_agent` only when concrete reusable code exists for them.

