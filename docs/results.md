# Results

`runs` is the complete, generated experiment history. `results` is a concise,
human-readable selection of evidence.

Each promoted result should contain:

- `summary.csv` with aggregate metrics;
- one or more clearly labelled figures;
- a short `README.md` explaining the setup and conclusion;
- the run IDs from which it was produced.

Compare algorithms within one environment because reward scales differ across
environments. Prefer mean evaluation return over several seeds, and show
variability when the number of runs makes that meaningful.

