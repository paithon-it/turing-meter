# Calibrating one person against one machine: the simulations

The calculations behind [`docs/PAIR-PROTOCOL.md`](../../docs/PAIR-PROTOCOL.md): how many cases a
person needs, what the intervals of `k` and of the angle can say, what each rule of use returns
inside the model, what the pitfalls cost. The program they test is `turing-meter pair`
(`src/turing_meter/pair.py`).

The scripts read the estimators of the experiment (`experiments/angle-between-errors/angle/`) by
path and write nothing inside it; set `TURING_METER_EXPERIMENT` to point elsewhere. They need
numpy and scipy (`uv run python analysis/pair-calibration/<script>.py` from the repository root);
`verify_chatbench.py` also needs pandas and the public ChatBench data
(`python -m angle.download data chatbench`, run inside the experiment). Results in
`results/` were produced on Python 3.11 with the locked numpy 2.4.6
(`uv run --python 3.11 --with pandas python <script>`): the Monte Carlo draws change in the last
digit with another numpy. About 20 minutes on four CPUs; the conversion table
(`conversion-m4.npz`, not committed) takes about 6 minutes the first time.

The `.json` files keep some Italian labels (scenario keys such as `gpt4o-come-nei-dati`); the `.txt`
files are in English.

| Script | What it does | Result |
|---|---|---|
| `estimators.py` | The estimators of the experiment in vector form, the conversion table on a grid, the generative process | |
| `sim_sample.py` | One person, 50 to 1,600 cases, six situations: estimates, intervals, coverage, readings | `sample.txt`, `sample.json` |
| `sim_sample.py --errors` | Precision of the angle by number of machine errors | `machine-errors.txt` |
| `sim_sample.py --verify` | The fast path against `measure.compare` | `verification.txt` |
| `sim_rules.py` | The alarm and the rules of use inside the model | `rules.txt` |
| `sim_pair.py` | Turing and yield of one person; repetition | `pair.txt` |
| `sim_group.py` | One person against a group, with `measure._one_pass` | `group.txt` |
| `sim_two_phase.py` | The two-phase design | `two-phase.txt` |
| `sim_pitfalls.py` | Keys, contamination, hard cases, yes/no, open answers | `pitfalls.txt` |
| `verify_chatbench.py` | Calibrations extracted from the real data; the machine changing its answer | `chatbench-check.txt` |
| `minor_figures.py` | Cost of the two errors, sentinel, memory, reliability, margin in errors | `minor-figures.txt` |
| `check_pair_tool.py` | `turing-meter pair` on sheets whose truth is known | `tool-check.txt` |
| `make_example.py` | The made-up sheet in `docs/examples/` | `docs/examples/pair-sheet-example.csv` |

The `.txt` files other than `sample.txt` are the standard output of their script
(`python <script> > results/<name>.txt`); the `.err` files hold what went to standard error and
are not committed.
