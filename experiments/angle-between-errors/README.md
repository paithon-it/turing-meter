> **English** · [Italiano](README.it.md)

# The angle between the errors

Code, protocol and results of the experiment behind the paithon.it article
["Intelligenza apparente"](https://paithon.it/intelligenza-apparente/) (in
Italian). This folder is part of the turing-meter repository, the instrument.

The experiment measures the angle between the errors of a person and of a
machine on the same items, the turing (how many times the same person
distinguishes right from wrong better with the machine beside them), and checks
whether the law of the ideal pair predicts what a pair can really do. The
turing is the ratio of a person's sensitivity d′ with an AI to their
sensitivity alone; the angle is the arccos of the correlation between human and
machine errors, measured on ChatBench, ChaosNLI (αNLI), HAIID and
model-vs-human, against 24 machines. Hypotheses and criteria are in
`PROTOCOL.md`, written before looking at the data (the Italian original is
`PROTOCOL.it.md`); what changed afterwards is in its "Deviations" section.

## How to cite

endjneer (2026). *Intelligenza apparente*. Version 1. Zenodo.
https://doi.org/10.5281/zenodo.24107649

Code and data: https://github.com/paithon-it/turing-meter, folder
`experiments/angle-between-errors`.

## What is here

- `angle/`: estimators (`sdt.py`, `measure.py`), data sources (`sources.py`,
  `download.py`, `items.py`), test of the law (`combiner.py`), analyses
  (`analysis*.py`, `run_chatbench.py`, `vaccaro.py`), summary and figures.
- `gpu/`: the scripts that run models and networks on a GPU.
- `tests/`: the estimators tested on simulated data with the design of each
  source. Protocol criterion: bias of ρ below 0.03.
- `results/`: the results the article is written on. The machines' responses
  are kept apart (see below).

## Rerunning it

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pandas scikit-learn pytest
.venv/bin/python -m pytest tests -q
.venv/bin/python -m angle.download data chatbench chaos haiid geirhos   # the public data, in data/
.venv/bin/python -m angle.items data                                    # the items, in data/items/
```

The ChaosNLI data (CC BY-NC 4.0) and the Vaccaro meta-analysis are not
redistributed. ChaosNLI is downloaded by `angle/download.py`; the table by
Vaccaro, Almaatouq and Malone (2024) must be taken by hand from the authors'
OSF repository
(https://osf.io/wrq7c/?view_only=b9e1e86079c048b4bfb03bee6966e560, the link
published in their paper) and saved as `data/vaccaro/Data_Extraction.csv`. The
scripts do not run code from the data authors.

On the GPU (NVIDIA with 24 GB, vLLM and PyTorch), from the experiment folder:

```bash
bash gpu/prepare.sh                     # models, images and data, in ~/rnd
bash gpu/queue.sh                       # vision networks and silicon people
bash gpu/run_llm.sh                     # language models
```

The responses end up in `gpu-results/` (they are not in this repository: they
are redone with the scripts in `gpu/`). Then, locally:

```bash
.venv/bin/python -m angle.analysis         GPU_RESULTS_DIR
.venv/bin/python -m angle.analysis_vision  GPU_RESULTS_DIR
.venv/bin/python -m angle.analysis_silicon SILICON_OUTCOMES_DIR
.venv/bin/python -m angle.run_chatbench
.venv/bin/python -m angle.vaccaro data/vaccaro/Data_Extraction.csv
.venv/bin/python -m angle.summary
.venv/bin/python -m angle.figure
```

The two tests of the instrument (protocol, "The instrument on trial"):

```bash
.venv/bin/python -m angle.calibration                        # simulation, about twenty minutes
.venv/bin/python -m angle.full_evidence GPU_RESULTS_DIR      # the ten vision networks
```

The four checks asked for by a review on 10 November 2026, exploratory
(protocol, "Deviations of 10 November 2026"), write `results/review.json`
without touching the other results:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m angle.review GPU_RESULTS_DIR   # a few minutes with two processes
```

`summary.json` has one row per hypothesis of the protocol.

## Licenses

- Code (`angle/`, `gpu/`, `tests/`): MIT license, see the repository
  `LICENSE` at `../../LICENSE`.
- Protocol and derived results (`PROTOCOL.md`, `PROTOCOL.it.md`, `results/`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). What derives from
  ChaosNLI stays under its license,
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- The source data are not here: ChatBench (CC BY 4.0), ChaosNLI (CC BY-NC 4.0),
  HAIID (MIT), model-vs-human and the Vaccaro meta-analysis remain under their
  authors' licenses.
- The article text: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
