> **English** · [Italiano](README.it.md)

# The angle between the errors

Code, protocol and results of the experiment behind the paithon.it article
["Intelligenza apparente"](https://paithon.it/intelligenza-apparente/) (in
Italian). This folder is part of the turing-meter repository, the instrument.

The experiment measures the angle between the errors of a person and of a
machine on the same items, the turing (how many times better the same person
picks out the right answer with the machine beside them), and checks
whether the law of the ideal pair predicts what a pair can really do. The
turing is the ratio of a person's sensitivity d′ with an AI to their
sensitivity alone; the angle is the arccos of the correlation between human and
machine errors, measured on ChatBench, ChaosNLI (αNLI), HAIID and
model-vs-human, against 24 machines. Hypotheses and criteria are in
`PROTOCOL.md`, written before looking at the data (the Italian original is
`PROTOCOL.it.md`); what changed afterwards is in its "Deviations" section.
Analyses A13 to A19 and the outcomes of A11 are in `PROTOCOL.md` only.

## How to cite

endjneer (2026). *Intelligenza apparente* (*Apparent Intelligence*). Zenodo.
https://doi.org/10.5281/zenodo.24107648 (all versions; the latest is the paper version, in
Italian and English)

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

## The closed models (added 14-15 November 2026)

Four Claude models (Haiku 5.5, Sonnet 5.5, Opus 5.5, Fable 5.2) answered the
ChatBench and αNLI items alone, one item per call, through Claude Code in
headless mode (`claude -p`: no tools, no MCP servers, no inherited settings, an
empty working folder, the letter constrained to the options by a response
schema). Thinking cannot be switched off on the larger models, so all four ran
at effort `low`. There are no probabilities, only verdicts, and the people never
worked with these models: the run gives k, the angle and the margin of the ideal
pair over the machine, not a turing. Exploratory (protocol, "Deviations of 14
November 2026"). The answers are in `closed-results/`, because they cannot be
redone exactly.

```bash
python -m closed.machines_claude probe claude-haiku-5-5                              # what the model receives
python -m closed.machines_claude run claude-opus-5-5 opus-5.5 chatbench,alphanli low  # resumes where it stopped
.venv/bin/python -m angle.frontier closed-results GPU_RESULTS_DIR                     # writes results/frontier.json
```

## Inside the conversations (added 15 November 2026)

What happens in the ChatBench conversations, exploratory and asked for by two
reviews (protocol, "Deviations of 15 November 2026"): where the points go
between the machine alone and the pair, extraction and filter, how often people
follow the chat when it is right and when it is wrong (against a linear
observer with no gating), always and never followers against identical people,
people split by skill (median over 300 random half splits, overall and within
mathematics), and the rule-of-three bound for a model with no error.

```bash
.venv/bin/python -m angle.conversations     # writes results/conversations.json, about half a minute
```

## After the sixth review (added 15 November 2026)

Two analyses asked for by a sixth review, exploratory (protocol, "After the
sixth review"). The ceiling of a pair when one of the two gives only a verdict:
the optimal observer under the article's model, for the rows of the article's
margin table and for the two ChatBench pairs. And, on αNLI, the closeness
between people and a machine net of the difficulty for a third population (the
open machines), and between two machines net of the difficulty for the people,
with the contested stories defined on half of the votes.

```bash
.venv/bin/python -m angle.verdict                                    # writes results/verdict-ceiling.json, about twenty minutes
.venv/bin/python -m angle.difficulty closed-results GPU_RESULTS_DIR  # writes results/difficulty.json, about ten minutes
```

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
