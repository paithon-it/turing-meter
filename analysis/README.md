# Analysis

Two reference analyses on public data, and the size of the next study. The same
checks run on any study the tool
measures.

## Is measured autonomy a quantity?

`michell_test.py` applies Luce and Tukey's conjoint measurement to METR's public
per-task results (Time Horizon 1.1). It is why turing-meter never puts different
agents on one latent scale.

```sh
curl -sLO https://metr.org/assets/task_results_1_1.yaml
python michell_test.py task_results_1_1.yaml            # 6 duration buckets
python michell_test.py task_results_1_1.yaml --bins 8   # sensitivity
```

### What explains the violations

Follow-ups added after two reviews (11-12 November 2026). They do not change the
test: they put the same question to other models, other buckets and parts of the
suite. Each simulation goes through the same buckets as the data. Shared code in
`michell_models.py`.

```sh
python michell_details.py task_results_1_1.yaml          # flips, horizons, runs per cell, the stopwatch read both ways, a worked example
python michell_collections.py task_results_1_1.yaml .    # SWAA, HCAST and RE-Bench per bucket; writes without_swaa.yaml and hcast_only.yaml
python michell_test.py hcast_only.yaml                   # the test inside one collection
python michell_random_buckets.py task_results_1_1.yaml   # 100 random divisions into 6 buckets, against Rasch
python michell_halflife.py task_results_1_1.yaml --sims 2000   # Ord's constant hazard, and a shape common to all agents
python michell_dimensions.py task_results_1_1.yaml       # tasks that weigh differently, and up to five dimensions
curl -sLO https://metr.org/assets/benchmark_results_1_1.yaml
python metr_horizons.py benchmark_results_1_1.yaml       # METR's published horizons
```

About a minute for `michell_dimensions.py` and six for `michell_random_buckets.py` on
four cores. Outputs and reading in section 5 of [`RESULTS.md`](RESULTS.md).

## Apparent intelligence on the reference meta-analysis

`composition_test.py` measures the intelligence factor and the turing on the
experiments collected by Vaccaro, Almaatouq and Malone (2024), each of which
measured humans alone, AI alone and humans with AI. Download
`Data_Extraction.csv` from the authors' OSF repository
(https://osf.io/wrq7c/?view_only=b9e1e86079c048b4bfb03bee6966e560, the share link published in the paper), then:

```sh
python composition_test.py Data_Extraction.csv
```

## How large the assimilation study must be

`assimilation_power.py` simulates the study of [`../docs/STUDY.md`](../docs/STUDY.md)
and its whole analysis: two arms, the same people before and after three months,
items whose difficulty person and machine share, people who differ in
sensitivity, noise and criterion, and a bootstrap inside every simulated study.
It checks that the primary statistic stays at zero when the arms learn by
different amounts, when people take the machine's lean, or when they weigh the
shared difficulty more, and it reports power and the chance of ruling out the
effect, by people per arm and at the corners of the pilot gate. No data needed
(about twenty minutes on four cores):

```sh
python assimilation_power.py --reps 400
```

Requires `numpy`, `scipy`, `pyyaml`. Neither dataset is redistributed here.
Results in [`RESULTS.md`](RESULTS.md).
