# Results

## 1. Is measured autonomy a quantity? (1 November 2026)

Data: METR `task_results_1_1.yaml` (Time Horizon 1.1). Output of `python michell_test.py task_results_1_1.yaml` (6 buckets, 200 simulations, seed 534):

```
23 agents, 228 tasks, 5207 cells, 27119 runs, 19153 successes

== Models (binomial log-likelihood, AIC, BIC) ==
M0: LL=   -6640.9  k= 24  AIC=  13329.8  BIC=  13526.8
M1: LL=   -6590.5  k= 46  AIC=  13273.1  BIC=  13650.6
M2: LL=   -3094.2  k=250  AIC=   6688.4  BIC=   8740.4
M3: LL=   -3015.9  k=272  AIC=   6575.9  BIC=   8808.4

[1] Per-agent slopes (M1) vs common slope (M0): LR=100.8, df=22, p=4.72e-12
    slopes (logit per doubling of human minutes): -0.73 to -0.39, median -0.58; common -0.58
    same question with free difficulties (M3 vs M2): LR=156.5, df=22, p=2.81e-22
    pairs whose order flips between the 50% and 80% horizon: 9 of 253 with METR's published fits (11 with ours)

[2] Rasch difficulty (from machine runs only) vs log2(human minutes), 156 identified tasks:
    r=0.740, R^2=0.547
    residual SD 2.11 logits, i.e. a factor 7.5 in human minutes at equal machine difficulty
    tasks failed by every agent: 13, solved by every agent: 59

[3] Conjoint measurement, 23 agents x 6 duration buckets (cells with >= 20 runs):
    buckets (human minutes): ['0.02-0.09', '0.09-2.11', '2.11-12.62', '12.62-51.09', '51.09-360.00', '360.00-1800.00']
    row independence: 71 of 253 agent pairs change order across buckets
    column independence: 3 of 15 bucket pairs change order across agents
    double cancellation: 4735 violations in 219416 cases with a true antecedent (2.16%)
    expected under M0 (common slope on human time (additive)): rows 60.3 (p=0.140); columns 1.1 (p=0.025); double cancellation 0.81% (p=0.000)
    expected under M1 (per-agent slope (METR)): rows 79.5 (p=0.830); columns 1.1 (p=0.030); double cancellation 0.88% (p=0.000)
    expected under M2 (Rasch, machine-estimated difficulty (additive)): rows 36.3 (p=0.000); columns 1.4 (p=0.040); double cancellation 1.57% (p=0.040)
    expected under M3 (machine difficulty, per-agent slope): rows 53.0 (p=0.015); columns 1.4 (p=0.035); double cancellation 1.21% (p=0.020)
    mean success rate per bucket: [1.   0.96 0.73 0.75 0.35 0.24]
    bucket 2 vs 4: the longer bucket does better for 5 agents
    bucket 3 vs 4: the longer bucket does better for 18 agents
    bucket 5 vs 6: the longer bucket does better for 1 agents
```

## Reading

1. **One slope does not fit all agents** (p ~ 5e-12). With METR's published fits, 9 of 253 pairs of agents swap order between the 50% and the 80% horizon: five nearly tied pairs, four far apart (up to almost a factor of two, Claude Opus 4.6 vs Gemini 3.1 Pro).
2. **Human time explains 55% of machine difficulty** (r = 0.74). Tasks taking a person 13-51 minutes are solved more often than tasks taking 2-13 minutes by 18 of 23 agents. 72 of 228 tasks carry no information on differences between agents (59 solved by all, 13 by none).
3. **Double cancellation fails beyond noise**: 2.16% violations against 0.8-0.9% expected with human time as difficulty, and 1.2-1.6% with difficulty estimated from the runs. The direction holds with 5 and 8 buckets.

Conclusion: the time horizon is a useful reference scale (like Rockwell hardness or an octane number), not an additive quantity. Every number turing-meter reports must state its threshold, task set and success rule.

## 2. Apparent intelligence on the reference meta-analysis (2 November 2026)

Data: Vaccaro, Almaatouq and Malone (2024), `Data_Extraction.csv` from the authors'
OSF repository,
https://osf.io/wrq7c/?view_only=b9e1e86079c048b4bfb03bee6966e560. Output of
`python composition_test.py Data_Extraction.csv` (2000 draws, seed 534):

```
Comparisons on the d' scale: 87 from 27 papers
  from accuracy: 76
  from sensitivity+specificity: 10
  from AUC: 1
Left out:
  accuracy: a condition at or below chance: 12
  d' as reported: a condition at or below chance: 2
  metric F1-Score: no d': 2
  metric Precision: no d': 2
  metric Recall: no d': 2
  metric Score: no d': 1
  metric Specificity: no d': 1

All: n = 87 from 27 papers
  share of the ideal, eta      median 0.69  [0.52, 0.80]  (each paper once: 0.69)
  share of the possible gain, g median 0.30  [0.19, 0.43]  (each paper once: 0.31)
  eta against the verdict ideal       0.76
  turing                      median 1.24  [1.18, 1.70]
  pair beats the better of the two: 31; pair below the person alone: 12

Person stronger (theta < 45): n = 20 from 8 papers  (fewer than 10 papers: the interval is optimistic)
  share of the ideal, eta      median 0.81  [0.78, 0.88]  (each paper once: 0.82)
  share of the possible gain, g median 0.24  [0.00, 0.51]  (each paper once: 0.24)
  eta against the verdict ideal       0.87
  turing                      median 1.07  [1.00, 1.16]
  pair beats the better of the two: 14; pair below the person alone: 5

Machine stronger (theta >= 45): n = 67 from 23 papers
  share of the ideal, eta      median 0.59  [0.48, 0.79]  (each paper once: 0.64)
  share of the possible gain, g median 0.32  [0.22, 0.43]  (each paper once: 0.33)
  eta against the verdict ideal       0.65
  turing                      median 1.39  [1.23, 1.79]
  pair beats the better of the two: 17; pair below the person alone: 7

Same people in both conditions: n = 26 from 17 papers
  share of the ideal, eta      median 0.75  [0.64, 0.81]  (each paper once: 0.78)
  share of the possible gain, g median 0.38  [0.30, 0.43]  (each paper once: 0.35)
  eta against the verdict ideal       0.82
  turing                      median 1.34  [1.19, 1.48]
  pair beats the better of the two: 11; pair below the person alone: 1

Different people: n = 61 from 11 papers
  share of the ideal, eta      median 0.59  [0.46, 0.81]  (each paper once: 0.63)
  share of the possible gain, g median 0.22  [0.12, 0.44]  (each paper once: 0.20)
  eta against the verdict ideal       0.65
  turing                      median 1.23  [1.07, 2.01]
  pair beats the better of the two: 20; pair below the person alone: 11

AI showed its confidence: n = 28 from 12 papers
  share of the ideal, eta      median 0.82  [0.66, 0.88]  (each paper once: 0.79)
  share of the possible gain, g median 0.46  [0.30, 0.61]  (each paper once: 0.33)
  eta against the verdict ideal       0.90
  turing                      median 1.23  [1.18, 1.36]
  pair beats the better of the two: 17; pair below the person alone: 1

AI showed only its verdict: n = 59 from 16 papers
  share of the ideal, eta      median 0.57  [0.48, 0.78]  (each paper once: 0.64)
  share of the possible gain, g median 0.20  [0.15, 0.38]  (each paper once: 0.26)
  eta against the verdict ideal       0.59
  turing                      median 1.24  [1.09, 1.97]
  pair beats the better of the two: 14; pair below the person alone: 11

Machine stronger, AI showed its confidence: n = 18 from 9 papers  (fewer than 10 papers: the interval is optimistic)
  share of the ideal, eta      median 0.77  [0.65, 0.90]  (each paper once: 0.69)
  share of the possible gain, g median 0.41  [0.30, 0.65]  (each paper once: 0.35)
  eta against the verdict ideal       0.83
  turing                      median 1.35  [1.21, 1.45]
  pair beats the better of the two: 8; pair below the person alone: 0

Machine stronger, AI showed only its verdict: n = 49 from 14 papers
  share of the ideal, eta      median 0.51  [0.33, 0.77]  (each paper once: 0.62)
  share of the possible gain, g median 0.25  [0.16, 0.40]  (each paper once: 0.30)
  eta against the verdict ideal       0.54
  turing                      median 1.51  [1.22, 2.01]
  pair beats the better of the two: 9; pair below the person alone: 7

Without the largest study (Lai_2020, 16 comparisons, 2 distinct groups of people alone):
  person stronger: n = 20, eta median 0.81, g median 0.24
  machine stronger: n = 51, eta median 0.66, g median 0.37
  AI showed its confidence: n = 28, eta median 0.82, g median 0.46
  AI showed only its verdict: n = 43, eta median 0.72, g median 0.25

Is the fall of eta with theta behaviour or arithmetic?
  rank correlation theta-eta: -0.81; with g shuffled across comparisons: -0.74 [-0.83, -0.65]
  rank correlation theta-g: -0.18; same people only: -0.02

Anchors:
  below the person alone (Tu < 1): 12/87
  above the better of the two: 31/87
  above confidence sharing (Bahrami 2010): 3/87; its median share of the ideal: 0.98
  above the ideal (eta > 1): 3/87
  person alone barely above chance (dH < 0.35, accuracy < 0.57): 21/87, from {'Lai_2020': 11, 'Liu_2021': 4, 'Cabrera_2023': 2, 'Lu_2022': 4}

Noise (accuracy rows): comparisons that hold in at least 95% of redraws,
by the number of items assumed behind each AI accuracy
  above the ideal: 3 observed, 2 with known noise; 20: 0, 50: 0, 100: 0, 200: 0, 1000: 0, exact: 0
  beats the better of the two: 31 observed, 26 with known noise; 20: 0, 50: 0, 100: 3, 200: 9, 1000: 12, exact: 13

Above the ideal:
  Zhao_2022  sensitivity+specificity  dH 0.92  dM 1.14  dS 1.60  eta 1.10
  Chen_2023  accuracy  dH 0.56  dM 0.64  dS 1.11  eta 1.31
  Liang_2022  accuracy  dH 0.94  dM 1.05  dS 1.41  eta 1.01
```

### Reading

The scale is `d'` and the ideal assumes that person and machine err independently
(90 degrees): the meta-analysis has no item-level data, so the angle cannot be
measured here (SPEC 2.1).

1. **People realize about a third of the possible gain.** Median gain share 0.30,
   95% interval over papers 0.19-0.43; on the 26 comparisons that measure the same
   people alone and with the machine, 0.38. As a share of the ideal, the pairs
   show 0.69 [0.52-0.80]; the observed turing is 1.24, against a median of 1.87
   apparent turings (`|S| / dH`, what the ideal would give).
2. **The share does not depend on who is stronger.** Gain share 0.24 when the
   person alone is the stronger partner (20 comparisons, 8 papers: the interval
   is optimistic), 0.32 when the machine is (67, 23 papers); 0.37 without Lai
   2020. The share of the ideal does fall (0.81 against 0.59), but that is
   arithmetic: `eta = cos theta + g (1 - cos theta)`, and the floor `cos theta`
   falls as the machine gets stronger. With the gain share shuffled at random
   across comparisons the rank correlation between `theta` and `eta` is still
   -0.74 [-0.83, -0.65], against -0.81 observed; between `theta` and the gain
   share it is -0.18, and -0.02 on the same people. The losses the meta-analysis
   found when the AI is stronger, and the "dominance trap" of Campagner, Cabitza
   and Crupi (2026), are about the pair failing to beat the AI (17 of 67 do), not
   about people using less of it: the ceiling rises faster than people follow.
3. **What the person receives may matter, and the data cannot say how much.**
   Gain share 0.46 when the AI showed its confidence (28 comparisons, 12 papers),
   0.20 when it showed only its verdict (59, 16 papers). Counting each paper once,
   0.33 against 0.26; within the machine-stronger group, 0.35 against 0.30. A
   difference between studies, not an experiment: test 3 of the SPEC is the
   experiment.
4. **Anchors.** 12 of 87 pairs are below the person alone; 31 beat the better of
   the two, the literature's synergy; 3 are above confidence sharing, the rule
   pairs of people follow (Bahrami et al. 2010), which is at 0.98 of the ideal in
   the median comparison; 3 are above the ideal. How many of the 31 beat the
   better partner beyond noise depends on the number of items behind each AI
   accuracy, which the studies do not report: 0 with 20 or 50 items, 3 with 100,
   9 with 200, 12 with 1000, 13 with an exact AI. None of the 3 above the ideal
   survives the noise.

### Limits

- **One scale.** `d'` from accuracy assumes an unbiased decider and balanced
  classes; from sensitivity and specificity it assumes equal-variance Gaussian
  evidence. On another scale every ratio changes. `d'` is used because the
  composition law is stated on it.
- **The angle is assumed.** With `phi` fixed, a pair below its ideal may have
  correlated errors or a poor connection; these data cannot tell which.
- **The ideal assumes the person can weigh the machine's evidence.** Most studies
  showed only a verdict; the verdict ideal is reported beside it.
- **Small denominators.** In 21 of 87 comparisons the person alone is barely above
  chance (accuracy below 0.57: Lai 2020 11, Liu 2021 4, Lu 2022 4, Cabrera 2023
  2). There the turing has an enormous interval: for the first Lai 2020 row,
  redrawing the two group accuracies with their standard errors, from 0.57 to
  more than 20 (the upper end depends on how close to chance the redraws fall). The 16 comparisons of Lai 2020 share two groups of people alone.
- **Selection by the stronger partner** uses the same noisy estimates.
- **Left out**: 14 comparisons with a condition at or below chance, and 8 rows
  whose metric does not determine `d'` (precision, recall, F1, a score, a
  specificity without its sensitivity).
- **Not the article's axes, and not the opposite reading.** The article composes
  will and technique; a person alone has both. And wanting the angle between human
  and machine small goes against signal detection theory, which wants the errors
  independent (SPEC 2.1).

## 3. How large the assimilation study must be (2 November 2026)

Superseded in part by section 4: after the review of the same day, the primary
statistic is no longer the pooled `rho`, and the bootstrap resamples people only.

Simulation of the study in `docs/STUDY.md`, under the model of
`src/turing_meter/apparent.py` with a difficulty that person and machine share.
Output of `python assimilation_power.py --reps 1000` (seed 2026):

```
The world simulated (latent values)
  machine d' = 2.00
  person  d' = 1.00 before, 1.22 after practice (both arms)
  useful angle (the person adds nothing below it): 60 degrees
  assimilation share a = 0.134
                       rho(person, machine)  angle   rho(person, person)
  before, both arms                  0.200   78.5                0.133
  after, control                     0.245   75.8                0.200
  after, machine arm                 0.345   69.8                0.214

Power to see a difference of 0.10 in rho (one-sided 5%, 1000 studies per cell)
forms    people/arm  items/person  person-items  naive|null  DiD|null  DiD|alt  sd   power
shared           25            50          1250      +0.046    -0.001   +0.095  0.101  0.24
shared           25           100          2500      +0.044    -0.002   +0.096  0.074  0.37
shared           25           200          5000      +0.045    -0.002   +0.096  0.049  0.62
shared           50            50          2500      +0.048    +0.001   +0.100  0.075  0.40
shared           50           100          5000      +0.038    -0.002   +0.098  0.052  0.60
shared           50           200         10000      +0.042    +0.000   +0.100  0.039  0.81
shared          100            50          5000      +0.047    +0.001   +0.099  0.052  0.58
shared          100           100         10000      +0.042    -0.001   +0.099  0.036  0.83
shared          100           200         20000      +0.045    +0.001   +0.100  0.026  0.98
unique           25            50          1250      +0.046    +0.000   +0.097  0.104  0.23
unique           25           100          2500      +0.041    -0.002   +0.099  0.072  0.42
unique           25           200          5000      +0.042    +0.001   +0.094  0.054  0.55
unique           50            50          2500      +0.040    -0.004   +0.098  0.077  0.38
unique           50           100          5000      +0.043    -0.001   +0.097  0.052  0.58
unique           50           200         10000      +0.043    -0.001   +0.098  0.038  0.84
unique          100            50          5000      +0.042    +0.001   +0.099  0.054  0.57
unique          100           100         10000      +0.042    -0.002   +0.098  0.037  0.85
unique          100           200         20000      +0.043    -0.001   +0.098  0.026  0.98
```

### Reading

1. **Practice alone closes the angle.** With no assimilation at all, `rho` in the
   machine arm rises by about +0.04 between the two tests, because people get less
   noisy and their remaining errors gather on the items that are hard for
   everyone, the machine included. A study without a control arm would report
   nearly half of the effect it looks for. The difference in differences against
   a control arm that practises on the same items is centred on zero.
2. **What counts is person-items.** Power follows the number of answers per arm
   and wave, not how they are split: about 10,000 for 80% power against a
   difference of 0.10 (50 people times 200 items, or 100 times 100). Shared and
   unique forms give the same power; shared forms also give the person-person
   correlation for free.
3. **The effect is far from the threshold.** In this world the useful angle is
   60 degrees and the effect takes the person from 78.5 to about 70: an
   assimilation of this size can be measured long before it reaches the
   threshold.

### Limits

- The same assimilation for everyone, and Gaussian evidence: both are to be
  checked on the pilot.
- The machine's quirks and the shared difficulty are drawn fresh for every
  form; real forms are fixed and balanced, which should help.

## 4. Whether the analysis of the assimilation study holds (2 November 2026, protocol v0.2)

A methodological review of protocol v0.1 found that the difference in
differences of the pooled `rho` holds only if the two arms learn by the same
amount, that a lean toward the machine's answer inflates it, and that the
bootstrap as written tripled its standard error. Version 0.2 measures the
alignment within each class of the right answer, corrects it for what practice
alone predicts, and resamples people only. The script now simulates the whole
analysis, bootstrap included, with the same people at both waves and people who
differ in sensitivity, noise and criterion. Output of
`python assimilation_power.py --reps 400` (seed 2026):

```
World: dM 2.0, dH 1.0 -> 1.22, rho0 0.2, useful angle 60 degrees; 50 people per arm at both waves, 200 items per test; 400 studies, 200 bootstrap draws each.
Columns: means of the three differences in differences; sd over studies and mean
bootstrap se of the adjusted one; P(yes) = 90% interval above 0, P(no) = below +0.10,
both for the adjusted; cls = P(yes) for the class rho; dHgap = d'H machine minus
control at T2.

scenario                                          pooled   class     adj     sd     se     P(yes) P(no)   cls  dHgap
null: equal learning                              +0.001  +0.001  +0.000  0.033  0.036  0.04±0.01  0.89  0.05  -0.00
null: machine arm learns more (dH 1.43)           +0.037  +0.040  -0.002  0.037  0.035  0.06±0.01  0.88  0.31  +0.21
null: machine arm takes the machine's lean        +0.147  -0.003  -0.002  0.042  0.041  0.05±0.01  0.81  0.04  +0.00
null: machine arm weighs shared difficulty x1.3   +0.069  +0.070  -0.002  0.040  0.035  0.07±0.01  0.88  0.62  -0.01
effect +0.10: equal learning                      +0.099  +0.102  +0.093  0.037  0.034  0.85±0.02  0.12  0.85  +0.00
effect +0.10: machine arm learns more             +0.135  +0.141  +0.092  0.037  0.035  0.83±0.02  0.10  0.97  +0.21

People per arm present at both waves
scenario                                          pooled   class     adj     sd     se     P(yes) P(no)   cls  dHgap
35 people, effect +0.10                           +0.097  +0.099  +0.092  0.042  0.041  0.72±0.02  0.10  0.73  +0.00
35 people, null                                   -0.001  -0.001  -0.000  0.042  0.042  0.05±0.01  0.76  0.04  +0.01
70 people, effect +0.10                           +0.098  +0.102  +0.093  0.030  0.029  0.94±0.01  0.11  0.94  -0.00
70 people, null                                   +0.001  +0.001  +0.001  0.029  0.030  0.04±0.01  0.97  0.05  -0.00

Corners of the pilot gate, 50 people per arm
scenario                                          pooled   class     adj     sd     se     P(yes) P(no)   cls  dHgap
corner dH 0.8, k = 3, rho0 0.13, effect           +0.101  +0.103  +0.095  0.041  0.040  0.75±0.02  0.07  0.81  +0.00
corner dH 0.8, k = 3, rho0 0.13, null             +0.003  +0.003  +0.002  0.039  0.041  0.06±0.01  0.79  0.06  +0.00
corner dH 1.5, k = 1.5, rho0 0.30, effect         +0.098  +0.102  +0.091  0.038  0.035  0.83±0.02  0.11  0.81  +0.01
corner dH 1.5, k = 1.5, rho0 0.30, null           +0.002  +0.001  +0.002  0.037  0.036  0.07±0.01  0.86  0.07  +0.00
```

### Reading

1. **Only the adjusted statistic survives every null.** The machine arm learning
   more raises the class `rho` by +0.04. Weighing the shared difficulty more,
   which is what learning the truth does when the machine is near the truth,
   raises it by +0.07. Taking the machine's lean raises the pooled `rho` by
   +0.15. The adjusted statistic stays within 0.002 of zero in all three, and
   declares assimilation in 4 to 7% of null studies, against 5% nominal.
2. **Power.** Against +0.10: 0.85 with 50 people per arm present at both tests,
   0.72 with 35, 0.94 with 70; 0.75 and 0.83 at the two corners of the pilot gate.
   The adjustment costs little: it absorbs about 0.007 of an effect of 0.10.
3. **Ruling out.** With no effect, the interval falls below +0.10 in 89% of
   studies with 50 people, 76% with 35 and 97% with 70. Effects below 0.10 cannot
   be ruled out at this size.
4. **The bootstrap fits.** People resampled within arm, the same draw at both
   waves, the forms fixed: the mean bootstrap standard error stays within 15% of
   the spread over studies, slightly low where the null is hardest, which is why
   the false positives reach 7%.

### Limits

- One family of worlds, with Gaussian evidence and the same assimilation for
  everyone.
- No component shared by people and not by the machine: it would break the
  adjustment, and only the control arm can show it (STUDY section 4).
- Dropouts at random only. In the review's own check, computing T0 on everyone
  and T2 on those who stay biased the difference by about 0.02 when dropout
  depended on the person's noise; the same people at both waves did not.
- 200 bootstrap draws per simulated study; the study itself uses 2,000.

