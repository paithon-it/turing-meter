# turing-meter: specification

Version 0.5, November 2026. endjneer, paithon.it.

turing-meter is the measuring instrument of *Intelligenza apparente*, an article
by the same author on [paithon.it](https://paithon.it/intelligenza-apparente/)
(2026, [doi:10.5281/zenodo.24107649](https://doi.org/10.5281/zenodo.24107649)).

## 1. Question

How much does a machine augment a human mind on a given kind of work, and how much
of what the two could do together actually shows?

Not "how intelligent is the machine". Measured on METR's public data, the best
scale we have for the autonomy of AI agents is not a quantity
(`analysis/michell_test.py`, `analysis/RESULTS.md`). turing-meter never places
different agents on one latent scale. It compares a mind with itself, on the same
work, alone and with a machine, and it compares the pair with the best a pair
like that could do.

## 2. Apparent intelligence

The article models the intelligence that shows in a piece of work as the composition
of two orthogonal components, like apparent power in an AC circuit: `S = P + jQ`,
`|S| = sqrt(P^2 + Q^2)`, with `P` natural intelligence and `Q` algorithmic
intelligence.

turing-meter reads that law as the **ideal of a pair**. Two observers who decide on
the same cases, each with noisy evidence, and who combine all their evidence in the
best way, reach a sensitivity that depends on how their errors are related. When
their errors are independent, sensitivities add in quadrature (Green and Swets
1966; Ernst and Banks 2002, for two senses of one person; Sorkin and Dai 1994,
for groups):

    |S| = sqrt(dH^2 + dM^2)

where `dH` and `dM` are the sensitivities of the person alone and of the machine
alone (section 3). Pairs of people fall short of it (Bahrami et al. 2010). Orthogonal, in this reading, means: **the two err on
different cases** (section 2.1).

Squares add because weights of evidence do. Under the model, the log-likelihood
ratio that one observation carries is normal, with mean `d'^2 / 2` in natural
units (on the yes/no convention of section 3) and variance twice the mean. Turing
found this at Bletchley Park, for a weight of evidence that is normal when the
hypothesis is true (Good 1985); the same result was published later, in 1954, by
Peterson, Birdsall and Fox. Weights of independent evidence add, and so do squared
sensitivities (`apparent.expected_weight_of_evidence`).

The instrument reports, per task family and protocol:

- **the turing** (Tu) `= dS / dH`, where `dS` is the sensitivity the person shows
  working with the machine, on the same cases as alone. A mind working alone is
  worth 1 Tu: every mind is its own standard. A pure number, like the per-unit
  values of electrical engineering, which give a quantity as a fraction of a base
  chosen for that machine; here the base is the person. No machine appears in the
  definition. A turing belongs to a person, or a group, on a task: the machine
  alone has none. Not defined at `dH = 0`.
- **the apparent intelligence** `|S|`, the ideal of the pair.
- **the apparent turings** `Tu_app = |S| / dH`: the ideal in the person's unit,
  what the two separate readings promise. The twin of apparent power in
  volt-amperes, with the turing as the twin of real power in watts; the two have
  the same dimension, and the two names say what was measured. One difference:
  volt-amperes are read without the angle, while `|S|` already contains it
  (section 2.1).
- **the intelligence factor** `eta = dS / |S| = Tu / Tu_app`: how much of the
  apparent intelligence shows in the work. The analogue of the power factor, real
  over apparent power. 0 at chance, 1 at the ideal. It describes how far the pair
  is from its ideal, not how the person behaves: see the gain share below.
- **the gain share** `g = (dS - dH) / (|S| - dH)`: how much of the possible gain
  the person realizes. 0 for a person who ignores the machine, 1 at the ideal.
  This is the number that describes behaviour.
- **the direction** `theta`, with `cos(theta) = dH / |S| = 1 / Tu_app`, the
  person's share of the apparent intelligence; it is also the `eta` of a person who
  ignores the machine. At 90 degrees `theta = atan(dM / dH)`: 45 degrees for equal
  partners, more when the machine is stronger, and `sin(theta)` is the `eta` of a
  person who always follows the machine.

By construction (these are definitions, not findings):

    Tu  = eta * Tu_app = eta / cos(theta)
    eta = cos(theta) + g (1 - cos(theta))

A stronger machine raises the apparent turings, the turing the person could reach,
even when people change nothing in how they work; `eta` says how many of them
arrive. It also lowers the floor of `eta`, so `eta` falls as the machine gets
stronger even when people realize the same share of the gain: comparisons across
machines of different strength use `g`, not `eta`.

**The square of the turing.** `Tu^2` is the weight of evidence an ideal observer
would need to do as well as the pair, over what it would need to do as well as the
person alone (`apparent.evidence_ratio`). Equivalently, it is the number of
hypothetical copies of the person, with independent errors and their evidence
pooled, that would decide as well as the pair. Real people are not such copies:
their errors correlate, and `n` observers whose errors correlate at `rho` reach at
most `1 / sqrt(rho)` times one of them. The efficiency `eta` of Tanner and Birdsall
(1958), defined on energies, is the square of an intelligence factor whose
reference is the ideal receiver instead of the ideal of the pair.

The power factor taken literally, `P / |S|`, has a counterpart on this scale, `cos(theta)`:
the part of the ideal the person already covers alone, the inverse of the apparent
turings. It equals `eta` only when `dS = dH`. The two value the same pair in
opposite ways: read as `P / |S|`, a person who relies fully on a strong machine has
a low factor; for the instrument, a pair that reaches its ideal has `eta = 1`. For
the electrician, bringing the power factor toward 1 is a virtue; for a pair,
`cos(theta)` near 1 means the machine adds little. turing-meter does not measure
`P / |S|`, and gives the name to `eta`.

### 2.1 The angle between the minds

Whether the errors are independent is a question for data. On items that both the
person alone and the machine alone have answered, the correlation `rho` of their
errors defines the angle between them, `phi = arccos(rho)`. Under the model, the
ideal of the pair becomes

    |S|^2 = (dH^2 + dM^2 - 2 rho dH dM) / (1 - rho^2)

the sensitivity of the best linear observer on two correlated Gaussian cues,
`d^T Sigma^-1 d` with `d = (dH, dM)` and `Sigma` their correlation matrix, as long
as both of its weights are non-negative. Above the useful angle (below), `|S|` is
also the length of the one vector whose orthogonal projections on two axes at
angle `phi` are `dH` and `dM`, and `cos(theta) = dH / |S|` is its projection on the
person's axis.

- 90 degrees: independent errors, the law of `|S|`;
- toward 0: they err on the same cases. The ideal falls to the stronger partner,
  which it reaches at `rho = weaker / stronger`. Beyond that point the
  unconstrained optimum (Sorkin and Dai 1994) would give the weaker partner a
  negative weight, using it only to subtract the noise the two share, and would
  rise above the stronger partner again. The instrument excludes negative weights
  by choice: a constraint, not a property of the model. Below that point the ideal
  is the stronger partner;
- beyond 90: each tends to be right where the other is wrong; the ideal exceeds
  the law.

The point where the ideal falls to the stronger partner is the **useful angle**,
`arccos(weaker / stronger)` (`apparent.useful_angle`): next to a machine `k` times
stronger, a person adds to the ideal only if their errors meet the machine's at
more than `arccos(1/k)`, that is 60 degrees for `k = 2`, 78.5 for 5, 84.3 for 10.
The better the machine, the closer to 90 the person must stay. Whether use moves
a person's angle is the question of the assimilation study (`docs/STUDY.md`).

In the person's unit, the ideal depends only on `k = dM / dH` and `phi`:

    Tu_app = sqrt(1 + k^2 - 2 k cos(phi)) / sin(phi)

`sqrt(1 + k^2)` at 90 degrees; below the useful angle it stops at `max(1, k)`
(`apparent.apparent_turings_from_ratio`, equal to `apparent.apparent_turings`).

The angle can also be read the other way, wanting the machine as close as
possible to the person, to reduce the angle. Here aligned errors are the worst
case for a pair, and 90 degrees is the best one that independent minds can count
on. The instrument takes the reading of signal detection theory, and says so.

`phi` needs many items. With 100 items answered by both, the 95% interval of
`phi` runs from about 68 to 118 degrees when the truth is 90; with 500, from 80 to
99. A study estimates `phi` per task family on the pooled items, not per person.

`phi` is measured, never assumed, whenever item-level data exist. It is measured
within each class of the right answer (`apparent.class_alignment`, and
`apparent.pooled_alignment` for many people, each with their own criterion):
within a class, being right and answering yes coincide, so the table of the two
minds' answers gives the correlation of their evidence whatever their criteria.
On right/wrong pooled over classes (`apparent.error_alignment`), two minds that
lean toward the same answer err together on one class and look aligned without
sharing any evidence. When no item-level data exist, the report says that `|S|`
assumes 90 degrees.

### 2.2 What the person receives

The ideal depends on what passes from the machine to the person. `|S|` assumes the
person can weigh the machine's evidence, that is its confidence. A person who
receives only the machine's verdict can at best reach the **verdict ideal**
(`apparent.verdict_ideal`), between the better of the two and `|S|`. Every report
gives `eta` against both.

The machine of the ideal is the one the person worked with. In a chat the machine
answers a person who asks, rephrases and insists; measured alone on the same items,
it is a different machine. An ideal computed on the machine alone can fall below
what the pair shows, and attributes to the person what the conversation drew from
the machine. Where the machine's answers inside the pair are recorded, `dM` is reported
both ways, on the same items: the turing does not change, the apparent turings and
`eta` do.

## 3. Magnitudes

Sensitivity `d'` of signal detection theory, equal-variance Gaussian model: zero
at chance, no ceiling, and the scale on which the composition law of section 2 is
stated. A ratio of two `d'` is a ratio on this scale and on no other; the report
says so (section 6, test 2).

- From hit and false-alarm rates (sensitivity and specificity): `d' = z(hit) - z(fa)`.
- From accuracy, for a balanced binary task and an unbiased decider:
  `d' = 2 z(accuracy)`.
- From the accuracy of a forced choice among `m` options, one of them right: the
  `d'` that solves `PC = E[Phi(Z + d')^(m - 1)]`, with `Z` standard normal and
  `Phi` its distribution function; with `m = 2`, `d' = sqrt(2) z(accuracy)`.
- From the area under the ROC curve: `d' = sqrt(2) z(AUC)`. Where the two
  variances differ (section 6, test 2), the same expression defines `d_a`, which
  takes the place of `d'`.

**Conventions.** The first two lines are the yes/no convention, and they agree at
a neutral criterion. A decision between two options read as yes/no and read as a
forced choice gives two values of `d'` a factor `sqrt(2)` apart at the same
accuracy. The turing, the apparent turings, `eta`, `g`, `theta` and the angles are
ratios or do not use `d'`, and do not depend on the convention as long as `dH`,
`dM` and `dS` follow the same one; differences in units of `d'` are not comparable
across conventions. The report states the convention.

A condition at or below chance has no magnitude and is reported, not used; the
turing is not defined at `dH = 0`. Precision, recall and F1 do not determine `d'`
without the base rate.

**Production tasks** (a result accepted or not by a check, produced over time):
accepted results per hour of the person's time. The machine alone has no hours of
the person, so only the turing is defined: no `|S|`, no apparent turings, no
`eta`.

## 4. Inputs

### 4.1 Task file

One directory per task, with a `task.yaml`:

```yaml
id: invoice-total-001
family: invoices                 # results are given per family
kind: production                 # decision | production
language: it
prompt: |
  Nella cartella inputs/ c'è una fattura. Scrivi nel file output/result.json
  il numero della fattura, la data (AAAA-MM-GG) e il totale in euro.
environment:
  image: python:3.12-slim
  files: [inputs/]
check:
  type: script                   # exit code 0 = success
  command: python check.py output/result.json
  protected: [check.py, expected.json]
chance: null                     # decision tasks: majority-class accuracy
human:                           # timed attempts, failures included
  - person: p01                  # an id from people.yaml
    condition: alone             # alone | with_machine
    minutes: 4.2
    passed: true
  - person: p01
    condition: with_machine
    protocol: series             # an id from people.yaml, with_machine only
    minutes: 2.9
    passed: true
```

People are judged by the same check as the machine. Rules:
- every task has a deterministic check; no LLM-as-judge;
- every path in a task file stays inside the task directory (task files may come
  from untrusted contributors);
- protected files never enter the agent's view of what can be edited; a change in
  their hash makes the run a failure and raises a tampering flag;
- decision tasks are answered item by item, by the person alone, the machine
  alone and the person with the machine, on the same items: `phi` needs it.

### 4.2 People and protocol

`people.yaml` declares who was measured and how they worked with the machine,
because the ideal belongs to the pair and to the way they are connected:

```yaml
people:
  p01:
    role: accounts clerk
    experience: 5+ years
    year: 2026
protocols:                                # one or more; test 3 needs two
  series:
    machine: anthropic/claude-sonnet-5-5   # model and version
    harness: reference                      # agent or interface
    final_decision: human                   # human | machine | shared
    shows_confidence: false                 # false: the verdict ideal applies
    order: counterbalanced                  # alone_first | machine_first | counterbalanced
```

People are pseudonymous ids. No names, no data that identify them.

### 4.3 Machine runs

The machine alone runs each task k times inside a fixed harness (Inspect, UK AI
Security Institute), with the same check and the tampering check. Raw logs are
JSONL; every report is rebuilt from them.

## 5. Analysis

Per task family and protocol, with intervals from a bootstrap at two levels, over
people and over items: both are samples, as in the multi-reader studies of
radiology (Dorfman, Berbaum and Metz 1992):

- `dH`, `dM`, `dS`, and `k = dM / dH`;
- `phi` from the items (section 2.1), or "assumed 90";
- `|S|`, the verdict ideal, `eta` against each, `theta`;
- the turing and the apparent turings, per person and for the group, with
  `Tu = eta * Tu_app`, and the yield on the machine, the same turing read
  from the machine's side, `Tu / k = dS / dM`;
- the gain share `g`, the number to compare across families and protocols;
- where the pair falls among the anchors: below the person alone (Tu < 1); above
  the better of the two (what the literature calls synergy or complementarity);
  above the confidence-sharing rule that pairs of people follow
  (`(dH + dM) / sqrt(2)`, Bahrami et al. 2010); above the ideal (`eta > 1`, which
  says that the model or the angle is wrong, or, below the useful angle, that the
  pair does what the constraint of non-negative weights excludes);
- the **adoption** `(dS - dH) / (dM - dH)`, only when the machine is stronger: how
  much of the gap the person lets through. It depends on the scale and is
  reported as a description, not a quantity.
- intervals from a bootstrap over papers or people say how many clusters they
  rest on; below ten, they are optimistic.

Never one number per model, never a ranking of models.

### 5.1 Reporting a turing

A turing is never given alone. Every value comes with:

- the task, the people, the machine (model, version, and where it was measured:
  section 2.2) and the circuit, that is how the machine is used (the protocol of
  section 4.2);
- whose turing it is, one person's or a group's (a group's sensitivities come from
  its pooled accuracies), and the design with its order (section 4.2): in a
  sequential design the same person answers a case alone first, then with the
  machine;
- `dH`, `dM` (or `k`) and `phi`, measured or assumed, on the convention of `d'`
  stated (section 3);
- its 95% interval, from the bootstrap over people and items.

Why:

- **Read beside `k`.** A person who copies every answer of a machine `k` times more
  sensitive is worth `k` turings, and neither `eta` nor `g` tells copying apart:
  for the copier `eta = k / Tu_app` and `g = (k - 1) / (Tu_app - 1)`, both close to
  1 near the useful angle. What the person adds to the machine shows only where
  `Tu > k`. The yield on the machine, `Tu / k = dS / dM`, reads it from that side:
  below 1, the person takes something away from the machine.
- **The base can move.** The definition holds no machine, and does not age with
  the machines it measures; the values are tied to people and cases. Values from
  different machines are set side by side only on the same people and the same
  cases. And if using the machine changes the person alone, the base moves: a
  person whose `dH` falls gains turings without the pair improving. Hence `dH` is
  always reported.
- **A ratio.** When `dH` is close to chance the denominator is small and
  uncertain, and the interval wide (`analysis/RESULTS.md`, section 2, Limits).
- **The unit has no ceiling; tests do.** When a pair answers every item right,
  `d'` can no longer be estimated, and a harder task is needed.

## 6. Validity tests

Each test states the result that would refute what it tests; the report prints the
outcome.

1. **The ideal is a ceiling.** With `phi` measured, `eta > 1` beyond noise in more
   than 5% of the pairs refutes the composition model for that family. Below the
   useful angle the ceiling is the constraint of non-negative weights (section
   2.1), and a pair above it refutes the constraint, not the composition. With
   `phi` assumed, `eta < 1` cannot tell correlated errors from a poor connection,
   and the test only excludes strongly opposed errors.
2. **Scale.** `d'` assumes equal-variance Gaussian evidence. Where confidence
   ratings exist, the slope of the z-ROC tests it; a slope far from 1 means `eta`
   must be computed on the unequal-variance model, and the report says so.
3. **Circuit.** Same people, same items, two protocols (machine confidence shown
   or not). If `eta` does not respond, it does not measure how the two are
   connected.
4. **Stability.** Same person, two sessions a week apart. If the turing or `eta`
   moves beyond its interval, it describes the session, not the mind.
5. **Reference data.** `analysis/composition_test.py` runs the analysis on the
   binary decision tasks of Vaccaro, Almaatouq and Malone (2024), which measured
   humans alone, AI alone and humans with AI. They have no item-level data, so
   `phi` is assumed 90 there. Results in `analysis/RESULTS.md`.
6. **Conjoint measurement.** `analysis/michell_test.py` on any task set: whether
   human time and machine ability combine as one quantity.

## 7. A second measure, for later

What we design in our image may come out different from us. `phi`
measures it on decisions. On open work, the same difference can be measured as the
evidence, in bits, that a judge gains about whether a piece of work was made by a
person or by a machine: zero for a perfect imitation. Not in the MVP.

## 8. Non-goals

No ranking of models. No price: the cost of work is a different question, and it
is kept out of the unit. No claim that the turing measures intelligence in itself:
it measures how much a given mind, on given work, shows more with a machine, on a
declared scale. No claim that the article's two axes, will and technique, are
measured: a person alone has both, and the instrument measures a person and a
machine.

## 9. Prior work

The parts are not new. Sensitivities that add in quadrature for independent
evidence: Green and Swets 1966; Ernst and Banks 2002. The ideal group, with
correlated observations: Sorkin and Dai 1994; Sorkin, Hays and West 2001. Pairs
of people: Bahrami et al. 2010, who found that pairs follow confidence sharing,
not the ideal, and that pairs of very different sensitivity do worse than the
better member; the turing is their collective benefit with the person, not the
better member, as denominator. The value of the AI's information against the
distortion in how people use it: Amin et al. 2026.
Error consistency between people and neural networks: Geirhos, Meding and
Wichmann 2020. Bayesian combination of human and machine classifications:
Steyvers et al. 2022; Kerrigan, Smyth and Steyvers 2021. Error correlation as
the condition for human-AI complementarity: Campagner, Cabitza and Crupi 2026
(who also name the "dominance trap"); Guo, Wu and Yiu 2026, with the threshold of
correlation beyond which no confidence-based rule gives complementarity. Human augmentation and the losses when the AI is stronger:
Vaccaro, Almaatouq and Malone 2024. People who take on the errors of the AI they
worked with: Vicente and Matute 2023; Vicente, Matute, Fregosi and Cabitza 2025;
Glickman and Sharot 2024; the literature for the assimilation study is in
`docs/STUDY.md`, section 3. The weight of evidence, and Turing's finding that its
variance is twice its mean when it is normal and the hypothesis is true: Good 1985
(Bayesian Statistics 2, pp. 249-270); the same result published in 1954 by
Peterson, Birdsall and Fox. The definition of `d'`, in use since 1954 and fixed by
Tanner and Birdsall 1958, with their efficiency `eta` on energies (JASA 30(10),
doi:10.1121/1.1909408). The same readers on the same cases,
with uncertainty over both: Dorfman, Berbaum and Metz 1992 (Investigative
Radiology 27(9), doi:10.1097/00004424-199209000-00015). What turing-meter adds is
the instrument: one protocol, item-level data, and the numbers of section 2
reported together.

## 10. Acceptance criteria

1. `analysis/composition_test.py` reproduces `analysis/RESULTS.md` from the public
   data.
2. A study with at least ten people, two task families and two protocols, and at
   least 500 decision items per family answered by the person alone and the
   machine alone, runs end to end and reports the numbers of section 5 and
   tests 1-4.
3. Every report rebuilds identically from the raw logs.
4. Tampering is detected on a planted test case.
