# turing-meter

> **English** · [Italiano](README.it.md)

**How much does a machine augment a human mind, and how much of what the two could
do together actually shows?**

`turing-meter` is the measuring instrument of
[*Intelligenza apparente*](https://paithon.it/intelligenza-apparente/) (*Apparent Intelligence*),
an article on paithon.it, in Italian with an English translation
(doi:[10.5281/zenodo.24107648](https://doi.org/10.5281/zenodo.24107648), all versions).
The article takes the apparent sensitivity of the ideal pair of Sorkin and Dai,
`|S| = sqrt(dH^2 + dM^2)` when the errors of the two are independent, and borrows
its vocabulary from AC circuits (apparent power, power factor). It conjectures that the
intelligence that shows in a piece of work composes will and craft at right angles,
and the instrument reads only the craft axis. This repository reads the law of `|S|`
as the ideal of a person working with a machine, and measures how close real pairs
get to it.

> Status: **design stage** (November 2026). The specification, the schema, the
> core computations, two reference analyses on public data and a first experiment
> ([`experiments/angle-between-errors/`](experiments/angle-between-errors/)) are here, and so is the tool for one person calibrating a model on their own cases, `turing-meter pair` ([protocol](docs/PAIR-PROTOCOL.md), [one-page card](docs/PAIR-CARD.md)); the machine runner is not written yet, and the protocol has not yet been tried with real people. See
> [`docs/PLAN.md`](docs/PLAN.md).

## The numbers

The same decisions are made three times: by the person alone (`dH`), by the
machine alone (`dM`), by the person with the machine (`dS`). Each is a sensitivity
`d'` of signal detection theory: zero at chance, no ceiling.

- **The apparent sensitivity** `|S| = sqrt(dH^2 + dM^2)` is the `d'` of the ideal
  observer of Sorkin and Dai (1994): the best two minds can do together when they
  err on different cases and pool all their cues (Green and Swets 1966). Two people
  in discussion reach it only if they are about equally good (Bahrami et al. 2010).
  Orthogonal, in this reading, means independent errors. It is the measurable part
  of the apparent intelligence of the title, its projection on the craft axis (the
  name changed on 15 November 2026; before, `|S|` itself was called apparent
  intelligence).
- **The intelligence factor** (*efficiency* in the article) `eta = dS / |S|`: how much of the apparent
  sensitivity shows in the work, as the power factor is real over apparent power.
  0 at chance, 1 at the ideal.
- **The gain share** `g = (dS - dH) / (|S| - dH)`: how much of the possible gain
  the person realizes, 0 if they ignore the machine. `eta` falls by arithmetic as
  the machine gets stronger; `g` is the number that describes behaviour.
- **The turing** (Tu) `= dS / dH`: the same people on the same cases, alone and
  with the machine. A mind working alone is worth 1 Tu: every mind is its own
  standard, as in the per-unit values of electrical engineering. No machine
  appears in the definition; the values belong to people and cases. `Tu` is a ratio,
  so it depends on its yardstick, the person: two turings are comparable only if the
  people they start from are the same, and for someone who alone answers almost at
  random the denominator tends to zero (there, look at `dS - dH`). `Tu^2` is the
  weight of evidence an ideal observer would need to do as well as the pair, over
  what it would need to do as well as the person.
- **The apparent turings** `Tu_app = |S| / dH`: the ideal in the person's unit,
  what the two separate readings promise, as volt-amperes beside the watts of the
  turing (volt-amperes are read without the angle; `|S|` already contains it). By
  construction `Tu = eta * Tu_app`. They depend only on `k = dM / dH` and the angle
  `phi` below: `sqrt(1 + k^2 - 2k cos(phi)) / sin(phi)`, `sqrt(1 + k^2)` at 90
  degrees.
- **The direction** `theta`, with `cos(theta) = dH / |S| = 1 / Tu_app`, the
  person's share of the apparent sensitivity; `theta = atan(dM / dH)` at 90
  degrees.
- **The angle between the minds** `phi = arccos(rho)`, where `rho` is the
  correlation of their errors on the same items. 90 degrees is the law of `|S|`; it
  is measured whenever item-level data exist, and it changes the ideal.
- **The useful angle** `arccos(weaker / stronger)`: next to a machine `k` times
  stronger, a person adds to the ideal only if the errors meet at more than
  `arccos(1/k)`, 78.5 degrees for `k = 5`. Below it the ideal is the stronger
  partner alone, because the instrument allows no negative weights: a choice, not
  a law. Whether working with a machine closes a person's angle is the question of
  the next study ([`docs/STUDY.md`](docs/STUDY.md)); that people take on the errors
  of the AI they worked with is already known (Vicente and Matute 2023; Glickman
  and Sharot 2024).

## What the public data say

On 87 binary decision comparisons from 27 papers of a meta-analysis of experiments
that measured humans alone, AI alone and humans with AI (Vaccaro, Almaatouq and
Malone, 2024), with the angle assumed at 90 degrees because those data have no
item-level answers:

- people realize a median **0.30** of the gain the ideal would give them (95%
  interval over papers 0.19-0.43; 0.38 on the comparisons that measure the same
  people), and the pairs show 0.69 of their apparent sensitivity;
- the share of the gain does **not** depend on who is stronger (0.24 when the
  person is, 0.32 when the machine is). The share of the ideal falls (0.81 against
  0.59), but by arithmetic: a stronger machine raises the ceiling faster than
  people follow;
- 0.46 when the AI showed its confidence, 0.20 when it showed only its verdict:
  a difference between studies, and fragile (0.33 against 0.26 counting each
  paper once);
- 31 pairs beat the better of the two alone; between 0 and 13 of them beyond
  noise, depending on how many items stand behind each AI accuracy, which the
  studies do not report. Only 3 beat the rule that pairs of people follow.

Details, limits and the second analysis (why different agents are never put on
one latent scale) in [`analysis/`](analysis/).

## The first experiment

[`experiments/angle-between-errors/`](experiments/angle-between-errors/) is the experiment of the article: the code, the
protocol written before any result (with its deviations, dated) and the derived
results. It measures the angle between the errors of people and machines on the
same items, on four public sources with item-level answers (ChatBench, ChaosNLI on
αNLI, HAIID, model-vs-human), against 28 machines: those the sources do not
provide, twelve open-weight models and ten vision networks, were run on a 24 GB
GPU, and four Claude models of November 2026 answered alone (no person ever worked
with them, so for those there is a margin, not a turing). What holds best:

- **The instrument is centred within its model.** In simulation it recovers the
  true turing within 1.2% in the seven situations subject to the criterion (of
  eight), with cautious intervals: the 95% interval contains the true value 98-100%
  of the time.
- **With GPT-4o people go up, and the pair holds the machine back.** A group of
  people goes from 58 to 77 percent correct, 1.65 turings, and the pair stays
  below the machine alone (88%; yield on the machine 0.76). With Llama 3.1 8B,
  weaker, 1.40 turings, and the pair passes the machine queried alone (yield 1.19),
  not the one of the conversation (0.92). Both are analyses added after the
  protocol.
- **Where the pair loses points.** With GPT-4o the pair is 11 points below the
  machine alone: 7.6 are lost in chats without a verdict (the machine writes but
  never names one of the four options) and 4.2 in the person's final choice. Which
  loss weighs more the data do not say: the interval of the difference includes
  zero.
- **The ceiling of a pair is a range.** With GPT-4o the ideal pair, under the model,
  does not beat the machine if it receives only the verdict, and beats it by 2.2%
  with pooled cues; without a model, someone who always knew whom to believe would
  reach 93.4% correct (17% above the machine in sensitivity). The pair fell short
  even of the lower end.
- **The angle is not right everywhere.** On αNLI and on HAIID's text it is below
  90 degrees; on ChatBench the interval is entirely below 90 in only 8 comparisons
  out of 20 (medians between 59 and 81 degrees by data family). On αNLI, with the
  answer key, it closes as the open-weight models grow, from 76 to 56 degrees; the
  157 stories where most readers contradict the key explain 30-43% of the descent.
- **The formula predicts the ceiling of a pair, not which pair gains most**, in a
  weak test: the useful angle separates two tendencies, and neither is clear-cut.

Every result comes with how solid it is, pre-specified or added after the numbers,
in the article (Appendix E) and in the protocol. About twenty later analyses were
written down before they ran, but they are not pre-registrations in the strict
sense, and no interval is corrected for multiple comparisons: an isolated result at
the margin should be taken with caution.

## Principles

- **A mind against itself.** No ranking of models, no single number per model.
- **The model can fail.** A pair above the ideal, beyond noise and with the angle
  measured, refutes the composition for that kind of work.
- **The circuit is part of the result.** Who takes the final decision, what the
  machine shows, in which order: the protocol is written into every report.
- **A turing comes with its context.** The task, the people, the machine, the
  circuit, `dH`, `k`, the angle and an interval. It is read beside `k`: a person
  who copies a machine `k` times more sensitive is worth `k` turings, and neither
  `eta` nor `g` tells copying apart; what the person adds shows only where `Tu`
  exceeds `k`. Details in [`docs/SPEC.md`](docs/SPEC.md), section 5.1.
- **No price.** The cost of work is a different question and stays out of the
  unit.
- **People are pseudonymous.** Timed attempts carry an id, a role, an experience
  band and a year; never a name.

## Built on

- [Inspect](https://inspect.aisi.org.uk/) (UK AI Security Institute, MIT) for
  running the machine alone in sandboxes.
- Signal detection theory for the scale and the ideal; Geirhos, Meding and
  Wichmann (2020) for error consistency between people and machines.
- Luce and Tukey's conjoint measurement, and Michell's critique of psychometrics,
  for the diagnostics.

## How to cite

endjneer (2026). *Intelligenza apparente* (*Apparent Intelligence*). Version 5. Zenodo. https://doi.org/10.5281/zenodo.24302609
(the record holds the Italian original and an English translation; all versions: https://doi.org/10.5281/zenodo.24107648)

Code and data: https://github.com/paithon-it/turing-meter

## License

- Code under the MIT License, © 2026 paithon.it.
- The protocol and the derived results of the experiment (`experiments/angle-between-errors/PROTOCOL*.md`,
  `experiments/angle-between-errors/results/`) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  what derives from ChaosNLI keeps its license,
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- The data of the sources are not redistributed: the scripts read them from
  their sources, under their authors' licenses.
- The text of the article: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
