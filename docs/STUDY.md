# The assimilation study: protocol

Version 0.2, November 2026. endjneer, paithon.it. Not run, not pre-registered. It is
a proposal for a research group to run with us (section 12): we bring the
question, the protocol, the code and the simulations; a partner brings what an
independent author cannot, which is an ethics committee, participants and
funding.

**Changes from 0.1**, after a methodological review on 2 November 2026:
- the alignment is measured within each class of the right answer, because a
  lean toward the machine's answer inflates the pooled one (section 4);
- the primary statistic is corrected for what practice alone predicts, because
  the two arms may learn by different amounts (section 4);
- the bootstrap resamples people only, with the forms fixed (section 7);
- the estimand is stated, a third arm is offered as an extension, and practice
  is counted in items, not minutes (section 5);
- the pilot gate is narrower and tested at its corners (sections 5 and 6);
- the design allows for dropouts and adds a three-outcome decision rule, ethics,
  cost and a conflict of interest.

## 1. Question

Does working with a machine move a person's **unassisted** errors toward the
machine's? The measure is `rho`, the correlation between the evidence of the person
alone and that of a frozen machine on the same items, and its angle
`phi = arccos(rho)`. The claim under test: with use, `rho` rises more than practice
alone explains, whatever happens to the person's accuracy.

Two things can look alike and are kept apart. **Erring on the same cases** is a
change in the evidence, and it is what the useful angle of section 2 is about.
**Leaning toward the same answer** is a change in the criterion: a person who
learns to say "no" as often as the machine does will err with it on one class
of cases without seeing anything the way it does. Both are reported; only the
first is the primary outcome.

In the words of the article: the machine does not make us stupid, it makes us
similar to it.

## 2. Why the instrument needs it

The weaker mind of a pair adds to the ideal only while `rho < weaker / stronger`
(`apparent.useful_angle`, SPEC section 2.1). Next to a machine `k` times stronger
on the `d'` scale, the errors must meet at more than `arccos(1/k)`:

| k = dM / dH | 2 | 3 | 5 | 10 |
|---|---|---|---|---|
| useful angle | 60 | 70.5 | 78.5 | 84.3 |

Two things move. As machines improve, `k` grows and the useful angle climbs
toward 90. If use closes the person's angle, the two meet, and from then on the
person adds nothing to the pair, not even in principle. Accuracy alone cannot see
the second movement: a person can grow more accurate while their errors move onto
the machine's.

## 3. What is already known

The phenomenon is not new, and the study must say so.

- People who classified with a biased AI repeated its errors once alone:
  Vicente and Matute 2023 (Sci Rep, doi:10.1038/s41598-023-42384-8).
- With 86 medical students and a control group, the error on the cases where
  the AI had advised wrongly was .26 against .01, and slightly higher elsewhere:
  Vicente, Matute, Fregosi and Cabitza 2025 (IJHCS,
  doi:10.1016/j.ijhcs.2025.103474). The figures come from the reanalysis in
  Herrera-Poyatos et al. 2026 (arXiv 2610.18765), which uses them descriptively
  because the original tests treat answers as independent, and which reads the
  effect as a criterion shifted on one class.
- Feedback on who was right changed the unassisted behaviour where advice alone
  did not: Cabitza and Vicente 2026 (IJHCS, doi:10.1016/j.ijhcs.2026.103855).
  The reanalysis warns that the probe had only positive answers (a more liberal
  criterion would look the same), that groups were assigned by month of birth,
  and that the feedback group did worse on cases with a known criterion.
- Interacting with a biased algorithm biased people's independent judgements,
  more than interacting with people did, and the effect grew over the
  interaction: Glickman and Sharot 2024 (Nat Hum Behav,
  doi:10.1038/s41562-024-02077-2).
- In Go, after an AI program was released, professionals' unassisted moves
  showed "higher alignment with AI's suggestions": Choi et al. 2025 (arXiv
  2310.08704). The authors have no clean control group; they compare younger and
  older players and read their estimate as a lower bound.
- Accuracy cannot tell better discrimination from a shifted criterion:
  Herrera-Poyatos et al. 2026. Alignment at equal accuracy, measured during
  assistance: arXiv 2610.24969, appendix D.
- Delegation and skill as coupled states, with a low-skill trap that grows with
  the machine's capability: Huang and Vishnoi 2026 (arXiv 2603.02950).
- Designs for repeated exposure: Wang and Baiocchi 2026 (arXiv 2607.27701). A
  parallel-group design like this one estimates the global effect efficiently,
  but cannot separate the immediate effect of the advice from its carry-over.
  This study wants the global effect three months on.

What this study adds is a synthesis: an item-level `rho` within class, against a
real machine (not one built to err on purpose), over months, against a control
arm, corrected for practice, tied to the threshold of section 2. The phenomenon,
the metric and the dynamics each exist elsewhere.

## 4. Hypotheses and decision

**Estimand.** The effect, three months on, of seeing the machine's verdict and
confidence during practice, on top of seeing the right answer, on the alignment
between the person's unassisted evidence and the frozen machine's, beyond what
practice alone explains. With the third arm of section 5, a second estimand: the
same without the right answer.

**Primary statistic.** For each arm and wave:
- `rho_HM`, person-machine alignment within each class of the right answer,
  people pooled with each one's own criterion (`apparent.pooled_alignment`);
- `rho_HH`, the same between people of the arm, over all pairs
  (`apparent.people_alignment`);
- the adjusted alignment `A = rho_HM - sqrt(q * rho_HH)`
  (`apparent.practice_adjusted`), with `q` estimated at T0 on both arms pooled.

Why the adjustment: when person and machine share only the difficulty of the
items, `rho_HM^2 = q * rho_HH` at any level of practice, where `q` is the share of
the frozen machine's evidence that is the shared difficulty. Less noise, or more
weight on what is hard for everyone (learning the truth, which near-optimal
machines also know), moves both sides together. A quirk of the machine taken up
by the person breaks the identity. The statistic is the difference in
differences of `A`: T2 minus T0, machine arm minus control.

**H1.** That difference is positive.

**Decision**, on the 90% two-sided percentile interval:

| Interval | Reading |
|---|---|
| above 0, reaching +0.10 | assimilation |
| above 0, below +0.10 | assimilation, smaller than the size of interest |
| below +0.10, containing 0 | no assimilation of the size of interest |
| containing 0 and +0.10 | inconclusive |

**Why +0.10.** It is the size the study can see with 80% power at a feasible
number of people (section 6). In the reference world (`k` = 2, `rho` 0.20 before)
the margin to the useful angle is 0.30, and +0.10 in three months would take a
third of it. Smaller effects could matter over years; this study cannot see them,
and says so. T1 is descriptive: whether the change is already there after one
month.

**Assumption check.** The adjustment assumes that what people share with each
other on an item, they share with the machine. Shared human heuristics that the
machine lacks would break it. The control arm tests this: its own change in `A`
should be near zero. If its 90% interval leaves ±0.05, the primary is reported
with that warning, and the raw difference of `rho_HM` beside it.

**Secondary outcomes**, in this order, without correction for multiplicity
(they describe; only the primary decides):
1. the difference in differences of `rho_HM` without the adjustment, with the
   validity condition that `d'H` at T2 differs between arms by less than 0.10;
2. the criterion: each arm's `c` and its distance from the machine's, per wave
   (assimilation of the lean, reported apart);
3. `rho` on right/wrong pooled over classes, as in version 0.1 and in prior work;
4. `d'H` by arm and wave (deskilling);
5. the margin `phi - useful_angle`;
6. strata: items the machine gets right and wrong (the q0 and q1 of arXiv
   2610.24969);
7. a probit model with crossed random effects, per class:
   `correct ~ arm * wave * machine_right + (1 + wave | person) + (1 | item)`.
   The term `arm:wave:machine_right` asks whether the gap between the items the
   machine gets right and those it gets wrong grows more in the machine arm. Its
   scale moves with the person's noise, so it is read with the validity
   condition of outcome 1.

## 5. Design

**Arms.** Core: two arms. The machine arm sees, on each practice item, the
machine's verdict and confidence, then the right answer; the control arm sees the
same items and the same right answer, no machine. Extension, for a partner with
the means: a third arm that sees the verdict and confidence and **not** the right
answer, the condition of real use, where a wrong machine is never contradicted.
Seeing the right answer lets people learn where the machine fails, so the core
design works against the hypothesis; the third arm does not.

**People.** Adults, pseudonymous ids only, randomized within strata of baseline
`d'`. 70 per arm are recruited so that about 50 remain at both tests with 30%
dropout. The analysis set is the people present at T0 and T2. Inclusion: fluent in
the language of the task, no training in the task's domain. Paid per session with
a completion bonus, or, cheaper, practice as coursework in a university course,
with the students as participants and the same ethics.

**Task family.** Binary decisions with a known answer, 10 to 20 seconds each, a
pool of at least 5,000 items. The machine's errors must be its own, never
planted, and learnable: predictable from what the person sees. Default: income
or credit decisions from a card of tabular features (a public census or lending
dataset, without sex or ethnicity), with a model trained once. A partner's own
domain with a real classifier (diagnosis from images or records, as in the
studies of section 3) is as good if it passes the gate. Second family, if
resources allow: real against generated images, where the detector's cues are
mostly invisible to people; there the hypothesis predicts no assimilation, which
tests the mechanism only if the two families have the same power.

**Pilot gate**, pre-registered on its own, twenty people:
- `dH` from 0.8 to 1.5 (66 to 77% for an unbiased person);
- `k = dM / dH` from 1.5 to 3, with `dM` at most 2.6 (90%);
- `rho_HM` within class at most 0.30, and `1/k - rho_HM` at least 0.20, so the
  person starts well inside the useful angle;
- the machine's errors predictable from the features on new items: a second
  model that predicts "the machine is wrong" reaches an AUC of 0.65. A machine
  near the best possible errs mostly where everyone errs, and leaves little to
  assimilate; if so, a weaker machine with errors of its own (fewer features,
  trained on a slice of the data);
- `rho_HH` reported, to know how much the shared difficulty weighs;
- the machine's errors do not track proxies of protected traits.
If a condition fails, the family or the machine changes before anyone is
randomized.

**Schedule.**
- T0: 200 items alone, no feedback.
- Twelve weekly sessions of 60 items each, time logged: the same items and the
  same number in every arm.
- T1 after one month, T2 after three: 200 items alone, no feedback.
- After T2: a questionnaire (strategy used, trust in the machine, attention to its
  verdict), then a written debriefing that explains the effect, and an optional
  session against it.

**Items.** Test forms disjoint from the practice items and from each other, the
same forms in both arms, balanced by class, with the machine's errors at their
natural rate, so that `rho` refers to the items the machine meets, and matched in
difficulty on a norming sample. Forms rotate across waves in a Latin square,
stratified by arm. Forty items return between T1 and T2, for each person's
consistency with themselves (a disattenuated `rho` is secondary; the primary
stays raw).

**Machine.** One model version, its answers and confidence on every item computed
once and stored with a hash. Nothing is called live.

## 6. Sample size

`analysis/assimilation_power.py` simulates the whole analysis: the same people at
both waves, differing in sensitivity, noise and criterion; forms balanced by class
and shared by the arms; the bootstrap of section 7 inside every simulated study.
Results in `analysis/RESULTS.md`, section 4.

In the reference world (`dM` 2.0, `dH` from 1.0 to 1.22 with practice, `rho` 0.20
before), with 50 people per arm present at both tests and 200 items per test (400
simulated studies, 200 bootstrap draws in each):

| Scenario | pooled `rho` | class `rho` | adjusted `A` | assimilation declared |
|---|---|---|---|---|
| no effect, equal learning | +0.001 | +0.001 | +0.000 | 4% |
| no effect, the machine arm learns more (`dH` 1.43) | +0.037 | +0.040 | -0.002 | 6% |
| no effect, the machine arm takes the machine's lean | +0.147 | -0.003 | -0.002 | 5% |
| no effect, the machine arm weighs the shared difficulty 1.3 times | +0.069 | +0.070 | -0.002 | 7% |
| +0.10, equal learning | +0.099 | +0.102 | +0.093 | 85% |
| +0.10, the machine arm learns more | +0.135 | +0.141 | +0.092 | 83% |

- **Only the adjusted statistic stays at zero in every null.** Pooled `rho` reads
  the lean as +0.15. Class `rho` reads unequal learning as +0.04 and the extra
  weight on the shared difficulty as +0.07, and would declare assimilation in 31%
  and 62% of those studies. The adjusted one declares it in 4 to 7%, against 5%
  nominal.
- **Power against +0.10:** 0.85 with 50 people per arm, 0.72 with 35, 0.94 with 70.
  At the corners of the gate it is 0.75 (`dH` 0.8, `k` = 3, `rho` 0.13) and 0.83
  (`dH` 1.5, `k` = 1.5, `rho` 0.30).
- **Ruling out +0.10 when there is no effect:** 0.89 with 50 people, 0.76 with 35,
  0.97 with 70. When the effect is exactly +0.10, the study wrongly rules it out
  about once in ten. The adjustment absorbs a little of the effect (+0.093 for
  +0.10), because people who take the machine's quirk also grow more alike.
- **The bootstrap matches the statistic:** its mean standard error is within 15%
  of the spread over studies in every scenario.

Hence 70 people per arm recruited, to keep 50 at both tests with 30% dropout. If
the pilot lands near the weak corner, the partner recruits more or narrows the
gate.

The simulation assumes Gaussian evidence and the same assimilation for everyone,
and it does not include a component that people share with each other and not
with the machine; the control arm checks the last one (section 4).

## 7. Analysis

- Statistics of section 4, with `apparent.pooled_alignment`,
  `apparent.people_alignment` and `apparent.practice_adjusted`; the code is
  frozen with a hash before data arrive.
- Intervals: percentile bootstrap, 2,000 draws, over people within arm, the same
  draw at every wave, forms fixed. The inference is about people, given the forms;
  generalisation to items rests on the forms being drawn at random from the pool
  and on the model of outcome 7. Coverage of the bootstrap was checked on the
  simulation (section 6); the seed is registered.
- Exclusions, fixed now: fewer than 8 of the 12 sessions; a median response time
  under 2 seconds on a test; on a test, `d'` below 0.2. Excluded people are
  reported by arm.
- Dropouts: the primary uses the people present at T0 and T2; a sensitivity
  analysis weights them by the inverse probability of staying, from baseline
  `d'`, arm and the week of leaving.
- The analyst sees the arms coded until the primary is computed.
- Checks that do not depend on the outcome: agreement with the machine during
  practice, from the logs (manipulation); alignment during assistance (positive
  control); the control arm's change in `A` (section 4).
- Deviations from this plan are logged with their date and reason; analyses
  added later are labelled as such.

## 8. What could fake the result, and the answer of the design

| Threat | Answer |
|---|---|
| Shared difficulty: less noise, more `rho` | control arm with the same practice; difference in differences |
| The machine arm learns more, or less | the practice adjustment `A`; validity condition on `d'H` |
| Learning the truth is learning a near-optimal machine | the adjustment; gate on the predictability of the machine's errors |
| A lean toward the machine's answer inflates pooled `rho` | `rho` within class; the criterion as a separate outcome |
| A person cannot align beyond their own consistency | 40 repeated items; disattenuated `rho` as secondary |
| Memory of the machine's answers | test items never seen in practice |
| The machine changes | answers stored once, hash and version logged |
| More reading in one arm | sessions counted in items, time logged |
| People leave unevenly | the same people at both waves; inverse probability weights |
| Regression to the mean | no selection on `rho` |
| A deliberate strategy ("what would the AI say?") | the end questionnaire; still assimilation, reported as such |

## 9. What each outcome would mean

- `A` up more than control: assimilation. The angle the law of `|S|` assumes is a
  resource that use consumes.
- `A` flat, the criterion moved toward the machine's: people take its lean, not
  its way of seeing. Easier to undo, since a criterion is set by a choice.
- `A` flat, `dH` up: people learn from the machine without taking its errors, the
  best case for a pair.
- `A` flat, `dH` down: deskilling without assimilation.
- `A` up in the tabular family and flat in the image family: the mechanism is
  learning the machine's visible rule.

## 10. A free complement

Chess has public games of the same players before and after the neural engines
of 2017 and 2020. With one frozen engine as the machine and a stronger one as the
truth, the overlap between a player's errors and the frozen engine's can be
followed over years. There is no control group and no randomization, and the two
engines belong to the same family, so alignment with one is nearly alignment with
the other: it can only suggest, but it costs nothing.

## 11. Ethics, data and cost

- **The study induces a bias on purpose**, since that is the outcome, on decisions
  about real people's income or credit. Removing sex and ethnicity does not
  remove proxies (marital status, relationship, country of origin, occupation):
  the gate checks that the machine's errors do not track them. Debriefing after T2
  is part of the protocol.
- **Ethics committee** before the pilot, at the partner's institution.
- **Data protection** (GDPR): consent as the legal basis; the partner as
  controller; pseudonymous ids with the key kept apart; answers, response times
  and the questionnaire, nothing else; retention stated; withdrawal with
  deletion; anonymised data shared at the end.
- **Load and cost.** Twelve sessions of about 20 minutes and three tests of about
  50 minutes: about 6.5 hours per person. With 140 people (two arms) at 12 euros an
  hour and a completion bonus, 10,000 to 15,000 euros plus the platform; half as
  much again with the third arm; much less as coursework.

## 12. Who runs it

The author of this protocol is independent, not an institution, and cannot
fund the study. It is offered to a research group.

**What we bring:** the question and its link to the threshold of section 2; this
protocol and its pre-registration draft; the code of `turing-meter`
(measurement, simulation, analysis), open to the partner; the analysis plan and
help to run it; the write-up.

**What a partner brings:** the ethics committee and the role of data controller;
participants (a panel, or a course); the session platform; funding; the blind
analyst of section 7; and, if they wish, their own task family.

**Authorship** by contribution (CRediT), agreed before the pilot.

**Conflict of interest.** The hypothesis supports the thesis of the author's
article *Intelligenza apparente* (paithon.it, 2026). This is declared in the pre-registration,
and it is one more reason for a blind analyst and a pre-registered report.

## 13. Before any data

1. A partner, and the committee's approval.
2. The pilot, pre-registered on its own with the gate as its decision rule.
3. Materials frozen with hashes: the item pool, the model and its stored answers,
   the forms, the Latin square, the session software, the analysis code, the seed.
4. Pre-registration of this protocol (OSF), with the deviations policy above.
5. The study.
