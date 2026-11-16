# Protocol: the angle between the errors, measured

Written on 9 November 2026 **before** computing any result. Its purpose is to
avoid choosing the analyses after seeing the numbers. Whatever changes later is
recorded at the end, in the "Deviations" section, with the reason.

## The problem

Article 534 defines the angle φ between the errors of a person and those of a
machine, then declares that it cannot measure it and fixes it at 90 degrees.
All its figures depend on that assumption, including "people realize one third
of the possible gain". Here the angle is measured, on public data with
item-by-item responses, and we check whether the law really predicts what a
pair can do.

No new people are measured: the human responses are public. The machines that
the sources do not provide run on an NVIDIA L4 GPU with 24 GB.

## Model and definitions

Signal detection theory, Gaussian noise with equal variance.

- **d′**. Yes/no: `2 z(a)` when the classes are balanced and the criterion is
  neutral, otherwise `z(hit) − z(fa)`. Forced choice among `m` options: invert
  `PC = ∫ φ(x − d) Φ(x)^(m−1) dx`. The ratios (turing, η, g) do not depend on
  the scale factor.
- **ρ, φ**. ρ is the correlation between the person's evidence and the
  machine's evidence on the same items, conditional on the correct answer;
  φ = arccos ρ. The shared difficulty of an item **counts** in ρ, because it is
  exactly what makes the second opinion useless. With many people and few
  items each, it is estimated with each person's personal threshold
  (`rho_personal_thresholds`), obtained from their accuracy with an empirical
  Bayes shrinkage.
- **Ideal** `|S|(φ) = sqrt((dH² + dM² − 2ρ dH dM) / (1 − ρ²))`, clipped at the
  stronger when `ρ` exceeds `min/max` (no negative weights).
- **Useful angle** `arccos(min(dH, dM) / max(dH, dM))`.
- **g(φ)** `= (dS − dH) / (|S|(φ) − dH)`, the share of the possible gain.

With more than two options the correlation of the correct answers is not that
of the evidence: it is converted with a table computed by simulation, for the
sensitivities of that comparison. The conversion is checked on simulated data.

## Data

| Code | Source | Task | People | Items | dS measured |
|---|---|---|---|---|---|
| CB | ChatBench (Chang, Anderson, Hofman 2025), CC BY 4.0 | MMLU, 4 options, mathematics/physics/ethics | yes, as person-question pairs | 396 | yes, *answer-first* condition |
| CN | ChaosNLI, αNLI (Nie, Zhou, Bansal 2020) | 2 options, abduction, truth from the original story | 100 votes per item | 1,532 | no |
| HA | HAIID (Vodrahalli et al. 2022) | 5 binary tasks, graded response | same person before and after | 152 | yes |
| GE | model-vs-human (Geirhos et al.) | 16 categories, 17 conditions | trial by trial | ~21,000 trials | no |

Machines: ChatBench provides GPT-4o and Llama 3.1 8B alone. On the L4, open
models of different generations and families run on the same questions
(probability of each option, without reasoning) and, for vision, a scale of
networks from 2012 onward. For HAIID, vision and language models on the
original items.

## Hypotheses and criteria

Fixed beforehand. "Passes" means that the 95 percent interval meets the
criterion.

- **H1, the angle is not a right angle.** Person-machine φ below 90 degrees on
  every data family. φ and its interval are reported for each comparison.
- **H2, machines resemble each other.** Machine-machine ρ greater than
  person-machine ρ. Replication of known results.
- **H3, the law predicts.** *The main test.* On each comparison a combiner is
  trained (logistic regression, five-fold cross-validation by item) that uses
  only what the pair really has: the person's answer and possible confidence,
  the machine's probability. The mean absolute error between the combiner's d′
  and the prediction with the measured φ must be smaller than that of the
  prediction at 90 degrees. With a person who gives only the verdict, the
  prediction is the ideal *at verdict level* with φ, computed by simulation:
  the law at full evidence would be an unreachable target.
- **H4, the third moves.** Where dS is measured (CB, HA), g(measured φ) is
  greater than g(90°). The prediction is an increase, the size is open. On the
  Vaccaro meta-analysis, g is reported for ρ from 0 to 0.6 as a sensitivity
  analysis, because there ρ cannot be measured.
- **H5, the useful angle.** For each pair we state whether φ exceeds the useful
  angle, and whether the combiner does better than the machine alone beyond the
  error. The two answers must agree; disagreements are reported.

## Estimation and uncertainty

Two-level bootstrap (items and people) with 2,000 repetitions, percentile
intervals. Medians and intervals are reported, not significance. Where the
machine gives 50 answers per item (GPT-4o and Llama from ChatBench), its
"correct answer" on an item is the fraction of correct samples.

Before touching the real data, the estimators pass a test on simulated data
with the design of each source: bias of ρ below 0.03 in absolute value;
otherwise the estimator is corrected. That check is in `tests/` and runs at
every change.

## Silicon people: testing the assimilation protocol

An earlier, unpublished protocol of ours proposes the measure
`A = ρHM − sqrt(q ρHH)` to separate the errors taken from the machine from the
mere effect of practice, but it has been tested only on a purpose-built
Gaussian generator. Here it is tested with real networks on real images, where
the shared difficulty is not a single factor.

Sixty small networks ("people") learn CIFAR-10 in stages, in three arms:
**control** (the correct answer after each trial), **neutral assisted** (plus
the opinion of a frozen machine, with no injected defects) and **assisted with
defect** (the same machine with a systematic error injected on two classes).
On the same never-seen images, at stages T0, T1, T3, each network is measured
alone.

We state in advance what it would mean for the measure to work:
- **False alarm**: in the comparison of control against neutral assisted, with
  arms of equal skill, the difference in the differences of A must have an
  interval that includes zero.
- **Sensitivity**: in the arm with the defect, A must grow more than in the
  control.
- **The counter-test**: naive ρHM must grow with practice in the control too,
  that is, the trap that A exists to avoid must really occur.

If the measure fails one of these tests, it must be written that way: a
network is not a person, but a measure that fails on a known case is not ready
for people.

## What cannot be concluded

None of the sources follows people over time: the question of assimilation
remains open. The human data are crowdworkers on short tasks, not specialists.
The machines that the sources provide are those of 2024-2025. A measured angle
holds for that task, those people and that machine.

## Deviations

Recorded as they come, with the date and the reason. Everything below was
decided **after** the first look at the data, and must be read as exploration.

- **9 November 2026, after the first ChatBench numbers (φ of GPT-4o and Llama
  with the machine alone).** With Llama the pair exceeds the ideal (η > 1):
  ChatBench's machine "alone" is a model queried without the person, whereas in
  the chat the person asks, corrects, insists. The fixed machine is the wrong
  model: they are measured at the wrong terminals. We add:
  - **M-in-chat**: for ChatBench also the answer that the machine gave *in that
    conversation* (`answer_letter` of `conversations.json`, where present),
    machine per trial, with the same formulas. Both are reported.
  - **H6, who decides first.** ChatBench randomly assigns whether the person
    answers alone before querying the machine (*answer-first*) or goes straight
    to it (*direct-to-AI*). Final accuracy and d′ are compared in the two
    conditions, per machine: it is the only randomized comparison in here on
    the order of who takes the initiative.
  - **H7, the scale.** On the same people and the same questions of ChatBench,
    the machines from the small model to the large: how φ, the useful angle,
    the possible turing and cos θ (the person's share of the ideal) change as
    the machine grows. It is the test of the hypothesis that the growth of the
    technology does not bring the machine closer to the human, and it says by
    itself what would falsify it.
- **9 November 2026, reading the HAIID method.** The "advice" shown to the
  participants, presented now as AI now as a person, was the **mean answer of
  about fifty people** (Vodrahalli et al., sec. 3.1), with a small random
  perturbation: it is not a machine. It is not used as a machine: it stays,
  with this label, as a control ("the crowd"), to know what angle and what
  share a human opinion gives. The HAIID machines are only those run on the L4
  on the original items.
- **9 November 2026, after the experiments were done, on the silicon people.**
  The first computation of the sensitivity criterion compared two different
  things: the arm with the defect was measured against the defective machine,
  the control against the healthy one. The correct comparison uses the same
  comparator for both: the control is also measured against the defective
  machine. Both numbers are reported, plus a direct measure that does not
  depend on A: on just the items where the defect changes the answer, how often
  the network gives exactly the injected wrong answer.
- **9 November 2026, after an independent review of the code**, all corrections
  of our own errors found after seeing the numbers:
  - *The silicon defect.* It was drawn at random at every pass, so it did not
    depend on the image: a network could inherit only a class tendency, which
    the measure A removes by construction. The first run could say nothing about
    the sensitivity of A. It was redone with a defect that depends only on the
    image (a fixed direction of the machine's features), and A is also reported
    without conditioning on the class.
  - *The person-person bootstrap.* Duplicating rows paired a trial with its
    copy and inflated the interval: now the weight sits on the item. The point
    estimate does not change.
  - *The d′ with a shifted criterion.* The protocol asked for z(hit) − z(fa) when
    the criterion is not neutral; the code used 2 z(a) for all, and some models
    on HAIID and αNLI lean strongly to one side. Now person, machine and
    combiner use the free criterion, with the ½ correction. The combiner also
    sees which option the machine chose.
  - *The conversion with m > 2.* It was computed on identical people and
    exceeded the protocol's error criterion on ChatBench. Now it simulates the
    people with their dispersion of sensitivity and the threshold estimated from
    few answers.
  - *The ρ grid.* It acted as a ceiling (0.97) and a floor (−0.6). Now it runs
    from −0.9 to 0.995; values at the limit are counted.
  - *H3.* With ρ > 0 the prediction at 90 degrees sits above the one with the
    angle, so when the combiner is below both the angle wins without merit. The
    informative rows are also reported, those in which the combiner falls
    between the two predictions, and the gain over the better of the two.
  - *The H5 interval* now also resamples the people.
- **9 November 2026, the combiner's d′.** For consistency with person and
  machine, the pair's d′ is computed class by class, with a free criterion. The
  ½ correction must be applied on the distinct items, as for the machine: on
  person × item rows it blew up when the combiner learned to always say the
  same class on about twenty items. On HAIID the items per task are 24-32, and
  the machine's d′ often touch the maximum: the test of the law is weak there,
  and is reported per family.
- **9 November 2026, after the second silicon run.** With the defect tied to
  the image, A within the class, as prescribed, still fails the sensitivity
  test, while the direct copy of the defect is visible. To understand why, two
  analyses were added, both exploratory: A without conditioning on the class and
  A on only the two classes on which the defect acts. The criterion declared in
  advance remains failed.
- **9 November 2026, after a second independent review:**
  - *The comparison at the two ChatBench terminals.* The "alone" row and the
    "in chat" row used different trials: the second keeps only the conversations
    with a recorded answer, and the others had a much lower accuracy. The
    machine alone computed on the same trials as the chat is added, and that is
    the one compared.
  - *The silicon bootstrap* resampled the networks with replacement and paired a
    network with its copy, inflating ρHH: now the weight sits on the network.
  - *The test of the law.* Rows with identical combiner and prediction (at the
    accuracy ceiling) were counted twice between "below" and "above": now they
    are set apart. Totals are also reported without HAIID, because there the
    combiner's d′ is at free criterion on 16 items per class and is not
    comparable with the simulated prediction. The choice was made after seeing
    the numbers.
- **9 November 2026, tuning the machines.** Qwen3.8 27B fits in the 24 GB only
  with more memory assigned and a shorter context. On the HAIID dermatology
  photos the small vision models (and Qwen3.8) work on the image reduced to 1024
  pixels per side: the original resolution does not fit in their context.
- **9 November 2026, bootstrap.** The protocol said 2,000 repetitions. These are
  used for all text and HAIID comparisons; for vision 600 (689 comparisons with
  m = 16) and for the silicon 300, because of the computing cost. The intervals
  are less stable, and the point estimates do not change.
- **9 November 2026, vision in the test of the law.** With the Geirhos data the
  person gives only a verdict: the combiner uses only the machine's probability,
  and the prediction is the ideal at verdict level.
- **9 November 2026, third review: the turing as a unit.** Descriptive
  additions, decided after seeing the numbers; no new hypothesis and no
  criterion.
  - *Apparent turings.* `|S| / dH` (`sdt.apparent_turings`): the ideal of the
    pair in the person's unit. It depends only on k = dM/dH and on φ; the
    measured turing is η times this. In the ChatBench table it takes the place
    of the g column, which stays in the text.
  - *The turing intervals.* The two-level bootstrap already computed them
    (`turing_ci`) and the article did not report them: now they accompany every
    value, also by subject (mathematics, physics, ethics).
  - *The turing on other sources.* On the comparisons of the Vaccaro
    meta-analysis (`vaccaro.turing_of_comparisons`: median, interval resampling
    the studies, how many below one) and on the HAIID crowd advice (already in
    `map.json`). It serves to show that the unit applies to machines of other
    years and to an aid that is not a machine; it proves nothing about the angle.
- **9 November 2026, after a third independent review**, corrections of our own
  errors and descriptive additions:
  - *The pair's sensitivity with two classes.* Person and machine had d′ at free
    criterion, the pair `2 z(a)`: now the pair uses the same convention. It
    touches only the five rows of the crowd advice on HAIID (the only ones with
    two classes and a pair response): the turing of sarcasm goes from 1.31 to
    1.33, that of the census from 1.16 to 1.16 with an interval that now
    excludes one. The rest is unchanged.
  - *The angle between two people on ChatBench.* It was computed on the correct
    answers without the conversion for the four options, and compared with that
    between person and machine, which is converted. Now it is converted too (two
    people with the group's mean sensitivity: an approximation): from 69 to 66
    degrees. The conclusion does not change, and is strengthened.
  - *The turing read from the machine's side.* `dS / dM` is added, with its
    interval from the same bootstrap: it says whether the person, next to the
    machine, adds or subtracts. And we report what would have been obtained by
    someone who had copied the machine (`dS = dM`), because neither η nor g
    distinguishes it from an ideal pair near the useful angle.
  - *Another scale.* The turing recomputed on the logistic scale of Luce's choice
    model, to say how much it depends on the Gaussian assumption.
  - *The ceiling of a crowd.* From the angle between two people, the limit
    reached by a crowd of peers who pool their cues: `1 / sqrt(ρ)`.

## The instrument on trial (added 9 November 2026, evening)

Written **before** computing these two tests, but after seeing all the results
above: they are tests of the unit of measure, not of the angle. They use no new
machines: the first is a simulation, the second works on the probabilities of
the ten vision networks already saved.

### T, the calibration

An instrument must be tested where the truth is known. Data are generated from
the model (Gaussian cues per option, four options) with the ChatBench design:
160 people with different sensitivities, 7 questions each out of 380, a machine
with sensitivity `dM` and correlation `ρ` with the people, and a pair that sums
the person's cues and the machine's cues with a weight `w`. The true value of
each quantity is what the same instrument returns on a huge sample of the same
process.

Scenarios: (A) ideal pair; (B) pair halfway; (C) weak machine followed too
much, turing below one; (D) the person copies the machine, turing equal to k;
(E) machine equal to the person; (F) weak person and thirty lone people, as in
physics; (G) like A, with the machine answering trial by trial. For each, at
least 150 replicas, and on each replica the same `compare` as in the article
with 300 resamples.

For turing, apparent turings, η and φ we report: bias, root mean squared error
and coverage of the 95 percent interval.

**Criterion.** For the turing, in scenarios A to E and in G: bias within 5
percent of the true value and coverage between 90 and 98 percent. In scenario F
worse is expected, and we report by how much. If the criterion fails the
article says so, and the turing intervals are read accordingly.

### E, the law at full evidence

The H3 test worked with people who give only a verdict. The ten networks
(2014-2023) left, for each of the 18,080 images, the probability of each
category: between two networks the evidence is full on both sides, and it is
the case in which the formula for the ideal should hold. In each pair of
networks the weaker acts as the "mind" and the stronger as the "machine".

For each pair and each condition (the same as in the vision analysis, without
the *cue-conflict*): `dA` and `dB` from the accuracy with sixteen options; `ρ`
with the article's instrument (tetrachoric of the correct answers, converted);
the prediction `|S|` with the measured angle and at 90 degrees; and the
sensitivity reached by a full-evidence combiner, which sums the logarithms of
the two probabilities with two learned weights and is evaluated on unseen
images. Conditions where one of the two is at chance level are excluded, and
those where the pair touches the accuracy ceiling are counted separately.

**Criteria.** (1) The mean absolute error of the prediction with the measured
angle is smaller than that at 90 degrees, with the 95 percent interval of the
difference, resampling the conditions, entirely below zero. (2) The median
ratio between what the combiner reaches and the ideal with the angle lies
between 0.9 and 1.1. If (2) fails upward, the formula underestimates what a
pair at full evidence can do, and this must be written.

Also reported, descriptively, is the table "a 2014 mind next to the machines
that came later": VGG16 as the mind, the other nine as machines, with k, φ,
apparent turings, the combiner's turing and η; and, per degradation family, how
turing and accuracy gain change as the difficulty grows.

**Limits declared beforehand.** A network is not a person. Pairs and conditions
are not independent: the networks resemble each other, and every network enters
nine pairs. The saved probabilities are rounded to five decimals. At full
evidence the Gaussian model is an approximation, and an overconfident network
violates it.

### Outcomes and deviations of the two tests (9 November 2026, night)

**T, the calibration.** The criterion **fails** in five scenarios out of six (A,
B, C, E, G), and on the cautious side: the bias of the turing is within 1
percent, but the 95 percent interval contains the true value 98-100 times out
of a hundred, because it is 1.3 to 1.7 times wider than necessary. Only D
passes, on the edge (98 percent). In F the bias is +4.1 percent and the root
mean squared error 19 percent of the true value. The article's turing intervals
must be read as intervals at about 99 percent. The mechanism was not isolated.

Deviations: 300 replicas for A and B, 150 for the others (the machine was busy;
the declared minimum was 150); the resampling seed changes at every replica,
otherwise all would have shared the same weights; the true value comes from the
accuracies on two million trials with the generative ρ (the JSON also has the
instrument itself on one million trials, as the letter of the protocol says:
for the turing it coincides); a person's minimum sensitivity is cut at 0.02.

Defects of the instrument found by the calibration, **declared and not yet
corrected**, because correcting them requires redoing all the runs:
- the correlation is computed on a grid of thresholds with step 0.02
  (`measure._fast_rho`, and the grid in `sdt.estimate_from_evidence`): with
  endless data ρ does not converge to the true value (0.16 against 0.20 in
  scenario A). On the replicas the error has a mean of about zero; on the real
  ChatBench data it shifts the angle by less than a degree (GPT-4o 74.5 against
  75.2 degrees);
- `measure._sd_sensitivity` and `sdt.shrink` underestimate the dispersion of the
  sensitivities by a factor (1 − 1/n); the effect on ρ is negligible.

**E, the law at full evidence.** Both criteria **pass**: mean absolute error
0.10 with the measured angle against 0.75 at 90 degrees (difference −0.65,
interval from −0.75 to −0.54, resampling 76 conditions), and median ratio to
the ideal 1.00. They pass, however, with little merit, and the article says so:
among these networks the median angle lies between 22 and 43 degrees, 1,383
units out of 2,615 fall below the useful angle, and there the prediction
coincides with the stronger network. A pair that copies the stronger passes the
same criteria with a smaller error (0.07). Above the useful angle the law
predicts a median gain of 1.3 percent, the combiner obtains none, and the rank
correlation between predicted and measured gain is 0.09. What the test supports
is the useful angle, not the law of the gain.

Deviations: 77 conditions instead of the 73 of `vision.json` (restricted to the
73 the result does not change); 63 units with the pair at chance level out of
sample, a state not foreseen, kept apart; the conversion of ρ is that of
`sdt.convert_rho_mafc` recomputed without rounding. The baseline "whoever
copies the machine" was added during the work.

Defect found, not corrected: `sdt.convert_rho_mafc` has the grid up to 0.95
and does not make the table monotone; it is not used by the article's analyses
except for the angle between two people on ChatBench, where ρ is far from the
edge.

## Deviations of 10 November 2026

- **10 November 2026, after a review.** A review asked for four robustness
  checks. They were decided after seeing all the results, so they are
  **exploratory**: they do not change hypotheses, criteria or numbers already
  written, and a favourable outcome does not count as confirmation. The reason
  for each is an alternative hypothesis that the reported results did not
  exclude. Code in `angle/review.py` (no existing function has changed), numbers
  in `results/review.json`, tests of the new code in `tests/test_review.py`. The
  whole computation takes about three minutes.
  - *The scale on αNLI without the contested keys.* Reason: as they grow, the
    machines might concentrate their residual errors on the items where the
    voters contest the key, and the tetrachoric, which assumes Gaussian cues,
    would read that concentration as correlation. Same estimator
    (`compare_votes`, 2,000 resamples), same ten machines, same two modes: (a)
    without the 157 items out of 1,532 in which the majority of the 100 votes
    goes against the key (the 8 tie items stay: 1,375 items); (b) with the most
    voted option as the key (without the 8 ties: 1,524 items). On the original
    items the same run gives back `map.json` to the digit. Outcome: the descent
    remains, shorter. Spearman between dM and φ −0.89 and −0.90 in (a), −0.93 and
    −0.94 in (b), against −0.96 and −0.95 (immediate answer and reasoning). All
    the angles rise, by 6-17 degrees, and the distance between the weakest
    machine and the strongest falls in (a) from 21 to 15 degrees in immediate
    answer (from 76 to 55 degrees before, from 83 to 68 now) and from 20 to 12
    with reasoning. A part of the descent, from 30 to 40 percent, comes from the
    contested items; the rest does not. In (b) the key is defined by the same
    votes with which the person is measured, and human accuracy is inflated by
    construction (0.88 against 0.84).
  - *The useful angle with free weights.* Reason: in E the combiner has
    non-negative weights, so below the useful angle it cannot exceed the
    stronger network except by noise, and the confirmation of the useful angle
    might come from the constraint. Same 2,615 valid units, same five-fold
    division, same ε; weights between −1,000 and 1,000 (no fold reaches the
    limit). The constrained combiner, redone on the same units, gives back the
    file's accuracies on all. Outcome: below the useful angle (1,383 units) the
    pair exceeds the stronger in 117 units with the constraint and in 411 out of
    1,354 without (another 7 touch the ceiling, 22 fall to chance level), but
    without the constraint it falls below it in 455: above and below are
    equivalent (sign test, p = 0.14), and the median ratio dS/dB stays 1.00
    (quartiles 0.98 and 1.01). Among the units that exceed it, the median ratio
    is 1.030 without the constraint and 1.018 with it. The formula without the
    constraint predicts there a median gain of 3.3 percent, and the rank
    correlation between predicted and measured gain is 0.14. The weight of the
    weak network is negative in all five folds in 476 units. Above the useful
    angle (1,232 units) the constraint hardly matters: without it, 533 units
    above the stronger and 364 below; with it, 496 and 337; median 1.00 in both.
  - *The turing without a chat answer.* Reason: the ChatBench "alone" row mixes
    the trials with and without an answer recorded in the chat, and those
    without had never been looked at alone. Same procedure as
    `run_chatbench.py` (`compare`, four options, 2,000 resamples) on the
    complementary trials. GPT-4o: 372 trials (124 people, 222 questions), person
    accuracy 0.49, machine 0.81, pair 0.50, turing 1.03 [0.77, 1.44], dS/dM 0.44
    [0.27, 0.60]. Llama 3.1 8B: 432 trials (128 people, 247 questions), 0.54,
    0.53 and 0.57, turing 1.11 [0.92, 1.38]. With the trials that have an answer
    in the chat (1,121 and 969) they make up the 1,493 and 1,401 of the "alone"
    row. For the calibration (T) these intervals must be read as at about 99
    percent. Addition of the same evening (fourth review): the trials without an
    answer split according to ChatBench's annotation of how the person used the
    chat (`user_effort`, assigned by a language model), without intervals, in
    `by_effort`. GPT-4o: "tried" 264 trials (person 0.49, machine 0.79, pair
    0.50), "low effort" 92, "known" 16. Llama 3.1 8B: "tried" 352 (0.52, 0.50,
    0.56), "low effort" 65, "known" 15. It describes trials chosen after the
    treatment: it does not say why the answer does not come out. After a fifth
    review (same evening), in `check_3`: `by_kind` splits the trials according to
    what the chat recorded (`answer`, a letter among the options;
    `out_of_options`, an annotated answer that is not a letter; `no_answer`),
    with accuracies and changes of answer. GPT-4o: 330 without an answer (person
    0.50, machine 0.84, pair 0.50; changes 29%, 8.2% toward the correct and 8.2%
    toward the wrong) and 42 out of options; where the chat answered, the
    switches to the correct are eight times those to the wrong.
    `in_chat_widened` counts the out-of-options answers as wrong: the machine in
    chat goes from 0.84 to 0.79 with Llama (alone 0.71 on the same trials) and
    from 0.92 to 0.88 with GPT-4o (alone 0.90). `low_effort_share`: 0.28 among
    the no-answer trials of GPT-4o, 0.07 over all of them.
  - *The uncertainty of the trends.* Reason: the rank correlations over ten
    machines were reported without uncertainty. Exact permutation p (the 10! =
    3,628,800 permutations of the ranks), percentile interval resampling the
    machines (10,000 times), the correlation leaving out one machine at a time;
    for αNLI also resampling the items with the same weights for all the
    machines. Outcome: vision −0.45, two-sided p 0.19 (one-sided 0.096),
    interval [−0.95, 0.43]. ChatBench in immediate answer −0.73, p 0.020,
    interval [−0.99, −0.06]; without gemma-4-26b −0.63, p 0.076 (one-sided
    0.038). ChatBench with reasoning −0.02, p 0.97. αNLI −0.96 and −0.95, p
    5·10⁻⁵ and 1·10⁻⁴, intervals [−1.00, −0.79] and [−1.00, −0.76], resampling
    the items [−0.99, −0.79] and [−0.99, −0.75]. With ten points the intervals
    over the machines are crude. For ChatBench and for vision there was no
    resampling at the level of items and people: the uncertainty of each φ
    (about twenty degrees on ChatBench) enters only through the dispersion among
    the machines.
  - *The width of the intervals.* The calibration (T) said "the mechanism was not
    isolated". For the turing on the ChatBench design the width (1.3-1.7 times
    what is needed) is compatible with Owen's (2007) pigeonhole bootstrap, which
    counts up to three times the residual variance, that is, up to √3 in width.
    It does not explain the angle (1.6-2.1 times) nor scenario F (2.4): there
    the mechanism remains not isolated. No new test, it is a reading of the
    literature.

## Deviations of 14 November 2026

Decided after all the results above, to answer one question: does the reading
hold with machines much stronger than GPT-4o? Exploratory throughout.

- **Four closed models.** Claude Haiku 5.5, Sonnet 5.5, Opus 5.5 and Fable 5.2
  answer the ChatBench (396) and αNLI (1,532) items alone, with the `letter`
  prompt unchanged, through Claude Code headless, at effort `low` (thinking
  cannot be switched off on the larger models), one draw per item, the letter
  constrained by a response schema. Verdicts only: the probability combiner of
  H3 does not apply. No answer was lost.
- **What is computed.** With the estimators of H1 and H7: k, φ, the useful
  angle, and the margin of the ideal pair over the stronger of the two,
  |S|/max(dH, dM) − 1, also at 90 degrees. On αNLI with the three readings of
  the key of check 1 of 10 November (the story's key, without the stories where
  most voters go against it, the majority as the key), for the closed and the
  open models alike. `angle/frontier.py`, `results/frontier.json`.
- **Limits known before reading the numbers.** On ChatBench the strongest
  models make almost no errors, and with no errors neither d′ nor φ exists: the
  accuracy is clipped at 0.9999, so k is a floor there. One draw per item, at
  one effort level, through a subscription product rather than the API: the
  model version is the one served on the day.

## Deviations of 15 November 2026

Asked for by two reviews of the article, after every result above. Exploratory
and descriptive: no interval is calibrated, and the trials with an answer in the
chat are a selection. `angle/conversations.py`, `results/conversations.json`.

- **Where the points go.** The distance in accuracy between the machine alone
  and the pair, split into the trials whose chat produced one of the options
  (the conversation: chat minus machine alone; the decision: pair minus chat)
  and the others. A partition of the trials, with no convention for the mute
  chats. A decomposition in d′ over all the trials was tried first, counting
  the mute chats as guesses: it depends on that convention (with the opposite
  one the signs of e and f reverse), and it was dropped.
- **Extraction and filter.** Tu = k·e·f on the trials with an answer, from the
  sensitivities already in `results/chatbench-source.json`.
- **Following the chat.** On the trials where the chat's letter differs from
  the person's answer alone: following when the chat is right and when it is
  wrong, with intervals resampling people, and people and questions together.
  Against a linear observer that adds a fixed bonus to the chat's option,
  calibrated on the pair's accuracy: it gives a similar difference, so the
  difference alone is not evidence of gating.
- **Always and never.** Shares of people with at least two disagreements who
  always or never followed, against people with identical rates.
- **Skill.** Half of each person's trials, drawn at random, classify; the other
  half measure. Median over 300 draws, lowest and highest third with ties.
  Each person had a single subject, so also within mathematics. A first single
  draw had been read in the article and was replaced: it was an outlier, and
  confounded with the subject.
- **Zero errors.** For a model with no error on ChatBench the item resampling
  cannot move the machine's accuracy: the bound comes from the rule of three
  (an error rate up to 3/396), with the person's sensitivity at its point and
  at the high end of its interval.

- **Later the same day, after a third review.** Four more descriptive analyses
  in the same module. *Group against person*: the people's spread of accuracy,
  net of binomial noise, fitted with a Beta distribution, gives the ratio of
  the average individual sensitivities with and without the machine, to set
  against the group turing (Macmillan and Kaplan, 1985); bootstrap over
  people. *Started wrong*: on the conversations with an answer, how much the
  chat beats the machine alone when the person's answer alone was right and
  when it was wrong, and how often a wrong chat repeats the person's wrong
  answer. *A hundred questions*: errors of the people, of the machine and of
  both, against independent errors. *Contact*: the four-factor decomposition
  in d′ over all trials, under the two conventions for a mute chat.

- **Later the same day, after a final review.** Two corrections. *Lowercase
  letters*: ChatBench records thirteen chat answers in lowercase («c») and the
  comparison with the key counted them as wrong; the letter is now normalized
  where the conversations are read (`sources.chatbench_conversations`), and
  `run_chatbench.py`, check 3 of `review.py` and `conversations.py` were run
  again. With GPT-4o three trials change, with Llama one; no conclusion
  changes. *The trap of the question*: whether a wrong chat repeats the
  person's wrong answer is compared not with one in three but with the share of
  the other people who, wrong on that question on their own, chose the chat's
  letter; intervals with Poisson weights over people, and over people and
  questions.

- **Later the same day, after a further review.** Agreement and disagreement
  (`conversations.py`, `agreement_and_disagreement`), descriptive: on the chats
  with a letter, how often the chat is wrong when it agrees with the person's
  answer alone and when it does not (intervals with Poisson weights over
  people, and over people and questions), who is right in a disagreement, by
  subject and by the confidence the person stated; following the chat in
  every disagreement against what the pair did. In the model: the weights of
  the ideal pair, (k − cos φ)/(1 − k cos φ); equal weights («the more
  confident one wins»); an observer that adds to the chat's option the bonus
  of the likelihood ratio.

## The devil's advocate (written 15 November 2026, before any run)

Written after the article's results and before any call to a machine in the two
new conditions. The code (`angle/devil.py`, `closed/devil_claude.py`,
`gpu/devil_llm.py`) was written and tested on synthetic trials first; this
section and the code are committed and pushed before the runs.

**The question.** The conversations show a trap: a wrong chat after a wrong
person repeats the person's answer more often than other people wrong on the
same question would. When the machine sees the person's answer, does it fall
into the person's error? And does asking it first for the arguments against
that answer remove the trap, and bring the pair above the machine alone?

**Data.** ChatBench, answer-first condition, phase 2: the 2,903 answers given
alone (329 people, 396 questions, 1,187 wrong). Each distinct pair (question,
answer given alone) is an item, 1,038 in all (`data/items/chatbench-devil.jsonl`).
The unit of analysis is the trial (person, question): in conditions A and B the
machine's verdict depends on the person's answer, so it is assigned per trial.

**Conditions.** One call per item and condition.

- *0, alone*: the `letter` prompt of `gpu/machines_llm.py`, without the person's
  answer. For Claude Haiku 5.5 the run already in
  `closed-results/chatbench__haiku-5.5__low.jsonl`; for Llama a new run.
- *A, told*: the same prompt with the line `I think the answer is X.` before
  the request, X the letter the person chose alone.
- *B, devil's advocate*: the line `I think the answer is X.`, then `First give,
  briefly, the strongest arguments against my answer. Then finish with a final
  line of the form: Answer: Y, where Y is the letter of the correct option (A
  to D).` Claude answers through a schema with two fields, the arguments first
  and then the letter.

**Machines.** Llama 3.1 8B Instruct, the open model the people of ChatBench
worked with (weights from the ungated mirror `unsloth/Llama-3.1-8B-Instruct`,
the same as Meta's), with vLLM and greedy decoding, the letter read from the
constrained probabilities as for the other open models; in B the arguments are
generated first, up to 700 tokens. Claude Haiku 5.5 through Claude Code in
headless mode, effort `low`, as for the other Claude runs. An item a machine
does not answer in some condition is dropped from all three, for that machine.

**Measures**, per machine.

1. The machine's accuracy on the 2,903 trials in 0, A and B, its d′ with four
   options, and the yield of B, d′_B / d′_0. B is a pair: the person proposes,
   the machine attacks and decides.
2. Accuracy on the trials where the person alone was wrong, and on those where
   the person was right, per condition.
3. The trap: on the trials where person and machine are both wrong, the share
   where the machine gives the person's letter, per condition.
4. Descriptive only: the angle between the person's errors and the machine's,
   per condition, with the experiment's estimator (`measure.compare`,
   machine per trial). In A and B the machine has seen the person's answer:
   the Gaussian model of the ideal pair does not hold, and no margin is
   computed.

**Hypotheses and criteria.** 95% percentile intervals, 2,000 replicas
resampling people and questions together (Poisson weights, as in
`measure`).

- *H1, the trap*: on the trials where the person was wrong, the machine told
  the answer is less accurate than the machine alone. Passes if the interval of
  acc_A − acc_0 lies below zero.
- *H2, the devil's advocate removes the trap*: on the same trials,
  acc_B − acc_A above zero.
- *H3, the pair above the machine*: on all trials, d′_B / d′_0 above one.
- *Reported whatever happens*, with no criterion: the cost on the trials where
  the person was right, acc_B − acc_0; the trap shares; the angles.

Three tests per machine, two machines, no correction for multiplicity.

**What it cannot say.** No person reads the arguments: the decision is the
machine's. A positive H3 says that, on these questions, a pair in which the
person's answer is the input of a contrarian machine beats the machine alone;
not that people reading the arguments would. One call per item: the variance
of the machine between calls is not in the intervals.

**Where it goes in the article.** If H3 passes for at least one machine, in
the body; otherwise in the appendix of what did not work. Either way with the
numbers of all three hypotheses for both machines.

**Deviations, the same day, after Claude Haiku 5.5's runs and before Llama's.**
Haiku's results had been seen; neither change depends on them.

- *Llama in 8 bits.* The bf16 weights (16 GB) would have left 2 GB free on the
  disk of the GPU machine, which is someone's work machine. Llama runs with the
  FP8 weights of the same model (`RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8`,
  9 GB, weights and activations in 8 bits).
- *Haiku alone, again.* The condition alone came from a run of 14 November,
  the other two from today: a difference between runs could pass for an effect
  of the conditions. Haiku alone is asked again today with the same prompt
  (`closed-results/chatbench__haiku-5.5-rerun__low.jsonl`). It is a check,
  decided after seeing the results, and is reported as such: the tests of the
  protocol stay on the run named above.

## After the sixth review (15 November 2026)

Written after every result above, and after a sixth review of the article. It
has nothing to do with the devil's advocate: that section, its code and its
results are not touched here. Everything below was decided **after** the
results: it is exploratory or descriptive, with no hypothesis and no criterion,
and a favourable outcome does not count as a confirmation. Code in
`angle/verdict.py` and `angle/difficulty.py` (no existing function has
changed), numbers in `results/verdict-ceiling.json` and
`results/difficulty.json`, tests in `tests/test_verdict.py` and
`tests/test_difficulty.py`.

- **The ceiling with a verdict.** Reason: the article's ideal pair pools the
  graded cues of the two, and the margin is measured against it; a person next
  to a machine usually gets its verdict, and a pair that exchanges verdicts
  cannot reach that ceiling. Method: the optimal observer (maximum a
  posteriori) that keeps its own graded cues and of the other sees only the
  option chosen, under the article's model (four options, one Gaussian cue per
  option, the noise of the two correlated option by option with ρ, as in
  `sdt._simulate_correct`). Given the observer's cues and a hypothesis on the
  true option, the probability of each verdict is a one-dimensional integral;
  the observer answers the option with the largest posterior. The cues are
  drawn (200,000 trials), the verdict is not: every trial is averaged over the
  four verdicts. The pair's errors are a ratio to those of the stronger member
  on the same trials, anchored to its exact error rate; where the observer is
  the stronger and almost never wrong, its cues are drawn with a wider noise
  and weighed back. Tested against two closed forms: independent errors with
  four options (the verdict is worth a fixed bonus ln(p/q)/d on its option),
  and correlated errors with two options (a one-dimensional integral). The
  observer is exact, so it is not held to non-negative weights as the
  article's ideal is: below the useful angle what it gains is reported as it
  comes.
  (a) The rows of the article's table of the margin: the person is the
  ChatBench group next to GPT-4o, right 57.7 times in a hundred on its own (the
  article prints 58; with exactly 0.58 the errors removed read one point higher
  than printed), the machine 1.2, 2, 3, 5 and 10 times as sensitive, at 75 and
  90 degrees. With shared cues the code gives back the table as printed.
  (b) The two ChatBench pairs, machine queried alone, all trials, at their
  measured sensitivities and angle, at the two ends of the angle's interval
  and at 90 degrees.
  Outcome, with only the machine's verdict and the person keeping the cues. At
  90 degrees the margin over the machine is 17.0 percent with a machine 1.2
  times as sensitive (30.2 with shared cues), 2.1 with one twice as sensitive
  (11.8), 0.1 with one three times (5.4), and nothing from five times on (2.0
  and 0.5). At 75 degrees: 8.9 percent (16.4), 0.1 (3.1), then nothing (0.3,
  0, 0). In errors of the machine removed, at 90 degrees: 17.7 percent (30.3
  with shared cues), 5.6 (28.2), 0.3 (27.2), nothing (26.5 and 25.7); at 75
  degrees: 9.5 (17.0), 0.3 (8.0), nothing (1.7, 0, 0). The quarter of the
  machine's errors that the ideal pair removes at 90 degrees whatever k exists
  only if the cues are shared. Monte Carlo standard errors: at most 0.15 points.
  With only the person's verdict and the machine keeping the cues, the case of
  the verdict combiner of H3 and H5: margin at 90 degrees 19.4, 7.0, 3.0, 1.1
  and 0.2 percent, at 75 degrees 11.0, 1.8, 0.1, 0.1 and 0.7; below the useful
  angle (five and ten times, at 75 degrees) the exact observer goes a fraction
  of a percent above the stronger, where the article's ideal stops at it.
  ChatBench. With GPT-4o (k = 2.16, 74.5 degrees): the ideal with shared cues
  is 1.020 times the machine's sensitivity, the ceiling with its verdict 1.000
  (1.000 and 1.007 at the ends of the angle's interval, 59 and 86 degrees;
  1.013 at 90), and the people reached 0.761 (0.673-0.849). With Llama 3.1 8B
  (k = 1.18, 75.1 degrees): 1.173 with shared cues, 1.098 with its verdict
  (1.047 and 1.132 at 62 and 82 degrees; 1.180 at 90), and the people reached
  1.189 (1.044-1.340), above both, because the machine queried alone is weaker
  than the one that talked with them («Deviations», 9 November).
  It is a model, not a measurement, and its errors are Monte Carlo errors, not
  sampling intervals. A machine that also gives its reasons, or a graded
  confidence, gives more than a verdict: the ceiling binds a pair that
  exchanges verdicts, not the people of ChatBench, who read a conversation.

- **Closeness net of difficulty, on αNLI.** Reason: in the angle two things
  end up together, how alike two minds are and how hard the items are for
  everyone. The article says so, says that the data to tell them apart exist,
  and the review asked to use them. For the pair nothing changes: the shared
  difficulty counts in ρ («Model and definitions») and the margin keeps the
  whole correlation. What changes is how much of the angle can be read as
  likeness.
  *Who the person is.* On αNLI the «person» is one vote of the hundred
  anonymous votes on a story (`measure.compare_votes`): there are no people to
  follow, only how many voted for the key. The correlation between person and
  machine is therefore, by construction, how much the machine fails where many
  people fail, and the difficulty for the people is the whole human side of
  it. A difficulty estimated on the people themselves would take everything
  out. It is shown only to say why it is not the analysis done here: with half
  of the votes of each story making the strata and the other half as the
  person, the angle with Fable 5.2 goes from 47.0 degrees to 62.7 with three
  bands (58.1-66.9), 82.9 with ten (79.3-86.2) and 86.6 with twenty-five
  (83.1-89.9). With three bands what is left is difficulty inside the band:
  an angle below 90 degrees there proves nothing.
  *Method.* The difficulty is estimated on a third population. Person and
  machine: the ten open machines in both modes, twenty answers per story (for
  an open machine the other nine, eighteen answers); the score of a story is
  how many are right. Machine and machine: the votes for the key. Strata: the
  consecutive levels of the score, merged from the hard end until each holds
  at least 60 stories (9 to 12 strata on the machines' score; 18 on the votes,
  15 without the contested stories): fine enough that little difficulty is
  left inside a stratum, large enough that a stratum has errors to count. With
  30 and with 120 stories the correlations move by at most 0.04; with the
  immediate answer alone as the third population (ten answers) those between
  person and machine are higher by up to 0.08, because a coarser score takes
  out less. Estimator: one latent
  correlation for all the strata, each stratum with its own two thresholds
  taken from its margins, at the maximum of the summed log-likelihood of the
  2×2 tables; with one stratum it is the article's tetrachoric. A stratum
  where one of the two makes no error carries no information and drops out.
  Next to it the Mantel-Haenszel common odds ratio, which needs no thresholds.
  Intervals: 2,000 resamples of the stories (Poisson weights), strata fixed;
  250 for the median over the 180 pairs of open machines. Test: on stories
  simulated with a known correlation inside each level of difficulty the
  estimator gives it back (0.29 for a true 0.30, where the correlation on all
  the stories together is 0.64; at the size of αNLI with eighteen strata and a
  machine right 88 times in a hundred, a mean of 0.30 over twelve corpora).
  *Outcome, person and machine*, net of the difficulty for the open machines.
  The angle opens and stays below 90 degrees with its whole interval for the
  four Claude models: Fable 5.2 from 47.0 (44.4-49.9) to 58.6 (55.4-61.8),
  Opus 5.5 from 47.9 to 59.9 (56.8-63.1), Sonnet 5.5 from 49.8 to 64.9
  (61.1-68.5), Haiku 5.5 from 55.3 to 74.7 (70.6-78.5). Without the 157
  contested stories: Fable from 67.5 to 72.2 (68.6-76.2), Opus from 70.2 to
  74.1 (70.5-77.9), Sonnet from 68.8 to 76.4 (72.9-79.7), Haiku from 71.3 to
  81.6 (78.6-84.8). Of the correlation between people and Fable three quarters
  remain (from 0.68 to 0.52, interval 0.47-0.57; 76 percent, 72-81), of that
  with Haiku less than half (from 0.57 to 0.26, 0.20-0.33; 46 percent, 37-55).
  The strongest open machines, net of the other nine, in immediate answer:
  Gemma 4 26B from 55.5 to 72.7 (69.1-76.5), Qwen3.8 27B from 52.9 to 70.3
  (66.2-74.2). For the seven weakest open machines the angle net of the others
  lies between 84.5 and 94.4 degrees over the two modes, and in eleven of the
  fourteen cases the interval includes 90 degrees or lies above it: what they
  share with the people is what is hard for the open machines.
  *Outcome, machine and machine*, net of the difficulty for the people. Opus
  5.5 and Fable 5.2 stay almost the same machine: from 0.95 (0.92-0.97) to
  0.83 (0.75-0.91). Sonnet and Fable from 0.88 to 0.59 (0.45-0.72). Haiku and
  Fable from 0.72 to 0.24 (0.05-0.41), Gemma 4 26B and Fable from 0.71 to 0.24
  (0.05-0.42), Gemma and Haiku from 0.77 to 0.52 (0.41-0.63). The median over
  the 180 pairs of open machines goes from 0.63 (0.60-0.66) to 0.45
  (0.43-0.49). The odds ratios say the same: Opus and Fable from 156 to 20
  (11-41), Haiku and Fable from 16 to 1.9 (1.1-3.1), Gemma and Fable from 15
  to 2.0 (1.2-3.3). Without the contested stories: Opus and Fable from 0.89 to
  0.85 (0.73-0.95), Haiku and Fable from 0.44 to 0.16 (−0.13 to 0.39), Gemma
  and Fable from 0.56 to 0.28 (0.01-0.51). With the strongest machines only
  the hard strata count: Fable makes no error in nine of the eighteen strata,
  and a comparison with it rests on 573 stories.
  *What it cannot say.* The score is a count of twenty answers, or of a
  hundred votes: the difficulty it leaves inside a stratum stays in the
  correlation, which is an upper bound on the likeness. And the open machines
  are not neutral judges: what they share with the machine measured goes out
  with the difficulty, and more of it the more that machine resembles them.
  Haiku and the weak open machines are close to the third population, Fable is
  not: the net angles of different machines are not a trend to read. There is
  no single «pure» angle, it depends on who measures the difficulty. Nothing
  here is calibrated.

- **The contested stories on half of the votes.** Reason: the article takes
  out the stories whose key most voters contradict, looking at the same votes
  that then measure the angle, and says so. Method: half of the hundred votes
  of each story, drawn at random, say whether it is contested; the other half
  is the person; same step as `compare_votes`. Point: mean over 200 splits.
  Interval: 2,000 repetitions, each with a new split and resampled stories.
  Outcome: the angle without the contested stories is 65.9 degrees for Fable
  5.2 (62.3-70.0) against 67.5 with the same votes, 68.3 for Opus 5.5
  (64.2-72.8) against 70.2, 67.8 for Sonnet 5.5 (64.1-71.9) against 68.8, 70.7
  for Haiku 5.5 (67.5-74.1) against 71.3, 69.0 for Gemma 4 26B in immediate
  answer (65.6-72.0) against 69.4. The margin of the ideal pair: 3.9 percent
  for Fable (2.3-5.7) against 4.5, 6.4 for Opus (4.2-8.9) against 7.2, 8.7 for
  Sonnet (6.0-11.3) against 9.2, 20.0 for Haiku (16.0-23.3) against 20.7. The
  choice made on the same votes is worth one or two degrees, and the reading
  does not change. The other half of that caution stays: taking out the
  stories hardest for the people narrows their range, whatever votes choose
  them.

- **The silicon people: the method, and the arm with the healthy machine.** No
  new computation: the numbers were already in `results/silicon.json` (300
  resamples) and the article did not describe the method. Method
  (`gpu/silicon.py`): CIFAR-10; twenty small convolutional networks per arm;
  three stages, with 4,000, 12,000 and 36,000 training images seen in all, six
  epochs each; the first stage alone in every arm. From the second stage the
  control minimizes the cross-entropy with the true label, the two assisted
  arms half of it plus half of the Kullback-Leibler divergence from the
  machine's output (weight 0.5). The machine is a ResNet-18 trained on the
  training set and on the 8,000 test images outside the exam, then frozen. Its
  defective copy, on the images it calls cat or dog whose features exceed a
  threshold along a fixed random direction (seven in ten), puts the other
  class two logits above. The exam is 2,000 test images that neither the
  networks nor the machine saw in training (the defect's threshold is set on
  them); on 277 the defect changes the machine's answer. So the networks
  always see the true label, and the machine's opinion is half of what they
  learn from: that the arm with the defect leans toward it is the direction
  the loss builds in. The size is not: where the machine was right before the
  defect, the target the loss aims at still has the true class first (at least
  one half, against at most 0.44 for the defect's class).
  The numbers. On the 277 items, the defect's wrong answer: 11.0 percent in
  the control, 11.2 with the healthy machine, 23.2 with the defective one;
  defective minus control 12.1 points (8.7-15.8), healthy minus control 0.2
  (−1.0 to 1.5); correct answers there 73.5, 73.9 and 61.1 percent. The arm
  with the healthy machine against the control, at the last stage:
  correlation of the correct answers with the machine 0.669 against 0.670
  (0.407 and 0.405 at the first), accuracy 83.6 against 83.3 percent;
  difference in the differences of A, without conditioning on the class,
  −0.005 (−0.041 to 0.032), within the class −0.016 (−0.141 to 0.071).
  Learning next to a healthy machine did not bring the networks closer to it
  than learning alone. The arm with the defect against the defective machine:
  correlation 0.539, against 0.413 for the control measured against the same
  machine (0.298 and 0.278 at the first stage); difference in the differences
  of A without conditioning on the class 0.106 (0.065-0.152), within the
  class, as the protocol prescribed, 0.019 (−0.059 to 0.094). The plain
  correlations have no interval: the networks' answers are not in this
  repository.

## The devil's advocate: outcome (15 November 2026)

Results in `results/devil.json`; the answers in `devil-results/` and
`closed-results/chatbench__haiku-5.5-rerun__low.jsonl`. Intervals as
pre-registered (2,000 replicas, Poisson weights over people and questions).

- **Claude Haiku 5.5.** H1 fails: told the person's wrong answer, the machine
  is 1.3 points *more* accurate than alone (−1.5 to +4.3). H2 fails: +1.5
  (−0.9 to +4.2). H3 passes: d′_B/d′_0 = 1.18 (1.07–1.31); with Haiku alone
  asked again the same day, 1.13 (1.03–1.26). Told the answer, without the
  request (not a hypothesis): 1.14 (1.03–1.27), 1.10 (1.00–1.22) with the
  rerun. Accuracy: alone 93.5 (rerun 94.5), told 96.5, devil's advocate 97.1.
  Output tokens per answer: about 170, 220 and 860. Between the two identical
  runs of Haiku alone, 28 of 396 questions change outcome.
- **Llama 3.1 8B (FP8), as pre-registered.** H1 passes: −34.5 points (−41.6 to
  −27.8). H2 passes: +52.0 (+45.3 to +58.3). H3 passes: 1.41 (1.09–1.89).
  Accuracy: alone 47.0, told 61.7, devil's advocate 56.7. But alone and told
  answered with the letter only, the devil's advocate after reasoning.
- **Llama, the reasoning controls (added after its results, before running
  them).** All three conditions reason: alone 72.0, told 72.3, devil's
  advocate 56.7. The H1 comparison: −18.5 (−24.8 to −12.2). The H2
  comparison: +10.0 (+3.6 to +16.5). The H3 comparison: 0.67 (0.57–0.77), the
  devil's-advocate pair below the machine alone. Where the person had been
  right the devil's advocate falls from 87.5 (told) to 54.1.
- **The trap** (both wrong, the machine gives the person's letter): Llama with
  the letter only 41 → 98 → 44 percent (alone, told, devil's advocate); Llama
  reasoning 48 → 85 → 44; Haiku 48 → 46 → 50, on few errors.
- **Angles**, descriptive (ρ between the person's errors and the machine's):
  Llama with the letter only 0.15, 0.99 (at the bound of the conversion) and
  −0.10 (−0.36 to 0.08); Llama reasoning, told, 0.74; Haiku −0.02, 0.26, 0.11.
- **Where it goes.** H3 passed for one machine, so by the rule above the
  experiment goes in the body of the article, with the pre-registered outcome
  for Llama and the controls in its appendix.

## After the seventh review: two new analyses (written 15 November 2026, before computing them)

Written after every result above, and before computing anything below. A
seventh review of the article asked two questions the data already in hand can
answer. Neither touches an earlier result. The code (`angle/homogenization.py`,
`angle/mute_chats.py`) is written and tested on synthetic data first; this
section and the code are committed and pushed before the runs on ChatBench.

### A. Do two people become more alike with the chat beside them?

**The question.** The article argues that whoever works with a machine can
take on its errors. On ChatBench two people answering alone have an angle of
66 degrees between their errors. If both answer the same question with the
same machine beside them, does the angle close?

**Data.** ChatBench, answer-first condition, phase 2 (`sources.chatbench_pairs`):
each person answers a question alone, then talks with the machine and answers
it again. Per arm: GPT-4o, 1,493 trials; Llama 3.1 8B, 1,401. The two answers
of each trial come from the same person on the same question, so the
comparison is paired.

**Measures**, per arm.

1. φ_HH: the angle between two different people on the same question, from
   their answers alone (`sdt.rho_person_pairs` on `acc`, the personal
   thresholds from the accuracy alone), converted to the cues as in the
   article (`convert_person_person`, with the mean accuracy alone).
2. φ_SS: the same, from the answers with the chat (`acc_ai`, thresholds from
   the accuracy with the chat, conversion with the mean accuracy with the
   chat). Same people, same questions, same pairs.
3. Δ = φ_HH − φ_SS, in degrees of the cues.
4. Same wrong letter: over the pairs of different people on the same question
   who are both wrong, the share who give the same letter; alone and with the
   chat, and the difference.
5. Descriptive: among the pairs both wrong with the same letter with the chat,
   the share where that letter is the one the chat gave; the crowd ceiling
   `1/sqrt(ρ)` alone and with the chat; Δ on the trials whose chat gave a
   letter and on those whose chat did not (a pair counts where both of its
   trials are in the subset).

**Uncertainty.** 95% percentile intervals, 2,000 replicas with Poisson weights
over the questions, the same weights for the two conditions (paired). People
are not resampled: the estimator pairs different people on the same question,
and the weights over questions already carry the dependence between the pairs.

**Hypotheses and criteria.**

- *A1, primary (GPT-4o)*: the angle closes with the chat. Passes if the
  interval of Δ lies above zero.
- *A2 (Llama)*: the same.
- *A3 (both arms)*: with the chat, two people both wrong give the same wrong
  letter more often. Passes, per arm, if the interval of the difference lies
  above zero.

No correction for multiplicity. Measure 5 has no criterion.

**What it cannot say.** It measures people *while* they use the machine, not
after: it says whether a crowd with the same machine beside it is less varied
than the same crowd alone, not whether people who worked with a machine stay
like it when it is gone (that is the longitudinal protocol). The answer with
the chat is always the second on the same question: without a condition that
answers twice alone, the effect of the machine is not separated from that of a
second look. The split by chats with and without a letter is the only partial
check, and it selects trials by how the conversation went.

**Where it goes.** If A1 passes, in the body of the article, next to the angle
between two people; otherwise in the appendix of what did not work. Either way
with A1, A2 and A3.

### B. The chats without an answer

**The question.** In 330 of the 1,493 GPT-4o trials ChatBench's automatic
annotation says the chat gave no answer. The article reads them as answers
lost in the conversation. A reviewer objects that the annotation may miss
answers the machine gave in words, without the letter.

**Data.** The 330 trials of `conversations.trials("gpt-4o")` with
`conv_has_answer` false; the 60 whose record has no message from the machine
are counted apart. The other 270 are examined.

**Measures.**

1. *Automatic, on all 270.* The text of each option, lightly normalized
   (lowercase, LaTeX delimiters and repeated spaces removed), is looked for in
   the machine's messages as a whole token (no letter or digit on either side).
   Options shorter than two characters after normalization are skipped. The
   trial is counted by how many options the machine names: none, exactly one,
   two or more. Where it names exactly one, whether it is the key, and whether
   the person chose it.
2. *Read, on a sample.* 50 of the 270, drawn with seed 534, each read with the
   question, the options and the conversation, without the key and without the
   person's answer, and put in the first category that applies:
   (a) *answer without the letter*: the machine states a final answer that
   matches exactly one option, in words or as an equivalent value;
   (b) *answer outside the options*: a final answer that matches none;
   (c) *partial help*: an explanation or an intermediate step, with no final
   answer to the question;
   (d) *missing question or clarification*: the machine asks for the question,
   the options or what is meant, or answers a different question because the
   message was incomplete;
   (e) *other*: off topic, a refusal, a greeting.
   The reader is an AI assistant, instructed with this list; ten of the fifty
   are read again by a second one, and the agreement is reported.

**Uncertainty.** Shares with 95% Wilson intervals; the sample shares are
carried to the 270 as estimates, not counts.

**Decision rule for the text.** If (a) is at least 10 of the 50, the article
stops calling the 330 «chats from which no answer came out»: it says that in an
estimated share of them the answer was in the chat without the letter, and
that the annotation missed it, and the decomposition of the 7.6 points gets
that caveat. Below 10, the article reports the check in one sentence and keeps
its reading. Either way the numbers go in the appendix.

## After the seventh review: outcome (15 November 2026)

Numbers in `results/homogenization.json` and `results/mute-chats.json`; the
sample read and the readings in `results/mute-chats-sample.jsonl` and
`results/mute-chats-labels.json`. Intervals as pre-registered.

### A. Two people with the chat beside them

- **GPT-4o** (1,493 trials, 168 people, 390 questions). φ_HH 70.4 degrees
  (59.1–76.3), φ_SS 72.0 (55.6–80.3): Δ = −1.6 (−11.2 to +11.5). **A1 fails.**
- **Llama 3.1 8B** (1,401 trials, 157 people, 387 questions). φ_HH 64.7
  (53.5–71.7), φ_SS 54.7 (43.3–63.5): Δ = +10.1 (−1.2 to +19.5). **A2 fails**,
  narrowly.
- **Same wrong letter.** GPT-4o 41.0 → 50.0 percent of the pairs both wrong
  (difference +9.0, −0.5 to +18.2); Llama 40.5 → 46.8 (+6.4, −3.1 to +16.2).
  **A3 fails** in both arms.
- *Descriptive.* Among the pairs both wrong with the same letter with the
  chat, both chats gave that letter in 5.5 percent (GPT-4o) and 14.4 (Llama),
  at least one in 34.6 and 52.5. Crowd ceiling `1/sqrt(ρ)`: GPT-4o 1.73 alone,
  1.80 with the chat; Llama 1.53 and 1.32. Where the chat gave a letter: GPT-4o
  66.6 → 63.1 (Δ 3.5, −14.4 to +29.4), Llama 52.0 → 20.9 (Δ 31.1, 8.2 to 43.5);
  where it gave none: GPT-4o 94.4 → 84.3, Llama 96.1 → 97.1. The subsets select
  trials by how the conversation went (500 replicas each).
- **Where it goes.** A1 failed, so by the rule above: the appendix of what did
  not work, with A1, A2 and A3; one sentence in the body says that the angle
  between two people does not visibly close.

### B. The chats without an answer

- 330 trials; 60 with no message from the machine (people right with the chat
  61.7 percent), 270 examined (47.8).
- **Automatic, on the 270.** No option named in 236, exactly one in 17 (6.3
  percent, 4.0–9.9), two or more in 17. Of the 17 with one option, it is the
  key in 5, and the person chose it in 9.
- **Read, on the sample of 50** (reader 1, all fifty): (a) answer without the
  letter 2 (4 percent, 1.1–13.5; 3 to 36 of the 270); (b) answer outside the
  options 8; (c) partial help 19; (d) missing question or clarification 10;
  (e) other 11. Reader 2, on the first ten, agrees on 8 (on one, (c) against
  (e); on another, (a) against (b): the machine's final answer, «I only», is
  not one of the options, so (b)).
- **Rule.** (a) is below 10 of 50: the article keeps its reading, reports the
  check in one sentence, and the numbers go in the appendix.
- *Note.* Reader 1 keyed its answers by line number of the sample file; they
  were mapped to the ids in `results/mute-chats-labels.json`.

## The random anchor (written 15 November 2026, before any run)

Written after every result above and before any call to a machine in the new
condition. The code (`angle/anchor.py`, `closed/anchor_claude.py`) was written
and tested on synthetic trials first (`tests/test_anchor.py`); this section and
the code are committed and pushed before the run.

**The question.** Told «I think the answer is X», with X the person's answer,
Claude Haiku 5.5 is more accurate than alone (96.5 against 93.5 percent; d′
ratio 1.14, 1.10 against Haiku alone asked again). Two explanations predict the
same gain. Either the person's answer carries information the machine uses, or
any candidate written in the prompt makes the machine check it (a format
effect, a kind of verification the prompt induces), whatever the candidate is.
A letter drawn at random separates them: it has the format and none of the
person's information.

**Data and items.** The 1,038 items of the devil's advocate (each distinct pair
of question and answer given alone), the same 2,903 trials. For each item a
letter R is drawn uniformly among the four options, fixed by the item's id and
the seed 534 (`anchor.anchor_letter`, SHA-256, not Python's hash), independently
of the person's answer and of the key: right one time in four, and equal to the
person's letter in 254 items, as chance gives. Items in
`data/items/chatbench-anchor.jsonl`.

**Condition.** *R, the random anchor*: the prompt of the told condition,
unchanged (`devil.prompt(c, "told")`), with X replaced by R. Claude Haiku 5.5
through Claude Code in headless mode, effort `low`, the one-field schema, one
call per item, as in `closed/devil_claude.py`. The other conditions are the
runs already in the repository: alone (the run named by the devil's protocol,
and the one asked again), told. An item not answered is dropped from all
conditions.

**Measures.** On the 2,903 trials:

1. The accuracy and d′ (four options) of the machine alone, told and anchored.
2. Accuracy when the letter in the prompt is right and when it is wrong, for the
   told and the anchored machine; the machine alone on the same splits.
3. The pull of a wrong letter: how often the machine gives the letter it was
   told when that letter is wrong, minus how often it gives it alone.
4. The told machine as the random letter predicts it: the anchored accuracy
   with a right letter times the person's accuracy, plus the anchored accuracy
   with a wrong letter times its complement. A told accuracy above the
   prediction would say the person's answers carry something a random letter
   with the same correctness does not; below, that the person's wrong answers
   pull harder than random wrong letters (they are plausible distractors).

**Hypotheses and criteria.** 95% percentile intervals, 2,000 replicas
resampling people and questions together (Poisson weights), as in the devil's
advocate.

- *R1, the format*: the anchored machine is more sensitive than the machine
  alone, d′_R / d′_0 above one. Passes if the interval lies above one with both
  runs of Haiku alone.
- *R2, the person's answer*: the told machine is more sensitive than the
  anchored machine, d′_A / d′_R above one. Passes if the interval lies above
  one.
- *Reported whatever happens*, with no criterion: measures 2-4.

**What it can and cannot say.** The told and anchored runs are on the same day
and the same items, so the variability between runs weighs less on R2 than on
the comparison with the machine alone (between the two runs of Haiku alone 28
of 396 questions change outcome); it is still not in the intervals. The
expected differences are small (the told and alone accuracies differ by three
points), and R2 may not have the power to pass even if the person's answer
helps: then measure 4 says more than the test.

**Where it goes in the article.** The article says that, for a machine that is
not sycophantic, the person's answer is one more hypothesis to check.

- R2 passes: the sentence stays, with the ratio of R2.
- R2 fails, R1 passes: the article says that the gain comes from the format, a
  candidate in the prompt, and not from the person's answer.
- Neither passes: the article says that the machine told our answer is not
  distinguishable from the machine told a random letter, and the gain over the
  machine alone is not attributed to the person.

In every case, the numbers in the appendix of the article.

## The random anchor: outcome (15 November 2026)

Results in `results/anchor.json`; the answers in
`devil-results/chatbench-anchor__haiku-5.5__told.jsonl` (1,038 items, none
missing, 219 output tokens per answer on average). Intervals as
pre-registered (2,000 replicas, Poisson weights over people and questions).

- **Accuracy** on the 2,903 trials: alone 93.5 (asked again 94.5), told 96.5,
  anchored 96.1. The random letter is the key in 22 percent of the trials.
- **R1, the format, fails as written.** d′_R / d′_0 = 1.12 (1.01–1.24) against
  the protocol's run alone, 1.08 (0.98–1.19) against Haiku alone asked again on
  the same day as the told and anchored runs. The criterion asked for both.
- **R2, the person's answer, fails.** d′_A / d′_R = 1.02 (0.94–1.11).
- **Measure 2.** Told: 97.1 with the person's letter right, 95.6 with it
  wrong. Anchored: 94.2 with the random letter right, 96.6 with it wrong. Alone
  on the same splits: 92.8 and 94.4 (by the person's letter), 90.5 and 94.3
  (by the random letter).
- **Measure 3, the pull of a wrong letter.** Told −0.7 points (−2.8 to +1.2),
  anchored −0.9 (−2.8 to +0.8); with Haiku alone asked again, −0.1 and −0.2.
  A wrong letter does not attract Haiku, whether the person's or random.
- **Measure 4.** The told accuracy predicted from the anchored machine is
  95.2; the observed 96.5; difference +1.3 points (−0.6 to +3.6).
- **Descriptive.** Output tokens per answer: alone about 175, told 217,
  anchored 219. On the 254 items where the random letter equals the person's
  the two prompts are identical; Haiku's letter differs in 7.
- **Rule.** Neither passes: the article says that the machine told our answer
  is not distinguishable from the machine told a random letter, and does not
  attribute the gain over the machine alone to the person.

## After the critical review (v4): the audit (written 16 November 2026, before running it)

A critical review of version 3 of the article asked to check what was decided
before and how. Two analyses come first, because the others use their
results. Both run on data already in the repository: no model is called.

### A8, the grid of thresholds

**The defect.** Declared in "T, the calibration": the correlation is computed
on a grid of thresholds with step 0.02 (`measure._fast_rho`, and the grid `zu`
in `sdt.estimate_from_evidence`). The article says that on ChatBench it shifts
the angle by less than a degree.

**What was checked before writing this, on simulated data only.** On the
huge samples of scenarios A and E (one million trials), with the grid the
instrument gives ρ = 0.163 and 0.287 against the true 0.20 and 0.30; with
exact thresholds in the measure only, 0.194 and 0.297; with exact thresholds
in the measure and in the conversion, 0.196 and 0.297. The grid is the cause,
and in scenario A it is worth about two degrees, not less than one. No real
data were measured with the corrected estimator before this section was
committed.

**The correction.** In both places the thresholds are used exactly (unique
values rounded to 10⁻⁶) instead of being binned on the grid. Nothing else
changes: the same seeds, the same resamples, the same conversion points.

**What is reported.** All the analyses that use the angle are rerun with the
corrected estimator (`analysis`, `run_chatbench`, `frontier`, `difficulty`,
`review`, `verdict`, `summary`, `figure`; `calibration` as well). For each
comparison: the angle before and after; per family, the median and the
largest shift. The calibration criteria are evaluated again.

**Rule for the article.** Version 4 uses the corrected estimator everywhere,
and the shift goes in appendix D in one paragraph with its numbers. If a
verdict of version 3 changes (the count of intervals below 90 degrees in H1,
H3, the table of H5, the sign of the rank correlation in H7, the angles of
tables 5, 7 and 8 beyond their intervals), the work stops and the author
decides before anything is written.

### A12, hypotheses H1-H5 with the criterion as written

**The problem.** The protocol says: "Passes means that the 95 percent
interval meets the criterion". `angle/summary.py` decides H3 by comparing two
means (line 80) and H2 by comparing two medians (line 64), without an
interval. On ChatBench H3 "passes" with a mean error of 0.0570 against 0.0574.
H1 says "on every data family" without saying whether per family or per
comparison.

**What is computed**, first on the results of version 3 (an audit of what
the article says) and then on those rerun after A8. Bootstrap over the
comparisons, stratified by data family, 2,000 replicas, seed 534.

- **H1.** Two readings, both reported. Per comparison: how many have the
  interval of φ below 90 degrees (as in version 3). Per family: the interval
  of the median φ over the comparisons; the family passes if its upper end is
  below 90.
- **H2.** Per family, the difference between the median ρ machine-machine and
  the median ρ person-machine, with its interval; passes if the lower end is
  above zero.
- **H3.** The mean over the comparisons of e_φ − e_90 (the absolute errors of
  the two predictions), with its interval; passes if the upper end is below
  zero. Overall, without HAIID and per family. The rows between the two
  predictions are counted, without an interval: there are too few.
- **H4** is not counted as a test of its own: with ρ > 0 the ideal at φ is
  lower than at 90 degrees, so whenever the pair beats the person the share of
  the gain at φ is larger. It is reported as a consequence of H1.
- **H5** stays as in version 3 (it already does not pass).

**Limit, declared now.** The bootstrap over comparisons does not carry the
uncertainty of each comparison's own estimates (its angle has its own
interval): the interval of H3 is too narrow, not too wide. A two-level
resampling would require rerunning every comparison with its replicas, and
is not done.

**Rule for the article.** Table 24 and appendix D use these verdicts. If one
differs from version 3 (H3 on ChatBench is the expected case), the work stops
and the author decides how it is written.

## After the critical review (v4): the new measures (written 16 November 2026, before computing them)

They run after A8, on the results rerun with the corrected estimator. No
model is called, except in A11, which waits for the author's approval. All
the bootstraps: Poisson weights, 2,000 replicas, seed 534, unless the
analysis they extend uses others (then the same as that analysis).

**A1, intervals for the margin and the two ceilings.** For the comparisons of
tables 5 (the two ChatBench pairs), 7 (today's Claude models on ChatBench)
and 8 (αNLI): the margin of the ideal pair over the stronger of the two, with
shared cues, |S|(φ)/max(d_H, d_M) − 1, and with the verdict only (the
machine's verdict to a person with graded cues, and the other way round,
`verdict.pair`). `measure.compare` keeps the replicas of (d_H, d_M, ρ); the
shared-cue margin is computed on each. The verdict-only margin is computed
on a grid of k and φ at the comparison's d_H and interpolated on each
replica; at the point estimate the interpolation must differ from the direct
computation by less than 0.1 points of margin, otherwise the grid is
refined. No hypothesis. Rule: every margin in tables 5, 7 and 8 has its
interval; in the prose a margin never appears without it; a margin whose
interval reaches zero is written as not distinguishable from zero.

**A2, the margin in correct answers.** For each margin of A1: the accuracy of
an observer with sensitivity d_stronger·(1 + margin), minus that of the
stronger, with the same number of options; on each replica. Rule: a column
"in correct answers" in the tables; in the prose only for the figure that
matters.

**A3, the errors removed.** The share of the machine's errors removed by the
ideal pair with shared cues at 90 degrees. Claim to check: for large k the
ratio of the pair's errors to the machine's tends to exp(−d_H²/4) in a choice
among m options (from the asymptotics of the error, which goes like
(m − 1)·Φ(−d/√2)), so it depends only on the person; with the yes/no
convention d′ = 2z(a), exp(−d_H²/8). Done analytically and numerically, for
m = 2 and 4, d_H from 0.5 to 2, k from 1 to 20. Criterion: "it does not depend
on k" is written only where the relative gap from the limit is below 5
percent; elsewhere the dependence is stated. It is a property of the model.

**A4, the weight of evidence.** W = d′²/2 (in nats; 1 nat = 10/ln 10 ≈ 4.34
decibans) for H, M and S, and ΔW_SH = W_S − W_H, ΔW_SM = W_S − W_M, with
intervals from the replicas of A1. Convention: in the equal-variance Gaussian
model d′²/2 is the expected log-likelihood ratio between the two classes for
one observation; in a choice among m options it is the same quantity for the
correct option against each wrong one, a convention and not the information
of the choice. No hypothesis. Rule: ΔW stands next to the turing; the zones
are also given by the signs of ΔW_SH and ΔW_SM.

**A5, the crowd.** ChatBench, condition answer-first, GPT-4o and Llama: the
ceiling of a large crowd whose members have sensitivity d′ and pairwise
correlation ρ is d′/√ρ (classic, for pooled graded evidence and equal
sensitivities). With ρ_HH and ρ_SS from "Two people with the chat beside
them", Tu_crowd = Tu·√(ρ_HH/ρ_SS). One bootstrap over people and questions
carries d′_H, d′_S and the two ρ with the same weights. Hypothesis (from the
review brief): the crowd gains less than the single person, Tu_crowd/Tu < 1.
Criterion: the upper end of the interval of Tu_crowd/Tu below 1. Rule: if it
passes, the article says the crowd gains less than the single person, with
the numbers; if not, that the crowd's gain is not distinguishable from the
single person's. Either way it is a consequence of the model on measured
inputs, and the angles between people are those of a test that did not pass.

**A6, the person as a verifier.** In the disagreements of table 16 (the chat
gives a letter among the options, different from the person's answer alone),
following the chat is the answer "the chat is right": d_v = z(follow | chat
right) − z(follow | chat wrong). The observer without gating
(`conversations.linear_observer`, a fixed bonus calibrated on the pair's
accuracy) gives d_v0. Premium = d_v − d_v0, interval from the bootstrap of
people and questions on d_v; d_v0 is computed once at the point (its
sampling error is not carried, declared). Per declared confidence of the
answer alone, only where there are at least 30 disagreements with the chat
wrong. Hypothesis: the person verifies beyond a fixed bonus, premium > 0.
Criterion: lower end above zero. Rule: if it fails, the text says that the
person is not distinguishable from an observer with a fixed weight, and the
word "premium" does not appear.

**A7, calibration with heterogeneous questions.** One new scenario H, like A
plus: the difficulty changes from question to question (the correct
option's mean shifted by δ_q ~ N(0, 0.5²), the same for person and machine),
and on each question one wrong option attracts (its mean raised by 0.6 for
everyone). The model of the article is then misspecified, and ρ has no
generative true value: the turing is judged against its true value from the
huge sample, with the criteria of the other scenarios (error at most 5
percent, coverage between 90 and 98); the angle against the instrument's own
value on the huge sample (coverage only). Rule: in appendix C as it comes.

**A9, the devil's advocate.** (i) `devil.bootstrap` already resamples people
and questions; check: the same with weights over the items (question ×
answer given alone) instead of the questions, change of the intervals
reported. (ii) The distinct calls per condition, counted from
`devil-results/`. (iii) The two stacks: Llama 3.1 8B in FP8 on vLLM, letter
only, against the Llama of ChatBench (accuracies recomputed on the same
questions); Claude Haiku 5.5 through Claude Code. Rule: one sentence each in
the body.

**A11, contamination (runs only with the author's approval).** ChatBench's
questions come from MMLU, αNLI is public since 2020. Test 1: the options
permuted. ChatBench: the options rotate so that the correct one lands on
another letter, chosen with SHA-256 of the seed 534 and the question, as in
the random anchor; the four Claude models with the settings of the frontier
run (4 × 396 calls). αNLI: the two hypotheses swap places; the eleven open
models, letter only (11 × 1,532 forward passes on the L4). Measure: the
accuracy original minus permuted, paired by question, with a bootstrap over
questions; then k, the angle, the margin and the ceilings on the permuted
questions. Criterion: the interval of the difference excludes zero. Rule: if
it does, k and margin "on new questions" go next to the original ones; if
not, the article says that the permutation does not move k and does not
exclude that the model remembers the content. In every case the k of
today's models is presented as an upper bound. With 396 correct answers out
of 396 (Fable) the test gives at most a lower bound, and it is said. Test 2
(questions reworded) is proposed only if test 1 shows an effect.

*Before any call (16 November 2026, after the author's approval):* the code is
`angle/contamination.py` (the permuted banks, the answers mapped back to the
original order of the options, the paired difference and its bootstrap) and
`gpu/run_contamination.sh`. One correction to the paragraph above: the open
models with letter-only results on αNLI are ten, not eleven (10 × 1,532
forward passes); the ten are run.

*Outcome on ChatBench (16 November 2026, 1,584 calls, none missing;
`results/contamination.json`).* Accuracy on the 396 items, original then
permuted: Haiku 0.922 and 0.932 (difference −0.010, −0.035 to +0.015), Sonnet
0.985 and 0.975 (+0.010, +0.003 to +0.020), Opus 0.987 and 0.990 (−0.003),
Fable 1.000 and 0.997 (+0.003). As written, the criterion passes for Sonnet
only. Added after, both descriptive: Sonnet's difference rests on 4 items,
all right before and wrong after, and the exact sign test gives p = 0.125;
Haiku asked the same unpermuted items a second time (the existing rerun,
`chatbench__haiku-5.5-rerun__low.jsonl`) changes 28 items, as many as the permutation (28, 12 worse and 16
better). Rule applied: for Sonnet k and margin on the permuted items go next
to the original ones (k 2.93 against 3.17); for the others the article says
that the permutation does not move k and does not exclude that the model
remembers the content. With one error on the permuted items, Fable's k
becomes estimable there: 3.90 (3.27 to 5.49). The αNLI part runs on the L4.

*Outcome on αNLI (16 November 2026, 10 × 1,532 forward passes, none missing;
`results/contamination.json`, answers mapped back in `contamination-results/`).*
The run started at 11:28 UTC, after the L4 had been free for ten minutes
(`aspetta.sh`). Eight machines finished; two failed and were run again at
19:30 UTC with the same code and data: qwen3.5-9b, whose download was reset
by the network, and qwen3.8-27b, which did not fit in memory with the default
settings and was run with `MEM_GPU=0.98 MAX_LEN=2048`, the settings of its
original αNLI run (`gpu/run_rest2.sh`). Accuracy original minus swapped, in
points: gemma-4-26b +1.0 (−0.3 to +2.3), granite-3.3-8b +0.1, olmo-3-7b −0.3,
phi-3.5-mini +1.1, qwen2.5-3b −2.6 (−6.2 to +0.9), qwen2.5-7b −1.0, qwen3-8b
−1.0, qwen3.5-9b +0.6, qwen3.8-27b −0.5, smollm3-3b +1.1. The criterion passes
for none of the ten. Rule applied: the article says that the swap does not
move k (it moves by at most 0.07, qwen2.5-3b 0.44 to 0.50) and does not
exclude that the models remember the stories. Added after, descriptive: the
stories that change outcome go from 109 (qwen3.8-27b) to 714 (qwen2.5-3b) of
1,532, split between the two directions (the exact sign test never goes below
p = 0.14); qwen2.5-3b and qwen3-8b choose the second position in about 71
and 64 percent of the stories, before and after the swap, whatever it holds
(`second_position_share`, also added after). One limit: the original runs
used an earlier vLLM, installed unpinned and whose version is not recorded
(this run: 0.33.0). The changed stories mix the swap and the software; only
the original stories run again with the same software would separate them
(15,320 more passes, not run).

## After the critical review (v4): outcome of the audit and of A3, A5, A6, A9 (16 November 2026)

**A8, the grid.** All the analyses that use the angle were rerun with exact
thresholds (`analysis`, `run_chatbench`, `analysis_vision`, `frontier`,
`devil`, `difficulty`, `review`, `full_evidence`, `summary`, `figure`,
`verdict`, `calibration`). The model-vs-human data were missing from the
local copy and were downloaded again from the authors' repository (90 CSV
files) before `analysis_vision` ran.
- On the huge samples of the calibration the instrument now gives the true
  value: scenario A ρ = 0.196 (true 0.20; it was 0.163), angle 78.7 degrees
  (true 78.5; it was 80.6); E 0.297 (0.30); F 0.449 (0.45). The bias, the
  coverage and the verdicts of the calibration are practically unchanged
  (coverage still above 98 percent in five scenarios of six).
- Shift of the angle, per family (median; extremes): ChatBench −0.02 (−1.9 to
  +3.1); the two ChatBench pairs −0.16 (−3.0 to +1.4); HAIID images −0.3 (−1.6
  to +0.1); HAIID text +0.02 (−0.03 to +0.55); αNLI 0 (no personal
  thresholds there); the 689 vision comparisons 0 (median). No angle leaves
  its version-3 interval, except Fable 5.2 on ChatBench, which has no errors:
  its angle cannot be estimated, and the estimate sits on an edge of the
  conversion table before and after (8.1 and 126.9 degrees).
- Table 5: GPT-4o 74.5 → 74.9 degrees (58.8–85.9), Llama 75.1 → 74.6
  (62.3–82.0).
- **Two verdicts of version 3 change**, both without changing their reading:
  in H1 the ChatBench comparisons with the interval below 90 degrees go from 7
  to 8 of 20; in H7 the rank correlation on ChatBench with reasoning goes from
  −0.02 to +0.10 (p 0.97 → 0.79). Other numbers quoted in the article move:
  without the strongest open model, ChatBench letter only, −0.63 (p 0.08) →
  −0.58 (p 0.11); the vision trend −0.45 (p 0.19) → −0.62 (p 0.06), still not
  distinguishable from chance at 5 percent; vision intervals below 90 degrees
  374 → 380 of 689; estimates at the edge of the table 30 → 29.
- **Rule.** A verdict changed: the work stops, and the author decides how it
  is written.

**A12, H1-H3 with the interval** (identical verdicts on the results of
version 3 and on the rerun ones; numbers of the rerun).
- H1 per family: all four pass (median φ with its interval): ChatBench 81.4
  (79.8–84.6), αNLI 64.4 (59.6–65.3), HAIID text 64.8 (57.6–70.9), HAIID
  images 78.3 (71.6–86.9). Per comparison: 8 of 20, 20 of 20, 16 of 18, 8 of
  16.
- H2: passes in the four families.
- H3: passes overall (mean e_φ − e_90 = −0.104, from −0.123 to −0.084),
  without HAIID (−0.124), on αNLI (−0.247) and on HAIID text (−0.147); **does
  not pass** on ChatBench (−0.001, from −0.020 to +0.017) nor on HAIID images
  (−0.004, from −0.041 to +0.034). Version 3 showed the ChatBench tie in its
  table; `summary.py` had written "passes".
- **Rule.** H3 differs from what `summary.json` said for two families: the
  author decides how it is written.

**A3, the errors removed.** With four options and the article's person (58
correct answers in a hundred, d_H = 1.09), the ideal pair at 90 degrees
removes 31.6 percent of the machine's errors with k = 1, 28.5 with k = 2,
27.6 with k = 3, 26.8 with k = 5, towards the limit 1 − exp(−d_H²/4) = 25.7.
The gap from the limit is below 5 percent from k = 1.5–2 on, for d_H between
1 and 1.25; up to k = 3 for d_H = 2. "About a quarter whatever k is" holds
from k ≈ 2, not for every k. A property of the model.

**A5, the crowd: does not pass.** Tu_crowd/Tu, GPT-4o 1.04 (0.51–2.68), Llama
0.86 (0.50–1.51); Tu_crowd 1.72 (0.87–4.40) and 1.20 (0.72–2.05) against the
single person's 1.65 and 1.40. With GPT-4o, in 145 replicas of 2,000 the
correlation between two people with the chat is not positive: the crowd's
ceiling is then unbounded and those replicas are left out, so the true upper
end is higher still. The crowd's gain is not distinguishable from the single
person's.

**A6, the verifier: does not pass.** d_v: GPT-4o 1.07 (0.47–1.72) against the
fixed weight's 1.01, premium +0.06 (−0.53 to +0.72); Llama 0.67 (0.13–1.23)
against 1.03, premium −0.36 (−0.90 to +0.20). By declared confidence, where
there are enough cases, no level passes. The person is not distinguishable
from an observer with a fixed weight.

**A9, the devil's advocate.** (i) With the weights on the items instead of the
questions the intervals become narrower (80 to 100 percent of the width): the
version-3 resampling was the prudent one. (ii) Distinct calls: 1,038 in each
condition with a letter in the prompt (told, devil, anchor), 396 for the
machine alone; behind them, 2,903 trials. (iii) The two Llamas, same 396
questions: in FP8 with the letter only 45.2 percent; ChatBench's Llama asked
for the letter only 41.5, in free text 63.9. The gap with the 67 percent of
the article is mostly the answer format, not the stack.

## After the critical review (v4): outcome of A1, A2, A4, A7 (16 November 2026)

**A1 and A2, margins and ceilings with their intervals** (`results/margins.json`).
- Table 5. GPT-4o: margin of the ideal pair with shared cues 2.2 percent (0
  to 8.3), that is +0.8 correct answers in a hundred (0 to 2.8); ceiling over
  the machine 1.022 (1.000–1.083); verdict-only ceiling 1.000 (1.000–1.011).
  Llama: 16.9 percent (5.9–26.4), +6.4 correct answers (2.3–9.5); 1.169
  (1.059–1.270); verdict-only 1.096 (1.018–1.188). The verdict grid needed
  no refinement with cubic interpolation (5 × 5; the point within 10⁻⁴ of the
  direct computation).
- Table 7, margin at 90 degrees from the interval of k. GPT-4o and Llama of
  ChatBench against all the people who answered alone (59 percent): in
  version 3 their k was computed outside the code; here it comes from the
  data: GPT-4o k = 2.08 (1.81–2.41), margin 11.0 percent (8.3–14.3), +3.5
  correct answers; Llama 1.23 (1.07–1.43), 28.8 percent (22.1–36.6), +10.6.
  Haiku 5.5 8.1 (5.8–10.9), Sonnet 5.5 4.9 (3.1–6.7), Opus 5.5 4.4
  (2.7–6.2). Fable 5.2, without errors: by the rule of three k ≥ 3.50 and a
  margin of at most 4.0 percent at the point sensitivity of the people; with
  the people at the high end of their interval, as table 21 reads the others,
  k ≥ 3.13 and at most 5.0 percent, the figures of version 3.
- Table 8 (αNLI): every margin with its interval, with the key of the story
  and without the contested stories.

**A4, the weight of evidence** (nats). GPT-4o: W_H 0.58, W_M 2.73, W_S 1.58;
ΔW_SH +1.00 (0.71–1.31), ΔW_SM −1.15 (−1.72 to −0.67): augmentation. Llama:
W_H 0.68, W_M 0.95, W_S 1.34; ΔW_SH +0.66 (0.41–0.94), ΔW_SM +0.39
(0.10–0.67): synergy. The signs give the same zones as the turing, and no
interval touches zero.

**A7, heterogeneous questions** (`results/calibration-h.json`, 300
replicas). The turing: error +1.2 percent (criterion met), coverage 99.0
percent (criterion not met, by excess of prudence, as in five scenarios of
version 3). The angle against the instrument's own value on a huge sample
(71.6 degrees): mean estimate 71.8, coverage 98.3 percent. With difficulty
changing from question to question and an attractive wrong option the
instrument keeps its behaviour.

## After the third review (v4): two more analyses (written 16 November 2026, before computing them)

**A13, the machine asked again.** Haiku 5.5 answered the 396 ChatBench items
twice, unpermuted (`chatbench__haiku-5.5__low.jsonl` and
`chatbench__haiku-5.5-rerun__low.jsonl`), and changes outcome on 28 of them.
A second call is then a second opinion whose errors are only partly the same.
Measure: the correlation between the correctness of the two runs on the same
items (tetrachoric, as between two machines in `machine_vs_machine`), its
angle, and the margin of the ideal pair "Haiku plus Haiku" (k = 1, formula 2
at that angle), next to the margin of the ideal pair "people plus Haiku" at
the measured angle and at 90 degrees (frontier, table 7). Intervals: 2,000
resamples of the items, seed 534. Descriptive, no criterion. Rule for the
article: the margin of the person is reported next to that of the machine
asked again; if the second is as large as the first or larger (overlapping
intervals included), the article says that the cheapest second opinion is the
machine asked again, and that the person's margin must be read against it.

**A14, an interval for where the points go.** The decomposition of the gap
between the pair and the machine alone with GPT-4o (`where_points_go`: the
conversation, the decision, the chats without an answer among the options)
is recomputed on 2,000 resamples with Poisson weights over people and over
questions, seed 534, as the other intervals of ChatBench. Criterion: the
interval of (points lost in the silent chats) minus (points lost in the
decision) lies above zero. Rule: if it does, the article keeps "the gain is
lost more in the silent chats than in the decision", with the interval; if
not, it says that the loss is split between the two, and the abstract and the
summary change accordingly.

*Outcome (16 November 2026, `results/third-review.json`).* A13: Haiku's two
runs change outcome on 28 of 396 items; correlation of correctness 0.78,
angle 35.7 degrees (23.6 to 50.3); margin of "Haiku plus Haiku" 2.7 percent
(0.0 to 8.1), against "people plus Haiku" 8.1 percent at 90 degrees (5.8 to
10.9) and 11.8 at the measured angle. The intervals overlap: by the rule, the
article reads the person's margin against that of the machine asked again.
The simulated conversion table of the correlation is built once at the point
estimate and reused in the resamples (rebuilding it each time takes hours).
A14: silent chats −7.6 points (−10.9 to −4.8), decision −4.2 (−6.8 to −1.6);
their difference 3.5 points (−0.3 to +7.7) touches zero: the criterion does
not pass, and the article says that the loss is split between the two.

## After the fourth review (v4): one more analysis and one correction (written 16 November 2026, before computing them)

**A15, H1, H2 and H3 resampling people and questions.** A12 resampled the
comparisons. But twenty machines measured against the same people on the same
questions are not twenty independent observations, and each comparison's own
uncertainty did not enter: the intervals of A12 are too narrow (the module
said so). A15 resamples what was sampled. In each of 2,000 replicas, seed
534, every person and every item of every task gets a Poisson(1) weight
(ChatBench: workers and questions; HAIID: participants and items, task by
task; αNLI: items only, since the votes per item do not tell people apart),
and the same weights serve every comparison on that task. Every comparison is
recomputed in the replica with the code of the map (`measure._one_pass` with
the comparison's conversion table; the step of `compare_votes` for αNLI), and
every pair of machines with the same item weights.
- H1: the median angle of the family in the replica; passes if the 97.5th
  percentile of the 2,000 medians is below 90 degrees.
- H2: the median correlation between machines minus the median between
  person and machine, per family; passes if the 2.5th percentile is above 0.
- H3: the mean, over the comparisons of a set, of |d_combo − pred_φ| −
  |d_combo − pred_90|; passes if the 97.5th percentile is below 0 (sets as in
  A12: all, without HAIID, each family). The combiner is not refitted in the
  replica: its choices are reweighted, as in `combiner.margin`. The two
  predictions follow the replica's d_H, d_M and ρ through their first-order
  expansion around the point, with derivatives by central differences on the
  same simulation (fixed seed, steps 0.05 in d and in ρ): refitting and
  re-simulating in every replica would take days on this machine.
Rule for the article: where A15 and A12 disagree, A15 holds, and the article
says that the verdicts of A12 rested on intervals that were too narrow. A
verdict that passes in A12 and not in A15 becomes "does not stand out" in the
text, with the A15 interval, and goes to appendix D.

**A13 as written (correction).** The protocol of A13 asked for the margin of
"Haiku plus Haiku" with k = 1. The code took each run's own sensitivity and
the better run as the reference: the difference between the two accuracies
is noise, and the margin came out too small (2.7 percent). Corrected: one
sensitivity for both, from the mean accuracy of the two runs; the correlation
converted at that sensitivity; margin sqrt(2/(1+ρ)) − 1; interval over the
items as before, recomputing the sensitivity and ρ in each resample. The
same rule for the article as in A13. There are two identical runs, not three:
the majority of three would need a new run.

*Outcome of the correction of A13 (16 November 2026, `results/third-review.json`,
key `a13_k1`).* One sensitivity for the two runs (d′ 2.69); correlation of
correctness 0.78, angle 35.6 degrees (23.7 to 50.4); margin of "Haiku plus
Haiku" 5.0 percent (2.2 to 10.5), against 2.7 with the code of A13. Next to
"people plus Haiku", 8.1 percent at 90 degrees (5.8 to 10.9), the intervals
overlap more than before: the rule gives the same reading.

**A deviation found with the fourth review: A3 on the article's person.** A3
computed the errors removed for a person right exactly 58 times in a hundred
(d′ 1.089), while the margin table of the article (`verdict.margin_table`)
uses the ChatBench group next to GPT-4o, right 57.7 times (d′ 1.080). The two
places of the article gave two limits for one person (25.7 and 25.3 percent).
A3 now reads the person from `results/chatbench-source.json`: limit 25.3
percent, 31.3 with k = 1, 28.2 with k = 2.

*Outcome of A15 (16 November 2026, `results/joint.json`, code `angle/joint.py`
at `4008516`).* At the point every comparison gives `map.json` back. With
people and items resampled together:
- H1: the median angle passes on ChatBench (81 degrees, 74 to 87), αNLI (64,
  62 to 66) and the HAIID texts (65, 53 to 71); on the HAIID images (78, 63 to
  93) it does not. A12 gave 72 to 87 there, and passed.
- H2: passes on ChatBench, αNLI and the HAIID texts; on the HAIID images the
  difference runs from −0.35 to +0.74 and does not. A12 passed there.
- H3: the same verdicts as A12, with wider intervals (all: −0.13 to −0.08;
  without HAIID: −0.14 to −0.11; αNLI and the HAIID texts pass; ChatBench and
  the HAIID images do not).
By the rule, the article says that the two verdicts on the HAIID images do not
stand out, and moves them to appendix D.

Deviation, found while running: a comparison that falls to chance in a
replica (accuracy at or below 1/m, or a negative sensitivity of one class) is
left out of that replica only, as `measure.compare` already did with its own
resamples (one HAIID comparison kept 864 of its 2,000). It happens only on
HAIID, with 32 items per task: 9,040 cells of 68,000 there (13 percent),
none on ChatBench or αNLI; the first run stopped on it, the second ran with
this rule. A check added after the protocol, with only the comparisons that
never fall to chance, covers ChatBench and αNLI and gives the same verdicts.

**The chat figure from the results.** The article's tree of a hundred trials
with GPT-4o had its numbers written in the figure's script. They now come from
`results/chat-branches.json` (`python -m angle.conversations branches`). One
number changes: the machine alone in the branch where the chat gives another
letter is 26.85 per hundred trials, which rounds to 26.8; the script had 26.9,
from rounding an already rounded value.

## After the fifth review (v4): A16, the third run of Haiku (written 16 November 2026, before running it)

**Why.** A13 reads Haiku's two runs as two opinions, but the margin of
"Haiku plus Haiku" (5.0 percent with k = 1) is an ideal: it assumes graded
evidence, and the Claude models give only a verdict. With two verdicts the 28
items on which the runs disagree are a tie. A third run gives the majority of
three, a number measured on the verdicts, with no Gaussian assumption. The
author approved the 396 calls on 16 November 2026, on the reviewer's advice.

**Data.** Haiku 5.5 on the 396 ChatBench items, unpermuted, with the code and
settings of the two existing runs (`closed/machines_claude.py`, effort low):
`python -m closed.machines_claude run claude-haiku-5-5 haiku-5.5-rerun2
chatbench low`, saved as `closed-results/chatbench__haiku-5.5-rerun2__low.jsonl`.
The analysis uses the items answered in all three runs; missing answers are
reported.

**Measures** (`angle/third_run.py`, `results/third-run.json`).
- The accuracy of each run, and of the majority of three: the letter given by
  at least two runs. When the three letters all differ, the item counts by
  expectation, as a tie broken at random: 1/3 if one of the three is right.
- The measured margin of the majority, in sensitivity: d(majority) / d(single)
  − 1, with d from the accuracy in a choice among four (`sdt.d_from_accuracy`,
  m = 4) and "single" the mean accuracy of the three runs (one machine, one
  sensitivity, as A13 with k = 1).
- Its interval: 2,000 resamples of the items, seed 534, the three runs
  resampled together.
- Descriptive: the number of items on which the three runs split 3-0, 2-1 and
  1-1-1; the ideal of three equal observers at A13's correlation,
  sqrt(3/(1 + 2ρ)) − 1, for reference.

**Hypothesis and criterion.** The majority of three is better than a single
run: the 2.5th percentile of the measured margin is above zero.

**Rule for the text.** The measured margin goes next to the ideal one wherever
the article gives "Haiku plus Haiku" (§2, appendix D), with its interval and
the words "measured" and "ideal". If the criterion passes, the article says
how much the majority of three gains, measured. If it does not, it says that
the majority of three, measured on the verdicts, does not stand out from a
single run, and the hypothesis goes to appendix D. In both cases "people plus
Haiku" remains an ideal, and the article says that a measured margin and an
ideal one are not the same kind of number.
