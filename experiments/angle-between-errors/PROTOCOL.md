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
