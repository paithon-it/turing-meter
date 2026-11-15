# Usage protocol: measuring a person-machine pair before trusting it

> **English** · [Italiano](PAIR-PROTOCOL.it.md). The Italian is the original; this is its translation.

Draft 0.1, 15 November 2026. It accompanies the article "Intelligenza Apparente" and uses its quantities. The article calls the user's own measurement the "baseline measurement", because it reserves "calibration" for the calibration of the instrument itself (its Appendix C). These pages use "calibration" for the user's baseline measurement.
It has not been tried with real people: the calculations are simulations made with the estimators in `experiments/angle-between-errors/`, plus two checks on the ChatBench data. Every figure says where it comes from:
**[article]** is a figure from the article, **[simulation]** a calculation of these pages inside the
model, **[data]** a calculation of these pages on the public ChatBench data.

## In brief

- **The question.** I want to use a model, for example GPT-4o, and I do not know what it is worth
  compared with me. Is a first test needed? Yes: *k*, how many times the machine is worth the person,
  is published by no one, because it depends on the person and on the task.
- **What is measured well.** *k*. With 100 cases it falls between 1.6 and 3.1 when it is 2.2; with 200
  between 1.75 and 2.8 [simulation]. That is enough to tell a much stronger machine from one on par.
- **What is not measured with a hundred cases.** The angle between the errors: with 100 cases it falls
  between 47 and 105 degrees when it is 75. It lives on the machine's errors: to know it within ten
  degrees you need between 50 and 80 of them, that is almost 700 cases with a machine at 88 per cent
  and more than 2,000 with one at 98 [simulation].
- **The rule of use** therefore rests on *k* and on the margin ceiling, not on comparing the angle with
  the useful angle, which a hundred cases cannot place on either side.
- **The test of the pair** is done on the same cases as the calibration, not on new cases: 200 cases
  tell whether the pair stands below the machine when the yield is 0.76 as with GPT-4o.
- **Repetition** is asymmetric. The machine is remeasured at every version, at no cost. A single
  person, with a hundred cases at a time, sees a drop of their own only if it is a collapse; an
  organization sees it.
- **The gravest pitfalls**: wrong keys when the machine is strong, cases the model has already seen, a
  single *k* where the kinds of case diverge, and the chat machine not being the calibration machine.

## 1. What it is for, and what it is not

It is for people who make decisions that have a verifiable right answer (a diagnosis later confirmed, a
classification, a check with a known outcome) and want to know how to divide the work with a model:
follow it, check it, or decide together.

It is not for work that has no key: writing, designing, choosing which question to ask. The article's
instrument measures two techniques on the same question, and whoever chose the question stays outside
it [article]. It does not measure time, unless it is timed separately. And it does not say what will
happen in six months: it holds for those cases, that version of the model, that way of querying it.

## 2. Four things to know first

**1. *k* is yours, and it changes with the task.** In the ChatBench group that worked with GPT-4o the
people's true accuracies have a standard deviation of 15 points around 58 per cent
[data, from the file `conversations.json` of the repository]. Someone one standard deviation below the
mean faces a machine worth 3.9 times them; someone one above, 1.5 times. And for the same people GPT-4o
is worth 2.33 times in mathematics, 3.18 in physics and 1.08 in ethics [article]. A *k* averaged over
everything describes no one.

**2. The machine's sensitivity is not a property of the model.** It belongs to the model on those
cases, with that prompt. Whoever produces the model can publish how it answers an open test; how much it
is worth on your cases you measure yourself.

**3. The angle rests on a single count.** Of the cases the machine gets wrong, how many do you also get
wrong? On ChatBench people get wrong 57 per cent of the questions GPT-4o gets wrong, and 40 per cent of
those it gets right [data]. If the two shares were equal the errors would be independent (90 degrees);
the further apart they are, the more the angle closes. This is why the precision of the angle depends
on how many errors the machine makes, not on how many cases you have.

**4. Calibration measures the machine alone, not the chat.** In ChatBench Llama 3.1 8B answered
correctly in 85 per cent of the conversations that produced an answer, and in 73 when asked alone on the
same ones; with GPT-4o, in more than a fifth of the trials no answer came out of the chat [article]. To
know how the pair does, you have to test the pair.

## 3. The design

### 3.1 The cases

- **Kinds of case.** First list the kinds of work the model will be used for. Each kind has its own
  calibration and its own rule.
- **Where they come from.** From real work, drawn at random: not the interesting cases, not the ones
  that are easy to find. Never from public tests, and if possible later than the date at which the model
  stopped learning.
- **The key.** The right answer must be fixed before anyone answers, from a source that is neither the
  person nor the machine (an outcome later ascertained, a second expert). A tenth of the keys, at random,
  must be reviewed by a second reader who does not know who answered what: the share of keys that do
  not hold up is a number in the report. After the test, all the keys of the cases the machine got wrong
  are also reviewed. This corrects only the errors against it: how much remains on the other side is
  told by the share found in the random tenth.
- **The format.** Four options, if the work allows it (section 3.5).
- **How many, and how to divide them.** For one person: 300 cases per kind, divided at random into three
  modules of 100. Module A is for calibration, B for testing the rule, C is kept aside for repetition.
  Randomly dividing a single pile makes the modules equally difficult on average: this is what makes
  it possible to compare the person of today with the person of a year from now. Module C must not be
  given to the machine until it is needed.
- **Order.** The cases are presented in random order, different for the person and for the machine.

### 3.2 The session

1. **The person alone.** They answer all the cases of the module without seeing the key or the machine.
   For each case they note the answer, the confidence on three levels and the time.
2. **The machine alone.** The same cases, one per call, with no memory between one case and the next,
   with a fixed prompt. Note: name and date of the version (not the generic name, which can move to a
   new version without notice), prompt text, temperature, tools switched on. One call per case is
   enough and does not bias the estimates; three or five tell you more about where the machine changes
   its answer (section 7).
3. **The person with the machine.** On the same cases, in a second round and in a different order, the
   person uses the chat as they would really use it. Note the answer that came out of the chat (or that
   none came out), the final answer and the time. Between the first round and the second no key is
   shown.
4. **Only at the end** are keys and results opened. From that moment the module's cases, for that
   person, are spent.

The third step is done on the same cases, not on new ones, for three reasons: the turing and the yield need all three answers on the same
cases; it is the design of ChatBench, the source of the figures to compare with; and it halves the
cases. The price is that the answer with the machine is a second attempt. In the data it weighs little:
where the chat stayed silent people stay at 50 per cent, and the order of the moves does not visibly
change the outcome [article]. New cases are needed afterwards, to test the rule (section 5).

### 3.3 How many cases: the calculations

**How they are made.** For each situation (true sensitivities, true angle, four options) 4,000
calibrations of a single person are simulated with the process in the repository, and the article's
estimators run on each one. The tables give the interval in which 95 per cent of the estimates fall:
what a single calibration can return. Check on the real data: drawing calibrations of 100 cases from
ChatBench, *k* and the angle oscillate as much as the model says (standard deviation of the logarithm
of *k* 0.171 against 0.170; of the angle 14.5 degrees against 14.8) [data].

**Person at 58 per cent, machine at 88, like GPT-4o on ChatBench.** True: *k* 2.16, angle 74.5
degrees, useful angle 62.5, margin 2 per cent.

| Cases | Machine errors | *k* | Angle | Measured margin | Zone A decided | "Angle above the useful angle" decided |
|---|---|---|---|---|---|---|
| 50 | 6 | 1.5-3.8 | 13-127 | 0-57% | 45% | 8% |
| 100 | 12 | 1.6-3.1 | 47-105 | 0-25% | 72% | 14% |
| 200 | 23 | 1.75-2.8 | 56-95 | 0-14% | 94% | 24% |
| 400 | 47 | 1.9-2.6 | 62-88 | 0-9% | 100% | 44% |
| 800 | 93 | 1.95-2.4 | 66-83 | 0-6% | 100% | 72% |
| 1,600 | 187 | 2.0-2.3 | 68-81 | 0-5% | 100% | 93% |

"Zone A decided" is the share of calibrations in which the whole interval of *k* lies above 1.5
(section 4). The last column is the share in which the interval lets us say that the angle exceeds the
useful angle, that is, that the person adds something at least in principle: with a hundred cases it
happens one time in seven, and the margin at stake is 2 per cent.

**On par, like GPT-4o on the ethics questions** (71 and 74.5 per cent). True: *k* 1.08, angle 70
degrees, margin 17 per cent.

| Cases | Machine errors | *k* | Angle | Measured margin | "*k* includes 1" | "Margin above zero" decided |
|---|---|---|---|---|---|---|
| 50 | 13 | 0.74-1.62 | 41-107 | 0-57% | 94% | 25% |
| 100 | 26 | 0.82-1.43 | 50-93 | 3-40% | 92% | 70% |
| 200 | 51 | 0.90-1.30 | 55-86 | 6-32% | 86% | 96% |
| 400 | 102 | 0.94-1.24 | 59-81 | 9-27% | 79% | 100% |

**The situations in between and the stronger machines.**

| True situation | With 100 cases | With 200 cases |
|---|---|---|
| *k* 1.18 (Llama 3.1 8B: 60.5 and 67%) | *k* 0.85-1.68; not distinguished from 1 in 87% | *k* 0.94-1.50; not distinguished in 74% |
| *k* 1.5 (65 and 82%) | *k* 1.13-2.05; "stronger machine" decided in 79% | *k* 1.22-1.83; decided in 99% |
| *k* 2.5 (machine at 93%) | *k* 1.85-3.8; zone A in 93%; angle 54-127 | *k* 2.0-3.3; zone A in 100%; angle 67-119 |
| *k* 3.2 (machine at 98%) | *k* 2.3-6.1; zone A in 97% (exact interval); angle unreadable | *k* 2.6-4.6; zone A always; angle unreadable |

**What can be said, then.**

- **With 50 cases**: who is stronger, if the distance is large. Nothing else.
- **With 100 cases**: *k* within about a third. A machine worth at least two and a half times the person
  (zone A in 93 per cent of calibrations) is told apart from one on par (the interval includes 1 in 92).
  With *k* around 2, zone A comes out seven times in ten. The angle cannot be read.
- **With 200 cases**: *k* within a quarter. Zone A comes out in 94 per cent of cases with *k* 2.2. On
  par, one can say that the ideal pair has a margin above zero. The yield of a pair like the ChatBench one can be told apart from 1. The angle still has an interval of forty degrees.
- **With 400 cases**: the angle within 13 degrees with a machine at 88 per cent.

**The practical rule for the angle.** Its standard deviation is about 45 degrees divided by the square
root of the machine's errors (between 35 and 60 depending on accuracies), and the 95 per cent interval
is twice that on each side.

| Machine errors | Angle interval, on each side | Cases with machine at 80% | at 88% | at 93% | at 98% |
|---|---|---|---|---|---|
| 10 | about 30, or more | 50 | 83 | 143 | 500 |
| 20 | 16-23 | 100 | 166 | 286 | 1,000 |
| 40 | 11-16 | 200 | 333 | 572 | 2,000 |
| 80 | 7-11 | 400 | 666 | 1,143 | 4,000 |
| 160 | 5-8 | 800 | 1,332 | 2,287 | 8,000 |

Below 10 errors the angle should not be reported; between 10 and 40 it is an indication.

**The intervals.** For a single person the cases are resampled. In simulation this interval contains the
true value between 92 and 97 times in a hundred, for *k*, angle and margin, when there are at least 100
cases and the machine has at least a dozen errors. With few errors it does not hold: with a machine at
98 per cent and 100 cases it contains the true *k* 84 times in a hundred, and the true angle 40. For
*k*, below five machine errors, the exact intervals of the two accuracies are used, wider and always
valid. They are not the article's intervals, which resample people and cases together and are 1.3 to
2.1 times wider than necessary [article].

### 3.4 When the machine errs very rarely

With a machine at 98 per cent, in 100 cases two errors are expected, and one calibration in eight sees
none [simulation]. Three things remain possible.

1. **A floor for *k*.** Zero errors in *n* cases are compatible with a true error up to 3/*n*: this is
   the rule of three, the same one the article uses for Fable 5.2. With 100 cases and no errors, a
   person around 58 per cent has a *k* of at least 1.9 [simulation, `check_pair_tool.py`].
2. **A ceiling for the margin.** The margin at 90 degrees falls as *k* rises: at the low end of the
   interval of *k* one has the widest margin an ideal pair could promise with independent errors. With
   100 cases and a machine at 98 per cent that ceiling is below 20 per cent in 97 per cent of
   calibrations, and below 15 in 73. It is the article's construction for the Claude models (margin at
   most 11, 6.7, 6.2 and 5 per cent). The limit is the same: beyond 90 degrees, if the errors offset
   each other, the margin exceeds the ceiling, and without machine errors it cannot be ruled out. In the
   article's measurements the median angle lies between 59 and 81 degrees across all sources.
3. **More errors without more hours**, with the two-phase design (section 3.7).

What must not be done is to choose the cases because the machine gets them wrong, without weighting
them: another *k* is measured. And with machines this strong the keys become the main problem
(section 7).

### 3.5 Multiple choice, yes/no, open answers

**Four options** is the most efficient format. At equal sensitivity, with yes/no cases *k* is more
uncertain: with 100 cases it falls between 1.3 and 4.1 instead of between 1.6 and 3.1, and about 250
are needed for the same precision [simulation]. The reason is that 58 per cent on four options is 33
points from chance, while the same sensitivity on yes/no gives 70 per cent, which is 20 from it.

**Yes/no** needs two more precautions. The classes must be balanced, and the calculations must be done
within each class (the yeses and the nos of the key). If person and machine both lean toward the same
wrong answer, they err together on one class, and a single calculation reads that lean as similarity:
with a moderate common lean the angle comes out 68 degrees instead of 75 [simulation]. And the alarm is
worth less: with two options a disagreement only means that one of the two is wrong, while with four two
errors rarely fall on the same option. In the model, at GPT-4o's values, the machine errs in 2.6 per
cent of agreements and 12.5 of disagreements with two options, in 6 and 19 with four.

**Open answers.** Sensitivity is derived from the accuracy and from the number of alternatives, and an
open answer has no such number. The same accuracies, 58 and 88 per cent, give *k* 5.8 if two
alternatives are counted, 2.1 with four, 1.7 with ten, 1.3 with a thousand [simulation]: *k*, useful
angle and margin depend on a convention that no one can fix. For open-answer work what remains are the
calculations that do not ask for it: the accuracies, the share of shared errors, the alarm (agreement
means the same answer in substance, judged by someone who does not know who gave it), and the distance
in points between the machine alone and the pair, as in the article's "Where the points go" table. Or
the cases are turned into closed choices, knowing that recognizing the right answer among four is not
the same job as producing it.

### 3.6 One person or a group

A group that pools the cases measures the group's *k* and the group's angle. Simulation with the
article's instrument, people at 58 per cent on average with ChatBench's dispersion, machine at 88, two
people more similar to each other than either is to the machine (66 degrees against 75 [article]). There
are 300 calibrations per row, which is why the first row differs slightly from the table in section 3.3:

| Design | Human answers | Cases with a key | Group *k* | Angle |
|---|---|---|---|---|
| 1 person, 100 cases | 100 | 100 | 1.6-3.1 | 52-105 |
| 1 person, 500 cases | 500 | 500 | 1.9-2.5 | 64-87 |
| 5 people, the same 100 cases | 500 | 100 | 1.6-3.5 | 59-93 |
| 5 people, 100 different cases each | 500 | 500 | 1.6-3.4 | 65-86 |
| 20 people, the same 100 cases | 2,000 | 100 | 1.8-3.0 | 63-87 |
| 20 people, 25 different cases each | 500 | 500 | 1.8-2.9 | 64-87 |
| 20 people, 100 different cases each | 2,000 | 2,000 | 1.8-2.7 | 68-80 |
| 50 people, 40 different cases each | 2,000 | 2,000 | 2.0-2.5 | 68-81 |

- **For the angle, distinct cases count**, that is, the machine's distinct errors. Twenty people with 25
  different cases each read the angle like one person with 500 cases. Having more people answer the same
  cases helps, but less and less: on the same 100 cases the standard deviation goes from 14 degrees with
  one person to 9 with five, 7 with twenty, 6 with fifty.
- **For the group's *k*, people count**: five are not enough, however many answers they give, because
  each has their own sensitivity.
- **If keys are scarce**, give everyone the same cases. **If hours are scarce**, give different cases.
- **Each person's *k*** remains their own: the machine's sensitivity comes from all the cases, the
  person's from their own. With 100 cases each, nine tenths of what distinguishes one person from
  another is a true difference; with 25 cases seven tenths, with ChatBench's nine cases per person less than half
  [data]. With 25 cases a person's *k* has an uncertainty of almost a third: not enough to assign them
  a zone.
- **Each person's angle**, with a group, is not available: what you get is the group's. Whether the angle
  is a property of the task or of the person we do not know (section 10).

### 3.7 The two-phase design

The person's hours are expensive; the machine's are free. Where an archive of cases with the key exists (a thousand closed files, each with its known outcome), the machine does them all. The person does 200, drawn at random within
two strata: half among the cases the machine got wrong, half among those it got right, without knowing
which are which. Each answer weighs as much as the archive cases it represents. With the same 200
answers [simulation]:

| Machine | Archive | Random cases: angle | Two-phase: angle | *k*, random | *k*, two-phase |
|---|---|---|---|---|---|
| at 88% | 1,000 | 55-94 | 63-86 | 1.8-2.7 | 1.7-3.0 |
| at 93% | 1,000 | 67-120 | 78-102 | 2.1-3.2 | 2.0-3.3 |
| at 98% | 1,000 (20 errors) | unreadable | 74-109 | 2.6-4.3 | 2.6-4.1 |
| at 98% | 3,000 (60 errors) | unreadable | 80-101 | 2.6-4.3 | 2.5-4.2 |

The angle gains as much as it would with two or three times the cases; *k* loses little, and nothing
when the machine is strong. Putting all the machine's errors and few right cases in the sample does not
pay: the angle does not improve and *k* gets worse. With only 100 answers *k* gets too much worse (from
1.7-3.0 to 1.6-3.3). The design has a second advantage: reviewing the keys of the cases the machine got
wrong is a check of the keys done where it matters most.

## 4. The rules of use

### 4.1 The zones

The rule is taken from the interval of *k*, for each kind of case.

| Zone | Interval of *k* | Margin ceiling at 90 degrees | What it means |
|---|---|---|---|
| A | all above 1.5 | at most 20% | The machine is clearly stronger |
| B | above 1, not all above 1.5 | between 20 and 41% | Stronger, not by much |
| C | includes 1 | 41% | They cannot be told apart |
| D, E | all below 1 | as B and A, with the roles swapped | The person is stronger |

The 1.5 threshold is a convention. It corresponds to a ceiling of 20 per cent with independent errors
and to a margin of 8.5 per cent at 75 degrees, the angle measured between people and GPT-4o. Whoever wants a lower
ceiling moves the threshold: 1.8 for 15 per cent, 2.2 for 10. In the article's data all the pairs with a
machine twice as strong, or more, stayed below the machine (yield between 0.66 and 0.76); between 1.6
and 1.9 the yield goes from 0.81 to 1.29 depending on the machine. Two machines alone do not fix a
threshold.

**The margin read in errors.** The margin is in sensitivity, and people at work count errors. At 90
degrees a person at 58 per cent always adds the same weight of evidence to the ideal pair, and the cut in
errors is almost the same at every *k*: between 26 and 30 per cent of the machine's errors. At an angle
of 75 degrees the cut narrows: 17 per cent with *k* 1.2, 8 with *k* 2, 2 with *k* 3 [simulation]. In
absolute numbers, with a machine at 98 per cent the ceiling is worth half an error per hundred cases.

### 4.2 What the numbers allow us to say

**In zone A.**

1. **The machine's answer stands, even in disagreement.** In the 464 disagreements between people and the
   chat, GPT-4o's chat was right 86 times in a hundred and the person 10. Anyone who had always followed
   it would have answered correctly 86 times in a hundred; people stopped at 77, and where they did not
   follow it they were right one time in four [article]. In the model, someone who has their own cue and
   only the verdict of a machine 2.2 times more sensitive cannot do better than always follow it
   [simulation].
2. **Even when very sure.** In mathematics, among those who said they were very sure, the person was
   right in 10 disagreements out of 46 and the chat in 35 [article]. The model gives the same picture: 19
   and 74 per cent. The rule "I follow the machine, except when I am very sure" costs 5 points of
   correct answers at GPT-4o's values, and 7 with a machine at 98 per cent [simulation].
3. **The disagreement is an alarm.** GPT-4o's chat erred in 3.5 per cent of agreements and 14.4 of
   disagreements, and three of its errors in four fell in a disagreement [article]. The alarm sounds in
   41 per cent of cases, and in those cases the machine is still right more than eight times in ten: it
   serves to choose what to verify, not whom to believe.
4. **Make sure an answer comes out.** Of the 11 points the pair lost against GPT-4o alone, almost 8 lie in the
   chats from which no answer came out; there the machine alone answered correctly 84 times in a
   hundred and people stayed at 50 [article]. It is a description, not a proven cause, but the remedy
   costs nothing: give the machine the whole case, with the options, and if it does not answer, ask
   again, without falling back on your own answer.
5. **Checking the machine adds no sensitivity.** With a yield of 0.76 the pair gives little more than
   half the weight of evidence of the machine alone, even at equal time [article]. If the signature
   stays, it is for accountability, or because the two errors do not cost the same (section 7).

**In zone C.**

1. **Neither is in charge.** On the ethics questions the disagreements split almost evenly: 49 and 37
   per cent with GPT-4o, 43 and 52 with Llama [article].
2. **Agreement is the useful part.** In the model, on par, the machine errs in 9 per cent of agreements
   and 52 of disagreements [simulation].
3. **A disagreement is decided with something else**: more time, a third source.

**In zone B** the numbers give no rule. The model promises the ideal pair between 8 and 40 per cent of
sensitivity, depending on *k* and the angle, and the article's data at these values of *k* go in both
directions.
Cases are added, and the decision is made on the test of the pair.

**In zones D and E**, with the person stronger, there is one datum and it goes in the direction of
caution: with Llama on ethics (*k* 0.49) people fell to 0.91 turing, and followed the machine's wrong
answers 62 per cent of the time against 26 for the model [article].

**What each rule yields, in the model.** Correct answers per hundred, four options [simulation,
`sim_rules.py`]. The person's confidence here is the distance between their first and second cue: a
person who knows exactly how sure they are.

| Situation | Person | Machine | Ideal pair | Equal weights | Weighs the verdict optimally | "Machine, unless very sure" |
|---|---|---|---|---|---|---|
| GPT-4o on ChatBench (*k* 2.16) | 57.7 | 88.3 | 89.0 | 85.1 | 88.3 | 83.4 |
| Machine at 93% (*k* 2.5, 90 degrees) | 57.7 | 93.0 | 94.9 | 92.7 | 93.1 | 87.4 |
| Machine at 98% (*k* 3.2, 90 degrees) | 57.7 | 98.0 | 98.5 | 96.9 | 98.0 | 90.7 |
| Middle zone (*k* 1.5) | 64.5 | 81.2 | 84.5 | 83.2 | 82.2 | 81.0 |
| Llama 3.1 8B (*k* 1.18) | 60.5 | 66.7 | 73.2 | 73.0 | 70.5 | 69.2 |
| On par (*k* 1.08) | 71.2 | 74.5 | 81.2 | 81.2 | 79.1 | 77.5 |

Two readings. With a strong machine the margin is taken only by pooling graded cues: with the verdict
alone the best is to follow it, and every rule that lets the person decide loses points. On par even a
crude rule gains 3 points, and the ideal almost 7.

### 4.3 What the numbers do not allow us to say

- **"Below the useful angle, follow the machine."** True in the model, unusable as a rule: with 100
  cases you do not know which side of the useful angle you are on (it is decided one time in seven at
  GPT-4o's values, less than one in two with 400 cases). And with GPT-4o the real pair lost a quarter of
  the machine's sensitivity while standing *above* the useful angle, 75 degrees against 62. What decides
  is *k*.
- **The weights of the ideal pair.** The ratio is 4.5 at the angle measured with GPT-4o and 2.5 at 86
  degrees, the high end of the interval, while the low end already lies below the useful angle
  [article]. A number that changes like this within an interval obtained with 1,500 trials is not a rule
  for someone who has 100.
- **"On par, opinions are weighed by confidence."** It is what the model says, not what was seen. On
  ethics the yield is 1.00 with an interval from 0.66 to 1.47; three disagreements in four come from
  very sure people, and in the ethics disagreements the person is right 37 times in a hundred [article].
  In the model, on par, someone very sure is right 70 times in a hundred [simulation]: declared
  confidence is not the model's. It must be tested, not prescribed.
- **"In a disagreement, recheck."** Rechecking alone, in the data, did worse than following. No datum in the article says what makes a third source worth consulting.
- **A rule from one's own confidence.** With 100 cases the disagreements are about forty, divided over
  three levels: too few to fix a personal threshold.
- **A gain beyond the model.** That people know when the machine errs is not proven: they follow the chat
  more often when it is right, but so would someone who always weighs it the same way [article].
- **Time.** Without a stopwatch the power formula stays empty.
- **A turing without *k*.** Whoever copies a machine *k* times more sensitive is worth *k* turings
  [article].

## 5. The test of the pair

**On the same cases as the calibration** (third step of the session) one reads the turing, the yield,
and where the points go between the machine alone and the pair: the conversation, the decision, the
chats without an answer. How many cases are needed [simulation, one person]:

| True pair | 100 cases | 200 cases | 400 cases |
|---|---|---|---|
| As with GPT-4o: turing 1.64, yield 0.76 | yield 0.60-0.93; "below the machine" decided in 68% | 0.65-0.88; 95% | 0.68-0.84; 100% |
| On par, ideal pair: turing 1.27, yield 1.17 | yield 0.98-1.41; "above the machine" in 39% | 1.04-1.33; 72% | 1.07-1.28; 93% |
| On par, the person listens little: yield 1.07 | 0.85-1.37; 8% | 0.91-1.26; 11% | 0.95-1.21; 21% |

That a pair stands below the machine shows with 200 cases when the distance is that of ChatBench. That a
pair on par beats both asks for 400 cases if the pair is ideal; if it gains less it does not show. A
turing above one shows sooner: with 100 cases always in the first case, in 64 per cent in the second.

**The test of the rule**, on module B. Once the rule is fixed, the person does the round again (alone,
then with the machine applying the rule) on new cases. The answers given alone are added to the
calibration: after module B the cases for *k* and for the angle are 200. The yield of module B is
compared with that of module A, and the difference between two yields is more uncertain than either: with
100 cases per module only a jump of 0.33 is seen (from 0.76 to more than 1.05), with 200 of 0.23, with
400 of 0.17 [simulation]. A single person cannot see a small improvement.

**Time.** With the two times per case, the pair's power relative to the person is the turing squared
times the ratio of the times; relative to the machine that is delegated to without looking, the yield
squared times the ratio of the times [article, formulas 8 and 9]. Checking pays off only if the yield
exceeds the square root of the ratio between the time of checking and that of delegating: with a check
that lasts six times the delegation, a yield of 2.4.

## 6. Repetition

**The machine.** At every new version the modules already used are given to the machine again: the
person's answers are in the archive, and *k*, angle and alarm are redone without an hour of work. This
holds if the cases have not ended up in the new version's training: it depends on the conditions under
which they were sent. Between one version and the next a sentinel is enough: thirty cases given again
every month. With the noise between two calls of GPT-4o on ChatBench (one answer in twenty changes
outcome), six or more changed answers out of thirty happen less than one time in a hundred [data]: they
are the signal to redo the calibration.

**The person.** Measured on new cases (module C), with the turing anchored to the first day and the
ratio between the sensitivity alone then and today [article]. But a single person sees little
[simulation, person at 58 per cent, two different modules]:

| Drop in sensitivity alone | Correct answers, from 58% to | 100 answers at a time | 200 | 400 | 1,000 | 3,000 |
|---|---|---|---|---|---|---|
| 10% | 54.3% | 8% | 10% | 16% | 34% | 76% |
| 20% | 50.9% | 17% | 28% | 50% | 87% | 100% |
| 30% | 47.4% | 33% | 54% | 84% | 100% | 100% |
| 40% | 44.0% | 52% | 79% | 98% | 100% | 100% |

The percentages are the probability of noticing the drop. With 100 cases at a time a person sees, one
time in two, only a drop of 40 per cent. An organization that pools 1,000 answers at a time (ten people
with 100 cases) sees a drop of 20 per cent almost nine times in ten, and with 3,000 a drop of 10 per cent three times in four, if the drop is common to all. The anchored turing is an organization-level measure.

**When.** The machine: at every version, and when the sentinel sounds. The person: once a year, or when
the way of working changes. The modules wear out: three modules are three measurements.

## 7. The pitfalls

| Pitfall | How much it shifts | What to do |
|---|---|---|
| **Wrong keys** | Machine at 88%: with 2% of wrong keys the angle goes from 74 to 71 degrees, with 5% to 66. Machine at 98%: with 1% from 90 to 80 degrees, with 2% to 74, and *k* from 3.2 to 2.9; a third and then half of its "errors" are errors of the key [simulation] | Blind second reading on a tenth of the keys and on all the cases the machine gets wrong; report the share that does not hold up |
| **Cases the model has already seen** | True machine at 80%: if it knows four cases in ten, 88% is measured and *k* goes from 1.76 to 2.15. On par, with six cases in ten seen, *k* goes from 1.08 to 1.59: from on par to stronger machine [simulation] | Own cases, never published, later than the training; module C must not be sent before use |
| **Mixed kinds of case** | GPT-4o is worth 3.18 times people in physics and 1.08 in ethics [article]; in the example sheet the total gives zone A, one kind of case gives zone C | One *k* and one rule per kind; at least 100 cases per kind |
| **Cases too hard for the person** | With 100 cases, the typical error of the sensitivity is 15% at 58 per cent correct answers, 30% at 40, 42% at 35 [simulation]: *k* and turing explode | Choose kinds of case where the person is at least at 50% on four options (75% on yes/no) |
| **Cases too easy** | At 98% one calibration of 100 cases in eight sees no errors; the angle does not exist | Floor for *k* and margin ceiling; two phases |
| **Version, prompt, temperature** | On ChatBench, between two identical calls, GPT-4o changes outcome on one question in twenty and Llama on one in five; but eight tenths of GPT-4o's error repeats, and the majority of five calls gains half a point [data] | Note the dated version, prompt, settings. Asking twice says where the machine is unstable, it does not correct its errors |
| **The person who remembers the cases** | If they remember the key of three cases in ten, *k* goes from 2.16 to 1.55 [simulation] | Each module only once; no key before the end |
| **The machine in the chat** | Llama: 85% in chat, 73% alone. GPT-4o: more than a fifth of trials without an answer [article] | The test of the pair, with the chat's answer noted case by case |
| **The two errors do not cost the same** | Sensitivity does not see it. Example calculation: positives one in ten, an omission costs twenty false alarms. A pair with yield 0.76 that moves the criterion costs 48 per hundred cases; the machine with a neutral criterion 35, with a cautious criterion that cannot be moved 55 [simulation] | On yes/no, count omissions and false alarms separately for person, machine and pair, and weigh them with your own costs and the true frequency of positives |
| **Dependent cases** | Ten files of the same client are not ten cases: the true intervals are wider | One case per client or per file; otherwise resample by groups |
| **Peeking and extending** | Lengthening the calibration until the hoped-for zone comes out raises the probability of getting the zone wrong | Fix the steps beforehand (100, then 200) and the zone that makes you stop |

On the keys there is a confirmation in the data. On αNLI, where in one story in ten the majority of
voters contradicts the key, the angle between people and Fable 5.2 is 47 degrees with all the stories
and 68 without the contested ones [article]. In simulation, on two-option cases with a person at 90 per
cent and a machine at 96, 10 per cent of wrong keys brings the angle from 68 to 33 degrees, and 2 per
cent to 53.

A contested key is not necessarily wrong. Where the second reader does not confirm the key, the
calculations are given twice, with all the cases and without the contested ones, as the article does.
The two readings are two extremes: removing cases by looking at the answers of the one being measured
opens the angle even on its own [article].

## 8. Who does what

**Whoever produces the model** cannot publish *k*, which is not theirs. They can publish what lets
anyone calculate it on an open test:

- **the answers question by question**, not only the accuracy: question identifier, version of the test
  and of the key, answer given, and the distribution of the answers over several calls (ChatBench has
  fifty per question);
- **the conditions**: dated version, prompt text, temperature, tools switched on, date;
- **on tests that have people's answers case by case**, because the angle with a reference population
  can only be calculated that way; and the angle with the previous version, which says how fast a
  calibration ages (Opus 5.5 and Fable 5.2 have a correlation of 0.95 [article]);
- **how often no usable answer comes out** in chat-style use: in the data it is the biggest loss, and
  no model card reports it;
- **results on new questions**, held by a third party and published only once the questions are retired: an open
  and repeatable test is also a test that the next model has read.

With these public answers, whoever uses the model is left with only their own half to do: answering 100
or 200 questions of that test, and getting *k* and the alarm without querying the machine. It is an
overestimate, because the test is public.

**Whoever uses the model** measures what is theirs: their own sensitivity on their own cases, the
confidence, the time, the pair. They keep the sheet and redo it at every version.

**An organization** does what a single person cannot:

- the archive of cases with the key, the second reading, the modules drawn at random, the cases kept secret;
- the machine queried once for everyone, with the conditions fixed and the answers kept;
- the two-phase design and the group, which give the angle;
- a *k* per kind of case and per band of experience, and the rule that follows: who signs, when a
  check is mandatory, who gets a disagreement;
- repetition, the sentinel, the turing anchored on the group;
- the decision, taken beforehand, about who sees each person's results: they are measures of a person,
  and with a few dozen cases each they are too uncertain to judge them.

## 9. The format

**The card.** One page: [`PAIR-CARD.md`](PAIR-CARD.md). What you need, the four steps, what is read with
100 and with 200 cases, the rule by zone, what not to read, when to redo it, five mistakes.

**The sheet.** A CSV with one row per case: [`examples/pair-sheet-template.csv`](examples/pair-sheet-template.csv).
Columns: `case`, `stratum`, `key`, `person`, `confidence`, `machine`, `chat`, `pair`, `seconds_alone`,
`seconds_pair`, and `weight` for the two-phase design. Only `key`, `person`, `machine` are required.
The program also accepts Italian column names (`caso`, `strato`, `chiave`, `persona`,
`sicurezza`, `macchina`, `chat`, `coppia`, `secondi_solo`, `secondi_coppia`, `peso`).

**The program.** `turing-meter pair`, in the repository's package:

    turing-meter pair docs/examples/pair-sheet-example.csv --options 4 --delegate-seconds 20

It returns, in English, the two sensitivities, *k* with its interval (resampling of the cases, and exact
when the errors are few), the angle with its interval and the "indicative" or "not estimable" warning,
the useful angle, the margin, the ceiling, the same numbers in errors per hundred, the alarm, the table
by confidence, turing and yield, where the points go, the power, *k* per kind of case with the warning
if the zones diverge, the zone and the rule. A complete report is in
[`examples/pair-report-example.txt`](examples/pair-report-example.txt) (fake sheet, 200 cases,
[`examples/pair-sheet-example.csv`](examples/pair-sheet-example.csv)). The kinds-of-case warning there
fires: zone A on the total, zone C on the atypical cases (`atypical`). With `--json file` it also
writes the measurement as JSON.

Tested on sheets whose truth is known (`analysis/pair-calibration/check_pair_tool.py`, and on a small
scale the tests in `tests/test_pair.py`): with 6,000 cases it recovers *k* 2.16, angle 74.4 and margin
2.0 per cent; on yes/no with a common lean it has no systematic error; with the weights of the two
phases it narrows the angle's interval from 39 to 23 degrees; with zero machine errors it gives a floor
for *k* and does not report the angle.

**What was already there and what was added.** The article's calculations live in
`experiments/angle-between-errors/angle/` and that folder stays as it is. The package
`src/turing_meter` took from there the functions for *m* options (`mchoice.py`, a copy of
`angle/sdt.py`) and built `pair.py` on top of them.

| Need | In the experiment | Added in the package |
|---|---|---|
| Sensitivity with *m* options, tetrachoric, ideal, useful angle | `angle/sdt.py` | Ported to `mchoice.py`, unchanged |
| Conversion from the correlation of the correct answers to that of the cues | `sdt.convert_rho_mafc`, `measure.conversion_table` | For a single person the first is used, with the exact threshold. The second rounds the thresholds, an approximation: with a machine around 98% its table stops growing with the correlation and no longer inverts reliably |
| Angle and intervals for many people with few cases | `measure.compare` | For a single person `measure.compare` discards one resample in three and has no exact intervals for few errors: here resampling of the cases and exact intervals |
| Floor for *k* with zero errors | `conversations.zero_errors` | A general function |
| Margin with interval, ceiling, errors per hundred | Only the formula | Everything |
| Agreement, disagreement, confidence, where the points go | `conversations.py`, tied to ChatBench's columns | The version on any sheet |
| Weights for the two phases | `_one_pass` accepts weights | Resampling by stratum with the weights |
| Reading the sheet, report, zones | Nothing | Everything |
| Yes/no for a single person, with the two kinds of error | `measure._one_pass` with the classes | The sensitivities and the correlation per class |
| Group: each person's *k* shrunk toward the mean | `sdt.shrink` for the thresholds | Only in simulation (`sim_group.py`) |
| Tests | The experiment's `tests/` | `tests/test_pair.py`, and the coverage of the interval for a single person in `analysis/pair-calibration/` |

**Where it lives.** Not in the experiment's folder: that is the article's record, with the protocol
written beforehand, the dated deviations, the cited results. A tool for those who use the model would
carry estimators that change (the exact threshold, the exact intervals) next to those that produced the
published numbers. The program is in the package (`src/turing_meter/pair.py`, command
`turing-meter pair`), the experiment keeps its frozen copy of `sdt.py`, the simulations of these pages
are in `analysis/pair-calibration/`, the protocol and the card in `docs/`.

## 10. What we do not know

1. **Whether the angle belongs to the task or to the person.** In the data no one has enough cases for an
   angle of their own. Using the group's angle for a person is a hypothesis.
2. **Whether a real pair takes the margin when there is one.** On par the model promises between 3 and 7
   points; in the data the yield is not distinguishable from one. Whether weighing opinions by declared
   confidence works with real people, who are not calibrated as the model supposes, no one has measured.
3. **What to do when the alarm sounds.** The data say what does not work, deciding alone. What makes a
   third source, or a second model, worth it is not there.
4. **Whether the machine's confidence is needed.** The margin in zone A is taken only with a graded cue
   from the machine. Whether the confidence a model states is such a cue is not measured here.
5. **Today's models.** No person has worked with them in the data: for them there is *k*, not a turing,
   and on exam questions the angle is not estimated. The zones, for the models actually in use, come
   from *k* alone.
6. **Today's chat.** Memory, tools, web search: whether today's chats stay as mute as ChatBench's, or
   are more compliant, we do not know. Nor do we know whether Llama's extra third in chat is owed to the people who ask.
7. **People.** The data come from workers recruited online on short tasks. For a specialist on their own
   work there is no measurement at all.
8. **Time and costs.** The instrument reads sensitivity per decision. No datum of the article has times
   case by case, or the costs of the two errors.
9. **Work without a key.** Open answers, long tasks, choices about what to ask: outside the instrument,
   and the conventions to make it fit change *k* from 1.3 to 5.8.
10. **How long a calibration lasts.** How much *k* and the angle change from one version to another, and
    whether a person's angle closes with use: the study over time has not been done.
11. **The threshold of the zones.** 1.5 is a convention, resting on two machines.
12. **How much the simulations are worth outside ChatBench.** The calculations assume Gaussian cues and
    independent cases. On the ChatBench data the oscillations of *k* and of the angle match; on other
    tasks it has not been tested, and the interval for a single person is verified only in simulation.
13. **How many keys are wrong**, in a real archive: it is known only after the second reading.

## Appendix. How to redo the calculations

The scripts are in [`analysis/pair-calibration/`](../analysis/pair-calibration/) and read the
estimators from `experiments/angle-between-errors/` without writing anything inside it; the path is changed
with the variable `TURING_METER_EXPERIMENT`. They need Python 3.11 or later, numpy and scipy (with the
repository: `uv run python analysis/pair-calibration/sim_sample.py`), plus pandas and the ChatBench data
(`python -m angle.download data chatbench`, inside the experiment) for `verify_chatbench.py`. All
together it runs in about twenty minutes on four processors; the conversion table
(`conversion-m4.npz`) is computed the first time, in six minutes. The folder's README says which
command writes which file.

| Script | What it does | Results (`analysis/pair-calibration/results/`) |
|---|---|---|
| `estimators.py` | The repository's estimators in vectorized form, the conversion table on a grid | |
| `sim_sample.py` | A single person, from 50 to 1,600 cases, six situations: estimates, intervals, coverage, readings | `sample.txt` |
| `sim_sample.py --errors` | The precision of the angle by number of machine errors | `machine-errors.txt` |
| `sim_sample.py --verify` | The fast route against `measure.compare`: same *k*, angle within a degree at the median, same interval width | `verification.txt` |
| `sim_rules.py` | The alarm and the rules in the model | `rules.txt` |
| `sim_pair.py` | Turing and yield of one person; repetition | `pair.txt` |
| `sim_group.py` | One person against a group, with `measure._one_pass` | `group.txt` |
| `sim_two_phase.py` | The two-phase design | `two-phase.txt` |
| `sim_pitfalls.py` | Keys, contamination, hard cases, yes/no, open answers | `pitfalls.txt` |
| `verify_chatbench.py` | The calibrations drawn from the real data; the machine that changes its answer | `chatbench-check.txt` |
| `minor_figures.py` | Costs of the two errors, sentinel, memory, reliability, margin in errors | `minor-figures.txt` |
| `check_pair_tool.py` | `turing-meter pair` tested on sheets whose truth is known | `tool-check.txt` |
| `make_example.py` | The fake sheet in `docs/examples/` | `docs/examples/pair-sheet-example.csv` |

Two caveats on the calculations. The fast route uses the tetrachoric with the exact threshold and the
table of `sdt.convert_rho_mafc`; the article's instrument, designed for many people with few cases, uses
rounded personal thresholds. On the same simulated calibrations the two give the same *k* and angles
that differ by a degree or less at the median, with a maximum of ten in one case in 150, where the
machine was at 93 per cent. And all the "decided" percentages depend on the threshold and on the level
of the interval: they are orders of magnitude, not guarantees.
