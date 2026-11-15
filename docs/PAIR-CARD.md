# Before trusting a model: calibration on one page

> **English** · [Italiano](PAIR-CARD.it.md)

Draft of 15 November 2026. The detail, with the calculations, is in [`PAIR-PROTOCOL.md`](PAIR-PROTOCOL.md).

**What it is for.** To find out, for one kind of work, how many times the machine is worth you (*k*), and to derive a rule of use from it. It applies to decisions with a verifiable right answer. It says nothing about work without a key.

## What you need

- **300 cases of your own per kind of work**, split at random into three modules of 100: one for calibration, one for testing the rule, one set aside to redo it in a year. Drawn at random from real work, never published, with the right answer verified by someone who is not you and is not the machine. In closed form, with four options, if possible. A hundred cases is the minimum to start.
- **A fixed model**: the dated version, the prompt text, the settings. All written down.
- **A sheet** with one row per case ([`examples/pair-sheet-template.csv`](examples/pair-sheet-template.csv)) and the program `turing-meter pair sheet.csv --options 4` (the report is in English).

## What you do

1. **Alone.** You answer all the cases of the module without seeing either the key or the machine. For each one you note how sure you are (1, 2, 3) and the time.
2. **The machine alone.** The same cases, one per call, always with the same prompt. Three calls per case is better: you see where it changes its answer.
3. **Together.** You redo the cases with the chat, as you would really use it. You note the answer that came out of the chat (or that none came out) and your final answer.
4. **Only at the end** you look at keys and results. Those cases, for you, are spent: they are not reused.

## What you read

| Number | What it says | With 100 cases | With 200 (after the second module) |
|---|---|---|---|
| *k*, with its interval | How many times the machine is worth you | From 1.6 to 3.1 if it is 2.2 | From 1.75 to 2.8 |
| Margin ceiling | The most a perfect pair would add to the stronger partner | Yes | Yes |
| Alarm | How often the machine errs where you agree and where you do not | Yes, roughly | Yes |
| Yield over the machine | Whether together you do better or worse than the machine alone | Only if the distance is large | Yes |
| Angle between the errors | How much you err on the same cases | **No** (from 47 to 105 degrees) | **No** (from 56 to 95) |

The angle lives on the machine's errors: to know it within ten degrees you need between 50 and 80 of them. Below 40 errors it is only an indication, below 10 it should not be read.

## The rule, by zone

- **Zone A: the whole interval of *k* above 1.5.** The machine's answer stands, even when you disagree and even when you are very sure. The disagreement is an alarm: those cases go to a third source, not to your second opinion. Get the answer out: the whole case in the chat, and if it does not answer, ask again.
- **Zone B: *k* above 1, not all above 1.5.** No fixed rule. Add cases and decide on the test of the pair (item 3).
- **Zone C: the interval includes 1.** Neither is in charge. When you agree you proceed; a disagreement is decided with something else (time, a third source).
- **Zones D and E: you are the stronger one.** The roles are swapped. Take care not to be dragged along by a wrong answer well put.

If the kinds of case give different zones, the rule is given kind by kind.

## What not to read

- An overall *k*, if the kinds of case diverge.
- A turing without *k* beside it: whoever copies a machine *k* times stronger is worth *k* turings.
- The calibration yield as a promise for the chat: in the chat the machine is not the same.

## When to redo it

- **At every new version of the model**: only the machine, on the same cases (if they have not ended up in its training).
- **Every 6-12 months, or when your way of working changes**: you, on new cases kept aside.

## Five mistakes that ruin the measurement

1. Cases taken from public tests: the machine may have seen them already, and *k* rises by itself.
2. Unverified keys: with a machine at 98 per cent, one wrong key in a hundred moves the angle by ten degrees.
3. Cases picked by hand because they are interesting: they measure the interesting cases.
4. Looking at the machine's answer before giving yours.
5. Reusing cases: if you remember three in ten, *k* goes from 2.2 to 1.6.
