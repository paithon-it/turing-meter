# Plan

Milestones, each one ending with something that runs and can be checked.

| # | Milestone | Status | Done when |
|---|---|---|---|
| M0 | Specification and reference analyses | done | `analysis/michell_test.py` and `analysis/composition_test.py` reproduce `analysis/RESULTS.md` from public data |
| M1 | Task schema and validator | done | `turing-meter validate tasks/` passes on the examples and rejects a malformed task with a clear message |
| M2 | Apparent intelligence | done | `apparent.py`: `d'` from rates, accuracy or AUC; the ideal of the pair at any angle; the intelligence factor, the turing and the anchors; the verdict ideal; the angle between the errors from item-level answers; with unit tests |
| M3 | Machine runs on Inspect | | one example task runs k = 5 times on one model, with JSONL logs and the tampering check |
| M4 | Human sessions | | a small local tool that presents a task, times the person alone and with the machine (order counterbalanced), runs the check and writes the attempts; pseudonymous ids only |
| M5 | Report | | one HTML page and one JSON file rebuilt from logs: the numbers of SPEC section 5 and tests 1-4 |
| M6 | First study | | ten people, two task families (one decision, one production), two protocols (machine confidence shown or not), every decision answered by the person alone, the machine alone and the pair on the same items |
| M7 | Assimilation study | protocol v0.2, reviewed; looking for a partner | A research group that runs it (ethics committee, participants, funding: `docs/STUDY.md` section 12); the pilot of twenty people, pre-registered on its own, passes the gate; then two arms of about 70 people (a third as an extension), tests before, after one month and after three, and the difference in differences of the practice-adjusted alignment with its interval. Sample size from `analysis/assimilation_power.py` |

Open questions to settle before M4:
- consent and data handling for the people who are timed (GDPR: pseudonymous
  ids, what is kept and for how long);
- which decision family has a known chance level and enough items to give each
  person a stable accuracy (at least 40 decisions per condition);
- which reference agent to freeze for the machine alone (Inspect's ReAct agent
  with bash and python tools is the default candidate).
