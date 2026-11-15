# turing-meter

Measures how much a person decides better with a machine beside them (the
turing) and the angle between their errors. The runner and the pair tool live
in `src/turing_meter/`, task packages in `tasks/`, analyses in `analysis/`, the
experiment of the article «Intelligenza Apparente» in
`experiments/angle-between-errors/` (its protocol, with every deviation, in
`PROTOCOL.md` and `PROTOCOL.it.md`).

## Security review

- **Every push**: `.githooks/pre-push` runs the repo skill
  `security-review-range` (headless, no shell) on exactly what is being pushed,
  and blocks on HIGH/MEDIUM findings with confidence >= 8/10, or when the review
  cannot run. Once per clone: `git config core.hooksPath .githooks`. Bypass
  (`SKIP_SECURITY_REVIEW=1 git push`) only with the maintainer's explicit ok.
  Reports in `.git/security-reviews/`. From an agent session, run `git push` in
  the background: the review takes 3-10 minutes.
- **Before every release**: `/security-review-range <released tag or
  sha>..HEAD`, always with an explicit range; tag each release.
- **Findings**: a defect in our code is fixed here, with a test that fails
  first; a defect in a dependency (Inspect, Docker, vLLM, numpy...) is reported
  upstream, never patched locally.
- Areas and exclusions: `.githooks/security-review-areas`. Hook test:
  `bash scripts/test/pre_push_hook_test.sh`.
