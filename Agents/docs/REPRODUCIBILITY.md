# Reproducibility

## Native evidence

A completed native run records the exact input hash, selected tool source hashes, prompt hashes, Python and dependency versions, public model configuration, Git commit, working-tree state, structured plan, raw metric artifacts, interpreted result, and report.

The manifest is evidence of what the runtime observed. It is not a substitute for archiving provider-side model versions or stochastic service behavior. For long-term reproduction, retain provider model snapshots when available and record any account-level configuration outside this repository.

## Run isolation

No generated file is shared across runs. Plans compile all mutable outputs into `runs/<run_id>`. Raw datasets and algorithm sources outside `Agents` are read-only. This prevents one run from overwriting the paper's results or becoming an implicit input to another run.

## Determinism boundary

Preprocessing and regression code follow their published behavior. LLM planning and interpretation may vary by provider version even at zero temperature. Typed schemas, validation, and metric grounding constrain this variability. EPSpec's scientific ranking model remains a configured external dependency in native mode.

## Demo boundary

Simulated outputs are deterministic fixtures derived from dataset and method identifiers. They exist to validate orchestration and parsers. They must never be cited as experiment results, compared with the paper's tables, or mixed into native result directories.

## Reproduction procedure

1. Checkout the manifest's Git revision.
2. Create an isolated environment from `pyproject.toml` and `uv.lock` when available.
3. Verify dataset and tool hashes against the manifest.
4. Configure the recorded provider model identifiers and approved credentials.
5. Run `doctor`.
6. Submit the recorded request or validated plan.
7. Compare structured metric artifacts before comparing narrative reports.

## Quality evidence

The test suite covers schema rejection, plan path containment, registry contracts, run storage, interruption and resume, cancellation state, simulator parsing, API lifecycle, metric grounding, evaluation cases, and the no-comment code policy. Build, lint, typing, and offline end-to-end checks are separate quality gates.
