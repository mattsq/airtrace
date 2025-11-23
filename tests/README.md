# Tests Guide

This suite emphasizes behavior-first coverage backed by real components. Use this guide when adding or reviewing tests.

## Principles
- Prefer integration-style flows that exercise real data modules, models, tasks, and trainers.
- Keep assertions meaningful: validate outputs, shapes, metadata, gradients, and side effects instead of object existence.
- Minimize mocking. Mock only true externalities (filesystem, network) or compatibility seams; avoid replacing the subject under test.
- When a stub fixture is necessary, document what it covers and what it intentionally skips.

## Mocking Boundaries
- ✅ Allowed: isolating external I/O, seeding randomness, patching optional dependencies.
- ⚠️ Use sparingly: monkeypatching compatibility shims (e.g., legacy path handling). Keep counts low and scoped.
- ❌ Avoid: mocking trainers, evaluation runners, data modules, or model builders that the test is meant to validate.

## Integration Expectations
- CLI coverage should run through real data-check/dry-run flows and ONNX export/inference where applicable.
- Data loaders should read real parquet fixtures rather than mocked stores.
- Model and transform tests should confirm finiteness, non-trivial outputs, and responsiveness to input changes.

## Quality Checks
- Run `./scripts/audit_tests.sh --detailed` to monitor weak assertions and over-mocking.
- Add clear comments when introducing any stub or monkeypatch to describe its scope.
- Keep new tests aligned with the remediation notes in `docs/audit/test_integrity_audit.md`.
