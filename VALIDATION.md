# Semantic ASR v0.1.0 validation

- Validated: `2026-08-28T22:44:23.173447+00:00`
- Bootstrap source commit: `5f07a1005cd524ee9105490602eea4e0de91e44a`
- Source archive SHA-256: `8687f0ab78ae996e158fb5a045d3a2792fce621ac469b7888173ddf42ef6489b`
- GitHub Actions run: `33217887825`
- Pytest: `40 passed in 1.80s`
- Compileall, Ruff modernization/format, rights registry, wheel build, clean installation and installed CLI smoke tests: passed.
- Raw audio/model weights committed: no.
- Likely secrets detected: no.

## Claim boundary

This gate validates source integrity, deterministic algorithms, contracts, optional CPU-PyTorch auxiliary heads, packaging and application smoke paths. It does not claim measured recognition-quality improvements because real Whisper/Qwen weights and a held-out Japanese speech corpus were not executed in this gate.
