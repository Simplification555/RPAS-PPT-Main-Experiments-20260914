# MASBench results and debugging snapshot — 2026-09-15

This is a partial-results and failure-evidence delivery, not a claim that all experiments passed.

| Method | Axis | Test result | Status |
|---|---|---|---|
| RPAS | breadth | 27/60 | Completed; output audit passed |
| RPAS | parallel | 16/60 | Completed; output audit passed |
| RPAS | depth | 19/60 for both tested candidates | Completed; output audit passed |
| RPAS | horizon | Not run | No valid parent |
| RPAS | robustness | Not run | Selection validity gate failed |
| AFlow | All five axes | Not run | No eligible workflow (formal_v2) |
| MaAS | depth | 16/60 | Test completed; validity thresholds failed |
| MaAS | Other four axes | Not run | No eligible checkpoint |

GAIA remains paused. Missing tests are not zero scores.

## Debugging artifacts

Extract `masbench-debug-artifacts.tar.gz` with `tar -xzf masbench-debug-artifacts.tar.gz`.
It contains experiment logs, call records, evaluation journals, candidate/selection results, run manifests, audit snapshots, and current runner/adapter source. Binary model/controller checkpoints and Python caches are excluded. Recognizable credentials are redacted; manifest hashes refer to the published bytes.

Use `outputs/formal_balanced/maas/{breadth,parallel}` for the continued runs, `outputs/formal/maas/{depth,horizon,robustness}` for the other MaAS axes, and `outputs/formal_v2/aflow` for the latest AFlow runs. Older outputs and interim audits are retained for diagnosis; their historical status may be superseded by final blocker files.

MaAS final logs explicitly report: `No MaAS checkpoint passed frozen selection validity gates; test remains unopened`. This is a gate failure, not evidence of a missing runner or a hung process. Earlier conversational diagnoses to that effect were incorrect.

Frozen settings: Qwen3.5-9B, temperature 0, top_p 1, thinking disabled, max_tokens 6144, data seed 2026, run seed 0. MaAS budget: 96 search workflows, 96 selection workflows, 60 test workflows when an eligible checkpoint exists. No gate was relaxed for this delivery.
