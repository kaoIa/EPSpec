# Operations

## Modes

`native` invokes the published algorithm implementations. It is the only mode suitable for generating scientific results.

`simulate` creates deterministic, schema-complete artifacts for integration tests, demonstrations, API verification, and deployment smoke tests. Every simulated result carries `simulated=true` and the report states that it is not a scientific result.

`offline` replaces the two orchestration LLM roles with deterministic local adapters. It does not silently change native scientific ranking. The packaged demo combines offline orchestration with simulated execution.

## Lifecycle

Runs transition through created, initialized, planning, awaiting clarification, awaiting approval, queued, executing, interpreting, and a terminal state. A run can be resumed only from an awaiting state. Terminal runs are immutable from the lifecycle API.

The API accepts a run and returns immediately with HTTP 202. Poll `/v1/runs/{run_id}` or consume `/v1/runs/{run_id}/events`. Clarification and approval payloads appear in `interruption`; submit the response to `/resume`.

## Concurrency

`EPSPEC_AGENT_MAX_CONCURRENCY` bounds comparison workers and API background workers. Each scientific algorithm runs in its own process, so imports, module globals, and runtime patches cannot leak into another tool. Start with one worker when memory is constrained.

## Timeouts and cancellation

`EPSPEC_WORKER_TIMEOUT` applies to each scientific worker. The parent checks both elapsed time and the run's cancellation flag. It terminates the worker, escalates to a kill after a bounded wait, and retains its request, partial outputs, and log.

Cancellation is immediate for queued and human-waiting runs. An executing run becomes terminal when its active worker observes the cancellation request.

## Local storage

SQLite is appropriate for a single-host public reference deployment and local reproducibility. Keep `.runtime` and `runs` on persistent storage. For multi-host or high-throughput deployment, replace the repository and checkpoint backends with service-backed equivalents before scaling workers horizontally.

Installed wheels discover a repository when launched from its root or `Agents` directory. Services launched elsewhere must set `EPSPEC_AGENTS_DIR` and `EPSPEC_PROJECT_ROOT` explicitly. The container image sets both paths to `/workspace/Agents` and `/workspace`.

## API routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/v1/capabilities` | Datasets and scientific tools |
| POST | `/v1/runs` | Create and queue a run |
| GET | `/v1/runs` | List runs |
| GET | `/v1/runs/{run_id}` | Inspect lifecycle state |
| POST | `/v1/runs/{run_id}/resume` | Answer clarification or approval |
| POST | `/v1/runs/{run_id}/cancel` | Request cancellation |
| GET | `/v1/runs/{run_id}/events` | Server-sent event stream |
| GET | `/v1/runs/{run_id}/artifacts` | Hash-addressed artifact inventory |
| GET | `/v1/runs/{run_id}/report` | Grounded Markdown report |

## Recovery

Human interrupts are durable because the graph checkpoint and thread identifier are persisted. Restart the process and issue the same resume command. A process crash during a native scientific worker may require restarting that stage; the retained worker request and log provide the audit trail.

Run `doctor` after dependency upgrades, provider changes, dataset relocation, or deployment migration.
