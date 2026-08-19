# Security

## Trust model

Scientific tools are repository code and therefore trusted code, but they are isolated from the long-lived control process. Natural-language requests, API payloads, MCP arguments, model outputs, imported plans, and generated artifacts are untrusted data.

## Controls

- Strict Pydantic models reject unknown fields and inconsistent intent combinations.
- Plan validation constrains raw input to `Data/Raw Data` and every mutable output to the current run directory.
- Interpretation-only plans may read model results only from a validated `Agents/runs/<source_run>` directory.
- Run identifiers use a restricted character set.
- Runtime artifacts use atomic writes.
- Scientific modules execute in subprocesses with bounded time and cancellation.
- API authentication can be enabled with `EPSPEC_SERVER_TOKEN`.
- Secrets are excluded from plan and worker JSON and redacted from events.
- Configuration reporting exposes only whether credentials are configured.
- Reports cannot introduce unsupported named metric values.
- Prompt capture is disabled by default.

## Credentials

Use role-specific keys with the minimum provider permissions. Store `.env` only on the deployment host. Do not commit it. In container or service deployments, use the platform's secret manager. Rotate any key that appears in logs, prompts, worker payloads, or issue reports.

## Network

Offline simulated mode performs no model-provider calls. Native baselines do not require the scientific ranking key. Native EPSpec methods require the configured scientific endpoint. Planning and interpretation endpoints are used only when offline orchestration is disabled.

## Public deployment

Set `EPSPEC_SERVER_TOKEN`, bind behind TLS, restrict CORS at the reverse proxy, limit request body size, enforce process and memory quotas, and retain audit events. The packaged API does not provide arbitrary filesystem paths or arbitrary Python tool registration.

## Reporting a vulnerability

Open a private security advisory on the repository with the affected version, reproduction, impact, and recommended mitigation. Do not include live credentials, private spectra, or complete provider responses.
