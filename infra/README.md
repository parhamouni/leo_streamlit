# `infra/`

Deployment artifacts for the multi-user web-app migration.

This directory will hold:

- `systemd/` — new unit files (`fence-api-v2.service`, `fence-worker.service`). The existing `fence-fast.service` and `fence-api.service` units in [../ops/systemd/](../ops/systemd/) are **not** modified.
- `nginx/` — reverse-proxy config for `api.<host>` → FastAPI on the AWS box.
- `deploy_frontend.md` — Vercel deployment notes for [../frontend/](../frontend/).
- `deployment_notes.md` — env vars, rollback, journalctl tips for the new stack.

Currently a placeholder. Populated incrementally per the migration plan at [.claude/plans/yes-i-get-you-refactored-hummingbird.md](../.claude/plans/yes-i-get-you-refactored-hummingbird.md).
