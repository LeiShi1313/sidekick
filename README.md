# Sidekick

Sidekick is a context-aware AI agent for conversations, people, and shared
knowledge. It connects chat platforms to an agent runtime and a standalone
memory system without making either layer depend on Telegram or QQ concepts.

## Modules

- `src/sidekick/chat`: platform-neutral messages, commands, identities,
  attachments, response transport, and conversation handling.
- `src/sidekick/telegram`: Telegram adapter and history/identity integration.
- `src/sidekick/onebot`: OneBot 11 adapter for QQ groups.
- `src/sidekick/wechat`: WeChat Linux connector integration and AI transport.
- `agent`: Pi-based agent loop, web/code tools, memory tools, session history,
  and run auditing.
- `memory`: pinned Hindsight deployment and local patches.
- `playground`: browser interface for exercising the agent and memory outside
  a chat platform.
- `proxy`: local hostname routing and the Sidekick service index.

TeleFire is not a Sidekick dependency. Telegram credentials and session files
are runtime inputs and are never stored in this repository.

## Development

Python requires 3.14 or newer.

```bash
uv sync
uv run pytest

cd agent
npm ci
npm test
```

The CLI loads only the installed Sidekick adapters:

```bash
uv run sidekick telegram ai --account default
uv run sidekick onebot ai
uv run sidekick wechat ai
uv run sidekick-memory-migrate --help
uv run sidekick-memory-benchmark --help
```

Configuration can be supplied through `.env`, environment variables, or
`~/.sidekick/config.toml`. Start from the committed `.env.example` files; all
secret values are placeholders.

The Telegram adapter accepts a comma-separated list of trusted Matrix relay bot
IDs in `SIDEKICK_TELEGRAM_MATRIX_BRIDGE_BOT_IDS`. The equivalent global
`[telegram]` setting is `matrix_bridge_bot_ids = [6332621450]`. Always use
numeric Telegram IDs; usernames are mutable. Messages from those bots are
decoded from the bridge's bold-name envelope before command parsing. Because
Telegram does not expose the Matrix MXID, each display name becomes a
non-canonical alias scoped to that bridge and Telegram chat. The alias can be
retained in the room memory bank, but it cannot receive private-bank grants or
merge across rooms.

To deny specific Telegram accounts before AI routing, set
`SIDEKICK_TELEGRAM_BLOCKED_USER_IDS` to a comma-separated list of numeric user
IDs. The equivalent global `[telegram]` setting is
`blocked_user_ids = [123456789]`. Denial takes precedence over individual
allow-list entries and open group access. Use numeric IDs because Telegram
usernames are mutable.

WeChat and OneBot/QQ AI responses receive the mandatory
`mainland-messaging-v1` output policy; Telegram does not. The policy asks the
model to audit its complete response before delivery, and guarded responses are
buffered until the run completes. `SIDEKICK_MAINLAND_BLOCKED_TERMS` can add a
JSON array of literal substrings that must never be sent, for example
`["restricted-example"]`. Matching is case-insensitive after Unicode and
visible Markdown normalization, and a match is replaced with a fixed neutral
reply without preserving an AI continuation. This first version checks final
text and attachment filenames; it does not inspect image pixels or arbitrary
file contents.

The WeChat adapter reads `SIDEKICK_WECHAT_URL` (default
`http://127.0.0.1:18188`) and the optional `SIDEKICK_WECHAT_TOKEN`. It requires
the connector's complete/current chat snapshot, durable event replay,
idempotent text send, and stable outbound message-ID capabilities. Its local
operational state defaults to `~/.sidekick/wechat.db`; it contains chat metadata
and outbound-send recovery state, not message history. The AI database stores
only the durable event cursor and pending message references. Sidekick refetches
message content from the connector when it executes work or builds context.
Connector history remains partial by contract: Sidekick can backfill only
messages retained in the connector's observation catalog, never a complete
WeChat chat history.
Retained non-text rows without a canonical `senderId` are ignored because they
cannot be safely attributed; text rows still require a canonical sender.
Bounded `chat_history` rows are retained as one provenance-marked text message
under the outer stable ID; nested media remains descriptive text rather than a
separate download or message.
Group messages prefer the connector's room-local member nickname, then the
member's global display name, then the stable WeChat ID. New memory episodes
capture that connector-provided label, while already-retained historical memory
is not rewritten.
Keep an unauthenticated connector bound to loopback; use its bearer token (and
TLS outside a trusted local network) whenever it is reachable by another host.

When `SIDEKICK_HINDSIGHT_URL` is set, `SIDEKICK_HINDSIGHT_TOKEN` is also
required. The WeChat channel then uses the same memory commands as the other
chat adapters. `/ai` receives an account-scoped Hindsight
memory target, replying with `/ai_memory` retains that stored reply chain,
`/ai_memory_backfill days <1-30>` or `/ai_memory_backfill messages <1-5000>`
performs a bounded best-effort connector backfill, and `/ai_memory_enable` starts
continuous ingestion for new messages in that chat. The configured Sidekick
owner can inspect or change a group's per-person AI cooldown with `/ai_limit`,
`/ai_limit <0-86400>`, or `/ai_limit default`; zero disables the cooldown but
keeps the one-active-run safeguard. The owner can also inspect or replace that
group's AI command with `/ai_prefix`, `/ai_prefix $ask`, or reset it with
`/ai_prefix default`. A group override replaces `/ai` only for that group;
fixed `/ai_*` management commands remain available, and their equivalent
group-prefixed forms use the override (for example, `$ask_access`). Continuous
and Dream ingestion are off per chat until explicitly enabled. Setting
`SIDEKICK_HINDSIGHT_URL` to an empty value disables memory; memory commands then
report that Hindsight is disabled instead of silently changing ingestion state.
A recalled or redacted WeChat message is excluded from later connector reads and
backfills, but recall cannot erase a copy that was already retained in
Hindsight; that memory must be revised or removed through the memory service.

The root Compose stack declares `wechat-host-ai` and `wechat-peer-ai`. Each
worker joins only its matching bridge network, uses its own WeChat operational
state and AI databases, and shares the existing Pi agent and Hindsight services.
Start or update them one at a time without recreating the Telegram or OneBot
adapters, completing the rollout gates below before advancing:

```bash
docker compose up -d --build wechat-host-ai
# Complete the generated-send rollout gates, then:
docker compose up -d --build wechat-peer-ai
```

The Compose healthcheck applies the same live capability gates and parses the
same bounded history window as the adapter bootstrap.
Optional connector bearer tokens are configured independently with
`SIDEKICK_WECHAT_HOST_TOKEN` and `SIDEKICK_WECHAT_PEER_TOKEN`.

The first upgrade to the shared inbound inbox is a controlled cursor cutover.
Stop the old worker, back up both of its SQLite databases, and verify its legacy
pending-work table is empty. Before the new worker starts, seed
`ai_inbound_sources` in the AI database with the configured
`SIDEKICK_ADAPTER_INSTANCE_ID`, the active WeChat account ID as its epoch, and
the old worker's durable cursor. Legacy message and revision tables stay
untouched but are no longer read. For rollback, copy the shared inbox cursor
back to `wechat_connectors.cursor` before starting an older worker.

### Generated-send rollout safety

The authenticated channel `/health` and `/v1/channels` responses expose
`adapter.indeterminateOutboundCount`. A non-zero value means a send may have
reached the native chat without a trustworthy receipt, so outgoing controls in
the affected chat remain fail-closed; `null` means the adapter cannot yet prove
the count. WeChat reconciles these sends from the connector's idempotency
ledger with bounded, persisted backoff. Telegram and OneBot retain genuinely
ambiguous sends in memory for deliberate operator review. The exposed value is
aggregate, so use adapter logs and active-run context to identify the affected
native chat. Do not restart either adapter while the value is non-zero; restart
clears that in-memory quarantine.

The first upgrade from a build without durable provenance needs a separate
pre-upgrade audit. The new counter has no record of ambiguous sends created by
the old build, and a zero after startup cannot prove that none exist. Before
stopping an old worker, quiesce new AI commands, verify that it has no active
runs, and wait 30 seconds for ordinary connector polling to finish. Audit the
old adapter and connector logs since the last clean start for terminal-unknown
or other post-admission send failures. Resolve every recorded request ID through
the connector send journal: proceed only when each is definitely failed or its
stable message ID and outbound event have been observed. If the audit is
incomplete or any operation remains unknown, keep the worker and affected chat
quarantined and do not treat the new worker's zero as a release gate.

After that first audit, update one provenance-aware worker at a time. WeChat
startup takes exclusive ownership of its state database, so a duplicate worker
must fail startup instead of overlapping. Verify authenticated health reports a
connector-wide zero, then smoke-test one text response and one attachment
response without a repeated command. Re-check that the value is zero after the
smoke tests before advancing to the next worker. Do not roll a WeChat worker
back to a build that predates durable generated-send provenance while the value
is non-zero or `null`: the older build cannot read the reconciliation ledger
and could treat a delayed generated echo as a manual command. Keep the new
worker running until reconciliation reaches zero, then quiesce and repeat the
active-run and send-journal audit before rollback.

### AI generation queue rollout safety

Authenticated `/health` and `/v1/channels` responses expose
`adapter.aiQueue` with content-free `queued`, `active`, `failedUnknown`,
`pendingIntake`, `oldestQueuedAt`, `oldestQueuedAgeSeconds`,
`oldestPendingIntakeAt`, and `oldestPendingIntakeAgeSeconds` fields. For a
planned restart, stop accepting new AI commands and wait for
`pendingIntake == 0`, `queued == 0`, and `active == 0`. Treat a missing (`null`)
queue snapshot as an unavailable gate, not as zero.

An older release cannot consume the durable generation queue. Before rolling
back, stop the worker, back up its AI SQLite database and WAL files, then run
the following transaction against that stopped database before starting the
older worker:

```sql
BEGIN IMMEDIATE;
UPDATE ai_inbound_work
SET status = 'unavailable',
    last_error_code = 'ROLLBACK_ABANDONED',
    lease_id = NULL,
    lease_trigger_cursor = NULL,
    current_version = NULL,
    updated_at = CAST(strftime('%s', 'now') AS REAL)
WHERE status = 'pending';
UPDATE ai_generation_jobs
SET status = 'cancelled',
    last_error_code = 'ROLLBACK_ABANDONED',
    lease_id = NULL,
    finished_at = CAST(strftime('%s', 'now') AS REAL),
    updated_at = CAST(strftime('%s', 'now') AS REAL)
WHERE status IN ('queued', 'running', 'cancel_requested');
COMMIT;
```

This deliberately abandons accepted work under the best-effort rollback
contract and prevents a later roll-forward from executing stale requests.

Mount each WeChat state database through its whole parent directory or a named
volume so SQLite WAL and ownership sidecars share the same filesystem identity.
Hard-linked databases and file-only bind mounts are unsupported.

## Containers

The stack is deliberately split by ownership so each layer can run on its own:

1. Create the external `ollama-embedding` network by starting the separately
   managed embedding stack.
2. Build Sidekick's pinned Hindsight control plane from an upstream checkout:

```bash
gh repo clone vectorize-io/hindsight ../hindsight
./memory/build-hindsight-control-plane.sh ../hindsight
```

3. Create `.env`, `agent/.env`, and `memory/.env` from their example files,
   then start each layer. Copy the same `MEMORY_API_TOKEN` into all three files,
   and copy the same `SIDEKICK_OPS_TOKEN` into `.env` and `agent/.env`. Keep the
   five Pi client tokens distinct and map each root adapter token to its matching
   agent token. Generate a separate `MEMORY_EGRESS_TOKEN` in `memory/.env`; it
   is used only between raw Hindsight and its fixed provider egress. Set each
   WeChat scope prefix to
   `wechat:account:<percent-encoded-connector-account-id>:`.

   Credential changes are an atomic migration: update all three environment
   files before restarting the memory, agent, and adapter layers. Existing Pi
   sessions without an authenticated owner binding deliberately cannot be
   resumed; start a new AI thread after this migration.

```bash
docker compose --env-file memory/.env -f memory/compose.yml up -d
docker compose --env-file agent/.env -f agent/compose.yml up -d
docker compose --env-file .env -f docker-compose.yml up -d
docker compose -f proxy/compose.yml up -d
```

Human-facing services are available through the dashboard proxy:

- `http://sidekick.localhost:18865`: service index.
- `http://playground.sidekick.localhost:18865`: agent playground.

Raw Hindsight is isolated on an internal backend network. Trusted clients use
the authenticated `memory-gateway`; its host port is loopback-only and its
health response contains no backend details. Hindsight reaches only the fixed
LLM and embedding endpoints through a separately authenticated egress gateway;
it never joins the shared embedding network. The browser-facing raw Hindsight
dashboard is intentionally not exposed. Dashboard routing is static and no
Sidekick container receives access to the Docker socket.

The adapter compose file expects the external `memory-platform` and
`agent-platform` networks created by the memory and agent projects. Runtime
databases, Telegram sessions, model credentials, and Hindsight data stay in
Docker volumes or local ignored files. Existing TeleFire state is not migrated
automatically; move session files and adapter databases, then republish
knowledge-directory entries as an explicit deployment operation before
replacing a live stack.
