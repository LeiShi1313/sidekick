# Sidekick

Sidekick is a context-aware AI agent for conversations, people, and shared
knowledge. It connects chat platforms to an agent runtime and a standalone
memory system without making either layer depend on Telegram or QQ concepts.

## Modules

- `src/sidekick/chat`: platform-neutral messages, commands, identities,
  attachments, response transport, and conversation handling.
- `src/sidekick/telegram`: Telegram adapter and history/identity integration.
- `src/sidekick/onebot`: OneBot 11 adapter for QQ groups.
- `src/sidekick/wechat`: WeChat Linux connector client, durable event
  projection, and AI transport.
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

The WeChat adapter reads `SIDEKICK_WECHAT_URL` (default
`http://127.0.0.1:18188`) and the optional `SIDEKICK_WECHAT_TOKEN`. It requires
the connector's complete/current chat snapshot, durable event replay,
idempotent text send, and stable outbound message-ID capabilities. Its local
projection defaults to `~/.sidekick/wechat.db`. Connector history remains
partial by contract: Sidekick can backfill only messages that this connector
instance has already observed and stored, never a complete WeChat chat history.
Retained non-text rows without a canonical `senderId` are ignored because they
cannot be safely attributed; text rows still require a canonical sender.
Bounded `chat_history` rows are retained as one provenance-marked text message
under the outer stable ID; nested media remains descriptive text rather than a
separate download or message.
Keep an unauthenticated connector bound to loopback; use its bearer token (and
TLS outside a trusted local network) whenever it is reachable by another host.

When `SIDEKICK_HINDSIGHT_URL` is set, `SIDEKICK_HINDSIGHT_TOKEN` is also
required. The WeChat channel then uses the same memory commands as the other
chat adapters. `/ai` receives an account-scoped Hindsight
memory target, replying with `/ai_memory` retains that stored reply chain,
`/ai_memory_backfill days <1-30>` or `/ai_memory_backfill messages <1-5000>`
performs the bounded best-effort local backfill, and `/ai_memory_enable` starts
continuous ingestion for new messages in that chat. Continuous and Dream
ingestion are off per chat until explicitly enabled. Setting
`SIDEKICK_HINDSIGHT_URL` to an empty value disables memory; memory commands then
report that Hindsight is disabled instead of silently changing ingestion state.
A recalled or redacted WeChat message is excluded from later local reads and
backfills, but recall cannot erase a copy that was already retained in
Hindsight; that memory must be revised or removed through the memory service.

The root Compose stack declares `wechat-host-ai` and `wechat-peer-ai`. Each
worker joins only its matching bridge network, uses its own WeChat projection
and AI state databases, and shares the existing Pi agent and Hindsight services.
Start or update both workers without recreating the Telegram or OneBot adapters:

```bash
docker compose up -d --build wechat-host-ai wechat-peer-ai
```

The Compose healthcheck applies the same live capability gates and parses the
same bounded history window as the adapter bootstrap.
Optional connector bearer tokens are configured independently with
`SIDEKICK_WECHAT_HOST_TOKEN` and `SIDEKICK_WECHAT_PEER_TOKEN`.

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
