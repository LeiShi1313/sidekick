# Sidekick

Sidekick is a context-aware AI agent for conversations, people, and shared
knowledge. It connects chat platforms to an agent runtime and a standalone
memory system without making either layer depend on Telegram or QQ concepts.

## Modules

- `src/sidekick/chat`: platform-neutral messages, commands, identities,
  attachments, response transport, and conversation handling.
- `src/sidekick/telegram`: Telegram adapter and history/identity integration.
- `src/sidekick/onebot`: OneBot 11 adapter for QQ groups.
- `agent`: Pi-based agent loop, web/code tools, memory tools, session history,
  and run auditing.
- `memory`: pinned Hindsight deployment and local patches.
- `playground`: browser interface for exercising the agent and memory outside
  a chat platform.
- `proxy`: local hostname routing for dashboards.

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
uv run sidekick-memory-migrate --help
uv run sidekick-memory-benchmark --help
```

Configuration can be supplied through `.env`, environment variables, or
`~/.sidekick/config.toml`. Start from the committed `.env.example` files; all
secret values are placeholders.

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
   then start each layer:

```bash
docker compose --env-file memory/.env -f memory/compose.yml up -d
docker compose --env-file agent/.env -f agent/compose.yml up -d
docker compose --env-file .env -f docker-compose.yml up -d
docker compose -f proxy/compose.yml up -d
```

The adapter compose file expects the external `memory-platform` and
`agent-platform` networks created by the memory and agent projects. Runtime
databases, Telegram sessions, model credentials, and Hindsight data stay in
Docker volumes or local ignored files. Existing TeleFire state is not migrated
automatically; move session files and adapter databases, then republish
knowledge-directory entries as an explicit deployment operation before
replacing a live stack.
