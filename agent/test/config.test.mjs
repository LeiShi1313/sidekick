import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import assert from "node:assert/strict";
import test from "node:test";

import {
  buildWebSearchConfig,
  loadConfig,
  parseToolGrants,
} from "../src/config.mjs";

test("builds mounted Codex search with bounded Exa fallback", () => {
  const config = buildWebSearchConfig();

  assert.deepEqual(config, {
    workflow: "none",
    allowBrowserCookies: false,
    openaiApiKey:
      "!node /app/src/codex-access-token.mjs /run/secrets/codex-auth.json",
    searchRouting: {
      providers: ["openai", "exa"],
      fallbackOn: ["transient", "quota", "network", "invalid-response"],
    },
    webSearch: { enabled: true },
    githubClone: { enabled: false },
    youtube: { enabled: false },
  });
  assert.equal("provider" in config, false);
});

test("loads the standalone agent configuration without Sidekick names", () => {
  Object.assign(process.env, {
    AI_BASE_URL: "http://provider.internal/v1",
    AI_API_KEY: "test-key",
    AI_CHAT_MODEL: "test-model",
    AI_IMAGE_MODEL: "gpt-image-2",
    AI_IMAGE_REQUEST_TIMEOUT: "180",
    AI_REASONING_EFFORT: "low",
    PI_IDENTITY_ALIAS_KEY: "test-identity-alias-key-that-is-strong",
    PI_AGENT_TELEGRAM_TOKEN: "telegram-agent-token-that-is-long-enough",
    PI_AGENT_ONEBOT_TOKEN: "onebot-agent-token-that-is-long-enough",
    PI_AGENT_WECHAT_HOST_TOKEN: "wechat-host-token-that-is-long-enough",
    PI_AGENT_WECHAT_PEER_TOKEN: "wechat-peer-token-that-is-long-enough",
    PI_AGENT_WECHAT_HOST_SCOPE_PREFIX: "wechat:account:wxid_host:",
    PI_AGENT_WECHAT_PEER_SCOPE_PREFIX: "wechat:account:wxid_peer:",
    PI_AGENT_PLAYGROUND_TOKEN: "playground-token-that-is-long-enough",
    PI_DATA_DIR: "/tmp/pi-agent-test",
    MEMORY_API_URL: "http://memory.internal:8888/",
    MEMORY_API_TOKEN: "memory-api-token-that-is-long-enough",
  });

  const config = loadConfig();

  assert.equal(
    config.engine.identityAliasKey,
    "test-identity-alias-key-that-is-strong",
  );
  assert.deepEqual(
    config.clients.map(({ id, adapterInstanceId, scopePrefix, cancelAny }) => ({
      id,
      adapterInstanceId: adapterInstanceId ?? null,
      scopePrefix: scopePrefix ?? null,
      cancelAny: cancelAny ?? false,
    })),
    [
      {
        id: "telegram",
        adapterInstanceId: "telegram-default",
        scopePrefix: "telegram:",
        cancelAny: false,
      },
      {
        id: "onebot",
        adapterInstanceId: "qq-default",
        scopePrefix: "qq:",
        cancelAny: false,
      },
      {
        id: "wechat-host",
        adapterInstanceId: "wechat-host",
        scopePrefix: "wechat:account:wxid_host:",
        cancelAny: false,
      },
      {
        id: "wechat-peer",
        adapterInstanceId: "wechat-peer",
        scopePrefix: "wechat:account:wxid_peer:",
        cancelAny: false,
      },
      {
        id: "playground",
        adapterInstanceId: null,
        scopePrefix: null,
        cancelAny: true,
      },
    ],
  );
  assert.equal(config.engine.baseUrl, "http://provider.internal/v1");
  assert.equal(config.engine.apiKey, "test-key");
  assert.equal(config.engine.model, "test-model");
  assert.equal(config.engine.imageModel, "gpt-image-2");
  assert.equal(config.engine.imageRequestTimeoutMs, 180_000);
  assert.equal(config.engine.reasoningEffort, "low");
  assert.equal(config.engine.memoryUrl, "http://memory.internal:8888");
  assert.equal(config.engine.memoryToken, "memory-api-token-that-is-long-enough");
  assert.equal(config.engine.workspaceDir, "/tmp/pi-agent-test/workspace");
  assert.equal(config.engine.auditDir, "/tmp/pi-agent-test/audit");

  delete process.env.MEMORY_API_TOKEN;
  assert.throws(() => loadConfig(), /MEMORY_API_TOKEN/);
  process.env.MEMORY_API_TOKEN = "memory-api-token-that-is-long-enough";

  delete process.env.AI_IMAGE_MODEL;
  assert.equal(loadConfig().engine.imageModel, null);
});

test("parses per-user tool grants with groups and fail-closed validation", () => {
  const allowedScopes = new Set(["wechat-peer"]);
  const parsed = parseToolGrants(
    JSON.stringify({
      scopes: { "wechat-peer": { deny: ["mcp"] } },
      users: {
        "telegram:user:419540347": { allow: ["web"] },
        "qq:user:123456": { allow: ["memory"], deny: ["fetch_content", "web"] },
      },
    }),
    allowedScopes,
  );

  assert.deepEqual(parsed.users["telegram:user:419540347"], {
    allow: ["web_search", "fetch_content"],
  });
  assert.deepEqual(parsed.users["qq:user:123456"], {
    allow: ["memory_*"],
    deny: ["fetch_content", "web_search"],
  });
  assert.deepEqual(parsed.scopes["wechat-peer"], { deny: ["mcp"] });

  assert.throws(
    () => parseToolGrants("not json", allowedScopes),
    /malformed JSON/,
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ users: { "chat:user:abc": { allow: ["nope"] } } }),
        allowedScopes,
      ),
    /unknown token "nope"/,
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ scopes: { unknown: { deny: ["mcp"] } } }),
        allowedScopes,
      ),
    /unknown scope "unknown"/,
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ users: { "bad id!": { allow: ["web"] } } }),
        allowedScopes,
      ),
    /bad user id/,
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ users: { "alice": { allow: ["web"] } } }),
        allowedScopes,
      ),
    /bad user id/,
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ users: { [`telegram:user:${"a".repeat(300)}`]: { allow: ["web"] } } }),
        allowedScopes,
      ),
    /bad user id/,
  );
  const bridgeId =
    "telegram:matrix-bridge:123%3A-100%3Aabcdef0123456789abcdef0123456789";
  assert.deepEqual(
    parseToolGrants(
      JSON.stringify({ users: { [bridgeId]: { deny: ["mcp"] } } }),
      allowedScopes,
    ).users[bridgeId],
    { deny: ["mcp"] },
  );
  assert.throws(
    () =>
      parseToolGrants(
        JSON.stringify({ users: { "chat:user:abc": { allow: ["web"], extra: [] } } }),
        allowedScopes,
      ),
    /accepts only allow and deny/,
  );
  assert.throws(
    () => parseToolGrants(JSON.stringify({ users: {} }), allowedScopes),
    /no user or scope rules/,
  );
});

test("wires the mounted tool grants file into the engine configuration", () => {
  const dir = mkdtempSync(join(tmpdir(), "sidekick-config-grants-"));
  const path = join(dir, "tool-grants.json");
  writeFileSync(
    path,
    JSON.stringify({
      users: { "telegram:user:419540347": { allow: ["web"] } },
    }),
  );
  process.env.PI_AGENT_TOOL_GRANTS_FILE = path;
  try {
    assert.deepEqual(JSON.parse(JSON.stringify(loadConfig().engine.toolGrants)), {
      scopes: {},
      users: {
        "telegram:user:419540347": {
          allow: ["web_search", "fetch_content"],
        },
      },
    });
    process.env.PI_AGENT_TOOL_GRANTS_FILE = join(dir, "missing.json");
    assert.throws(() => loadConfig(), /Unreadable tool grants configuration/);
  } finally {
    delete process.env.PI_AGENT_TOOL_GRANTS_FILE;
    rmSync(dir, { recursive: true, force: true });
  }
});
