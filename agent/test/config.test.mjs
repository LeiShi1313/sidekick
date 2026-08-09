import assert from "node:assert/strict";
import test from "node:test";

import { loadConfig } from "../src/config.mjs";

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
});
