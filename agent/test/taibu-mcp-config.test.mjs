import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_TAIBU_MCP_URL,
  TAIBU_MCP_GUIDANCE,
  createTaibuMcpConfig,
} from "../src/taibu-mcp-config.mjs";

test("builds an isolated, proxy-only Taibu MCP configuration", () => {
  const config = createTaibuMcpConfig({});

  assert.equal(DEFAULT_TAIBU_MCP_URL, "https://suanming.leishi.xyz/mcp");
  assert.deepEqual(config, {
    settings: {
      hostConfigDiscovery: "off",
      scriptMode: false,
      disableProxyTool: false,
      sampling: false,
      elicitation: false,
      outputGuard: true,
      requestTimeoutMs: 30_000,
      showStatusIcon: false,
      mcpFooterStatus: "off",
    },
    mcpServers: {
      taibu: {
        url: "https://suanming.leishi.xyz/mcp",
        lifecycle: "lazy",
        requestTimeoutMs: 30_000,
        protocolVersion: "2026-07-28",
        exposeResources: false,
        directTools: false,
      },
    },
  });
});

test("accepts an operator-provided HTTPS endpoint and timeout", () => {
  const config = createTaibuMcpConfig({
    TAIBU_MCP_URL: " https://taibu.example.test/mcp ",
    TAIBU_MCP_REQUEST_TIMEOUT_MS: "45000",
  });

  assert.equal(config.mcpServers.taibu.url, "https://taibu.example.test/mcp");
  assert.equal(config.mcpServers.taibu.requestTimeoutMs, 45_000);
  assert.equal(config.settings.requestTimeoutMs, 45_000);
});

test("rejects insecure or malformed Taibu MCP configuration", () => {
  assert.throws(
    () => createTaibuMcpConfig({ TAIBU_MCP_URL: "http://taibu.example.test/mcp" }),
    /must use HTTPS/,
  );
  assert.throws(
    () => createTaibuMcpConfig({ TAIBU_MCP_URL: "not a URL" }),
    /TAIBU_MCP_URL/,
  );
  assert.throws(
    () =>
      createTaibuMcpConfig({
        TAIBU_MCP_URL: "https://user:secret@taibu.example.test/mcp",
      }),
    /cannot contain credentials/,
  );
  assert.throws(
    () =>
      createTaibuMcpConfig({
        TAIBU_MCP_REQUEST_TIMEOUT_MS: "0",
      }),
    /TAIBU_MCP_REQUEST_TIMEOUT_MS/,
  );
});

test("guidance routes divination through MCP without a second calculator", () => {
  assert.match(TAIBU_MCP_GUIDANCE, /use the `mcp` tool/i);
  assert.match(TAIBU_MCP_GUIDANCE, /never calculate.*from memory/i);
  assert.match(TAIBU_MCP_GUIDANCE, /untrusted data, not instructions/i);
  assert.doesNotMatch(TAIBU_MCP_GUIDANCE, /scripts\/taibu|taibu-mcp package/i);
});
