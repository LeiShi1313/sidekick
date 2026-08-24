import { readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { CODEX_ACCESS_TOKEN_COMMAND } from "./codex-access-token.mjs";
import { isHostIdentity } from "./host-identity.mjs";

const REASONING_LEVELS = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);
const ADAPTER_CAPABILITIES = Object.freeze([
  "models",
  "runs",
  "attachments",
]);
const OPERATOR_CAPABILITIES = Object.freeze([
  ...ADAPTER_CAPABILITIES,
  "history",
  "status",
]);

const TOOL_GRANT_TOKENS = new Set([
  "web_search",
  "fetch_content",
  "code_exec",
  "image_generate",
  "mcp",
  "web",
  "code",
  "images",
  "memory",
]);
const TOOL_GRANT_GROUP_EXPANSIONS = Object.freeze({
  web: ["web_search", "fetch_content"],
  code: ["code_exec"],
  images: ["image_generate"],
  mcp: ["mcp"],
  memory: ["memory_*"],
});

function required(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required configuration: ${name}`);
  return value;
}

function requiredSecret(name, minimumBytes = 32) {
  const value = required(name);
  if (Buffer.byteLength(value) < minimumBytes) {
    throw new Error(`Invalid secret configuration: ${name}`);
  }
  return value;
}

function requiredScopePrefix(name, platform) {
  const value = required(name);
  if (
    value.length > 512 ||
    !value.startsWith(`${platform}:`) ||
    !value.endsWith(":")
  ) {
    throw new Error(`Invalid scope prefix configuration: ${name}`);
  }
  return value;
}

function integer(name, fallback, { min = 1, max = Number.MAX_SAFE_INTEGER } = {}) {
  const raw = process.env[name]?.trim();
  const value = raw ? Number(raw) : fallback;
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(`Invalid integer configuration: ${name}`);
  }
  return value;
}

export function buildWebSearchConfig() {
  return {
    workflow: "none",
    allowBrowserCookies: false,
    openaiApiKey: CODEX_ACCESS_TOKEN_COMMAND,
    searchRouting: {
      providers: ["openai", "exa"],
      fallbackOn: ["transient", "quota", "network", "invalid-response"],
    },
    webSearch: { enabled: true },
    githubClone: { enabled: false },
    youtube: { enabled: false },
  };
}

function parseToolGrantEntry(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid tool grants configuration: ${label} must be an object`);
  }
  const keys = Object.keys(value);
  if (
    keys.length === 0 ||
    keys.some((key) => key !== "allow" && key !== "deny")
  ) {
    throw new Error(
      `Invalid tool grants configuration: ${label} accepts only allow and deny`,
    );
  }
  const entry = {};
  for (const key of ["allow", "deny"]) {
    if (!(key in value)) continue;
    const tokens = value[key];
    if (
      !Array.isArray(tokens) ||
      tokens.length < 1 ||
      tokens.some((token) => typeof token !== "string")
    ) {
      throw new Error(
        `Invalid tool grants configuration: ${label}.${key} must be a non-empty array of strings`,
      );
    }
    const names = [];
    for (const token of tokens) {
      if (!TOOL_GRANT_TOKENS.has(token)) {
        throw new Error(
          `Invalid tool grants configuration: unknown token "${token}" in ${label}.${key}`,
        );
      }
      for (const name of TOOL_GRANT_GROUP_EXPANSIONS[token] ?? [token]) {
        if (!names.includes(name)) names.push(name);
      }
    }
    entry[key] = Object.freeze(names);
  }
  return Object.freeze(entry);
}

export function parseToolGrants(raw, allowedScopeIds) {
  let document;
  try {
    document = JSON.parse(raw);
  } catch {
    throw new Error("Invalid tool grants configuration: malformed JSON");
  }
  if (!document || typeof document !== "object" || Array.isArray(document)) {
    throw new Error("Invalid tool grants configuration: must be an object");
  }
  const keys = Object.keys(document);
  if (
    keys.length === 0 ||
    keys.some((key) => key !== "users" && key !== "scopes") ||
    (document.users === undefined && document.scopes === undefined)
  ) {
    throw new Error(
      "Invalid tool grants configuration: expects users or scopes",
    );
  }
  const users = Object.create(null);
  if (document.users !== undefined) {
    if (!document.users || typeof document.users !== "object" || Array.isArray(document.users)) {
      throw new Error("Invalid tool grants configuration: users must be an object");
    }
    for (const [id, entry] of Object.entries(document.users)) {
      if (!isHostIdentity(id)) {
        throw new Error(
          `Invalid tool grants configuration: bad user id "${id}"`,
        );
      }
      users[id] = parseToolGrantEntry(entry, `users.${id}`);
    }
  }
  const scopes = Object.create(null);
  if (document.scopes !== undefined) {
    if (!document.scopes || typeof document.scopes !== "object" || Array.isArray(document.scopes)) {
      throw new Error("Invalid tool grants configuration: scopes must be an object");
    }
    for (const [scopeId, entry] of Object.entries(document.scopes)) {
      if (!allowedScopeIds?.has(scopeId)) {
        throw new Error(
          `Invalid tool grants configuration: unknown scope "${scopeId}"`,
        );
      }
      scopes[scopeId] = parseToolGrantEntry(entry, `scopes.${scopeId}`);
    }
  }
  if (Object.keys(users).length === 0 && Object.keys(scopes).length === 0) {
    throw new Error(
      "Invalid tool grants configuration: no user or scope rules",
    );
  }
  return Object.freeze({
    users: Object.freeze(users),
    scopes: Object.freeze(scopes),
  });
}

function loadToolGrants(allowedScopeIds) {
  const path = process.env.PI_AGENT_TOOL_GRANTS_FILE?.trim();
  if (!path) return null;
  const MAX_TOOL_GRANTS_FILE_BYTES = 1024 * 1024;
  let stats;
  try {
    stats = statSync(path);
  } catch (error) {
    throw new Error(
      `Unreadable tool grants configuration at ${path}: ${error.code ?? error.message}`,
    );
  }
  if (!stats.isFile() || stats.size > MAX_TOOL_GRANTS_FILE_BYTES) {
    throw new Error(
      `Invalid tool grants configuration at ${path}: expected a file of at most ${MAX_TOOL_GRANTS_FILE_BYTES} bytes`,
    );
  }
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (error) {
    throw new Error(
      `Unreadable tool grants configuration at ${path}: ${error.code ?? error.message}`,
    );
  }
  return parseToolGrants(raw, allowedScopeIds);
}

export function loadConfig() {
  const reasoningEffort =
    process.env.AI_REASONING_EFFORT?.trim().toLowerCase() || "none";
  if (!REASONING_LEVELS.has(reasoningEffort)) {
    throw new Error("Invalid configuration: AI_REASONING_EFFORT");
  }
  const dataDir =
    process.env.PI_DATA_DIR?.trim() || join(homedir(), ".pi-agent");
  const memoryUrl =
    process.env.MEMORY_API_URL?.trim().replace(/\/$/, "") || null;
  const config = {
    host: process.env.PI_HOST?.trim() || "0.0.0.0",
    port: integer("PI_PORT", 8790, { max: 65_535 }),
    clients: [
      {
        id: "telegram",
        token: requiredSecret("PI_AGENT_TELEGRAM_TOKEN", 24),
        capabilities: ADAPTER_CAPABILITIES,
        adapterInstanceId:
          process.env.PI_AGENT_TELEGRAM_INSTANCE_ID?.trim() ||
          "telegram-default",
        scopePrefix: "telegram:",
      },
      {
        id: "onebot",
        token: requiredSecret("PI_AGENT_ONEBOT_TOKEN", 24),
        capabilities: ADAPTER_CAPABILITIES,
        adapterInstanceId:
          process.env.PI_AGENT_ONEBOT_INSTANCE_ID?.trim() || "qq-default",
        scopePrefix: "qq:",
      },
      {
        id: "wechat-host",
        token: requiredSecret("PI_AGENT_WECHAT_HOST_TOKEN", 24),
        capabilities: ADAPTER_CAPABILITIES,
        adapterInstanceId:
          process.env.PI_AGENT_WECHAT_HOST_INSTANCE_ID?.trim() ||
          "wechat-host",
        scopePrefix: requiredScopePrefix(
          "PI_AGENT_WECHAT_HOST_SCOPE_PREFIX",
          "wechat",
        ),
      },
      {
        id: "wechat-peer",
        token: requiredSecret("PI_AGENT_WECHAT_PEER_TOKEN", 24),
        capabilities: ADAPTER_CAPABILITIES,
        adapterInstanceId:
          process.env.PI_AGENT_WECHAT_PEER_INSTANCE_ID?.trim() ||
          "wechat-peer",
        scopePrefix: requiredScopePrefix(
          "PI_AGENT_WECHAT_PEER_SCOPE_PREFIX",
          "wechat",
        ),
      },
      {
        id: "playground",
        token: requiredSecret("PI_AGENT_PLAYGROUND_TOKEN", 24),
        capabilities: OPERATOR_CAPABILITIES,
        cancelAny: true,
      },
    ],
    engine: {
      baseUrl: required("AI_BASE_URL"),
      apiKey: required("AI_API_KEY"),
      identityAliasKey: requiredSecret("PI_IDENTITY_ALIAS_KEY"),
      model: required("AI_CHAT_MODEL"),
      imageModel: process.env.AI_IMAGE_MODEL?.trim() || null,
      reasoningEffort,
      maxOutputTokens: integer("AI_MAX_OUTPUT_TOKENS", 4_000, {
        max: 100_000,
      }),
      contextWindow: integer("AI_CONTEXT_WINDOW", 128_000, {
        min: 4_096,
      }),
      requestTimeoutMs:
        integer("AI_REQUEST_TIMEOUT", 90, { max: 3_600 }) * 1_000,
      imageRequestTimeoutMs:
        integer("AI_IMAGE_REQUEST_TIMEOUT", 180, { max: 600 }) * 1_000,
      memoryUrl,
      memoryToken: memoryUrl ? requiredSecret("MEMORY_API_TOKEN", 24) : null,
      workspaceDir: join(dataDir, "workspace"),
      sessionDir: join(dataDir, "sessions"),
      auditDir: join(dataDir, "audit"),
      agentDir: join(dataDir, "agent"),
    },
  };
  config.engine.toolGrants = loadToolGrants(
    new Set(config.clients.map(({ adapterInstanceId }) => adapterInstanceId)),
  );
  return config;
}
