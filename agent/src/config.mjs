import { homedir } from "node:os";
import { join } from "node:path";

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

export function buildWebSearchConfig({ baseUrl, model }) {
  let responsesUrl;
  try {
    responsesUrl = new URL(baseUrl);
  } catch {
    throw new Error("Invalid configuration: AI_BASE_URL");
  }
  if (
    responsesUrl.protocol !== "http:" &&
    responsesUrl.protocol !== "https:"
  ) {
    throw new Error("Invalid configuration: AI_BASE_URL");
  }
  responsesUrl.pathname = `${responsesUrl.pathname.replace(/\/+$/, "")}/responses`;
  responsesUrl.search = "";
  responsesUrl.hash = "";

  return {
    workflow: "none",
    allowBrowserCookies: false,
    openaiApiKey: "$AI_API_KEY",
    openaiResponsesUrl: responsesUrl.toString(),
    openaiSearchModel: model,
    searchRouting: {
      providers: ["openai", "exa"],
      fallbackOn: ["transient", "quota", "network", "invalid-response"],
    },
    webSearch: { enabled: true },
    githubClone: { enabled: false },
    youtube: { enabled: false },
  };
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
  return {
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
}
