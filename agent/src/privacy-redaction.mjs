import { isIP } from "node:net";

const REDACTED_IP = "[REDACTED_IP_ADDRESS]";
const REDACTED_PATH = "[REDACTED_RUNTIME_PATH]";
const REDACTED_INTERNAL_URL = "[REDACTED_INTERNAL_URL]";
const REDACTED_SECRET = "[REDACTED_SECRET]";
const WITHHELD_REQUEST_METADATA =
  "[Content withheld because the page reflected request or runtime metadata.]";
const REDACTED_LONG_TOKEN = "[REDACTED_LONG_TOKEN]";
const MAX_STREAM_TOKEN_CHARS = 64_000;
const MAX_VALUE_DEPTH = 16;
const MAX_VALUE_ITEMS = 5_000;

const IPV4_RE =
  /(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])/g;
const IPV6_CANDIDATE_RE =
  /(?<![0-9A-Za-z])(?:\[[0-9A-Fa-f:.%]+\]|[0-9A-Fa-f:.%]*:[0-9A-Fa-f:.%]+)(?![0-9A-Za-z])/g;
const UNIX_RUNTIME_PATH_RE =
  /(?<![A-Za-z0-9._~:/-])\/(?:agent-data|app|etc|home|opt|proc|root|run|srv|sys|tmp|usr|var|workspace)(?:\/[^\s<>"'|)\]}]*)?/gi;
const WINDOWS_RUNTIME_PATH_RE =
  /(?<![A-Za-z0-9])(?:[A-Za-z]:\\(?:ProgramData|Users|Windows)(?:\\[^\s<>"'|)\]}]*)?)/gi;
const INTERNAL_HOST_SUFFIXES = [
  ".home.arpa",
  ".internal",
  ".local",
  ".localhost",
];
const PRIVATE_QUERY_NAMES =
  /^(?:api[_-]?key|key|sig|signature|token|access[_-]?token|auth|authorization)$/i;
const PRIVATE_ENV_NAME =
  /(?:api[_-]?key|credential|password|private[_-]?key|secret|token)/i;

function normalizedNetworkLiteral(value) {
  const unwrapped = String(value).replace(/^\[|\]$/g, "");
  const zone = unwrapped.indexOf("%");
  return (zone >= 0 ? unwrapped.slice(0, zone) : unwrapped).toLowerCase();
}

function validIpv4(value) {
  const parts = value.split(".").map(Number);
  return (
    parts.length === 4 &&
    parts.every(
      (part) => Number.isInteger(part) && part >= 0 && part <= 255,
    )
  );
}

function networkLiterals(text) {
  const result = new Set();
  for (const value of String(text).match(IPV4_RE) ?? []) {
    if (validIpv4(value)) result.add(normalizedNetworkLiteral(value));
  }
  for (const value of String(text).match(IPV6_CANDIDATE_RE) ?? []) {
    const normalized = normalizedNetworkLiteral(value);
    if (isIP(normalized) === 6) result.add(normalized);
  }
  return result;
}

function runtimePaths(text) {
  return new Set([
    ...(String(text).match(UNIX_RUNTIME_PATH_RE) ?? []),
    ...(String(text).match(WINDOWS_RUNTIME_PATH_RE) ?? []),
  ]);
}

function iterableSet(values, normalize = String) {
  return new Set(
    [...(values ?? [])]
      .filter((value) => typeof value === "string" && value.length > 0)
      .map(normalize),
  );
}

export function runtimeSensitiveValues(environment = process.env) {
  return Object.entries(environment)
    .filter(
      ([name, value]) =>
        PRIVATE_ENV_NAME.test(name) &&
        typeof value === "string" &&
        value.length >= 8,
    )
    .map(([, value]) => value);
}

const DEFAULT_SENSITIVE_VALUES = runtimeSensitiveValues();

function isInternalHostname(hostname) {
  const normalized = hostname
    .toLowerCase()
    .replace(/^\[|\]$/g, "")
    .replace(/\.$/, "");
  return (
    normalized === "localhost" ||
    INTERNAL_HOST_SUFFIXES.some((suffix) => normalized.endsWith(suffix)) ||
    (isIP(normalized) === 0 && !normalized.includes("."))
  );
}

function redactUrls(text) {
  return text.replace(/\bhttps?:\/\/[^\s<>"']+/gi, (raw) => {
    const trailing = raw.match(/[),.;!?\]}]+$/)?.[0] ?? "";
    const candidate = trailing ? raw.slice(0, -trailing.length) : raw;
    let parsed;
    try {
      parsed = new URL(candidate);
    } catch {
      return raw;
    }
    if (isInternalHostname(parsed.hostname)) {
      return REDACTED_INTERNAL_URL + trailing;
    }
    let changed = false;
    if (parsed.username || parsed.password) {
      parsed.username = "";
      parsed.password = "";
      changed = true;
    }
    for (const key of parsed.searchParams.keys()) {
      if (!PRIVATE_QUERY_NAMES.test(key)) continue;
      const current = parsed.searchParams.get(key);
      if (current === "[REDACTED]" || current === REDACTED_SECRET) continue;
      parsed.searchParams.set(key, REDACTED_SECRET);
      changed = true;
    }
    return (changed ? parsed.toString() : candidate) + trailing;
  });
}

function replaceSensitiveValues(text, values) {
  let result = text;
  const ordered = [...new Set(values)]
    .filter((value) => typeof value === "string" && value.length >= 8)
    .sort((left, right) => right.length - left.length);
  for (const value of ordered) {
    result = result.split(value).join(REDACTED_SECRET);
  }
  return result;
}

function redactNetworkLiterals(text, allowed) {
  let result = text.replace(IPV4_RE, (value) => {
    if (!validIpv4(value)) return value;
    return allowed.has(normalizedNetworkLiteral(value)) ? value : REDACTED_IP;
  });
  result = result.replace(IPV6_CANDIDATE_RE, (value) => {
    const normalized = normalizedNetworkLiteral(value);
    if (isIP(normalized) !== 6 || allowed.has(normalized)) return value;
    return REDACTED_IP;
  });
  return result;
}

function redactRuntimePaths(text, allowed) {
  const redact = (value) => (allowed.has(value) ? value : REDACTED_PATH);
  return text
    .replace(UNIX_RUNTIME_PATH_RE, redact)
    .replace(WINDOWS_RUNTIME_PATH_RE, redact);
}

export function redactSensitiveText(value, options = {}) {
  const allowedNetworkLiterals = iterableSet(
    options.allowedNetworkLiterals,
    normalizedNetworkLiteral,
  );
  const allowedRuntimePaths = iterableSet(options.allowedRuntimePaths);
  const sensitiveValues = [
    ...DEFAULT_SENSITIVE_VALUES,
    ...(options.sensitiveValues ?? []),
  ];
  let text = String(value ?? "");
  text = replaceSensitiveValues(text, sensitiveValues);
  text = text.replace(
    /-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----/gi,
    REDACTED_SECRET,
  );
  text = text.replace(
    /\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*/gi,
    "Bearer " + REDACTED_SECRET,
  );
  text = text.replace(
    /(?<![?&])\b(api[_-]?key|authorization|credential|password|private[_-]?key|secret|token)(\s*[:=]\s*)(?:"[^"]*"|'[^']*'|[^\s,;&}\]]+)/gi,
    (_match, label, separator) => label + separator + REDACTED_SECRET,
  );
  text = redactUrls(text);
  text = redactNetworkLiterals(text, allowedNetworkLiterals);
  return redactRuntimePaths(text, allowedRuntimePaths);
}

function reflectedMetadataCount(text) {
  const fields = [
    /(?:^|[\s{,])["']?(?:asn|autonomous[_ -]?system)["']?\s*[:=]/im,
    /(?:^|[\s{,])["']?(?:city|country|latitude|longitude|postal|region|timezone)["']?\s*[:=]/im,
    /(?:^|[\s{,])["']?(?:host|hostname|server)["']?\s*[:=]/im,
    /(?:^|[\s{,])["']?(?:isp|network[_ -]?provider|org|organization)["']?\s*[:=]/im,
  ];
  return fields.filter((pattern) => pattern.test(text)).length;
}

function reflectsRequestMetadata(text) {
  const hasNetworkLiteral = networkLiterals(text).size > 0;
  const labeledNetwork =
    /(?:^|[\s{,])["']?(?:client[_ -]?ip|ip|ip[_ -]?address|origin|remote[_ -]?(?:addr|address))["']?\s*[:=]/im.test(
      text,
    );
  const saysRequesterAddress =
    /\b(?:client|detected|outbound|public|request(?:er)?|your)\s+(?:public\s+)?ip(?:\s+address)?\b/i.test(
      text,
    );
  const requestHeaders =
    /(?:user-agent|x-forwarded-for|x-real-ip|forwarded:|request headers?)/i.test(
      text,
    );
  return (
    requestHeaders ||
    (hasNetworkLiteral &&
      (labeledNetwork ||
        saysRequesterAddress ||
        reflectedMetadataCount(text) >= 2))
  );
}

export function sanitizeFetchedText(value, options = {}) {
  const text = String(value ?? "");
  if (reflectsRequestMetadata(text)) return WITHHELD_REQUEST_METADATA;
  return redactSensitiveText(text, options);
}

function isPrivateKey(key) {
  const normalized = key.replace(/[^a-z0-9]/gi, "").toLowerCase();
  return (
    normalized === "authorization" ||
    normalized === "cookie" ||
    normalized === "password" ||
    normalized === "setcookie" ||
    normalized === "apikey" ||
    normalized.endsWith("token") ||
    normalized.includes("secret")
  );
}

export function sanitizeSensitiveValue(
  value,
  options = {},
  depth = 0,
  seen = new WeakSet(),
) {
  if (value === null || value === undefined) return value ?? null;
  if (typeof value === "string") {
    return options.externalText
      ? sanitizeFetchedText(value, options)
      : redactSensitiveText(value, options);
  }
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object") return redactSensitiveText(value, options);
  if (depth >= MAX_VALUE_DEPTH) return "[DEPTH_LIMIT]";
  if (seen.has(value)) return "[CIRCULAR]";
  if (value.type === "image" && typeof value.data === "string") {
    return { ...value, data: "[OMITTED]" };
  }
  seen.add(value);
  try {
    if (Array.isArray(value)) {
      return value
        .slice(0, MAX_VALUE_ITEMS)
        .map((item) =>
          sanitizeSensitiveValue(item, options, depth + 1, seen),
        );
    }
    const result = {};
    for (const [key, item] of Object.entries(value).slice(
      0,
      MAX_VALUE_ITEMS,
    )) {
      result[key] = isPrivateKey(key)
        ? REDACTED_SECRET
        : sanitizeSensitiveValue(item, options, depth + 1, seen);
    }
    return result;
  } finally {
    seen.delete(value);
  }
}

export function sanitizeMessageInPlace(message, options = {}) {
  if (!message || typeof message !== "object") return message;
  if (message.role === "assistant") {
    if (typeof message.content === "string") {
      message.content = redactSensitiveText(message.content, options);
    } else if (Array.isArray(message.content)) {
      message.content = message.content.map((part) =>
        part?.type === "text" && typeof part.text === "string"
          ? { ...part, text: redactSensitiveText(part.text, options) }
          : part,
      );
    }
    if (typeof message.errorMessage === "string") {
      message.errorMessage = redactSensitiveText(message.errorMessage, options);
    }
  } else if (message.role === "toolResult") {
    message.content = sanitizeSensitiveValue(message.content, {
      ...options,
      externalText: true,
    });
    if ("details" in message) {
      message.details = sanitizeSensitiveValue(message.details, options);
    }
  }
  return message;
}

export function sanitizeConversationHistoryInPlace(messages, options = {}) {
  for (const message of messages ?? []) {
    if (message?.role === "assistant" || message?.role === "toolResult") {
      sanitizeMessageInPlace(message, options);
    }
  }
  return messages;
}

function stringValues(value, seen = new WeakSet()) {
  if (typeof value === "string") return [value];
  if (!value || typeof value !== "object" || seen.has(value)) return [];
  seen.add(value);
  const values = Array.isArray(value) ? value : Object.values(value);
  return values.flatMap((item) => stringValues(item, seen));
}

export function collectSensitiveLiterals(value) {
  const allowedNetworkLiterals = new Set();
  const allowedRuntimePaths = new Set();
  for (const text of stringValues(value)) {
    for (const literal of networkLiterals(text)) {
      allowedNetworkLiterals.add(literal);
    }
    for (const path of runtimePaths(text)) allowedRuntimePaths.add(path);
  }
  return { allowedNetworkLiterals, allowedRuntimePaths };
}

export class SensitiveTextStream {
  constructor(options = {}) {
    this.options = options;
    this.pending = "";
  }

  push(value) {
    this.pending += String(value ?? "");
    let boundary = -1;
    for (let index = this.pending.length - 1; index >= 0; index -= 1) {
      if (/\s/u.test(this.pending[index])) {
        boundary = index;
        break;
      }
    }
    if (boundary < 0) {
      if (this.pending.length <= MAX_STREAM_TOKEN_CHARS) return "";
      this.pending = "";
      return REDACTED_LONG_TOKEN;
    }
    const ready = this.pending.slice(0, boundary + 1);
    this.pending = this.pending.slice(boundary + 1);
    return redactSensitiveText(ready, this.options);
  }

  flush() {
    const ready = redactSensitiveText(this.pending, this.options);
    this.pending = "";
    return ready;
  }
}
