import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";
import { chmodSync, existsSync } from "node:fs";
import {
  chmod,
  mkdir,
  open,
  readFile,
  readdir,
  rename,
  unlink,
} from "node:fs/promises";
import { join } from "node:path";

import {
  redactSensitiveText,
  sanitizeMessageInPlace,
} from "./privacy-redaction.mjs";

const OMITTED_TOOL_RESULT = "[Tool result omitted after use]";
const OMITTED_CUSTOM_MESSAGE = "[Custom message omitted after use]";
const PUBLIC_AGENT_CWD = "/workspace";
const MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024;
const MAX_SAFE_SUMMARY_CHARS = 12_000;
const SESSION_BINDING_VERSION = 1;
const hardenedManagers = new WeakSet();

function secureSessionFile(manager) {
  const path = manager.getSessionFile();
  if (path && existsSync(path)) chmodSync(path, 0o600);
}

function safeWebEntryMetadata(data) {
  const type = data?.type === "search" || data?.type === "fetch"
    ? data.type
    : "unknown";
  const timestamp = Number.isSafeInteger(data?.timestamp)
    ? data.timestamp
    : Date.now();
  return { type, timestamp, omitted: true };
}

function safeCustomEntry(customType, data) {
  if (customType === "web-search-results") return safeWebEntryMetadata(data);
  return { omitted: true };
}

function safeContentPart(part) {
  if (!part || typeof part !== "object") return part;
  if (part.type === "thinking") return null;
  if (part.type === "toolCall") {
    return {
      type: "toolCall",
      id: String(part.id ?? ""),
      name: String(part.name ?? ""),
      arguments: {},
    };
  }
  if (part.type === "image") {
    return { type: "text", text: "[Image omitted after use]" };
  }
  if (part.type === "text") return { type: "text", text: part.text };
  return null;
}

export function sessionSafeMessage(message, state = {}) {
  let copy = structuredClone(message);
  if (copy.role === "user" && state.userMessageContent !== undefined) {
    copy.content = state.userMessageContent;
  }
  copy = stripMessageInjectedContext(copy);
  sanitizeMessageInPlace(copy, state.privacyOptions);
  if (copy.role === "assistant" && Array.isArray(copy.content)) {
    copy.content = copy.content.map(safeContentPart).filter(Boolean);
  } else if (copy.role === "user" && Array.isArray(copy.content)) {
    copy.content = copy.content.map(safeContentPart).filter(Boolean);
  } else if (copy.role === "toolResult") {
    copy.content = [{ type: "text", text: OMITTED_TOOL_RESULT }];
    delete copy.details;
  }
  return copy;
}

function messageText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("");
}

function safeConversationSummary(manager, state) {
  const turns = [];
  for (const entry of manager.getBranch()) {
    if (entry?.type !== "message") continue;
    const message = sessionSafeMessage(entry.message, {
      privacyOptions: state.privacyOptions,
    });
    if (!new Set(["user", "assistant"]).has(message.role)) continue;
    const text = messageText(message.content).trim();
    if (!text) continue;
    turns.push(`${message.role === "user" ? "User" : "Assistant"}: ${text}`);
  }
  const heading = "Earlier conversation (tool output and hidden context omitted):\n";
  const body = turns.join("\n\n");
  const available = MAX_SAFE_SUMMARY_CHARS - heading.length;
  return heading + (body.length <= available ? body : body.slice(-available));
}

export function hardenSessionPersistence(manager, getState = () => ({})) {
  if (hardenedManagers.has(manager)) return manager;
  hardenedManagers.add(manager);

  const appendMessage = manager.appendMessage.bind(manager);
  manager.appendMessage = (message) => {
    const entryId = appendMessage(sessionSafeMessage(message, getState()));
    secureSessionFile(manager);
    return entryId;
  };

  const appendCustomEntry = manager.appendCustomEntry.bind(manager);
  manager.appendCustomEntry = (customType, data) => {
    const entryId = appendCustomEntry(customType, safeCustomEntry(customType, data));
    secureSessionFile(manager);
    return entryId;
  };

  const appendCustomMessageEntry = manager.appendCustomMessageEntry.bind(manager);
  manager.appendCustomMessageEntry = (customType, _content, display, _details) => {
    const entryId = appendCustomMessageEntry(
      customType,
      OMITTED_CUSTOM_MESSAGE,
      display,
      undefined,
    );
    secureSessionFile(manager);
    return entryId;
  };

  if (typeof manager.appendCompaction === "function") {
    const appendCompaction = manager.appendCompaction.bind(manager);
    manager.appendCompaction = (
      _summary,
      firstKeptEntryId,
      tokensBefore,
      _details,
      fromHook,
    ) => {
      const entryId = appendCompaction(
        safeConversationSummary(manager, getState()),
        firstKeptEntryId,
        tokensBefore,
        undefined,
        fromHook,
      );
      secureSessionFile(manager);
      return entryId;
    };
  }

  if (typeof manager.branchWithSummary === "function") {
    const branchWithSummary = manager.branchWithSummary.bind(manager);
    manager.branchWithSummary = (entryId, _summary, _details, fromHook) => {
      const summaryId = branchWithSummary(
        entryId,
        safeConversationSummary(manager, getState()),
        undefined,
        fromHook,
      );
      secureSessionFile(manager);
      return summaryId;
    };
  }

  secureSessionFile(manager);
  return manager;
}

const OMITTED_CONTEXT_BLOCKS = new Set([
  "host_access_advisory",
  "untrusted_memory_context",
  "untrusted_reference_context",
]);

function stripInjectedContext(value) {
  if (typeof value !== "string") return value;
  const lines = value.split("\n");
  const kept = [];
  let omittedBlock = null;
  for (const line of lines) {
    const open = /^<([a-z_]+)>$/.exec(line)?.[1];
    if (omittedBlock === null && OMITTED_CONTEXT_BLOCKS.has(open)) {
      omittedBlock = open;
    } else if (omittedBlock !== null && line === `</${omittedBlock}>`) {
      omittedBlock = null;
    } else if (omittedBlock === null) {
      kept.push(line);
    }
  }
  return kept.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function stripMessageInjectedContext(message) {
  const copy = structuredClone(message);
  if (copy.role !== "user") return copy;
  if (typeof copy.content === "string") {
    copy.content = stripInjectedContext(copy.content);
  } else if (Array.isArray(copy.content)) {
    copy.content = copy.content.map((part) =>
      part?.type === "text"
        ? { ...part, text: stripInjectedContext(part.text) }
        : part,
    );
  }
  return copy;
}

function safeSessionEntry(entry, privacyOptions) {
  const copy = structuredClone(entry);
  if (copy.type === "session") {
    copy.cwd = PUBLIC_AGENT_CWD;
    delete copy.parentSession;
  } else if (copy.type === "message") {
    copy.message = sessionSafeMessage(
      stripMessageInjectedContext(copy.message),
      { privacyOptions },
    );
  } else if (copy.type === "custom") {
    copy.data = safeCustomEntry(copy.customType, copy.data);
  } else if (copy.type === "custom_message") {
    copy.content = OMITTED_CUSTOM_MESSAGE;
    delete copy.details;
  } else if (copy.type === "compaction" || copy.type === "branch_summary") {
    copy.summary = "Earlier conversation omitted during privacy migration.";
    delete copy.details;
  } else if (copy.type === "session_info") {
    copy.name = redactSensitiveText(copy.name, privacyOptions);
  } else if (copy.type === "label") {
    copy.label = redactSensitiveText(copy.label, privacyOptions);
  }
  return copy;
}

function parseSession(content) {
  const entries = [];
  const lines = content.split("\n");
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (!entry || typeof entry !== "object") {
        throw new Error("Malformed session entry");
      }
      entries.push(entry);
    } catch (error) {
      if (index === lines.length - 1 && !content.endsWith("\n")) break;
      throw error;
    }
  }
  return entries;
}

async function replaceSessionFile(path, content) {
  const temporary = `${path}.${randomUUID()}.tmp`;
  let handle;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(content, "utf8");
    await handle.close();
    handle = null;
    await rename(temporary, path);
    await chmod(path, 0o600);
  } finally {
    await handle?.close();
    await unlink(temporary).catch(() => {});
  }
}

function bindingDigest({ sessionId, principalId, scopeId, key }) {
  if (
    typeof sessionId !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(sessionId) ||
    typeof principalId !== "string" ||
    principalId.length < 1 ||
    principalId.length > 128 ||
    typeof scopeId !== "string" ||
    scopeId.length < 1 ||
    scopeId.length > 512 ||
    typeof key !== "string" ||
    Buffer.byteLength(key) < 32
  ) {
    throw new Error("Session binding context is invalid");
  }
  return createHmac("sha256", key)
    .update("sidekick:session-binding:v1\0")
    .update(sessionId)
    .update("\0")
    .update(principalId)
    .update("\0")
    .update(scopeId)
    .digest("hex");
}

function bindingPath(sessionDir, sessionId) {
  return join(sessionDir, "bindings", `${sessionId}.json`);
}

export async function bindSession(
  sessionDir,
  sessionId,
  { principalId, scopeId, key },
) {
  const directory = join(sessionDir, "bindings");
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await chmod(directory, 0o700);
  const payload = `${JSON.stringify({
    version: SESSION_BINDING_VERSION,
    binding: bindingDigest({ sessionId, principalId, scopeId, key }),
  })}\n`;
  const handle = await open(bindingPath(sessionDir, sessionId), "wx", 0o600);
  try {
    await handle.writeFile(payload, "utf8");
    await handle.sync();
  } finally {
    await handle.close();
  }
}

export async function assertSessionBinding(
  sessionDir,
  sessionId,
  { principalId, scopeId, key },
) {
  const expected = bindingDigest({ sessionId, principalId, scopeId, key });
  let raw;
  try {
    raw = await readFile(bindingPath(sessionDir, sessionId), "utf8");
  } catch {
    throw new Error("Agent session is unavailable");
  }
  if (raw.length > 512) throw new Error("Agent session is unavailable");
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("Agent session is unavailable");
  }
  if (
    !parsed ||
    parsed.version !== SESSION_BINDING_VERSION ||
    typeof parsed.binding !== "string" ||
    !/^[a-f0-9]{64}$/.test(parsed.binding)
  ) {
    throw new Error("Agent session is unavailable");
  }
  const actualBuffer = Buffer.from(parsed.binding, "hex");
  const expectedBuffer = Buffer.from(expected, "hex");
  if (!timingSafeEqual(actualBuffer, expectedBuffer)) {
    throw new Error("Agent session is unavailable");
  }
}

export async function scrubSessionDirectory(directory, privacyOptions = {}) {
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await chmod(directory, 0o700);
  const names = await readdir(directory);
  let scanned = 0;
  let rewritten = 0;
  for (const name of names) {
    if (!name.endsWith(".jsonl")) continue;
    scanned += 1;
    const path = join(directory, name);
    const handle = await open(path, "r");
    let raw;
    let mode;
    try {
      const info = await handle.stat();
      if (!info.isFile() || info.size > MAX_SESSION_FILE_BYTES) {
        throw new Error("Session file is too large");
      }
      mode = info.mode & 0o777;
      raw = await handle.readFile("utf8");
    } finally {
      await handle.close();
    }
    const scopedPrivacyOptions = {
      ...privacyOptions,
      identityScope: `legacy-session:${name}`,
    };
    const safe = `${parseSession(raw)
      .map((entry) => JSON.stringify(safeSessionEntry(entry, scopedPrivacyOptions)))
      .join("\n")}\n`;
    if (safe !== raw) {
      await replaceSessionFile(path, safe);
      rewritten += 1;
    } else if (mode !== 0o600) {
      await chmod(path, 0o600);
      rewritten += 1;
    }
  }
  return { scanned, rewritten };
}
