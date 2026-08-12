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
const MAX_SAFE_SUMMARY_ASSISTANT_CHARS = 2_000;
const MAX_SAFE_SUMMARY_REQUEST_CHARS = 2_500;
const MAX_SAFE_SUMMARY_LABEL_CHARS = 256;
const MAX_SAFE_SUMMARY_TARGETS = 16;
const SESSION_BINDING_VERSION = 1;
const PUBLIC_ASSISTANT_STOP_REASONS = new Set(["stop", "length"]);
const MEMORY_MUTATION_TOOLS = new Set([
  "memory_update_requester",
  "memory_update_participant",
]);
const PARTICIPANT_TARGET_RE =
  /^(?:reply_author|direct_chat_participant|mention_[1-9][0-9]*)$/;
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

function safeMutationArguments(part) {
  if (!MEMORY_MUTATION_TOOLS.has(part.name)) return {};
  const supplied = part.arguments;
  if (!supplied || typeof supplied !== "object" || Array.isArray(supplied)) {
    return {};
  }
  const result = {};
  if (supplied.operation === "set" || supplied.operation === "clear") {
    result.operation = supplied.operation;
  }
  if (
    part.name === "memory_update_participant" &&
    typeof supplied.target === "string" &&
    PARTICIPANT_TARGET_RE.test(supplied.target)
  ) {
    result.target = supplied.target;
  }
  return result;
}

function safeMutationDetails(toolName, supplied) {
  if (
    !MEMORY_MUTATION_TOOLS.has(toolName) ||
    !supplied ||
    typeof supplied !== "object" ||
    Array.isArray(supplied)
  ) {
    return null;
  }
  const result = {};
  for (const field of ["saved", "cleared", "unavailable", "conflict"]) {
    if (typeof supplied[field] === "boolean") result[field] = supplied[field];
  }
  if (
    toolName === "memory_update_participant" &&
    typeof supplied.target === "string" &&
    PARTICIPANT_TARGET_RE.test(supplied.target)
  ) {
    result.target = supplied.target;
  }
  if (
    typeof supplied.reason === "string" &&
    /^[a-z][a-z0-9_]{0,63}$/.test(supplied.reason)
  ) {
    result.reason = supplied.reason;
  }
  return result;
}

function safeMutationReceipt(toolName, details, isError) {
  const subject =
    toolName === "memory_update_participant" ? "Participant" : "Requester";
  const outcome = !isError && details?.saved
    ? "was saved"
    : !isError && details?.cleared
      ? "was cleared"
      : "was not changed";
  return [{ type: "text", text: `${subject} customization ${outcome}.` }];
}

function safeContentPart(part) {
  if (!part || typeof part !== "object") return part;
  if (part.type === "thinking") return null;
  if (part.type === "toolCall") {
    return {
      type: "toolCall",
      id: String(part.id ?? ""),
      name: String(part.name ?? ""),
      arguments: safeMutationArguments(part),
    };
  }
  if (part.type === "image") {
    return { type: "text", text: "[Image omitted after use]" };
  }
  if (part.type === "text") return { type: "text", text: part.text };
  return null;
}

function safeAssistantContent(message) {
  if (!Array.isArray(message.content)) return [];
  if (message.stopReason === "toolUse") {
    return message.content
      .filter((part) => part?.type === "toolCall")
      .map(safeContentPart)
      .filter(Boolean);
  }
  if (!PUBLIC_ASSISTANT_STOP_REASONS.has(message.stopReason)) return [];
  return message.content.map(safeContentPart).filter(Boolean);
}

export function sessionSafeMessage(message, state = {}) {
  let copy = structuredClone(message);
  const hasTrustedUserContent =
    copy.role === "user" && state.userMessageContent !== undefined;
  if (hasTrustedUserContent) {
    copy.content = state.userMessageContent;
  } else {
    copy = stripMessageInjectedContext(copy);
  }
  sanitizeMessageInPlace(copy, state.privacyOptions);
  if (copy.role === "assistant") {
    copy.content = safeAssistantContent(copy);
  } else if (copy.role === "user" && Array.isArray(copy.content)) {
    copy.content = copy.content.map(safeContentPart).filter(Boolean);
  } else if (copy.role === "toolResult") {
    if (MEMORY_MUTATION_TOOLS.has(copy.toolName)) {
      const details = safeMutationDetails(copy.toolName, copy.details);
      copy.content = safeMutationReceipt(copy.toolName, details, copy.isError);
      if (details && Object.keys(details).length > 0) copy.details = details;
      else delete copy.details;
    } else {
      copy.content = [{ type: "text", text: OMITTED_TOOL_RESULT }];
      delete copy.details;
    }
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

function boundedSummaryText(value, max) {
  const text = String(value ?? "").trim();
  if (text.length <= max) return text;
  const marker = "\n[...truncated...]\n";
  const available = max - marker.length;
  const start = Math.ceil(available / 2);
  const end = Math.floor(available / 2);
  return `${text.slice(0, start)}${marker}${text.slice(-end)}`;
}

function safeSummaryPayload(value, max) {
  const escaped = String(value ?? "")
    .replace(/&(?!amp;|lt;|gt;)/g, "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  return boundedSummaryText(escaped, max);
}

function promptBlockContent(value, name) {
  const open = `<${name}>`;
  const close = `</${name}>`;
  const start = value.indexOf(open);
  if (start < 0) return "";
  const contentStart = start + open.length;
  const end = value.indexOf(close, contentStart);
  if (end < 0) return "";
  return value.slice(contentStart, end).trim();
}

function promptBlock(value, name, max) {
  const supplied = promptBlockContent(value, name);
  if (!supplied) return "";
  const content = safeSummaryPayload(supplied, max);
  return `<${name}>\n${content}\n</${name}>`;
}

function safeSummaryIdentity(value) {
  const content = promptBlockContent(value, "host_request_identity");
  const actor = content.match(
    /^Host-resolved current requester actor ID: (actor_[a-f0-9]{16})$/m,
  )?.[1];
  if (!actor) return "";
  const suppliedLabel = content.match(/^Untrusted display label: ([^\n]*)$/m)?.[1];
  const label = safeSummaryPayload(
    suppliedLabel ?? "not provided",
    MAX_SAFE_SUMMARY_LABEL_CHARS,
  ).replaceAll("\n", " ");
  return (
    "<host_request_identity>\n" +
    `Host-resolved current requester actor ID: ${actor}\n` +
    `Untrusted display label: ${label}\n` +
    "</host_request_identity>"
  );
}

function safeSummaryBindings(value) {
  const content = promptBlockContent(value, "host_participant_bindings");
  const bindings = [];
  for (const line of content.split("\n")) {
    const match = line.match(
      /^Target handle: ([a-z0-9_]+) \| Actor ID: (actor_[a-f0-9]{16}) \| Untrusted display label: ([^\n]*)$/,
    );
    if (!match || !PARTICIPANT_TARGET_RE.test(match[1])) continue;
    const label = safeSummaryPayload(
      match[3],
      MAX_SAFE_SUMMARY_LABEL_CHARS,
    ).replaceAll("\n", " ");
    bindings.push(
      `Target handle: ${match[1]} | Actor ID: ${match[2]} | ` +
        `Untrusted display label: ${label}`,
    );
    if (bindings.length >= MAX_SAFE_SUMMARY_TARGETS) break;
  }
  if (bindings.length === 0) return "";
  return (
    "<host_participant_bindings>\n" +
    "These host-resolved handles remain bound to their actor IDs.\n" +
    `${bindings.join("\n")}\n` +
    "</host_participant_bindings>"
  );
}

function safeSummaryUserText(value) {
  const safe = stripInjectedContext(value, SUMMARY_OMITTED_CONTEXT_BLOCKS);
  const requestStart = safe.indexOf("<current_request>");
  const authority = requestStart < 0 ? safe : safe.slice(0, requestStart);
  const blocks = [
    safeSummaryIdentity(authority),
    safeSummaryBindings(authority),
    promptBlock(safe, "current_request", MAX_SAFE_SUMMARY_REQUEST_CHARS),
  ].filter(Boolean);
  return blocks.length > 0
    ? blocks.join("\n\n")
    : safeSummaryPayload(safe, MAX_SAFE_SUMMARY_REQUEST_CHARS);
}

function summaryRecords(manager, state) {
  const records = [];
  for (const entry of manager.getBranch()) {
    if (entry?.type !== "message") continue;
    const message = sessionSafeMessage(entry.message, {
      privacyOptions: state.privacyOptions,
    });
    if (message.role === "user") {
      const text = safeSummaryUserText(messageText(message.content));
      if (text) {
        records.push({
          text: `User: ${text}`,
          priority: text.includes("<host_participant_bindings>"),
        });
      }
    } else if (message.role === "assistant") {
      const text = safeSummaryPayload(
        messageText(message.content),
        MAX_SAFE_SUMMARY_ASSISTANT_CHARS,
      );
      if (text) records.push({ text: `Assistant: ${text}`, priority: false });
    } else if (
      message.role === "toolResult" &&
      MEMORY_MUTATION_TOOLS.has(message.toolName)
    ) {
      const target = message.details?.target
        ? ` Target: ${message.details.target}.`
        : "";
      records.push({
        text: `Memory update receipt: ${messageText(message.content)}${target}`,
        priority: true,
      });
    }
  }
  return records;
}

function safeConversationSummary(manager, state) {
  const heading = "Earlier conversation (tool output and hidden context omitted):\n";
  const available = MAX_SAFE_SUMMARY_CHARS - heading.length;
  const records = summaryRecords(manager, state);
  const selected = new Set();
  let used = 0;
  for (const priority of [true, false]) {
    for (let index = records.length - 1; index >= 0; index -= 1) {
      if (selected.has(index) || records[index].priority !== priority) continue;
      const separator = selected.size > 0 ? 2 : 0;
      if (used + separator + records[index].text.length > available) continue;
      selected.add(index);
      used += separator + records[index].text.length;
    }
  }
  const body = [...selected]
    .sort((left, right) => left - right)
    .map((index) => records[index].text)
    .join("\n\n");
  return heading + body;
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
  "requester_memory_context",
  "untrusted_memory_context",
  "untrusted_reference_context",
]);
const SUMMARY_OMITTED_CONTEXT_BLOCKS = new Set([
  ...OMITTED_CONTEXT_BLOCKS,
  "host_conversation_continuity",
  "untrusted_conversation_context",
]);

function stripInjectedContext(value, omittedBlocks = OMITTED_CONTEXT_BLOCKS) {
  if (typeof value !== "string") return value;
  const lines = value.split("\n");
  const lastCloseByBlock = new Map();
  for (const [index, line] of lines.entries()) {
    const close = /^<\/([a-z_]+)>$/.exec(line)?.[1];
    if (omittedBlocks.has(close)) {
      lastCloseByBlock.set(close, index);
    }
  }
  const kept = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const open = /^<([a-z_]+)>$/.exec(line)?.[1];
    if (omittedBlocks.has(open)) {
      const closeIndex = lastCloseByBlock.get(open);
      if (closeIndex === undefined || closeIndex < index) break;
      index = closeIndex;
      continue;
    }
    kept.push(line);
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
