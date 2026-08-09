import { createHash, randomUUID } from "node:crypto";
import {
  appendFile,
  chmod,
  mkdir,
  open,
  rename,
  readdir,
  unlink,
} from "node:fs/promises";
import { join } from "node:path";

import { redactSensitiveText } from "./privacy-redaction.mjs";

const RUN_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MAX_LIST_LIMIT = 100;
const MAX_EVENT_BYTES = 2 * 1024 * 1024;
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_STRING_CHARS = 1024 * 1024;
const MAX_ARRAY_ITEMS = 5_000;
const MAX_OBJECT_KEYS = 1_000;
const MAX_DEPTH = 16;
const CURRENT_AUDIT_VERSION = 2;

function safeString(value, max = 1_000) {
  return typeof value === "string" ? bounded(value, max) : null;
}

function safeInteger(value) {
  return Number.isSafeInteger(value) && value >= 0 ? value : null;
}

function suppliedCount(data, countKey, collectionKey) {
  return safeInteger(data[countKey]) ??
    (Array.isArray(data[collectionKey]) ? data[collectionKey].length : 0);
}

function suppliedChars(data, countKey, textKey) {
  return safeInteger(data[countKey]) ??
    (typeof data[textKey] === "string" ? data[textKey].length : 0);
}

function safeSourceHandle(value) {
  return typeof value === "string" && /^source_[0-9]+$/.test(value)
    ? value
    : null;
}

function modelMetadata(value) {
  const model = objectValue(value);
  return {
    id: safeString(model.id, 256),
    provider: safeString(model.provider, 128),
    api: safeString(model.api, 128),
    reasoning: model.reasoning === true,
    thinkingLevel: safeString(model.thinkingLevel, 64),
  };
}

function memoryHttpMetadata(type, data) {
  const result = {
    exchangeId: safeString(data.exchangeId, 128),
    operation: safeString(data.operation, 128),
    variant: safeString(data.variant, 128),
    toolCallId: safeString(data.toolCallId, 256),
  };
  if (type === "memory.http.request") {
    return {
      ...result,
      method: safeString(data.method ?? objectValue(data.request).method, 16),
    };
  }
  if (type === "memory.http.response") {
    const response = Object.keys(objectValue(data.response)).length > 0
      ? objectValue(data.response)
      : data;
    return {
      ...result,
      status: safeInteger(response.status),
      ok: response.ok === true,
      usable: response.usable === true,
      failureReason: safeString(response.failureReason, 128),
      durationMs: safeInteger(response.durationMs),
      bodyBytes: safeInteger(response.bodyBytes),
    };
  }
  return {
    ...result,
    durationMs: safeInteger(data.durationMs),
    errorName: safeString(objectValue(data.error).name, 128),
  };
}

export function minimizeAuditData(type, value = {}) {
  const data = objectValue(value);
  switch (type) {
    case "run.request":
      return {
        sessionId: safeString(data.sessionId, 128),
        parentEntryId: safeString(data.parentEntryId, 128),
        promptChars: suppliedChars(data, "promptChars", "prompt"),
        contextCount: suppliedCount(data, "contextCount", "context"),
        imageCount: safeInteger(data.imageCount) ?? 0,
        toolPolicy: safeString(data.toolPolicy, 32),
        model: safeString(data.model, 256),
        memoryEnabled:
          data.memoryEnabled === true ||
          objectValue(data.memory).primaryBankId != null ||
          objectValue(data.memory).scopeId != null,
        includeMemorySnapshot: data.includeMemorySnapshot === true,
      };
    case "memory.context":
      return {
        memoryEnabled:
          data.memoryEnabled === true || data.primaryBankId != null,
        queryCount: suppliedCount(data, "queryCount", "queries"),
        memoryCount: suppliedCount(data, "memoryCount", "memories"),
        recall: {
          status: safeString(objectValue(data.recall).status, 64),
        },
      };
    case "memory.directory.policy":
      return {
        requesterOwner:
          data.requesterOwner === true || objectValue(data.requester).owner === true,
        grantedBankCount: suppliedCount(
          data,
          "grantedBankCount",
          "grantedBankIds",
        ),
        participantCount: suppliedCount(
          data,
          "participantCount",
          "participants",
        ),
        allowedBankCount:
          data.allowedBankCount === null || data.allowedBankIds === null
            ? null
            : suppliedCount(data, "allowedBankCount", "allowedBankIds"),
      };
    case "memory.directory.result":
      return {
        status: safeString(data.status, 64),
        referenceCount: suppliedCount(data, "referenceCount", "references"),
      };
    case "memory.capabilities.issued":
      return {
        sourceCount: suppliedCount(data, "sourceCount", "sources"),
        stopReason: safeString(data.stopReason, 128),
      };
    case "memory.http.request":
    case "memory.http.response":
    case "memory.http.error":
      return memoryHttpMetadata(type, data);
    case "session.opened":
      return {
        sessionId: safeString(data.sessionId, 128),
        requestedSessionId: safeString(data.requestedSessionId, 128),
        parentEntryId: safeString(data.parentEntryId, 128),
        requestedParentEntryId: safeString(data.requestedParentEntryId, 128),
      };
    case "model.input":
      return {
        model: modelMetadata(data.model),
        tools: (Array.isArray(data.tools) ? data.tools : [])
          .map((tool) => safeString(tool, 128))
          .filter(Boolean)
          .slice(0, 64),
        promptChars: suppliedChars(data, "promptChars", "prompt"),
        sessionMessageCount: suppliedCount(
          data,
          "sessionMessageCount",
          "sessionMessagesBeforePrompt",
        ),
        imageCount: safeInteger(data.imageCount) ?? 0,
      };
    case "model.turn.started":
      return { turn: safeInteger(data.turn) };
    case "model.turn.completed":
      return {
        turn: safeInteger(data.turn),
        durationMs: safeInteger(data.durationMs),
        assistantTextChars:
          safeInteger(data.assistantTextChars) ??
          messageTextLength(data.message),
        toolResultCount: suppliedCount(
          data,
          "toolResultCount",
          "toolResults",
        ),
      };
    case "tool.started":
      return {
        turn: safeInteger(data.turn),
        toolCallId: safeString(data.toolCallId, 256),
        toolName: safeString(data.toolName, 128),
      };
    case "tool.completed": {
      const details = resultDetails(data);
      return {
        turn: safeInteger(data.turn),
        toolCallId: safeString(data.toolCallId, 256),
        toolName: safeString(data.toolName, 128),
        isError: data.isError === true,
        unavailable: data.unavailable === true || details.unavailable === true,
        durationMs: safeInteger(data.durationMs),
        sourceHandle: safeSourceHandle(
          data.sourceHandle ?? details.sourceHandle ?? objectValue(data.args).reference,
        ),
      };
    }
    case "memory.access.warning":
      return {
        unavailableBankCount:
          safeInteger(data.unavailableBankCount) ??
          (Array.isArray(data.unavailableBankIds)
            ? data.unavailableBankIds.length
            : 0),
      };
    case "memory.access.denied":
      return {
        historicalSourceCount: safeInteger(data.historicalSourceCount) ?? 0,
        unavailableSourceCount: safeInteger(data.unavailableSourceCount) ?? 0,
        reason: safeString(data.reason, 128),
      };
    case "run.completed":
      return {
        sessionId: safeString(data.sessionId, 128),
        entryId: safeString(data.entryId, 128),
        answerChars: suppliedChars(data, "answerChars", "answer"),
      };
    case "run.failed":
      return {
        code: safeString(data.code, 128) ?? "FAILED",
        sessionId: safeString(data.sessionId, 128),
      };
    case "audit.scrubbed":
      return { reason: safeString(data.reason, 128) ?? "legacy_unreadable" };
    default:
      return { omitted: true };
  }
}

function messageTextLength(message) {
  const content = objectValue(message).content;
  if (typeof content === "string") return content.length;
  if (!Array.isArray(content)) return 0;
  return content.reduce(
    (total, part) =>
      total +
      (part?.type === "text" && typeof part.text === "string"
        ? part.text.length
        : 0),
    0,
  );
}

function isPrivateKey(key) {
  const normalized = key.replace(/[^a-z0-9]/gi, "").toLowerCase();
  return (
    normalized === "authorization" ||
    normalized === "cookie" ||
    normalized === "setcookie" ||
    normalized === "password" ||
    normalized === "errormessage" ||
    normalized === "thinkingsignature" ||
    normalized === "apikey" ||
    normalized.endsWith("token") ||
    normalized.includes("secret")
  );
}

function sanitizedUrl(value) {
  if (typeof value !== "string") return sanitizeAuditValue(value);
  try {
    const parsed = new URL(value);
    parsed.username = "";
    parsed.password = "";
    for (const key of parsed.searchParams.keys()) {
      const normalized = key.replace(/[^a-z0-9]/gi, "").toLowerCase();
      if (
        isPrivateKey(key) ||
        normalized === "key" ||
        normalized === "sig" ||
        normalized.includes("signature")
      ) {
        parsed.searchParams.set(key, "[REDACTED]");
      }
    }
    return redactSensitiveText(bounded(parsed.toString()));
  } catch {
    return redactSensitiveText(bounded(value));
  }
}

function bounded(value, max = MAX_STRING_CHARS) {
  const text = String(value ?? "");
  return text.length <= max ? text : `${text.slice(0, max - 3)}...`;
}

function imageMetadata(value) {
  const supplied = typeof value.data === "string" ? value.data : "";
  if (
    supplied === "[OMITTED]" &&
    Number.isInteger(value.sizeBytes) &&
    value.sizeBytes >= 0
  ) {
    return {
      type: "image",
      mimeType: bounded(value.mimeType, 256),
      sizeBytes: value.sizeBytes,
      data: "[OMITTED]",
    };
  }
  const padding = supplied.endsWith("==") ? 2 : supplied.endsWith("=") ? 1 : 0;
  return {
    type: "image",
    mimeType: bounded(value.mimeType, 256),
    sizeBytes: Math.max(0, Math.floor((supplied.length * 3) / 4) - padding),
    data: "[OMITTED]",
  };
}

export function sanitizeAuditValue(value, depth = 0, seen = new WeakSet()) {
  if (value === null || value === undefined) return value ?? null;
  if (typeof value === "string") return redactSensitiveText(bounded(value));
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object") return bounded(value, 1_000);
  if (depth >= MAX_DEPTH) return "[DEPTH_LIMIT]";
  if (seen.has(value)) return "[CIRCULAR]";
  if (value.type === "thinking") {
    return { type: "thinking", thinking: "[OMITTED]" };
  }
  seen.add(value);
  try {
    if (value.type === "image" && "data" in value) return imageMetadata(value);
    if (Array.isArray(value)) {
      return value
        .slice(0, MAX_ARRAY_ITEMS)
        .map((item) => sanitizeAuditValue(item, depth + 1, seen));
    }
    const result = {};
    for (const [key, item] of Object.entries(value).slice(0, MAX_OBJECT_KEYS)) {
      const normalized = key.replace(/[^a-z0-9]/gi, "").toLowerCase();
      if (isPrivateKey(key)) result[key] = "[REDACTED]";
      else if (normalized === "url" || normalized.endsWith("url")) {
        result[key] = sanitizedUrl(item);
      } else if (
        (normalized === "urls" || normalized.endsWith("urls")) &&
        Array.isArray(item)
      ) {
        result[key] = item
          .slice(0, MAX_ARRAY_ITEMS)
          .map((url) => sanitizedUrl(url));
      } else result[key] = sanitizeAuditValue(item, depth + 1, seen);
    }
    return result;
  } finally {
    seen.delete(value);
  }
}

function encodeEvent(event) {
  const line = JSON.stringify(event);
  if (Buffer.byteLength(line) <= MAX_EVENT_BYTES) return `${line}\n`;
  const digest = createHash("sha256").update(line).digest("hex");
  return `${JSON.stringify({
    ...event,
    data: {
      truncated: true,
      originalBytes: Buffer.byteLength(line),
      sha256: digest,
    },
  })}\n`;
}

class RunAuditRecorder {
  constructor(path, runId) {
    this.path = path;
    this.runId = runId;
    this.sequence = 0;
    this.tail = Promise.resolve();
  }

  record(type, data = {}) {
    const safeType = bounded(type, 128);
    const event = {
      version: CURRENT_AUDIT_VERSION,
      sequence: (this.sequence += 1),
      timestamp: new Date().toISOString(),
      runId: this.runId,
      type: safeType,
      data: sanitizeAuditValue(minimizeAuditData(safeType, data)),
    };
    const line = encodeEvent(event);
    this.tail = this.tail.then(() =>
      appendFile(this.path, line, { encoding: "utf8", mode: 0o600 }),
    );
    return this.tail;
  }

  flush() {
    return this.tail;
  }
}

function parseAudit(content, expectedRunId) {
  const events = [];
  const lines = content.split("\n");
  for (const [index, line] of lines.entries()) {
    if (!line.trim()) continue;
    let event;
    try {
      event = JSON.parse(line);
    } catch (error) {
      if (index === lines.length - 1 && !content.endsWith("\n")) break;
      throw error;
    }
    if (
      !event ||
      (event.version !== 1 && event.version !== CURRENT_AUDIT_VERSION) ||
      event.runId !== expectedRunId ||
      !Number.isInteger(event.sequence) ||
      typeof event.timestamp !== "string" ||
      typeof event.type !== "string"
    ) {
      throw new Error("Malformed run audit");
    }
    events.push({
      ...event,
      version: CURRENT_AUDIT_VERSION,
      data: sanitizeAuditValue(minimizeAuditData(event.type, event.data)),
    });
  }
  events.sort((left, right) => left.sequence - right.sequence);
  return events;
}

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

function stringValue(value, max = 1_000) {
  return typeof value === "string" ? bounded(value, max) : null;
}

function durationValue(value) {
  return Number.isInteger(value) && value >= 0 ? value : null;
}

function elapsedMs(startedAt, finishedAt) {
  const start = Date.parse(startedAt ?? "");
  const finish = Date.parse(finishedAt ?? "");
  return Number.isFinite(start) && Number.isFinite(finish) && finish >= start
    ? finish - start
    : null;
}

function lastEvent(events, type) {
  return [...events].reverse().find((event) => event.type === type) ?? null;
}

function resultDetails(data) {
  return objectValue(objectValue(data.result).details);
}

function summarizeTools(events) {
  const calls = new Map();
  for (const event of events) {
    if (event.type !== "tool.started" && event.type !== "tool.completed") {
      continue;
    }
    const data = objectValue(event.data);
    const callId = stringValue(data.toolCallId, 256);
    const key = callId ?? `${event.type}:${event.sequence}`;
    const call = calls.get(key) ?? { callId, started: null, completed: null };
    if (event.type === "tool.started" && !call.started) call.started = event;
    if (event.type === "tool.completed") call.completed = event;
    calls.set(key, call);
  }

  return [...calls.values()]
    .map((call) => {
      const startedData = objectValue(call.started?.data);
      const completedData = objectValue(call.completed?.data);
      const name =
        stringValue(startedData.toolName, 128) ??
        stringValue(completedData.toolName, 128) ??
        "unknown";
      const failed =
        completedData.isError === true || completedData.unavailable === true;
      const sourceHandle = safeSourceHandle(completedData.sourceHandle);
      return {
        callId: call.callId,
        name,
        status: call.completed
          ? failed
            ? "failed"
            : "completed"
          : "in_progress",
        durationMs: durationValue(completedData.durationMs),
        query: null,
        source:
          name === "memory_query_source"
            ? { handle: sourceHandle, displayName: null, bankId: null }
            : null,
        eventSequence: call.started?.sequence ?? call.completed?.sequence ?? null,
      };
    })
    .sort((left, right) =>
      (left.eventSequence ?? Number.MAX_SAFE_INTEGER) -
      (right.eventSequence ?? Number.MAX_SAFE_INTEGER),
    );
}

function initialRecallStatus(context, events) {
  const recordedStatus = objectValue(context.recall).status;
  if (
    ["unknown", "in_progress", "completed", "partial", "failed"].includes(
      recordedStatus,
    )
  ) {
    return recordedStatus;
  }
  const exchanges = new Map();
  for (const event of events) {
    if (!event.type.startsWith("memory.http.")) continue;
    const data = objectValue(event.data);
    if (data.operation !== "recall" || data.toolCallId) continue;
    const key = stringValue(data.exchangeId, 256) ?? `${event.type}:${event.sequence}`;
    if (event.type === "memory.http.request" && !exchanges.has(key)) {
      exchanges.set(key, "in_progress");
    } else if (event.type === "memory.http.response") {
      const response = Object.keys(objectValue(data.response)).length > 0
        ? objectValue(data.response)
        : data;
      exchanges.set(
        key,
        response.usable === true
          ? "completed"
          : response.usable === false || response.ok === false
            ? "failed"
            : "unknown",
      );
    } else if (event.type === "memory.http.error") {
      exchanges.set(key, "failed");
    }
  }
  const statuses = [...exchanges.values()];
  if (statuses.length === 0) return "unknown";
  if (statuses.includes("in_progress")) return "in_progress";
  if (statuses.includes("unknown")) return "unknown";
  const completed = statuses.includes("completed");
  const failed = statuses.includes("failed");
  if (completed && failed) return "partial";
  return completed ? "completed" : "failed";
}

function memoryRoute(primaryBankId, tools) {
  if (!primaryBankId) return "off";
  const sourceTools = tools.filter(
    (tool) => tool.name === "memory_query_source",
  );
  if (sourceTools.length > 0) {
    if (sourceTools.some((tool) => tool.status === "completed")) {
      return "cross_bank_queried";
    }
    return sourceTools.every((tool) => tool.status === "failed")
      ? "cross_bank_failed"
      : "cross_bank_attempted";
  }
  const discoveryTools = tools.filter(
    (tool) => tool.name === "memory_find_sources",
  );
  if (discoveryTools.some((tool) => tool.status === "completed")) {
    return "source_discovery_only";
  }
  if (discoveryTools.length > 0) {
    return discoveryTools.every((tool) => tool.status === "failed")
      ? "cross_bank_failed"
      : "cross_bank_attempted";
  }
  return "current_bank_only";
}

function summarize(events) {
  const requestEvent = events.find((event) => event.type === "run.request") ?? null;
  const openedEvent = lastEvent(events, "session.opened");
  const modelEvent = lastEvent(events, "model.input");
  const contextEvent = events.find((event) => event.type === "memory.context") ?? null;
  const directoryEvent = events.find(
    (event) => event.type === "memory.directory.result",
  ) ?? null;
  const capabilityEvent = events.find(
    (event) => event.type === "memory.capabilities.issued",
  ) ?? null;
  const terminalEvent = [...events].reverse().find(
    (event) => event.type === "run.completed" || event.type === "run.failed",
  ) ?? null;
  const completedEvent = terminalEvent?.type === "run.completed" ? terminalEvent : null;
  const failedEvent = terminalEvent?.type === "run.failed" ? terminalEvent : null;
  const request = objectValue(requestEvent?.data);
  const opened = objectValue(openedEvent?.data);
  const completed = objectValue(completedEvent?.data);
  const failed = objectValue(failedEvent?.data);
  const context = objectValue(contextEvent?.data);
  const directory = objectValue(directoryEvent?.data);
  const capabilityData = objectValue(capabilityEvent?.data);
  const model = objectValue(objectValue(modelEvent?.data).model);
  const startedAt = stringValue(events[0]?.timestamp, 64);
  const finishedAt = stringValue(terminalEvent?.timestamp, 64);
  const memoryEnabled = request.memoryEnabled === true || context.memoryEnabled === true;
  const tools = summarizeTools(events);
  const directoryStatus = ["available", "unavailable", "disabled"].includes(
    directory.status,
  )
    ? directory.status
    : "unknown";
  const warnings = events
    .filter((event) => event.type === "memory.access.warning")
    .map((event) => {
      const data = objectValue(event.data);
      return {
        kind: "memory_access",
        unavailableBankCount: safeInteger(data.unavailableBankCount) ?? 0,
        eventSequence: event.sequence,
      };
    });
  const sessionId = stringValue(
    completed.sessionId ?? opened.sessionId ?? request.sessionId,
    128,
  );
  const parentEntryId = stringValue(
    opened.parentEntryId ?? request.parentEntryId,
    128,
  );

  return {
    status: terminalEvent
      ? terminalEvent.type === "run.completed"
        ? "completed"
        : "failed"
      : "in_progress",
    startedAt,
    finishedAt,
    durationMs: elapsedMs(startedAt, finishedAt),
    prompt: "",
    eventCount: events.length,
    session: {
      kind:
        request.sessionId != null ||
        request.parentEntryId != null ||
        parentEntryId != null
          ? "continuation"
          : "root",
      id: sessionId,
      parentEntryId,
      entryId: stringValue(completed.entryId, 128),
    },
    model: modelEvent
      ? {
          id: stringValue(model.id, 256),
          provider: stringValue(model.provider, 128),
          thinkingLevel: stringValue(model.thinkingLevel, 64),
        }
      : null,
    memory: {
      enabled: memoryEnabled,
      primaryBankId: null,
      route: memoryRoute(memoryEnabled ? "enabled" : null, tools),
      initialRecall: contextEvent
        ? {
            status: initialRecallStatus(context, events),
            queries: [],
            queryCount: safeInteger(context.queryCount) ?? 0,
            memoryCount: safeInteger(context.memoryCount) ?? 0,
            eventSequence: contextEvent.sequence,
          }
        : null,
      directory: directoryEvent
        ? {
            status: directoryStatus,
            query: null,
            sourceCount:
              safeInteger(capabilityData.sourceCount) ??
              safeInteger(directory.referenceCount) ??
              0,
            eventSequence: directoryEvent.sequence,
          }
        : null,
    },
    tools,
    warnings,
    failure: failedEvent
      ? {
          code: stringValue(failed.code, 128) ?? "FAILED",
          message: "Run failed",
          eventSequence: failedEvent.sequence,
        }
      : null,
  };
}

function listSummary(runId, summary) {
  return {
    runId,
    sessionId: summary.session.id,
    entryId: summary.session.entryId,
    status: summary.status,
    startedAt: summary.startedAt,
    finishedAt: summary.finishedAt,
    prompt: summary.prompt,
    memoryEnabled: summary.memory.enabled,
    memoryScopeId: null,
    eventCount: summary.eventCount,
  };
}

function unreadableAuditEvent(runId) {
  return {
    version: CURRENT_AUDIT_VERSION,
    sequence: 1,
    timestamp: new Date().toISOString(),
    runId,
    type: "audit.scrubbed",
    data: { reason: "legacy_unreadable" },
  };
}

async function replaceFile(path, content) {
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

export class RunAuditStore {
  constructor(directory) {
    this.directory = directory;
  }

  #path(runId) {
    return join(this.directory, `${runId}.jsonl`);
  }

  async start(runId) {
    if (!RUN_ID_RE.test(runId ?? "")) throw new Error("Invalid run id");
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    await chmod(this.directory, 0o700);
    const handle = await open(this.#path(runId), "wx", 0o600);
    await handle.close();
    return new RunAuditRecorder(this.#path(runId), runId);
  }

  async scrub() {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    await chmod(this.directory, 0o700);
    const names = await readdir(this.directory);
    let scanned = 0;
    let rewritten = 0;
    for (const name of names) {
      if (!name.endsWith(".jsonl")) continue;
      const runId = name.slice(0, -6);
      if (!RUN_ID_RE.test(runId)) continue;
      scanned += 1;
      const path = this.#path(runId);
      const handle = await open(path, "r");
      let raw;
      let mode;
      let tooLarge;
      try {
        const info = await handle.stat();
        mode = info.mode & 0o777;
        tooLarge = !info.isFile() || info.size > MAX_FILE_BYTES;
        raw = tooLarge ? "" : await handle.readFile("utf8");
      } finally {
        await handle.close();
      }
      let events;
      try {
        events = tooLarge
          ? [unreadableAuditEvent(runId)]
          : parseAudit(raw, runId);
      } catch {
        events = [unreadableAuditEvent(runId)];
      }
      const safe = events.map((event) => encodeEvent(event)).join("");
      if (tooLarge || raw !== safe) {
        await replaceFile(path, safe);
        rewritten += 1;
      } else if (mode !== 0o600) {
        await chmod(path, 0o600);
        rewritten += 1;
      }
    }
    return { scanned, rewritten };
  }

  async get(runId) {
    if (!RUN_ID_RE.test(runId ?? "")) return null;
    const path = this.#path(runId);
    let handle;
    try {
      handle = await open(path, "r");
      const info = await handle.stat();
      if (!info.isFile() || info.size > MAX_FILE_BYTES) return null;
      const events = parseAudit(await handle.readFile("utf8"), runId);
      return { runId, summary: summarize(events), events };
    } catch {
      return null;
    } finally {
      await handle?.close();
    }
  }

  async list({ limit = 50, cursor = null, sessionId = null } = {}) {
    const pageSize = Math.max(1, Math.min(MAX_LIST_LIMIT, Number(limit) || 50));
    let names;
    try {
      names = await readdir(this.directory);
    } catch {
      names = [];
    }
    const summaries = [];
    for (const name of names) {
      if (!name.endsWith(".jsonl")) continue;
      const runId = name.slice(0, -6);
      if (!RUN_ID_RE.test(runId)) continue;
      const audit = await this.get(runId);
      if (!audit || audit.events.length === 0) continue;
      const item = listSummary(runId, audit.summary);
      if (sessionId && item.sessionId !== sessionId) continue;
      summaries.push(item);
    }
    summaries.sort((left, right) => {
      const started = String(right.startedAt).localeCompare(String(left.startedAt));
      return started || left.runId.localeCompare(right.runId);
    });
    const cursorIndex = cursor
      ? summaries.findIndex((item) => item.runId === cursor)
      : -1;
    const start = cursor ? (cursorIndex < 0 ? summaries.length : cursorIndex + 1) : 0;
    const selected = summaries.slice(start, start + pageSize);
    return {
      items: selected,
      total: summaries.length,
      nextCursor:
        start + selected.length < summaries.length
          ? selected.at(-1)?.runId ?? null
          : null,
    };
  }
}
