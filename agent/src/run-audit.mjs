import { createHash } from "node:crypto";
import {
  appendFile,
  mkdir,
  open,
  readdir,
} from "node:fs/promises";
import { join } from "node:path";

const RUN_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MAX_LIST_LIMIT = 100;
const MAX_EVENT_BYTES = 2 * 1024 * 1024;
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_STRING_CHARS = 1024 * 1024;
const MAX_ARRAY_ITEMS = 5_000;
const MAX_OBJECT_KEYS = 1_000;
const MAX_DEPTH = 16;

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
    return bounded(parsed.toString());
  } catch {
    return bounded(value);
  }
}

function bounded(value, max = MAX_STRING_CHARS) {
  const text = String(value ?? "");
  return text.length <= max ? text : `${text.slice(0, max - 3)}...`;
}

function imageMetadata(value) {
  const supplied = typeof value.data === "string" ? value.data : "";
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
  if (typeof value === "string") return bounded(value);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object") return bounded(value, 1_000);
  if (depth >= MAX_DEPTH) return "[DEPTH_LIMIT]";
  if (seen.has(value)) return "[CIRCULAR]";
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
    const event = {
      version: 1,
      sequence: (this.sequence += 1),
      timestamp: new Date().toISOString(),
      runId: this.runId,
      type: bounded(type, 128),
      data: sanitizeAuditValue(data),
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
      event.version !== 1 ||
      event.runId !== expectedRunId ||
      !Number.isInteger(event.sequence) ||
      typeof event.timestamp !== "string" ||
      typeof event.type !== "string"
    ) {
      throw new Error("Malformed run audit");
    }
    events.push(event);
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

function registerCapability(capabilities, value) {
  const capability = objectValue(value);
  const handle = stringValue(capability.handle, 32);
  if (!handle) return;
  capabilities.set(handle, {
    handle,
    displayName: stringValue(capability.displayName, 512),
    bankId: stringValue(capability.bankId, 512),
  });
}

function sourceCapabilities(events) {
  const capabilities = new Map();
  for (const event of events) {
    const data = objectValue(event.data);
    if (event.type === "memory.capabilities.issued") {
      for (const source of Array.isArray(data.sources) ? data.sources : []) {
        registerCapability(capabilities, source);
      }
    }
    if (
      event.type === "tool.completed" &&
      data.toolName === "memory_find_sources"
    ) {
      const references = resultDetails(data).references;
      for (const source of Array.isArray(references) ? references : []) {
        registerCapability(capabilities, source);
      }
    }
  }
  return capabilities;
}

function sourceForTool(name, args, details, capabilities) {
  if (name !== "memory_query_source") return null;
  const handle = stringValue(details.sourceHandle ?? args.reference, 32);
  const capability = handle ? capabilities.get(handle) : null;
  return {
    handle,
    displayName:
      stringValue(details.displayName, 512) ?? capability?.displayName ?? null,
    bankId: stringValue(details.bankId, 512) ?? capability?.bankId ?? null,
  };
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

  const capabilities = sourceCapabilities(events);
  return [...calls.values()]
    .map((call) => {
      const startedData = objectValue(call.started?.data);
      const completedData = objectValue(call.completed?.data);
      const name =
        stringValue(startedData.toolName, 128) ??
        stringValue(completedData.toolName, 128) ??
        "unknown";
      const args = {
        ...objectValue(startedData.args),
        ...objectValue(completedData.args),
      };
      const details = resultDetails(completedData);
      const failed =
        completedData.isError === true || details.unavailable === true;
      return {
        callId: call.callId,
        name,
        status: call.completed
          ? failed
            ? "failed"
            : "completed"
          : "in_progress",
        durationMs: durationValue(completedData.durationMs),
        query: stringValue(args.query ?? args.question, 2_000),
        source: sourceForTool(name, args, details, capabilities),
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
      const response = objectValue(data.response);
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
  const directoryRequestEvent = events.find((event) => {
    const data = objectValue(event.data);
    return (
      event.type === "memory.http.request" &&
      data.operation === "directory.recall" &&
      !data.toolCallId
    );
  }) ?? null;
  const terminalEvent = [...events].reverse().find(
    (event) => event.type === "run.completed" || event.type === "run.failed",
  ) ?? null;
  const completedEvent = terminalEvent?.type === "run.completed" ? terminalEvent : null;
  const failedEvent = terminalEvent?.type === "run.failed" ? terminalEvent : null;
  const request = objectValue(requestEvent?.data);
  const opened = objectValue(openedEvent?.data);
  const completed = objectValue(completedEvent?.data);
  const failed = objectValue(failedEvent?.data);
  const requestMemory = objectValue(request.memory);
  const context = objectValue(contextEvent?.data);
  const directory = objectValue(directoryEvent?.data);
  const capabilityData = objectValue(capabilityEvent?.data);
  const model = objectValue(objectValue(modelEvent?.data).model);
  const startedAt = stringValue(events[0]?.timestamp, 64);
  const finishedAt = stringValue(terminalEvent?.timestamp, 64);
  const primaryBankId = stringValue(
    requestMemory.primaryBankId ?? requestMemory.scopeId ?? context.primaryBankId,
    512,
  );
  const tools = summarizeTools(events);
  const directoryStatus = ["available", "unavailable", "disabled"].includes(
    directory.status,
  )
    ? directory.status
    : "unknown";
  const directoryRequest = objectValue(
    objectValue(directoryRequestEvent?.data).request,
  );
  const directoryBody = objectValue(directoryRequest.body);
  const initialSources = Array.isArray(capabilityData.sources)
    ? capabilityData.sources
    : [];
  const warnings = events
    .filter((event) => event.type === "memory.access.warning")
    .map((event) => {
      const data = objectValue(event.data);
      return {
        kind: "memory_access",
        unavailableBankCount: Array.isArray(data.unavailableBankIds)
          ? data.unavailableBankIds.length
          : 0,
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
    prompt: stringValue(request.prompt, 300) ?? "",
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
      primaryBankId,
      route: memoryRoute(primaryBankId, tools),
      initialRecall: contextEvent
        ? {
            status: initialRecallStatus(context, events),
            queries: (Array.isArray(context.queries) ? context.queries : [])
              .map((query) => stringValue(query, 2_000))
              .filter(Boolean)
              .slice(0, 32),
            memoryCount: Array.isArray(context.memories)
              ? context.memories.length
              : 0,
            eventSequence: contextEvent.sequence,
          }
        : null,
      directory: directoryEvent
        ? {
            status: directoryStatus,
            query: stringValue(directoryBody.query, 2_000),
            sourceCount: initialSources.length,
            eventSequence: directoryEvent.sequence,
          }
        : null,
    },
    tools,
    warnings,
    failure: failedEvent
      ? {
          code: stringValue(failed.code, 128) ?? "FAILED",
          message:
            stringValue(failed.message, 1_000) ??
            stringValue(objectValue(failed.error).message, 1_000) ??
            "Run failed",
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
    memoryScopeId: summary.memory.primaryBankId,
    eventCount: summary.eventCount,
  };
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
    const handle = await open(this.#path(runId), "wx", 0o600);
    await handle.close();
    return new RunAuditRecorder(this.#path(runId), runId);
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
    for (const name of names.slice(0, 20_000)) {
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
