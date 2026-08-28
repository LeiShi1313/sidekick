import assert from "node:assert/strict";
import {
  appendFile,
  chmod,
  mkdtemp,
  readFile,
  rm,
  stat,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { RunAuditStore } from "../src/run-audit.mjs";

const RUN_ID = "11111111-1111-4111-8111-111111111111";
const SECOND_RUN_ID = "22222222-2222-4222-8222-222222222222";

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "sidekick-run-audit-"));
  return {
    root,
    store: new RunAuditStore(root),
    close: () => rm(root, { recursive: true, force: true }),
  };
}

test("records ordered append-only operational events", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await Promise.all([
      audit.record("run.request", {
        prompt: "Who owns deploys?",
        sessionId: null,
        systemPrompt: "Answer carefully.",
        memory: { scopeId: "chat:engineering" },
        authorization: "Bearer secret",
        provider: { errorMessage: "provider credential detail" },
        callbackUrl: "https://user:pass@example.test/path?api_key=query-secret&view=full",
        image: {
          type: "image",
          mimeType: "image/png",
          data: "aW1hZ2UtcGF5bG9hZA==",
        },
      }),
      audit.record("memory.http.request", {
        exchangeId: "recall-plain",
        request: {
          method: "POST",
          url: "http://memory/v1/default/banks/chat/memories/recall",
          body: { query: "Who owns deploys?", apiKey: "secret-key" },
        },
      }),
    ]);
    await audit.record("session.opened", {
      sessionId: "session-1",
      parentEntryId: null,
    });
    await audit.record("run.completed", {
      sessionId: "session-1",
      entryId: "entry-1",
      answer:
        "Alice owns deploys from 203.0.113.42 at " +
        "/home/example-service/private/workspace.",
    });
    await audit.flush();

    const result = await app.store.get(RUN_ID);
    assert.equal(result.runId, RUN_ID);
    assert.deepEqual(result.events.map((event) => event.sequence), [1, 2, 3, 4]);
    assert.deepEqual(result.events.map((event) => event.type), [
      "run.request",
      "memory.http.request",
      "session.opened",
      "run.completed",
    ]);
    assert.equal(result.events[0].data.promptChars, 17);
    assert.equal(result.events[0].data.memoryEnabled, true);
    assert.equal(result.events[1].data.method, "POST");
    assert.doesNotMatch(
      JSON.stringify(result),
      /Bearer secret|secret-key|provider credential detail|query-secret|user:pass|203\.0\.113\.42|example-service/,
    );
  } finally {
    await app.close();
  }
});

test("persists only allowlisted operational audit metadata", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "PRIVATE_USER_PROMPT",
      context: [{ kind: "reference", text: "PRIVATE_QUOTED_CONTEXT" }],
      systemPrompt: "PRIVATE_SYSTEM_PROMPT",
      sessionId: null,
      toolPolicy: "delegated",
      origin: { scopeId: "PRIVATE_CHAT_ID" },
      memory: { primaryBankId: "PRIVATE_MEMORY_BANK" },
    });
    await audit.record("model.input", {
      model: { id: "test-model", provider: "test-provider" },
      prompt: "PRIVATE_ENRICHED_PROMPT",
      systemPrompt: "PRIVATE_MODEL_SYSTEM_PROMPT",
      sessionMessagesBeforePrompt: ["PRIVATE_SESSION_HISTORY"],
      tools: ["web_search"],
    });
    await audit.record("tool.completed", {
      turn: 1,
      toolCallId: "call-1",
      toolName: "web_search",
      args: { query: "PRIVATE_TOOL_ARGUMENT" },
      result: { content: "PRIVATE_TOOL_RESULT" },
      isError: false,
      durationMs: 5,
    });
    await audit.record("run.completed", {
      sessionId: "session-1",
      entryId: "entry-1",
      answer: "PRIVATE_FINAL_ANSWER",
    });
    await audit.flush();

    const raw = await readFile(`${app.root}/${RUN_ID}.jsonl`, "utf8");
    assert.doesNotMatch(
      raw,
      /PRIVATE_USER_PROMPT|PRIVATE_QUOTED_CONTEXT|PRIVATE_SYSTEM_PROMPT|PRIVATE_CHAT_ID|PRIVATE_MEMORY_BANK|PRIVATE_ENRICHED_PROMPT|PRIVATE_MODEL_SYSTEM_PROMPT|PRIVATE_SESSION_HISTORY|PRIVATE_TOOL_ARGUMENT|PRIVATE_TOOL_RESULT|PRIVATE_FINAL_ANSWER/,
    );
    const result = await app.store.get(RUN_ID);
    assert.equal(result.events[0].version, 2);
    assert.deepEqual(result.events[0].data, {
      sessionId: null,
      parentEntryId: null,
      promptChars: 19,
      contextCount: 1,
      imageCount: 0,
      toolPolicy: "delegated",
      model: null,
      memoryEnabled: true,
      includeMemorySnapshot: false,
    });
    assert.deepEqual(result.events[1].data, {
      model: {
        id: "test-model",
        provider: "test-provider",
        api: null,
        reasoning: false,
        thinkingLevel: null,
      },
      tools: ["web_search"],
      promptChars: 23,
      sessionMessageCount: 1,
      imageCount: 0,
    });
    assert.deepEqual(result.events[2].data, {
      turn: 1,
      toolCallId: "call-1",
      toolName: "web_search",
      isError: false,
      unavailable: false,
      durationMs: 5,
      sourceHandle: null,
    });
    assert.deepEqual(result.events[3].data, {
      sessionId: "session-1",
      entryId: "entry-1",
      answerChars: 20,
    });
  } finally {
    await app.close();
  }
});

test("records the current requester owner attestation", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("memory.directory.policy", {
      requesterIsOwner: true,
      grantedBankIds: [],
      participants: [],
      allowedBankIds: null,
    });
    await audit.flush();

    const result = await app.store.get(RUN_ID);
    assert.equal(result.events[0].data.requesterOwner, true);
  } finally {
    await app.close();
  }
});

test("records requester-memory outcomes without tool arguments or content", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("tool.completed", {
      toolCallId: "call-requester-memory",
      toolName: "memory_update_requester",
      args: { customization: "PRIVATE_CUSTOMIZATION_ARGUMENT" },
      result: {
        content: "PRIVATE_CUSTOMIZATION_RESULT",
        details: { saved: true, cleared: false, conflict: false },
      },
      isError: false,
      durationMs: 7,
    });
    await audit.flush();

    const result = await app.store.get(RUN_ID);
    assert.deepEqual(result.events[0].data, {
      turn: null,
      toolCallId: "call-requester-memory",
      toolName: "memory_update_requester",
      isError: false,
      unavailable: false,
      saved: true,
      cleared: false,
      conflict: false,
      durationMs: 7,
      sourceHandle: null,
    });
    assert.doesNotMatch(
      JSON.stringify(result),
      /PRIVATE_CUSTOMIZATION_ARGUMENT|PRIVATE_CUSTOMIZATION_RESULT/,
    );
  } finally {
    await app.close();
  }
});

test("rewrites legacy audit payloads through the current allowlist", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.flush();
    await appendFile(
      `${app.root}/${RUN_ID}.jsonl`,
      `${JSON.stringify({
        version: 1,
        sequence: 1,
        timestamp: new Date().toISOString(),
        runId: RUN_ID,
        type: "run.request",
        data: {
          prompt: "PRIVATE_LEGACY_PROMPT",
          systemPrompt: "PRIVATE_LEGACY_SYSTEM_PROMPT",
        },
      })}\n`,
    );
    await chmod(`${app.root}/${RUN_ID}.jsonl`, 0o644);

    const result = await app.store.scrub();
    const raw = await readFile(`${app.root}/${RUN_ID}.jsonl`, "utf8");

    assert.deepEqual(result, { scanned: 1, rewritten: 1 });
    assert.doesNotMatch(raw, /PRIVATE_LEGACY/);
    assert.match(raw, /"version":2/);
    assert.equal((await stat(`${app.root}/${RUN_ID}.jsonl`)).mode & 0o777, 0o600);
  } finally {
    await app.close();
  }
});

test("redacts sensitive values while reading legacy audit events", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", { prompt: "Inspect the run" });
    await audit.flush();
    await appendFile(
      `${app.root}/${RUN_ID}.jsonl`,
      `${JSON.stringify({
        version: 1,
        sequence: 2,
        timestamp: new Date().toISOString(),
        runId: RUN_ID,
        type: "run.completed",
        data: {
          answer:
            "Legacy 203.0.113.42 at /home/example-service/private/workspace",
        },
      })}\n`,
    );

    const result = await app.store.get(RUN_ID);
    const serialized = JSON.stringify(result);
    assert.doesNotMatch(serialized, /203\.0\.113\.42|example-service/);
    assert.equal(result.events[1].data.answerChars, 62);
  } finally {
    await app.close();
  }
});

test("recovers complete events before a crash-truncated final line", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", { prompt: "Recover me" });
    await audit.flush();
    await appendFile(`${app.root}/${RUN_ID}.jsonl`, '{"version":1,"sequence":2');

    const result = await app.store.get(RUN_ID);
    assert.deepEqual(result.events.map((event) => event.type), ["run.request"]);
  } finally {
    await app.close();
  }
});

test("lists run summaries by session and reports terminal state", async () => {
  const app = await fixture();
  try {
    const first = await app.store.start(RUN_ID);
    await first.record("run.request", {
      prompt: "First run",
      sessionId: null,
      memory: { scopeId: "chat:engineering" },
    });
    await first.record("session.opened", { sessionId: "session-1" });
    await first.record("run.completed", {
      sessionId: "session-1",
      entryId: "entry-1",
    });
    await first.flush();

    const secondId = "22222222-2222-4222-8222-222222222222";
    const second = await app.store.start(secondId);
    await second.record("run.request", {
      prompt: "Second run",
      sessionId: "session-2",
    });
    await second.record("run.failed", { code: "PROVIDER_ERROR" });
    await second.flush();

    const page = await app.store.list({ limit: 20, sessionId: "session-1" });
    assert.equal(page.total, 1);
    assert.equal(page.items[0].runId, RUN_ID);
    assert.equal(page.items[0].sessionId, "session-1");
    assert.equal(page.items[0].entryId, "entry-1");
    assert.equal(page.items[0].status, "completed");
    assert.equal(page.items[0].memoryEnabled, true);
    assert.equal(page.items[0].memoryScopeId, null);
    assert.equal(page.items[0].eventCount, 3);

    const all = await app.store.list({ limit: 20 });
    assert.equal(all.total, 2);
    assert(all.items.some((item) => item.status === "failed"));
  } finally {
    await app.close();
  }
});

test("projects current-bank run decisions into a diagnostic summary", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "Did dog bro appear today?",
      sessionId: null,
      parentEntryId: null,
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await audit.record("memory.context", {
      primaryBankId: "telegram:chat:-1001",
      queries: [
        "2026-07-23 dog bro",
        "Requester personalization context for chat:user:alice",
      ],
      memories: [
        {
          id: "memory-1",
          text: "Dog bro appeared in the room today.",
          type: "observation",
          entities: ["chat:user:alice"],
          occurredStart: "2026-07-23",
          occurredEnd: null,
          mentionedAt: "2026-07-23T10:30:00Z",
          documentId: "conversation:7",
          chunkId: "chunk-7",
        },
      ],
      recall: {
        status: "completed",
        attemptedCount: 2,
        completedCount: 2,
        failedCount: 0,
      },
      customizations: ["Call me Captain.", "Use headings by default."],
      requesterMemory: {
        customizationStatus: "available",
        ownerCustomizationStatus: "available",
        evidenceStatus: "completed",
      },
      renderedContext: "PRIVATE_DUPLICATE_RENDERED_CONTEXT",
      renderedRequesterContext: "PRIVATE_REQUESTER_RENDERED_CONTEXT",
      access: { token: "PRIVATE_MEMORY_ACCESS_TOKEN" },
    });
    await audit.record("memory.http.request", {
      operation: "directory.recall",
      toolCallId: null,
      request: { body: { query: "Did dog bro appear today?" } },
    });
    await audit.record("memory.directory.result", {
      status: "available",
      references: [{ bankId: "telegram:chat:-2002" }],
    });
    await audit.record("memory.capabilities.issued", {
      sources: [
        {
          handle: "source_1",
          bankId: "telegram:chat:-2002",
          displayName: "Other group",
        },
      ],
    });
    await audit.record("session.opened", {
      sessionId: "session-1",
      parentEntryId: null,
    });
    await audit.record("model.input", {
      model: {
        id: "gpt-5",
        provider: "openai",
        thinkingLevel: "high",
      },
    });
    await audit.record("tool.started", {
      toolCallId: "call-current-1",
      toolName: "memory_query_current",
      args: { query: "2026-07-23 @dota2pp" },
    });
    await audit.record("memory.http.response", {
      operation: "current.recall",
      toolCallId: "call-current-1",
      response: { ok: true, status: 200, durationMs: 9 },
    });
    await audit.record("tool.completed", {
      toolCallId: "call-current-1",
      toolName: "memory_query_current",
      args: { query: "2026-07-23 @dota2pp" },
      result: {
        details: {
          bankId: "telegram:chat:-1001",
          memoryIds: ["memory-2"],
        },
      },
      isError: false,
      durationMs: 12,
    });
    await audit.record("run.completed", {
      sessionId: "session-1",
      entryId: "entry-1",
      answer: "Yes.",
    });
    await audit.flush();

    const { summary } = await app.store.get(RUN_ID);

    assert.equal(summary.status, "completed");
    assert.equal(summary.prompt, "");
    assert(Number.isInteger(summary.durationMs));
    assert.deepEqual(summary.session, {
      kind: "root",
      id: "session-1",
      parentEntryId: null,
      entryId: "entry-1",
    });
    assert.deepEqual(summary.model, {
      id: "gpt-5",
      provider: "openai",
      thinkingLevel: "high",
    });
    assert.deepEqual(summary.memory, {
      enabled: true,
      primaryBankId: "telegram:chat:-1001",
      route: "current_bank_only",
      initialRecall: {
        status: "completed",
        queries: [
          "2026-07-23 dog bro",
          "Requester personalization context for chat:user:alice",
        ],
        memories: [
          {
            id: "memory-1",
            text: "Dog bro appeared in the room today.",
            type: "observation",
            entities: ["chat:user:alice"],
            occurredStart: "2026-07-23",
            occurredEnd: null,
            mentionedAt: "2026-07-23T10:30:00Z",
            documentId: "conversation:7",
            chunkId: "chunk-7",
          },
        ],
        queryCount: 2,
        memoryCount: 1,
        eventSequence: 2,
      },
      directory: {
        status: "available",
        query: null,
        sourceCount: 1,
        eventSequence: 4,
      },
    });
    assert.deepEqual(summary.tools, [
      {
        callId: "call-current-1",
        name: "memory_query_current",
        status: "completed",
        durationMs: 12,
        query: null,
        source: null,
        eventSequence: 8,
      },
    ]);
    assert.deepEqual(summary.warnings, []);
    assert.equal(summary.failure, null);
    const detail = await app.store.get(RUN_ID);
    const memoryEvent = detail.events.find(
      (event) => event.type === "memory.context",
    );
    assert.equal(memoryEvent.data.primaryBankId, "telegram:chat:-1001");
    assert.deepEqual(memoryEvent.data.queries, summary.memory.initialRecall.queries);
    assert.deepEqual(memoryEvent.data.memories, summary.memory.initialRecall.memories);
    assert.deepEqual(memoryEvent.data.recall, {
      status: "completed",
      attemptedCount: 2,
      completedCount: 2,
      failedCount: 0,
    });
    assert.equal(memoryEvent.data.customizationCount, 2);
    assert.equal("customizations" in memoryEvent.data, false);
    assert.deepEqual(memoryEvent.data.requesterMemory, {
      customizationStatus: "available",
      ownerCustomizationStatus: "available",
      evidenceStatus: "completed",
    });
    assert.doesNotMatch(
      JSON.stringify(detail),
      /Call me Captain|Use headings by default|PRIVATE_DUPLICATE_RENDERED_CONTEXT|PRIVATE_REQUESTER_RENDERED_CONTEXT|PRIVATE_MEMORY_ACCESS_TOKEN/,
    );
  } finally {
    await app.close();
  }
});

test("distinguishes source discovery from successful cross-bank retrieval", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "Go to the Arch group and check the release discussion",
      sessionId: "session-1",
      parentEntryId: "entry-parent",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await audit.record("memory.context", { queries: ["release"], memories: [] });
    await audit.record("memory.directory.result", {
      status: "available",
      references: [{ bankId: "telegram:chat:-2002" }],
    });
    await audit.record("memory.capabilities.issued", {
      sources: [
        {
          handle: "source_1",
          bankId: "telegram:chat:-2002",
          displayName: "Known group",
        },
      ],
    });
    await audit.record("session.opened", {
      sessionId: "session-1",
      parentEntryId: "entry-parent",
    });
    await audit.record("model.input", {
      model: { id: "gpt-5", provider: "openai", thinkingLevel: "medium" },
    });
    await audit.record("tool.started", {
      toolCallId: "call-find-1",
      toolName: "memory_find_sources",
      args: { query: "Arch group" },
    });
    await audit.record("tool.completed", {
      toolCallId: "call-find-1",
      toolName: "memory_find_sources",
      args: { query: "Arch group" },
      result: {
        details: {
          references: [
            {
              handle: "source_2",
              bankId: "telegram:chat:-3003",
              displayName: "Arch Linux 中文群",
            },
          ],
        },
      },
      isError: false,
      durationMs: 14,
    });
    await audit.record("tool.started", {
      toolCallId: "call-source-1",
      toolName: "memory_query_source",
      args: { reference: "source_2", query: "release discussion" },
    });
    await audit.record("memory.http.request", {
      operation: "source.recall",
      variant: "source_2",
      toolCallId: "call-source-1",
      request: { body: { query: "release discussion" } },
    });
    await audit.record("memory.http.response", {
      operation: "source.recall",
      variant: "source_2",
      toolCallId: "call-source-1",
      response: { ok: true, status: 200, durationMs: 18 },
    });
    await audit.record("tool.completed", {
      toolCallId: "call-source-1",
      toolName: "memory_query_source",
      args: { reference: "source_2", query: "release discussion" },
      result: {
        details: {
          sourceHandle: "source_2",
          displayName: "Arch Linux 中文群",
          bankId: "telegram:chat:-3003",
          memoryIds: ["memory-9"],
        },
      },
      isError: false,
      durationMs: 23,
    });
    await audit.record("memory.access.warning", {
      unavailableBankIds: ["telegram:chat:-4004"],
    });
    await audit.record("run.completed", {
      sessionId: "session-1",
      entryId: "entry-child",
      answer: "They discussed the release.",
    });
    await audit.flush();

    const { summary } = await app.store.get(RUN_ID);

    assert.equal(summary.memory.route, "cross_bank_queried");
    assert.deepEqual(summary.session, {
      kind: "continuation",
      id: "session-1",
      parentEntryId: "entry-parent",
      entryId: "entry-child",
    });
    assert.deepEqual(summary.tools, [
      {
        callId: "call-find-1",
        name: "memory_find_sources",
        status: "completed",
        durationMs: 14,
        query: null,
        source: null,
        eventSequence: 7,
      },
      {
        callId: "call-source-1",
        name: "memory_query_source",
        status: "completed",
        durationMs: 23,
        query: null,
        source: {
          handle: "source_2",
          displayName: null,
          bankId: null,
        },
        eventSequence: 9,
      },
    ]);
    assert.deepEqual(summary.warnings, [
      {
        kind: "memory_access",
        unavailableBankCount: 1,
        eventSequence: 13,
      },
    ]);
  } finally {
    await app.close();
  }
});

test("reports failed cross-bank attempts and incomplete runs conservatively", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "Check another group",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await audit.record("memory.capabilities.issued", {
      sources: [
        {
          handle: "source_1",
          bankId: "telegram:chat:-2002",
          displayName: "Other group",
        },
      ],
    });
    await audit.record("tool.started", {
      toolCallId: "call-source-1",
      toolName: "memory_query_source",
      args: { reference: "source_1", query: "today" },
    });
    await audit.record("memory.http.response", {
      operation: "source.recall",
      toolCallId: "call-source-1",
      response: { ok: true, status: 200, durationMs: 2 },
    });
    await audit.record("tool.completed", {
      toolCallId: "call-source-1",
      toolName: "memory_query_source",
      args: { reference: "source_1", query: "today" },
      result: {
        content: [{ type: "text", text: "Source unavailable" }],
        details: {
          sourceHandle: "source_1",
          displayName: "Other group",
          bankId: "telegram:chat:-2002",
          unavailable: true,
        },
      },
      isError: false,
      durationMs: 3,
    });
    await audit.flush();

    const { summary } = await app.store.get(RUN_ID);

    assert.equal(summary.status, "in_progress");
    assert.equal(summary.durationMs, null);
    assert.equal(summary.memory.route, "cross_bank_failed");
    assert.equal(summary.tools[0].status, "failed");
    assert.deepEqual(summary.tools[0].source, {
      handle: "source_1",
      displayName: null,
      bankId: null,
    });
  } finally {
    await app.close();
  }
});

test("reports source discovery without claiming that another bank was queried", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "Find the named group",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await audit.record("tool.started", {
      toolCallId: "call-find-1",
      toolName: "memory_find_sources",
      args: { query: "Named group" },
    });
    await audit.record("tool.completed", {
      toolCallId: "call-find-1",
      toolName: "memory_find_sources",
      args: { query: "Named group" },
      result: { details: { references: [] } },
      isError: false,
      durationMs: 5,
    });
    await audit.flush();

    const { summary } = await app.store.get(RUN_ID);

    assert.equal(summary.memory.route, "source_discovery_only");
    assert.equal(summary.tools[0].name, "memory_find_sources");
    assert.equal(summary.tools[0].status, "completed");
  } finally {
    await app.close();
  }
});

test("classifies failed and unfinished source discovery as cross-bank outcomes", async () => {
  const app = await fixture();
  try {
    const failed = await app.store.start(RUN_ID);
    await failed.record("run.request", {
      prompt: "Check another group",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await failed.record("tool.started", {
      toolCallId: "call-find-failed",
      toolName: "memory_find_sources",
      args: { query: "another group" },
    });
    await failed.record("tool.completed", {
      toolCallId: "call-find-failed",
      toolName: "memory_find_sources",
      result: { details: { unavailable: true, references: [] } },
      isError: false,
      durationMs: 4,
    });
    await failed.flush();

    const unfinished = await app.store.start(SECOND_RUN_ID);
    await unfinished.record("run.request", {
      prompt: "Check another group",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await unfinished.record("tool.started", {
      toolCallId: "call-find-running",
      toolName: "memory_find_sources",
      args: { query: "another group" },
    });
    await unfinished.flush();

    assert.equal(
      (await app.store.get(RUN_ID)).summary.memory.route,
      "cross_bank_failed",
    );
    assert.equal(
      (await app.store.get(SECOND_RUN_ID)).summary.memory.route,
      "cross_bank_attempted",
    );
  } finally {
    await app.close();
  }
});

test("uses the parsed automatic recall outcome instead of a successful HTTP status", async () => {
  const app = await fixture();
  try {
    const audit = await app.store.start(RUN_ID);
    await audit.record("run.request", {
      prompt: "What happened?",
      memory: { primaryBankId: "telegram:chat:-1001" },
    });
    await audit.record("memory.http.request", {
      exchangeId: "recall-1",
      operation: "recall",
      toolCallId: null,
      request: { body: { query: "What happened?" } },
    });
    await audit.record("memory.http.response", {
      exchangeId: "recall-1",
      operation: "recall",
      toolCallId: null,
      durationMs: 10,
      response: {
        ok: true,
        status: 200,
        body: { invalid: "payload" },
      },
    });
    await audit.record("memory.context", {
      queries: ["What happened?"],
      memories: [],
      recall: {
        status: "failed",
        attemptedCount: 1,
        completedCount: 0,
        failedCount: 1,
      },
    });
    await audit.flush();

    const { summary } = await app.store.get(RUN_ID);

    assert.equal(summary.memory.initialRecall.status, "failed");
  } finally {
    await app.close();
  }
});

test("does not read audit paths for malformed or unknown run identities", async () => {
  const app = await fixture();
  try {
    assert.equal(await app.store.get("../../secret"), null);
    assert.equal(
      await app.store.get("33333333-3333-4333-8333-333333333333"),
      null,
    );
  } finally {
    await app.close();
  }
});
