import assert from "node:assert/strict";
import { appendFile, mkdtemp, rm } from "node:fs/promises";
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

test("records ordered append-only events and redacts credential-shaped fields", async () => {
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
      answer: "Alice owns deploys.",
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
    assert.equal(result.events[0].data.authorization, "[REDACTED]");
    assert.equal(result.events[0].data.provider.errorMessage, "[REDACTED]");
    assert.equal(
      result.events[0].data.callbackUrl,
      "https://example.test/path?api_key=%5BREDACTED%5D&view=full",
    );
    assert.deepEqual(result.events[0].data.image, {
      type: "image",
      mimeType: "image/png",
      sizeBytes: 13,
      data: "[OMITTED]",
    });
    assert.equal(
      result.events[1].data.request.body.apiKey,
      "[REDACTED]",
    );
    assert.doesNotMatch(
      JSON.stringify(result),
      /Bearer secret|secret-key|provider credential detail|query-secret|user:pass/,
    );
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
    assert.equal(page.items[0].memoryScopeId, "chat:engineering");
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
      queries: ["2026-07-23 dog bro"],
      memories: [{ id: "memory-1" }, { id: "memory-2" }],
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
    assert.equal(summary.prompt, "Did dog bro appear today?");
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
      primaryBankId: "telegram:chat:-1001",
      route: "current_bank_only",
      initialRecall: {
        status: "unknown",
        queries: ["2026-07-23 dog bro"],
        memoryCount: 2,
        eventSequence: 2,
      },
      directory: {
        status: "available",
        query: "Did dog bro appear today?",
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
        query: "2026-07-23 @dota2pp",
        source: null,
        eventSequence: 8,
      },
    ]);
    assert.deepEqual(summary.warnings, []);
    assert.equal(summary.failure, null);
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
        query: "Arch group",
        source: null,
        eventSequence: 7,
      },
      {
        callId: "call-source-1",
        name: "memory_query_source",
        status: "completed",
        durationMs: 23,
        query: "release discussion",
        source: {
          handle: "source_2",
          displayName: "Arch Linux 中文群",
          bankId: "telegram:chat:-3003",
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
      displayName: "Other group",
      bankId: "telegram:chat:-2002",
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
