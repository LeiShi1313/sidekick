import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { SessionManager } from "@earendil-works/pi-coding-agent";

import {
  hardenSessionPersistence,
  scrubSessionDirectory,
} from "../src/session-persistence.mjs";

const IDENTITY_ALIAS_KEY = "test-identity-alias-key-that-is-strong";

const usage = {
  input: 1,
  output: 1,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 2,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

test("rewrites legacy sessions without changing their entry tree", async () => {
  const root = await mkdtemp(join(tmpdir(), "sidekick-session-scrub-"));
  const sessionDir = join(root, "sessions");
  try {
    const manager = SessionManager.create(join(root, "private-workspace"), sessionDir, {
      id: "11111111-1111-4111-8111-111111111111",
    });
    manager.appendMessage({
      role: "user",
      content:
        "<host_request_identity>\nHost-resolved current requester actor ID: telegram:user:123456\n</host_request_identity>\n\n" +
        "<untrusted_memory_context>\nPRIVATE_RECALLED_MEMORY\n</untrusted_memory_context>\n\n" +
        "<current_request>\nKeep this human request\n</current_request>",
      timestamp: 1,
    });
    manager.appendMessage({
      role: "assistant",
      content: [
        { type: "thinking", thinking: "PRIVATE_REASONING" },
        {
          type: "toolCall",
          id: "call-1",
          name: "web_search",
          arguments: { query: "PRIVATE_TOOL_ARGUMENT" },
        },
      ],
      api: "openai-completions",
      provider: "openai-compatible",
      model: "test-model",
      usage,
      stopReason: "toolUse",
      timestamp: 2,
    });
    manager.appendCustomEntry("web-search-results", {
      id: "result-1",
      type: "search",
      timestamp: 3,
      queries: [{ content: "PRIVATE_WEB_RESULT" }],
    });
    manager.appendMessage({
      role: "toolResult",
      toolCallId: "call-1",
      toolName: "web_search",
      content: [{ type: "text", text: "PRIVATE_TOOL_RESULT" }],
      details: {},
      isError: false,
      timestamp: 4,
    });
    const finalId = manager.appendMessage({
      role: "assistant",
      content: [{ type: "text", text: "Keep this useful answer." }],
      api: "openai-completions",
      provider: "openai-compatible",
      model: "test-model",
      usage,
      stopReason: "stop",
      timestamp: 5,
    });
    const path = manager.getSessionFile();
    const entryIds = manager.getEntries().map((entry) => entry.id);
    await chmod(path, 0o644);

    assert.deepEqual(
      await scrubSessionDirectory(sessionDir, {
        identityAliasKey: IDENTITY_ALIAS_KEY,
      }),
      { scanned: 1, rewritten: 1 },
    );
    const raw = await readFile(path, "utf8");
    assert.doesNotMatch(
      raw,
      /private-workspace|PRIVATE_RECALLED_MEMORY|PRIVATE_REASONING|PRIVATE_TOOL_ARGUMENT|PRIVATE_WEB_RESULT|PRIVATE_TOOL_RESULT|telegram:user:123456/,
    );
    assert.match(raw, /Keep this human request/);
    assert.match(raw, /Keep this useful answer\./);
    assert.match(raw, /actor_[a-f0-9]{16}/);
    assert.match(raw, /Tool result omitted after use/);
    assert.match(raw, /"cwd":"\/workspace"/);
    assert.equal((await stat(path)).mode & 0o777, 0o600);

    const reopened = SessionManager.open(path, sessionDir, "/workspace");
    assert.deepEqual(reopened.getEntries().map((entry) => entry.id), entryIds);
    assert(reopened.getEntry(finalId));
    assert.deepEqual(
      await scrubSessionDirectory(sessionDir, {
        identityAliasKey: IDENTITY_ALIAS_KEY,
      }),
      { scanned: 1, rewritten: 0 },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("minimizes live tool details and compaction content", async () => {
  const root = await mkdtemp(join(tmpdir(), "sidekick-session-live-"));
  const sessionDir = join(root, "sessions");
  try {
    const manager = hardenSessionPersistence(
      SessionManager.create("/workspace", sessionDir),
      () => ({
        privacyOptions: {
          identityAliasKey: IDENTITY_ALIAS_KEY,
          identityScope: "telegram:chat:-1001",
        },
      }),
    );
    const userId = manager.appendMessage({
      role: "user",
      content:
        "<untrusted_reference_context>\nPRIVATE_REFERENCE\n</untrusted_reference_context>\n\n" +
        "<untrusted_memory_context>\nPRIVATE_MEMORY\n</untrusted_memory_context>\n\n" +
        "<current_request>\nKeep this request\n</current_request>",
      timestamp: 1,
    });
    manager.appendMessage({
      role: "toolResult",
      toolCallId: "call-1",
      toolName: "memory_query_source",
      content: [{ type: "text", text: "PRIVATE_TOOL_RESULT" }],
      details: {
        bankId: "qq:group:686743769",
        displayName: "Private Leadership",
        memoryIds: ["private-memory-1"],
      },
      isError: false,
      timestamp: 2,
    });
    manager.appendMessage({
      role: "assistant",
      content: [{ type: "text", text: "Keep this answer" }],
      api: "openai-completions",
      provider: "openai-compatible",
      model: "test-model",
      usage,
      stopReason: "stop",
      timestamp: 3,
    });
    manager.appendCompaction(
      "PRIVATE_COMPACTION_SUMMARY telegram:user:123456",
      userId,
      100,
      { raw: "PRIVATE_COMPACTION_DETAILS" },
    );

    const raw = await readFile(manager.getSessionFile(), "utf8");
    assert.match(raw, /Keep this request/);
    assert.doesNotMatch(
      raw,
      /PRIVATE_REFERENCE|PRIVATE_MEMORY|PRIVATE_TOOL_RESULT|qq:group:686743769|Private Leadership|private-memory-1|PRIVATE_COMPACTION|bank_[a-f0-9]{32}/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
