import assert from "node:assert/strict";
import test from "node:test";

import sharp from "sharp";

import { createAgentServer } from "../src/http-service.mjs";

const validRun = {
  runId: "11111111-1111-4111-8111-111111111111",
  sessionId: null,
  parentEntryId: null,
  prompt: "Calculate 6 * 7",
  context: [],
  systemPrompt: "Answer concisely.",
  toolPolicy: "delegated",
  identity: {
    requester: { id: "telegram:user:40", label: "Alice" },
    anchors: [{ id: "telegram:user:40", label: "Alice" }],
    requesterCanCustomize: true,
  },
  origin: {
    scopeId: "telegram:chat:-1001",
    adapterInstanceId: "telegram-default",
  },
};

const malformedJpegBytes = Buffer.from(
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBAUEBAYFBQUGBgYHCQ4JCQgICRINDQoOFRIWFhUSFBQXGiEcFxgfGRQUHScdHyIjJSUlFhwpLCgkKyEkJST/2wBDAQYGBgkICREJCREkGBQYJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAACAAIDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAABgj/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdAAyqX//Z",
  "base64",
);
const malformedOversizedJpegBytes = Buffer.from(
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBAUEBAYFBQUGBgYHCQ4JCQgICRINDQoOFRIWFhUSFBQXGiEcFxgfGRQUHScdHyIjJSUlFhwpLCgkKyEkJST/2wBDAQYGBgkICREJCREkGBQYJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAABBkEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFgEBAQEAAAAAAAAAAAAAAAAAAAYI/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AnQCGapAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAf/2Q==",
  "base64",
);

const modelImageBytes = await sharp({
  create: {
    width: 64,
    height: 32,
    channels: 3,
    background: { r: 20, g: 80, b: 160 },
  },
})
  .jpeg({ quality: 82 })
  .toBuffer();
const oversizedDimensionImageBytes = await sharp({
  create: {
    width: 1601,
    height: 1,
    channels: 3,
    background: { r: 20, g: 80, b: 160 },
  },
})
  .jpeg({ quality: 82 })
  .toBuffer();

const OPERATOR_TOKEN = "test-agent-token-that-is-long-enough";

const OPERATOR_CLIENT = {
  id: "operator",
  token: OPERATOR_TOKEN,
  capabilities: ["models", "runs", "attachments", "history", "status"],
  cancelAny: true,
};

async function listen(engine, clients = [OPERATOR_CLIENT]) {
  const server = createAgentServer({
    engine,
    clients,
    logger: { info() {}, error() {} },
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return {
    server,
    baseUrl: `http://127.0.0.1:${port}`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

test("streams a run as NDJSON", async () => {
  const engine = {
    async *run(request) {
      assert.deepEqual(request, validRun);
      yield { type: "run_started", runId: request.runId, sessionId: "session-1" };
      yield { type: "text_delta", delta: "42", reset: true };
      yield {
        type: "run_completed",
        sessionId: "session-1",
        entryId: "entry-1",
        answer: "42",
      };
    },
    async cancel() {
      return false;
    },
  };
  const app = await listen(engine);
  try {
    const response = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify(validRun),
    });
    assert.equal(response.status, 200);
    assert.match(response.headers.get("content-type"), /application\/x-ndjson/);
    const events = (await response.text())
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    assert.equal(events.at(-1).type, "run_completed");
    assert.equal(events.at(-1).entryId, "entry-1");
  } finally {
    await app.close();
  }
});

test("accepts one bounded JPEG model input and rejects invalid image arrays", async () => {
  const received = [];
  const engine = {
    async *run(request) {
      received.push(request);
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  };
  const app = await listen(engine);
  const headers = {
    "content-type": "application/json",
    authorization: `Bearer ${OPERATOR_TOKEN}`,
  };
  const image = {
    mimeType: "image/jpeg",
    data: modelImageBytes.toString("base64"),
  };
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        ...validRun,
        context: Array.from({ length: 4 }, (_, index) => ({
          kind: "reference",
          text: `${index}${"x".repeat(15_999)}`,
        })),
        images: [image],
      }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();
    assert.equal(received.length, 1);
    assert.equal(received[0].images.length, 1);
    assert.equal(received[0].images[0].mimeType, "image/jpeg");
    assert.deepEqual(received[0].images[0].data, modelImageBytes);

    for (const images of [
      [image, image],
      [{ ...image, mimeType: "image/png" }],
      [{ ...image, unexpected: true }],
      [{ ...image, data: `${image.data}=` }],
      [{ mimeType: "image/jpeg", data: "not-base64" }],
      [{ mimeType: "image/jpeg", data: "/9j/2Q==" }],
      [{
        mimeType: "image/jpeg",
        data: malformedJpegBytes.toString("base64"),
      }],
      [{
        mimeType: "image/jpeg",
        data: malformedOversizedJpegBytes.toString("base64"),
      }],
      [{
        mimeType: "image/jpeg",
        data: modelImageBytes.subarray(0, -10).toString("base64"),
      }],
      [{
        mimeType: "image/jpeg",
        data: oversizedDimensionImageBytes.toString("base64"),
      }],
    ]) {
      const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
        method: "POST",
        headers,
        body: JSON.stringify({ ...validRun, images }),
      });
      assert.equal(rejected.status, 400);
    }

    const oversized = Buffer.concat([
      Buffer.from([0xff, 0xd8, 0xff]),
      Buffer.alloc(2 * 1024 * 1024),
    ]).toString("base64");
    const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        ...validRun,
        images: [{ mimeType: "image/jpeg", data: oversized }],
      }),
    });
    assert.equal(rejected.status, 400);
    assert.equal(received.length, 1);
  } finally {
    await app.close();
  }
});

test("isolates adapter credentials by capability and instance identity", async () => {
  const owners = [];
  const origins = [];
  const clients = [
    {
      id: "telegram",
      token: "telegram-agent-token-that-is-long-enough",
      capabilities: ["models", "runs", "attachments"],
      adapterInstanceId: "telegram-default",
    },
    {
      id: "operator",
      token: "operator-agent-token-that-is-long-enough",
      capabilities: ["models", "runs", "attachments", "history", "status"],
      cancelAny: true,
    },
  ];
  const app = await listen({
    async *run(request, owner) {
      origins.push(request.origin);
      owners.push(owner);
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel(_runId, owner) {
      owners.push(owner);
      return true;
    },
    async listSessions() {
      return { items: [], total: 0, nextCursor: null };
    },
  }, clients);
  const adapterHeaders = {
    "content-type": "application/json",
    authorization: "Bearer telegram-agent-token-that-is-long-enough",
  };
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: adapterHeaders,
      body: JSON.stringify({
        ...validRun,
        origin: {
          scopeId: "telegram:chat:-1001",
          adapterInstanceId: "telegram-default",
        },
      }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();

    const impersonation = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: adapterHeaders,
      body: JSON.stringify({
        ...validRun,
        runId: "22222222-2222-4222-8222-222222222222",
        origin: {
          scopeId: "qq:group:42",
          adapterInstanceId: "qq-default",
        },
      }),
    });
    assert.equal(impersonation.status, 403);

    const history = await fetch(`${app.baseUrl}/v1/sessions`, {
      headers: { authorization: adapterHeaders.authorization },
    });
    assert.equal(history.status, 403);

    const cancel = await fetch(
      `${app.baseUrl}/v1/runs/${validRun.runId}/cancel`,
      { method: "POST", headers: { authorization: adapterHeaders.authorization } },
    );
    assert.equal(cancel.status, 200);
    assert.deepEqual(origins, [
      {
        scopeId: "telegram:chat:-1001",
        adapterInstanceId: "telegram-default",
      },
    ]);
    assert.deepEqual(owners, ["telegram", "telegram"]);
  } finally {
    await app.close();
  }
});

test("restricts a WeChat credential to its exact connector account", async () => {
  let calls = 0;
  const app = await listen(
    {
      async *run() {
        calls += 1;
        yield {
          type: "run_completed",
          sessionId: "s",
          entryId: "e",
          answer: "ok",
        };
      },
      async cancel() {
        return false;
      },
    },
    [
      {
        id: "wechat-host",
        token: "wechat-host-token-that-is-long-enough",
        capabilities: ["runs"],
        adapterInstanceId: "wechat-host",
        scopePrefix: "wechat:account:wxid_host:",
      },
    ],
  );
  const headers = {
    "content-type": "application/json",
    authorization: "Bearer wechat-host-token-that-is-long-enough",
  };
  const requestFor = (account, runId) => ({
    ...validRun,
    runId,
    origin: {
      scopeId: `wechat:account:${account}:chat:room%40chatroom`,
      adapterInstanceId: "wechat-host",
    },
    identity: {
      requester: {
        id: `wechat:account:${account}:user:alice`,
        label: "Alice",
      },
      anchors: [
        {
          id: `wechat:account:${account}:user:alice`,
          label: "Alice",
        },
      ],
    },
  });
  try {
    const own = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify(
        requestFor("wxid_host", "22222222-2222-4222-8222-222222222222"),
      ),
    });
    assert.equal(own.status, 200);
    await own.text();

    const peer = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify(
        requestFor("wxid_peer", "33333333-3333-4333-8333-333333333333"),
      ),
    });
    assert.equal(peer.status, 403);
    assert.equal(calls, 1);
  } finally {
    await app.close();
  }
});

test("rejects duplicate service credentials", () => {
  assert.throws(
    () =>
      createAgentServer({
        engine: {},
        clients: [
          {
            id: "first",
            token: "duplicate-agent-token-that-is-long-enough",
            capabilities: ["runs"],
          },
          {
            id: "second",
            token: "duplicate-agent-token-that-is-long-enough",
            capabilities: ["history"],
          },
        ],
      }),
    /unique/i,
  );
});

test("a rejected duplicate run does not cancel the existing owner", async () => {
  const existingOwner = Symbol("existing-run-owner");
  const cancelAttempts = [];
  let originalActive = true;
  const app = await listen({
    async *run(_request, requestOwner) {
      assert.notEqual(requestOwner, existingOwner);
      throw new Error("Agent run is already active");
    },
    async cancel(_runId, requestOwner) {
      cancelAttempts.push(requestOwner);
      if (requestOwner === existingOwner) originalActive = false;
      return requestOwner === existingOwner;
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify(validRun),
    });
    assert.equal(response.status, 200);
    const events = (await response.text())
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    assert.equal(events.at(-1).type, "run_failed");
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(cancelAttempts, []);
    assert.equal(originalActive, true);
  } finally {
    await app.close();
  }
});

test("rejects invalid run input before invoking the engine", async () => {
  let called = false;
  const app = await listen({
    async *run() {
      called = true;
    },
    async cancel() {
      return false;
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({ ...validRun, prompt: "", toolPolicy: "owner-ish" }),
    });
    assert.equal(response.status, 400);
    assert.equal(called, false);
    assert.deepEqual(await response.json(), {
      error: { code: "INVALID_REQUEST", message: "Invalid run request" },
    });
  } finally {
    await app.close();
  }
});

test("accepts a no-tools model run", async () => {
  let received;
  const app = await listen({
    async *run(request) {
      received = request;
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({ ...validRun, toolPolicy: "none" }),
    });
    assert.equal(response.status, 200);
    await response.text();
    assert.equal(received.toolPolicy, "none");
  } finally {
    await app.close();
  }
});

test("accepts a bounded model selection and rejects malformed model ids", async () => {
  const received = [];
  const app = await listen({
    async *run(request) {
      received.push(request);
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  });
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({ ...validRun, model: "gpt-5.4-mini" }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();
    assert.equal(received[0].model, "gpt-5.4-mini");

    for (const model of ["invalid model", 123]) {
      const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: "Bearer test-agent-token-that-is-long-enough",
        },
        body: JSON.stringify({ ...validRun, model }),
      });
      assert.equal(rejected.status, 400);
    }
    assert.equal(received.length, 1);
  } finally {
    await app.close();
  }
});

test("accepts only a strict bounded run origin independent of memory", async () => {
  const received = [];
  const app = await listen({
    async *run(request) {
      received.push(request);
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  });
  const headers = {
    "content-type": "application/json",
    authorization: "Bearer test-agent-token-that-is-long-enough",
  };
  const origin = {
    scopeId: "wechat:account:wxid%40bridge:chat:room/42",
    adapterInstanceId: "wechat-peer",
  };
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers,
      body: JSON.stringify({ ...validRun, origin }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();
    assert.deepEqual(received[0].origin, origin);
    assert.equal(received[0].memory, undefined);

    for (const invalidOrigin of [
      null,
      { scopeId: "", adapterInstanceId: "wechat-peer" },
      { scopeId: "scope", adapterInstanceId: "" },
      { scopeId: "x".repeat(513), adapterInstanceId: "wechat-peer" },
      { scopeId: "scope", adapterInstanceId: "x".repeat(129) },
      { scopeId: "scope" },
      { ...origin, debug: true },
    ]) {
      const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
        method: "POST",
        headers,
        body: JSON.stringify({ ...validRun, origin: invalidOrigin }),
      });
      assert.equal(rejected.status, 400);
    }
    assert.equal(received.length, 1);
  } finally {
    await app.close();
  }
});

test("serves the authenticated model catalog without provider details", async () => {
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async listModels() {
      return {
        defaultModel: "gpt-5.6-sol",
        models: ["gpt-5.4-mini", "gpt-5.6-sol"],
      };
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/models`, {
      headers: {
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
    });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      defaultModel: "gpt-5.6-sol",
      models: ["gpt-5.4-mini", "gpt-5.6-sol"],
    });

    const unauthenticated = await fetch(`${app.baseUrl}/v1/models`);
    assert.equal(unauthenticated.status, 401);
  } finally {
    await app.close();
  }
});

test("returns a stable error when the model catalog is unavailable", async () => {
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async listModels() {
      throw new Error("provider credential detail");
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/models`, {
      headers: {
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
    });
    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      error: {
        code: "MODEL_CATALOG_UNAVAILABLE",
        message: "Model catalog unavailable",
      },
    });
  } finally {
    await app.close();
  }
});

test("accepts a bounded memory target and rejects scope injection", async () => {
  let received;
  const app = await listen({
    async *run(request) {
      received = request;
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  });
  const memory = {
    primaryBankId: "telegram:chat:-1001",
    requesterIsOwner: false,
    grantedBankIds: ["qq:group:686743769"],
    participants: [
      {
        id: "telegram:user:41",
        label: "Bob",
        allowed: true,
        bankIds: ["telegram:chat:-1002"],
      },
    ],
    query: "What does Alice prefer?",
  };
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        context: [{ kind: "reference", text: "Alice joined the discussion." }],
        memory,
        includeMemorySnapshot: true,
      }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();
    assert.deepEqual(received.memory, memory);
    assert.equal(received.includeMemorySnapshot, true);

    const rejectedSnapshotFlag = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        includeMemorySnapshot: "yes",
      }),
    });
    assert.equal(rejectedSnapshotFlag.status, 400);

    const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        memory: { ...memory, primaryBankId: "../../other-bank" },
      }),
    });
    assert.equal(rejected.status, 400);

    const rejectedNumericBank = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        memory: { ...memory, primaryBankId: 1001 },
      }),
    });
    assert.equal(rejectedNumericBank.status, 400);

    const rejectedInjectedReferences = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        memory: {
          ...memory,
          sourceCapabilities: [{ handle: "source_1" }],
        },
      }),
    });
    assert.equal(rejectedInjectedReferences.status, 400);

    const rejectedNumericRequester = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        identity: {
          ...validRun.identity,
          requester: { ...validRun.identity.requester, id: 40 },
        },
      }),
    });
    assert.equal(rejectedNumericRequester.status, 400);

    const rejectedParticipantGrants = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        memory: {
          ...memory,
          participants: [
            {
              id: "telegram:user:41",
              label: "Bob",
              allowed: false,
              bankIds: ["telegram:chat:-1002"],
            },
          ],
        },
      }),
    });
    assert.equal(rejectedParticipantGrants.status, 400);
  } finally {
    await app.close();
  }
});

test("accepts exact Matrix bridge actors and rejects invalid identities", async () => {
  const received = [];
  const app = await listen({
    async *run(request) {
      received.push(request);
      yield { type: "run_completed", sessionId: "s", entryId: "e", answer: "ok" };
    },
    async cancel() {
      return false;
    },
  });
  const bridgeActorId =
    "telegram:matrix-bridge:6332621450%3A-1001%3A0123456789abcdef0123456789abcdef";
  try {
    const accepted = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        identity: {
          requester: { id: bridgeActorId, label: "SteamedFish" },
          anchors: [{ id: bridgeActorId, label: "SteamedFish" }],
        },
      }),
    });
    assert.equal(accepted.status, 200);
    await accepted.text();
    assert.equal(received.length, 1);
    assert.deepEqual(received[0].identity, {
      requester: { id: bridgeActorId, label: "SteamedFish" },
      anchors: [{ id: bridgeActorId, label: "SteamedFish" }],
      requesterCanCustomize: false,
    });

    const writableBridge = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        ...validRun,
        identity: {
          requester: { id: bridgeActorId, label: "SteamedFish" },
          anchors: [{ id: bridgeActorId, label: "SteamedFish" }],
          requesterCanCustomize: true,
        },
      }),
    });
    assert.equal(writableBridge.status, 400);

    const invalidIds = [
      "telegram:chat:-1001",
      "telegram:chat:-1001:matrix-bridge:forged",
      "telegram:matrix-bridge:",
      "telegram:matrix-bridge:6332621450%ZZ-1001%3A0123456789abcdef0123456789abcdef",
      "qq:matrix-bridge:6332621450%3A-1001%3A0123456789abcdef0123456789abcdef",
    ];
    for (const id of invalidIds) {
      const rejected = await fetch(`${app.baseUrl}/v1/runs`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: "Bearer test-agent-token-that-is-long-enough",
        },
        body: JSON.stringify({
          ...validRun,
          runId: "22222222-2222-4222-8222-222222222222",
          identity: {
            requester: { id, label: "Invalid identity" },
            anchors: [{ id, label: "Invalid identity" }],
          },
        }),
      });
      assert.equal(rejected.status, 400, id);
    }
    assert.equal(received.length, 1);
  } finally {
    await app.close();
  }
});

test("cancels an active run by its caller-provided id", async () => {
  let cancelled;
  const app = await listen({
    async *run() {},
    async cancel(runId) {
      cancelled = runId;
      return true;
    },
  });
  try {
    const response = await fetch(
      `${app.baseUrl}/v1/runs/${validRun.runId}/cancel`,
      {
        method: "POST",
        headers: {
          authorization: "Bearer test-agent-token-that-is-long-enough",
        },
      },
    );
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { cancelled: true });
    assert.equal(cancelled, validRun.runId);
  } finally {
    await app.close();
  }
});

test("rejects unauthenticated run and cancellation requests", async () => {
  let called = false;
  const app = await listen({
    async *run() {
      called = true;
    },
    async cancel() {
      called = true;
      return true;
    },
  });
  try {
    const run = await fetch(`${app.baseUrl}/v1/runs`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(validRun),
    });
    const cancel = await fetch(
      `${app.baseUrl}/v1/runs/${validRun.runId}/cancel`,
      { method: "POST" },
    );
    assert.equal(run.status, 401);
    assert.equal(cancel.status, 401);
    assert.equal(called, false);
  } finally {
    await app.close();
  }
});

test("health reveals no provider or credential details", async () => {
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/health`);
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { status: "ok" });
  } finally {
    await app.close();
  }
});

test("serves authenticated session history and run audits", async () => {
  const calls = [];
  const engine = {
    async *run() {},
    async cancel() {
      return false;
    },
    async listSessions(options) {
      calls.push(["listSessions", options]);
      return {
        items: [
          {
            id: "session-1",
            name: "Deployment",
            messageCount: 4,
            firstMessage: "Inspect deployment",
          },
        ],
        total: 1,
        nextCursor: null,
      };
    },
    async getSession(sessionId) {
      calls.push(["getSession", sessionId]);
      return sessionId === "session-1"
        ? { id: sessionId, leafId: "entry-1", entries: [] }
        : null;
    },
    async listRunAudits(options) {
      calls.push(["listRunAudits", options]);
      return {
        items: [{ runId: validRun.runId, sessionId: "session-1" }],
        total: 1,
        nextCursor: null,
      };
    },
    async getRunAudit(runId) {
      calls.push(["getRunAudit", runId]);
      return runId === validRun.runId ? { runId, events: [] } : null;
    },
  };
  const app = await listen(engine);
  const headers = {
    authorization: "Bearer test-agent-token-that-is-long-enough",
  };
  try {
    const list = await fetch(
      `${app.baseUrl}/v1/sessions?limit=20&q=deploy&cursor=session-0`,
      { headers },
    );
    assert.equal(list.status, 200);
    assert.equal((await list.json()).items[0].id, "session-1");

    const session = await fetch(`${app.baseUrl}/v1/sessions/session-1`, {
      headers,
    });
    assert.equal(session.status, 200);
    assert.equal((await session.json()).leafId, "entry-1");

    const audits = await fetch(
      `${app.baseUrl}/v1/runs?limit=10&sessionId=session-1`,
      { headers },
    );
    assert.equal(audits.status, 200);
    assert.equal((await audits.json()).items[0].runId, validRun.runId);

    const audit = await fetch(
      `${app.baseUrl}/v1/runs/${validRun.runId}/audit`,
      { headers },
    );
    assert.equal(audit.status, 200);
    assert.equal((await audit.json()).runId, validRun.runId);

    assert.deepEqual(calls, [
      [
        "listSessions",
        { limit: 20, cursor: "session-0", query: "deploy" },
      ],
      ["getSession", "session-1"],
      [
        "listRunAudits",
        { limit: 10, cursor: null, sessionId: "session-1" },
      ],
      ["getRunAudit", validRun.runId],
    ]);
  } finally {
    await app.close();
  }
});

test("serves active runs through the strict authenticated status query", async () => {
  let activeCalls = 0;
  let historyCalled = false;
  const item = {
    runId: validRun.runId,
    sessionId: "session-1",
    scopeId: "qq:group:42",
    adapterInstanceId: "qq-primary",
    modelId: "test-model",
    startedAt: "2026-07-31T10:00:00.000Z",
    updatedAt: "2026-07-31T10:00:01.000Z",
    phase: "model_running",
    currentTool: null,
  };
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    listActiveRuns() {
      activeCalls += 1;
      return { items: [item], total: 1 };
    },
    async listRunAudits() {
      historyCalled = true;
      return { items: [], total: 0, nextCursor: null };
    },
  });
  const headers = {
    authorization: "Bearer test-agent-token-that-is-long-enough",
  };
  try {
    const response = await fetch(`${app.baseUrl}/v1/runs?status=active`, {
      headers,
    });
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { items: [item], total: 1 });

    for (const query of [
      "status=completed",
      "status=active&limit=1",
      "status=active&status=active",
    ]) {
      const rejected = await fetch(`${app.baseUrl}/v1/runs?${query}`, {
        headers,
      });
      assert.equal(rejected.status, 400);
      assert.deepEqual(await rejected.json(), {
        error: { code: "INVALID_REQUEST", message: "Invalid run query" },
      });
    }
    assert.equal(activeCalls, 1);
    assert.equal(historyCalled, false);
  } finally {
    await app.close();
  }
});

test("validates history queries and returns stable missing-resource errors", async () => {
  let called = false;
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async listSessions() {
      called = true;
      return { items: [], total: 0, nextCursor: null };
    },
    async getSession() {
      return null;
    },
    async listRunAudits() {
      called = true;
      return { items: [], total: 0, nextCursor: null };
    },
    async getRunAudit() {
      return null;
    },
  });
  const headers = {
    authorization: "Bearer test-agent-token-that-is-long-enough",
  };
  try {
    for (const path of [
      "/v1/sessions?limit=0",
      "/v1/sessions?limit=101",
      `/v1/sessions?q=${"x".repeat(201)}`,
      "/v1/sessions?cursor=../../secret",
      "/v1/runs?sessionId=../../secret",
    ]) {
      const response = await fetch(`${app.baseUrl}${path}`, { headers });
      assert.equal(response.status, 400);
      assert.deepEqual(await response.json(), {
        error: { code: "INVALID_REQUEST", message: "Invalid history request" },
      });
    }
    assert.equal(called, false);

    const missingSession = await fetch(
      `${app.baseUrl}/v1/sessions/missing-session`,
      { headers },
    );
    assert.equal(missingSession.status, 404);
    assert.deepEqual(await missingSession.json(), {
      error: { code: "NOT_FOUND", message: "Session not found" },
    });

    const missingAudit = await fetch(
      `${app.baseUrl}/v1/runs/33333333-3333-4333-8333-333333333333/audit`,
      { headers },
    );
    assert.equal(missingAudit.status, 404);
    assert.deepEqual(await missingAudit.json(), {
      error: { code: "NOT_FOUND", message: "Run audit not found" },
    });
  } finally {
    await app.close();
  }
});

test("rejects unauthenticated history requests", async () => {
  let called = false;
  const engine = {
    async *run() {},
    async cancel() {
      return false;
    },
    async listSessions() {
      called = true;
    },
    async getSession() {
      called = true;
    },
    async listRunAudits() {
      called = true;
    },
    async getRunAudit() {
      called = true;
    },
    async listActiveRuns() {
      called = true;
    },
  };
  const app = await listen(engine);
  try {
    for (const path of [
      "/v1/sessions",
      "/v1/sessions/session-1",
      "/v1/runs",
      "/v1/runs?status=active",
      `/v1/runs/${validRun.runId}/audit`,
    ]) {
      const response = await fetch(`${app.baseUrl}${path}`);
      assert.equal(response.status, 401);
    }
    assert.equal(called, false);
  } finally {
    await app.close();
  }
});

test("describes one bounded attachment through the authenticated API", async () => {
  let received;
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async describeAttachment(request) {
      received = request;
      return "Description: a red square.\nVisible text: none.";
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/attachments/describe`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        kind: "image",
        mimeType: "image/jpeg",
        filename: "sample.jpg",
        data: jpeg.toString("base64"),
      }),
    });

    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), {
      description: "Description: a red square.\nVisible text: none.",
    });
    assert.equal(received.kind, "image");
    assert.equal(received.mimeType, "image/jpeg");
    assert.equal(received.filename, "sample.jpg");
    assert.deepEqual(received.data, jpeg);
  } finally {
    await app.close();
  }
});

test("rejects invalid or unauthenticated attachment analysis", async () => {
  let called = false;
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async describeAttachment() {
      called = true;
      return "unused";
    },
  });
  try {
    const invalid = await fetch(`${app.baseUrl}/v1/attachments/describe`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({ kind: "audio", text: "not supported" }),
    });
    const mislabeledImage = await fetch(
      `${app.baseUrl}/v1/attachments/describe`,
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: "Bearer test-agent-token-that-is-long-enough",
        },
        body: JSON.stringify({
          kind: "image",
          mimeType: "image/jpeg",
          data: Buffer.from("not-an-image").toString("base64"),
        }),
      },
    );
    const unauthenticated = await fetch(
      `${app.baseUrl}/v1/attachments/describe`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          kind: "text",
          mimeType: "text/plain",
          text: "private document",
        }),
      },
    );

    assert.equal(invalid.status, 400);
    assert.equal(mislabeledImage.status, 400);
    assert.equal(unauthenticated.status, 401);
    assert.equal(called, false);
  } finally {
    await app.close();
  }
});

test("rejects an invalid attachment description from the engine", async () => {
  const app = await listen({
    async *run() {},
    async cancel() {
      return false;
    },
    async describeAttachment() {
      return "x".repeat(4_001);
    },
  });
  try {
    const response = await fetch(`${app.baseUrl}/v1/attachments/describe`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer test-agent-token-that-is-long-enough",
      },
      body: JSON.stringify({
        kind: "text",
        mimeType: "text/plain",
        text: "bounded source",
      }),
    });

    assert.equal(response.status, 502);
    assert.deepEqual(await response.json(), {
      error: { code: "ANALYSIS_FAILED", message: "Attachment analysis failed" },
    });
  } finally {
    await app.close();
  }
});
