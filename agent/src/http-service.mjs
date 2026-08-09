import { createHash, timingSafeEqual } from "node:crypto";
import { createServer } from "node:http";

import sharp from "sharp";

import { isModelId } from "./model-id.mjs";

const MAX_BODY_BYTES = 64 * 1024;
const MAX_RUN_BODY_BYTES = 3 * 1024 * 1024;
const MAX_ATTACHMENT_BODY_BYTES = 3 * 1024 * 1024;
const MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024;
const MAX_MODEL_IMAGE_DIMENSION = 1_600;
const MAX_ATTACHMENT_TEXT_CHARS = 50_000;
const MAX_MEMORY_ANCHORS = 64;
const MAX_BANK_GRANTS = 64;
const MAX_PARTICIPANTS = 16;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const IDENTIFIER_RE = /^[A-Za-z0-9_-]{1,128}$/;
const BANK_ID_RE = /^[A-Za-z0-9][A-Za-z0-9:_.%-]{0,255}$/;
const MIME_RE = /^[a-z0-9][a-z0-9.+-]{0,63}\/[a-z0-9][a-z0-9.+-]{0,127}$/;
const IMAGE_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const CLIENT_CAPABILITIES = new Set([
  "models",
  "runs",
  "attachments",
  "history",
  "status",
]);

function json(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
  });
  response.end(body);
}

async function readJson(request, maxBytes = MAX_BODY_BYTES) {
  if (!request.headers["content-type"]?.toLowerCase().startsWith("application/json")) {
    throw new Error("invalid content type");
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) throw new Error("request too large");
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function isBoundedString(value, min, max) {
  return typeof value === "string" && value.length >= min && value.length <= max;
}

function hasOnlyKeys(value, allowed) {
  return Object.keys(value).every((key) => allowed.has(key));
}

function isBankId(value) {
  return typeof value === "string" && BANK_ID_RE.test(value);
}

function boundedBankIds(value) {
  if (!Array.isArray(value) || value.length > MAX_BANK_GRANTS) return null;
  const unique = new Set(value);
  if (
    unique.size !== value.length ||
    value.some((item) => !isBankId(item))
  ) {
    return null;
  }
  return [...value];
}

function isHostIdentity(value) {
  return isBankId(value) && /:(?:user|channel):/.test(value);
}

function listOptions(url, kind) {
  const allowed =
    kind === "sessions"
      ? new Set(["limit", "cursor", "q"])
      : new Set(["limit", "cursor", "sessionId"]);
  for (const key of url.searchParams.keys()) {
    if (!allowed.has(key) || url.searchParams.getAll(key).length !== 1) {
      return null;
    }
  }
  const rawLimit = url.searchParams.get("limit");
  const limit = rawLimit === null ? 50 : Number(rawLimit);
  const cursor = url.searchParams.get("cursor");
  if (
    !Number.isInteger(limit) ||
    limit < 1 ||
    limit > 100 ||
    (cursor !== null && !IDENTIFIER_RE.test(cursor))
  ) {
    return null;
  }
  if (kind === "sessions") {
    const query = url.searchParams.get("q") ?? "";
    if (query.length > 200) return null;
    return { limit, cursor, query };
  }
  const sessionId = url.searchParams.get("sessionId");
  if (sessionId !== null && !IDENTIFIER_RE.test(sessionId)) return null;
  return { limit, cursor, sessionId };
}

function isActiveRunQuery(url) {
  const keys = [...url.searchParams.keys()];
  return (
    keys.length === 1 &&
    keys[0] === "status" &&
    url.searchParams.getAll("status").length === 1 &&
    url.searchParams.get("status") === "active"
  );
}

function validateModelImages(value) {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 1) return null;
  const images = [];
  for (const image of value) {
    if (
      !image ||
      typeof image !== "object" ||
      Array.isArray(image) ||
      !hasOnlyKeys(image, new Set(["mimeType", "data"])) ||
      image.mimeType !== "image/jpeg"
    ) {
      return null;
    }
    const data = decodeBase64(image.data);
    if (!data || detectedImageMimeType(data) !== "image/jpeg") return null;
    images.push({ mimeType: "image/jpeg", data });
  }
  return images;
}

async function modelImagesAreDecodable(images = []) {
  for (const image of images) {
    try {
      const decoder = sharp(image.data, {
        failOn: "warning",
        limitInputPixels: MAX_MODEL_IMAGE_DIMENSION ** 2,
      });
      const metadata = await decoder.metadata();
      if (
        metadata.format !== "jpeg" ||
        !Number.isInteger(metadata.width) ||
        !Number.isInteger(metadata.height) ||
        metadata.width < 1 ||
        metadata.height < 1 ||
        metadata.width > MAX_MODEL_IMAGE_DIMENSION ||
        metadata.height > MAX_MODEL_IMAGE_DIMENSION ||
        (metadata.pages ?? 1) !== 1
      ) {
        return false;
      }
      await decoder.raw().toBuffer();
    } catch {
      return false;
    }
  }
  return true;
}

export function validateRunRequest(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const sessionId = value.sessionId;
  const parentEntryId = value.parentEntryId;
  const includeMemorySnapshot = value.includeMemorySnapshot;
  const model = value.model;
  const suppliedOrigin = value.origin;
  const suppliedIdentity = value.identity;
  const images = validateModelImages(value.images);
  const isRoot = sessionId === null && parentEntryId === null;
  const isContinuation =
    typeof sessionId === "string" &&
    IDENTIFIER_RE.test(sessionId) &&
    typeof parentEntryId === "string" &&
    IDENTIFIER_RE.test(parentEntryId);
  if (
    !UUID_RE.test(value.runId ?? "") ||
    (!isRoot && !isContinuation) ||
    !isBoundedString(value.prompt, 1, 16_000) ||
    !isBoundedString(value.systemPrompt, 1, 32_000) ||
    !new Set(["owner", "delegated", "none"]).has(value.toolPolicy) ||
    !(
      model === undefined ||
      isModelId(model)
    ) ||
    !(
      includeMemorySnapshot === undefined ||
      typeof includeMemorySnapshot === "boolean"
    ) ||
    images === null ||
    !Array.isArray(value.context) ||
    value.context.length > 4
  ) {
    return null;
  }
  if (
    !suppliedOrigin ||
    typeof suppliedOrigin !== "object" ||
    Array.isArray(suppliedOrigin) ||
    !hasOnlyKeys(
      suppliedOrigin,
      new Set(["scopeId", "adapterInstanceId"]),
    ) ||
    !isBoundedString(suppliedOrigin.scopeId, 1, 512) ||
    !isBoundedString(suppliedOrigin.adapterInstanceId, 1, 128)
  ) {
    return null;
  }
  const origin = {
    scopeId: suppliedOrigin.scopeId,
    adapterInstanceId: suppliedOrigin.adapterInstanceId,
  };
  if (
    !suppliedIdentity ||
    typeof suppliedIdentity !== "object" ||
    Array.isArray(suppliedIdentity) ||
    !hasOnlyKeys(suppliedIdentity, new Set(["requester", "anchors"])) ||
    !Array.isArray(suppliedIdentity.anchors) ||
    suppliedIdentity.anchors.length < 1 ||
    suppliedIdentity.anchors.length > MAX_MEMORY_ANCHORS
  ) {
    return null;
  }
  const requester = suppliedIdentity.requester;
  if (
    !requester ||
    typeof requester !== "object" ||
    Array.isArray(requester) ||
    !hasOnlyKeys(requester, new Set(["id", "label"])) ||
    !isHostIdentity(requester.id) ||
    !(
      requester.label === null ||
      requester.label === undefined ||
      isBoundedString(requester.label, 1, 256)
    )
  ) {
    return null;
  }
  const anchors = [];
  const seenIdentityIds = new Set();
  for (const anchor of suppliedIdentity.anchors) {
    if (
      !anchor ||
      typeof anchor !== "object" ||
      Array.isArray(anchor) ||
      !hasOnlyKeys(anchor, new Set(["id", "label"])) ||
      !isHostIdentity(anchor.id) ||
      !(
        anchor.label === null ||
        anchor.label === undefined ||
        isBoundedString(anchor.label, 1, 256)
      ) ||
      seenIdentityIds.has(anchor.id)
    ) {
      return null;
    }
    seenIdentityIds.add(anchor.id);
    anchors.push({ id: anchor.id, label: anchor.label ?? null });
  }
  if (anchors[0].id !== requester.id) return null;
  const identity = {
    requester: { id: requester.id, label: requester.label ?? null },
    anchors,
  };
  const context = [];
  for (const item of value.context) {
    if (
      !item ||
      typeof item !== "object" ||
      item.kind !== "reference" ||
      !isBoundedString(item.text, 1, 16_000)
    ) {
      return null;
    }
    context.push({ kind: item.kind, text: item.text });
  }
  let memory;
  if (value.memory !== undefined) {
    const supplied = value.memory;
    if (
      !supplied ||
      typeof supplied !== "object" ||
      Array.isArray(supplied) ||
      !hasOnlyKeys(
        supplied,
        new Set([
          "primaryBankId",
          "requesterIsOwner",
          "grantedBankIds",
          "participants",
          "query",
        ]),
      ) ||
      !isBankId(supplied.primaryBankId) ||
      !(
        supplied.query === undefined ||
        supplied.query === null ||
        isBoundedString(supplied.query, 1, 8_000)
      ) ||
      !Array.isArray(supplied.participants) ||
      supplied.participants.length > MAX_PARTICIPANTS
    ) {
      return null;
    }
    const grantedBankIds = boundedBankIds(supplied.grantedBankIds);
    if (
      typeof supplied.requesterIsOwner !== "boolean" ||
      grantedBankIds === null ||
      (supplied.requesterIsOwner && grantedBankIds.length > 0)
    ) {
      return null;
    }
    const participants = [];
    const participantIds = new Set();
    for (const participant of supplied.participants) {
      const bankIds = boundedBankIds(participant?.bankIds);
      if (
        !participant ||
        typeof participant !== "object" ||
        Array.isArray(participant) ||
        !hasOnlyKeys(
          participant,
          new Set(["id", "label", "allowed", "bankIds"]),
        ) ||
        !isHostIdentity(participant.id) ||
        participant.id === identity.requester.id ||
        participantIds.has(participant.id) ||
        !(
          participant.label === null ||
          participant.label === undefined ||
          isBoundedString(participant.label, 1, 256)
        ) ||
        typeof participant.allowed !== "boolean" ||
        bankIds === null ||
        (!participant.allowed && bankIds.length > 0)
      ) {
        return null;
      }
      participantIds.add(participant.id);
      participants.push({
        id: participant.id,
        label: participant.label ?? null,
        allowed: participant.allowed,
        bankIds,
      });
    }
    memory = {
      primaryBankId: supplied.primaryBankId,
      requesterIsOwner: supplied.requesterIsOwner,
      grantedBankIds,
      participants,
      ...(supplied.query ? { query: supplied.query } : {}),
    };
  }
  return {
    runId: value.runId,
    sessionId,
    parentEntryId,
    prompt: value.prompt,
    context,
    systemPrompt: value.systemPrompt,
    toolPolicy: value.toolPolicy,
    identity,
    origin,
    ...(model ? { model } : {}),
    ...(includeMemorySnapshot ? { includeMemorySnapshot: true } : {}),
    ...(memory ? { memory } : {}),
    ...(images.length > 0 ? { images } : {}),
  };
}

function decodeBase64(value) {
  if (
    typeof value !== "string" ||
    value.length < 4 ||
    value.length > Math.ceil(MAX_ATTACHMENT_BYTES / 3) * 4 + 4 ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(value)
  ) {
    return null;
  }
  const decoded = Buffer.from(value, "base64");
  const expected = value.replace(/=+$/, "");
  const actual = decoded.toString("base64").replace(/=+$/, "");
  if (
    decoded.length === 0 ||
    decoded.length > MAX_ATTACHMENT_BYTES ||
    actual !== expected
  ) {
    return null;
  }
  return decoded;
}

function detectedImageMimeType(data) {
  if (
    data.length >= 3 &&
    data[0] === 0xff &&
    data[1] === 0xd8 &&
    data[2] === 0xff
  ) {
    return "image/jpeg";
  }
  if (
    data.length >= 8 &&
    data.subarray(0, 8).equals(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    )
  ) {
    return "image/png";
  }
  if (
    data.length >= 12 &&
    data.subarray(0, 4).toString("ascii") === "RIFF" &&
    data.subarray(8, 12).toString("ascii") === "WEBP"
  ) {
    return "image/webp";
  }
  return null;
}

export function validateAttachmentRequest(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const mimeType = value.mimeType;
  const filename = value.filename;
  if (
    typeof mimeType !== "string" ||
    !MIME_RE.test(mimeType) ||
    !(
      filename === null ||
      filename === undefined ||
      isBoundedString(filename, 1, 200)
    )
  ) {
    return null;
  }
  if (value.kind === "image" && IMAGE_MIME_TYPES.has(mimeType)) {
    const data = decodeBase64(value.data);
    if (
      !data ||
      detectedImageMimeType(data) !== mimeType ||
      value.text !== undefined
    ) {
      return null;
    }
    return { kind: "image", mimeType, filename: filename ?? null, data };
  }
  if (
    value.kind === "text" &&
    isBoundedString(value.text, 1, MAX_ATTACHMENT_TEXT_CHARS) &&
    value.data === undefined
  ) {
    return {
      kind: "text",
      mimeType,
      filename: filename ?? null,
      text: value.text,
    };
  }
  return null;
}

function writeNdjson(response, event) {
  return response.write(`${JSON.stringify(event)}\n`);
}

function preparedClients(clients) {
  if (!Array.isArray(clients) || clients.length < 1 || clients.length > 32) {
    throw new Error("Agent service clients are invalid");
  }
  const ids = new Set();
  const tokens = new Set();
  return clients.map((client) => {
    if (
      !client ||
      typeof client !== "object" ||
      !IDENTIFIER_RE.test(client.id ?? "") ||
      typeof client.token !== "string" ||
      client.token.length < 24 ||
      !Array.isArray(client.capabilities) ||
      client.capabilities.length < 1 ||
      new Set(client.capabilities).size !== client.capabilities.length ||
      client.capabilities.some((item) => !CLIENT_CAPABILITIES.has(item)) ||
      !(
        client.adapterInstanceId === undefined ||
        IDENTIFIER_RE.test(client.adapterInstanceId)
      ) ||
      !(
        client.scopePrefix === undefined ||
        isBoundedString(client.scopePrefix, 1, 512)
      ) ||
      !(
        client.cancelAny === undefined ||
        typeof client.cancelAny === "boolean"
      )
    ) {
      throw new Error("Agent service client is invalid");
    }
    if (ids.has(client.id)) throw new Error("Agent client IDs must be unique");
    if (tokens.has(client.token)) {
      throw new Error("Agent client tokens must be unique");
    }
    ids.add(client.id);
    tokens.add(client.token);
    return {
      id: client.id,
      tokenDigest: createHash("sha256")
        .update(`Bearer ${client.token}`)
        .digest(),
      capabilities: new Set(client.capabilities),
      adapterInstanceId: client.adapterInstanceId ?? null,
      scopePrefix: client.scopePrefix ?? null,
      cancelAny: client.cancelAny === true,
    };
  });
}

function authenticate(request, clients) {
  const actual = request.headers.authorization ?? "";
  const actualDigest = createHash("sha256").update(actual).digest();
  let authenticated = null;
  for (const client of clients) {
    if (timingSafeEqual(actualDigest, client.tokenDigest)) authenticated = client;
  }
  return authenticated;
}

function requireCapability(response, principal, capability) {
  if (principal.capabilities.has(capability)) return true;
  json(response, 403, {
    error: { code: "FORBIDDEN", message: "Forbidden" },
  });
  return false;
}

function principalAllowsRun(principal, run) {
  if (principal.adapterInstanceId === null) return true;
  if (run.origin.adapterInstanceId !== principal.adapterInstanceId) return false;
  if (
    principal.scopePrefix !== null &&
    !run.origin.scopeId.startsWith(principal.scopePrefix)
  ) {
    return false;
  }
  if (
    principal.scopePrefix !== null &&
    (run.identity.anchors.some(
      ({ id }) => !id.startsWith(principal.scopePrefix),
    ) || !run.identity.requester.id.startsWith(principal.scopePrefix))
  ) {
    return false;
  }
  if (
    run.memory &&
    run.memory.primaryBankId !== run.origin.scopeId
  ) {
    return false;
  }
  return true;
}

export function createAgentServer({ engine, clients, logger = console }) {
  const serviceClients = preparedClients(clients);
  return createServer(async (request, response) => {
    response.setHeader("x-content-type-options", "nosniff");
    response.setHeader("cache-control", "no-store");
    const url = new URL(request.url ?? "/", "http://agent.invalid");

    if (request.method === "GET" && url.pathname === "/health") {
      json(response, 200, { status: "ok" });
      return;
    }

    const principal = authenticate(request, serviceClients);
    if (!principal) {
      json(response, 401, {
        error: { code: "UNAUTHORIZED", message: "Unauthorized" },
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/models") {
      if (!requireCapability(response, principal, "models")) return;
      try {
        json(response, 200, await engine.listModels());
      } catch (error) {
        logger.error("Model catalog request failed", {
          errorType: error instanceof Error ? error.name : "UnknownError",
        });
        json(response, 502, {
          error: {
            code: "MODEL_CATALOG_UNAVAILABLE",
            message: "Model catalog unavailable",
          },
        });
      }
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/sessions") {
      if (!requireCapability(response, principal, "history")) return;
      const options = listOptions(url, "sessions");
      if (!options) {
        json(response, 400, {
          error: { code: "INVALID_REQUEST", message: "Invalid history request" },
        });
        return;
      }
      try {
        json(response, 200, await engine.listSessions(options));
      } catch {
        json(response, 500, {
          error: {
            code: "HISTORY_UNAVAILABLE",
            message: "Session history unavailable",
          },
        });
      }
      return;
    }

    const sessionMatch = url.pathname.match(
      /^\/v1\/sessions\/([A-Za-z0-9_-]{1,128})$/,
    );
    if (request.method === "GET" && sessionMatch) {
      if (!requireCapability(response, principal, "history")) return;
      try {
        const session = await engine.getSession(sessionMatch[1]);
        if (!session) {
          json(response, 404, {
            error: { code: "NOT_FOUND", message: "Session not found" },
          });
        } else {
          json(response, 200, session);
        }
      } catch {
        json(response, 500, {
          error: {
            code: "HISTORY_UNAVAILABLE",
            message: "Session history unavailable",
          },
        });
      }
      return;
    }

    if (request.method === "GET" && url.pathname === "/v1/runs") {
      if (url.searchParams.has("status")) {
        if (!requireCapability(response, principal, "status")) return;
        if (!isActiveRunQuery(url)) {
          json(response, 400, {
            error: { code: "INVALID_REQUEST", message: "Invalid run query" },
          });
          return;
        }
        try {
          json(response, 200, await engine.listActiveRuns());
        } catch {
          json(response, 500, {
            error: {
              code: "RUNS_UNAVAILABLE",
              message: "Active runs unavailable",
            },
          });
        }
        return;
      }
      if (!requireCapability(response, principal, "history")) return;
      const options = listOptions(url, "runs");
      if (!options) {
        json(response, 400, {
          error: { code: "INVALID_REQUEST", message: "Invalid history request" },
        });
        return;
      }
      try {
        json(response, 200, await engine.listRunAudits(options));
      } catch {
        json(response, 500, {
          error: {
            code: "HISTORY_UNAVAILABLE",
            message: "Run history unavailable",
          },
        });
      }
      return;
    }

    const auditMatch = url.pathname.match(
      /^\/v1\/runs\/([0-9a-f-]+)\/audit$/i,
    );
    if (request.method === "GET" && auditMatch) {
      if (!requireCapability(response, principal, "history")) return;
      if (!UUID_RE.test(auditMatch[1])) {
        json(response, 400, {
          error: { code: "INVALID_REQUEST", message: "Invalid history request" },
        });
        return;
      }
      try {
        const audit = await engine.getRunAudit(auditMatch[1]);
        if (!audit) {
          json(response, 404, {
            error: { code: "NOT_FOUND", message: "Run audit not found" },
          });
        } else {
          json(response, 200, audit);
        }
      } catch {
        json(response, 500, {
          error: {
            code: "HISTORY_UNAVAILABLE",
            message: "Run history unavailable",
          },
        });
      }
      return;
    }

    if (request.method === "POST" && url.pathname === "/v1/attachments/describe") {
      if (!requireCapability(response, principal, "attachments")) return;
      let attachment;
      try {
        attachment = validateAttachmentRequest(
          await readJson(request, MAX_ATTACHMENT_BODY_BYTES),
        );
      } catch {
        attachment = null;
      }
      if (!attachment) {
        json(response, 400, {
          error: { code: "INVALID_REQUEST", message: "Invalid attachment request" },
        });
        return;
      }
      try {
        const description = await engine.describeAttachment(attachment);
        if (!isBoundedString(description, 1, 4_000)) {
          throw new Error("Invalid attachment description");
        }
        json(response, 200, { description });
      } catch (error) {
        logger.error("Attachment analysis failed", {
          errorType: error instanceof Error ? error.name : "UnknownError",
        });
        json(response, 502, {
          error: { code: "ANALYSIS_FAILED", message: "Attachment analysis failed" },
        });
      }
      return;
    }

    const cancelMatch = url.pathname.match(
      /^\/v1\/runs\/([0-9a-f-]+)\/cancel$/i,
    );
    if (request.method === "POST" && cancelMatch) {
      if (!requireCapability(response, principal, "runs")) return;
      const runId = cancelMatch[1];
      if (!UUID_RE.test(runId)) {
        json(response, 400, {
          error: { code: "INVALID_REQUEST", message: "Invalid run id" },
        });
        return;
      }
      json(response, 200, {
        cancelled: await engine.cancel(
          runId,
          principal.cancelAny ? null : principal.id,
        ),
      });
      return;
    }

    if (request.method !== "POST" || url.pathname !== "/v1/runs") {
      json(response, 404, {
        error: { code: "NOT_FOUND", message: "Not found" },
      });
      return;
    }
    if (!requireCapability(response, principal, "runs")) return;

    let run;
    try {
      run = validateRunRequest(await readJson(request, MAX_RUN_BODY_BYTES));
      if (run && !(await modelImagesAreDecodable(run.images))) run = null;
    } catch {
      run = null;
    }
    if (!run) {
      json(response, 400, {
        error: { code: "INVALID_REQUEST", message: "Invalid run request" },
      });
      return;
    }
    if (!principalAllowsRun(principal, run)) {
      json(response, 403, {
        error: { code: "FORBIDDEN", message: "Forbidden" },
      });
      return;
    }

    response.writeHead(200, {
      "content-type": "application/x-ndjson; charset=utf-8",
      "transfer-encoding": "chunked",
    });
    const requestOwner = principal.id;
    let completed = false;
    response.on("close", () => {
      if (!completed) void engine.cancel(run.runId, requestOwner);
    });
    try {
      for await (const event of engine.run(run, requestOwner)) {
        if (response.destroyed) break;
        writeNdjson(response, event);
      }
      completed = true;
    } catch (error) {
      completed = true;
      logger.error("Agent run failed", {
        runId: run.runId,
        errorType: error instanceof Error ? error.name : "UnknownError",
      });
      if (!response.destroyed) {
        writeNdjson(response, {
          type: "run_failed",
          code: "AGENT_ERROR",
          message: "Agent run failed",
        });
      }
    } finally {
      if (!response.destroyed) response.end();
    }
  });
}
