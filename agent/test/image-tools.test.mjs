import assert from "node:assert/strict";
import test from "node:test";

import {
  createBoundedImageFetch,
  createImageGenerationGate,
  createImageTools,
} from "../src/image-tools.mjs";


const JPEG_BYTES = Buffer.from([
  0xff, 0xd8, 0xff, 0x00, 0x00, 0x00, 0xff, 0xd9,
]);
const PNG_BYTES = Buffer.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
]);


function fixture(result = { data: [{ b64_json: JPEG_BYTES.toString("base64") }] }) {
  const calls = [];
  const artifacts = [];
  const client = {
    images: {
      async generate(request) {
        calls.push(request);
        if (result instanceof Error) throw result;
        return result;
      },
    },
  };
  const tools = createImageTools({
    client,
    model: "gpt-image-2",
    onArtifact: (toolCallId, artifact) => {
      artifacts.push({ toolCallId, artifact });
    },
  });
  return { tool: tools[0], calls, artifacts };
}


test("generates one bounded JPEG without returning bytes to the model", async () => {
  const app = fixture();

  const result = await app.tool.execute("call-image-1", {
    prompt: "A fox and Devon Rex cat visiting Xiamen",
  });

  assert.deepEqual(app.calls, [
    {
      model: "gpt-image-2",
      prompt: "A fox and Devon Rex cat visiting Xiamen",
      n: 1,
      size: "1024x1024",
      quality: "medium",
      output_format: "jpeg",
      output_compression: 85,
    },
  ]);
  assert.deepEqual(app.artifacts, [
    {
      toolCallId: "call-image-1",
      artifact: {
        filename: "generated-image.jpg",
        mimeType: "image/jpeg",
        displayAs: "image",
        data: JPEG_BYTES.toString("base64"),
      },
    },
  ]);
  assert.equal(result.terminate, true);
  assert.match(result.content[0].text, /generated and queued/i);
  assert.equal(result.details.sizeBytes, JPEG_BYTES.length);
  assert.doesNotMatch(JSON.stringify(result), new RegExp(JPEG_BYTES.toString("base64")));
  await assert.rejects(
    app.tool.execute("call-image-2", { prompt: "Generate another" }),
    /one image/i,
  );
  assert.equal(app.calls.length, 1);
});


test("rejects malformed and oversized image responses", async () => {
  const malformed = fixture({ data: [{ b64_json: "not base64" }] });
  await assert.rejects(
    malformed.tool.execute("call-image-1", { prompt: "A landscape" }),
    /invalid image/i,
  );
  await assert.rejects(
    malformed.tool.execute("call-image-2", { prompt: "Try again" }),
    /one image/i,
  );
  assert.equal(malformed.calls.length, 1);
  assert.deepEqual(malformed.artifacts, []);

  const oversized = fixture({
    data: [
      {
        b64_json: Buffer.concat([
          Buffer.from([0xff, 0xd8, 0xff]),
          Buffer.alloc(5 * 1024 * 1024),
        ]).toString("base64"),
      },
    ],
  });
  await assert.rejects(
    oversized.tool.execute("call-image-2", { prompt: "A landscape" }),
    /invalid image/i,
  );
  assert.deepEqual(oversized.artifacts, []);
});


test("does not expose provider failures to the model", async () => {
  const app = fixture(new Error("secret provider details"));

  await assert.rejects(
    app.tool.execute("call-image-1", { prompt: "A landscape" }),
    /^Error: Image generation is unavailable$/,
  );
});


test("passes cancellation to a pending provider request", async () => {
  const controller = new AbortController();
  let providerSignal;
  const client = {
    images: {
      async generate(_request, options) {
        providerSignal = options.signal;
        return new Promise((_resolve, reject) => {
          options.signal.addEventListener(
            "abort",
            () => reject(options.signal.reason),
            { once: true },
          );
        });
      },
    },
  };
  const [tool] = createImageTools({
    client,
    model: "gpt-image-2",
    onArtifact: () => assert.fail(),
  });

  const pending = tool.execute(
    "call-image-1",
    { prompt: "A landscape" },
    controller.signal,
  );
  controller.abort(new Error("cancelled by user"));

  await assert.rejects(pending, /cancelled by user/i);
  assert.equal(providerSignal, controller.signal);
});


test("rejects a chunked provider response before it can grow without bound", async () => {
  const boundedFetch = createBoundedImageFetch(async () => {
    const body = new ReadableStream({
      start(controller) {
        for (let index = 0; index < 8; index += 1) {
          controller.enqueue(Buffer.alloc(1024 * 1024));
        }
        controller.close();
      },
    });
    return new Response(body, { status: 200 });
  });

  await assert.rejects(
    boundedFetch("https://provider.example/v1/images/generations"),
    /oversized response/i,
  );
});


test("allows only one image provider request at a time", async () => {
  let finish;
  let calls = 0;
  const providerResult = new Promise((resolve) => {
    finish = resolve;
  });
  const client = {
    images: {
      async generate() {
        calls += 1;
        return providerResult;
      },
    },
  };
  const gate = createImageGenerationGate();
  const options = {
    client,
    model: "gpt-image-2",
    onArtifact: () => {},
    tryAcquire: () => gate.tryAcquire(),
  };
  const [first] = createImageTools(options);
  const [second] = createImageTools(options);

  const pending = first.execute("call-image-1", { prompt: "A landscape" });
  await assert.rejects(
    second.execute("call-image-2", { prompt: "Another landscape" }),
    /busy/i,
  );
  assert.equal(calls, 1);

  finish({ data: [{ b64_json: JPEG_BYTES.toString("base64") }] });
  await pending;
  const [third] = createImageTools(options);
  await third.execute("call-image-3", { prompt: "After the first" });
  assert.equal(calls, 2);
});


test("uses the actual PNG metadata when a provider ignores JPEG output", async () => {
  const app = fixture({ data: [{ b64_json: PNG_BYTES.toString("base64") }] });

  const result = await app.tool.execute("call-image-1", {
    prompt: "A landscape",
  });

  assert.equal(app.artifacts[0].artifact.filename, "generated-image.png");
  assert.equal(app.artifacts[0].artifact.mimeType, "image/png");
  assert.equal(result.details.mimeType, "image/png");
});


test("returns no image tool when image generation is disabled", () => {
  assert.deepEqual(
    createImageTools({ client: null, model: null, onArtifact: () => {} }),
    [],
  );
});
