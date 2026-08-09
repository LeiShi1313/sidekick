import assert from "node:assert/strict";
import test from "node:test";

import { createImageTools } from "../src/image-tools.mjs";


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
