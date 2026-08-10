import assert from "node:assert/strict";
import test from "node:test";

import { MAX_GENERATED_IMAGE_RESPONSE_BYTES } from "../src/image-tools.mjs";
import { createNativeImageCaptureFetch } from "../src/native-image-output.mjs";


const JPEG_DATA_URL = `data:image/jpeg;base64,${Buffer.from([
  0xff, 0xd8, 0xff, 0x00, 0x00, 0x00, 0xff, 0xd9,
]).toString("base64")}`;


function sseResponse(deltas) {
  const body = deltas
    .map((delta) =>
      `data: ${JSON.stringify({
        id: "chatcmpl-native-image-test",
        choices: [{ index: 0, delta, finish_reason: null }],
      })}\n\n`,
    )
    .join("");
  return new Response(`${body}data: [DONE]\n\n`, {
    headers: { "content-type": "text/event-stream" },
  });
}


test("captures one validated native image while preserving the provider stream", async () => {
  const outputs = [];
  const source = sseResponse([
    {
      images: [
        { type: "image_url", image_url: { url: JPEG_DATA_URL } },
      ],
    },
  ]);
  const expectedBody = await source.clone().text();
  const fetchNativeImage = createNativeImageCaptureFetch({
    fetchImpl: async () => source,
    onImage: (output) => outputs.push(output),
  });

  const response = await fetchNativeImage("https://provider.example/v1/chat");

  assert.equal(await response.text(), expectedBody);
  assert.equal(outputs.length, 1);
  assert.equal(outputs[0].artifact.mimeType, "image/jpeg");
  assert.equal(outputs[0].artifact.data, JPEG_DATA_URL.split(",")[1]);
  assert.equal(outputs[0].sizeBytes, 8);
});


test("rejects malformed, mismatched, and multiple native images", async (t) => {
  const pngLabeledAsJpeg = `data:image/png;base64,${JPEG_DATA_URL.split(",")[1]}`;
  const cases = [
    { name: "malformed data URL", images: [{ image_url: { url: "https://example.test/image.jpg" } }] },
    { name: "MIME mismatch", images: [{ image_url: { url: pngLabeledAsJpeg } }] },
    {
      name: "multiple images",
      images: [
        { image_url: { url: JPEG_DATA_URL } },
        { image_url: { url: JPEG_DATA_URL } },
      ],
    },
  ];
  for (const scenario of cases) {
    await t.test(scenario.name, async () => {
      const fetchNativeImage = createNativeImageCaptureFetch({
        fetchImpl: async () => sseResponse([{ images: scenario.images }]),
        onImage() {},
      });
      const response = await fetchNativeImage(
        "https://provider.example/v1/chat",
      );
      await assert.rejects(response.text(), /invalid native image/);
    });
  }
});


test("rejects an oversized unterminated provider event", async () => {
  const oversized = `data: ${"x".repeat(MAX_GENERATED_IMAGE_RESPONSE_BYTES)}`;
  const fetchNativeImage = createNativeImageCaptureFetch({
    fetchImpl: async () =>
      new Response(oversized, {
        headers: { "content-type": "text/event-stream" },
      }),
    onImage() {},
  });

  const response = await fetchNativeImage("https://provider.example/v1/chat");
  await assert.rejects(response.text(), /oversized native image response/);
});


test("rejects a completion that combines native image output and a tool call", async () => {
  const fetchNativeImage = createNativeImageCaptureFetch({
    fetchImpl: async () =>
      sseResponse([
        {
          images: [
            {
              type: "image_url",
              image_url: { url: JPEG_DATA_URL },
            },
          ],
        },
        {
          tool_calls: [
            {
              index: 0,
              id: "call-image-after-native",
              type: "function",
              function: { name: "image_generate", arguments: "{}" },
            },
          ],
        },
      ]),
    onImage() {},
  });

  const response = await fetchNativeImage("https://provider.example/v1/chat");
  await assert.rejects(response.text(), /conflicting native image output/);
});
