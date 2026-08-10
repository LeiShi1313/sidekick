import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";


export const IMAGE_TOOL_NAME = "image_generate";
const MAX_GENERATED_IMAGE_BYTES = 5 * 1024 * 1024;
const MAX_GENERATED_IMAGE_BASE64_CHARS =
  Math.ceil(MAX_GENERATED_IMAGE_BYTES / 3) * 4;
export const MAX_GENERATED_IMAGE_RESPONSE_BYTES =
  MAX_GENERATED_IMAGE_BASE64_CHARS + 64 * 1024;


export function createBoundedImageFetch(fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== "function") {
    throw new Error("Image provider fetch is unavailable");
  }
  return async (input, init) => {
    const response = await fetchImpl(input, init);
    const suppliedLength = Number(response.headers.get("content-length"));
    if (
      Number.isFinite(suppliedLength) &&
      suppliedLength > MAX_GENERATED_IMAGE_RESPONSE_BYTES
    ) {
      await response.body?.cancel();
      throw new Error("Image provider returned an oversized response");
    }
    if (response.body === null) return response;

    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        total += value.byteLength;
        if (total > MAX_GENERATED_IMAGE_RESPONSE_BYTES) {
          throw new Error("Image provider returned an oversized response");
        }
        chunks.push(Buffer.from(value));
      }
    } catch (error) {
      await reader.cancel(error).catch(() => {});
      throw error;
    }
    return new Response(Buffer.concat(chunks, total), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}


export function createImageGenerationGate() {
  let active = false;
  return {
    tryAcquire() {
      if (active) return null;
      active = true;
      let released = false;
      return () => {
        if (released) return;
        released = true;
        active = false;
      };
    },
  };
}


export function decodeGeneratedImage(value) {
  if (
    typeof value !== "string" ||
    value.length < 8 ||
    value.length > MAX_GENERATED_IMAGE_BASE64_CHARS ||
    !/^[A-Za-z0-9+/]+={0,2}$/.test(value)
  ) {
    return null;
  }
  const data = Buffer.from(value, "base64");
  if (
    data.length < 8 ||
    data.length > MAX_GENERATED_IMAGE_BYTES ||
    data.toString("base64") !== value
  ) {
    return null;
  }
  if (
    data[0] === 0xff &&
    data[1] === 0xd8 &&
    data[2] === 0xff &&
    data.at(-2) === 0xff &&
    data.at(-1) === 0xd9
  ) {
    return {
      data,
      filename: "generated-image.jpg",
      mimeType: "image/jpeg",
    };
  }
  if (
    data.subarray(0, 8).equals(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    )
  ) {
    return {
      data,
      filename: "generated-image.png",
      mimeType: "image/png",
    };
  }
  return null;
}


export function decodeGeneratedImageDataUrl(value) {
  if (typeof value !== "string") return null;
  const match = /^data:(image\/(?:jpeg|png));base64,(.+)$/.exec(value);
  if (!match) return null;
  const image = decodeGeneratedImage(match[2]);
  if (!image || image.mimeType !== match[1]) return null;
  return { ...image, encoded: match[2] };
}


export function createImageTools({
  client,
  model,
  onArtifact,
  tryAcquire = () => () => {},
}) {
  if (!client || !model) return [];
  if (typeof onArtifact !== "function") {
    throw new Error("Image artifact receiver is unavailable");
  }
  let attempted = false;
  return [
    defineTool({
      name: IMAGE_TOOL_NAME,
      label: "Generate image",
      description:
        "Generate one original image from a detailed text prompt when the user asks to create, draw, or render an image. The host delivers the image directly to the chat. This tool creates new images; it does not search for existing images or edit an input image.",
      promptSnippet:
        "Image creation is host-controlled. When the user asks for an original image, always call image_generate exactly once. Never return image bytes or an image URL directly. After the tool succeeds, briefly tell the user the generated image is attached.",
      parameters: Type.Object({
        prompt: Type.String({
          minLength: 1,
          maxLength: 16_000,
          description: "A complete, detailed description of the image to create.",
        }),
      }),
      async execute(toolCallId, { prompt }, signal) {
        if (attempted) {
          throw new Error("Only one image can be generated per request");
        }
        attempted = true;
        const release = tryAcquire();
        if (typeof release !== "function") {
          throw new Error("Image generation is busy. Try again later.");
        }
        try {
          let response;
          try {
            response = await client.images.generate(
              {
                model,
                prompt,
                n: 1,
                size: "1024x1024",
                quality: "medium",
                output_format: "jpeg",
                output_compression: 85,
              },
              { signal },
            );
          } catch (error) {
            if (signal?.aborted) throw error;
            if (error?.code === "moderation_blocked") {
              throw new Error(
                "Image generation was blocked by safety checks. Ask the user to revise the prompt.",
              );
            }
            throw new Error("Image generation is unavailable");
          }
          const encoded = response?.data?.[0]?.b64_json;
          const image = decodeGeneratedImage(encoded);
          if (!image || response.data.length !== 1) {
            throw new Error("Image provider returned an invalid image");
          }
          signal?.throwIfAborted();
          await onArtifact(toolCallId, {
            filename: image.filename,
            mimeType: image.mimeType,
            displayAs: "image",
            data: encoded,
          });
          return {
            content: [
              {
                type: "text",
                text: "One image was generated and queued for delivery by the host.",
              },
            ],
            details: {
              filename: image.filename,
              mimeType: image.mimeType,
              sizeBytes: image.data.length,
            },
          };
        } finally {
          release();
        }
      },
    }),
  ];
}
