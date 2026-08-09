import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";


export const IMAGE_TOOL_NAME = "image_generate";
const MAX_GENERATED_IMAGE_BYTES = 5 * 1024 * 1024;
const MAX_GENERATED_IMAGE_BASE64_CHARS =
  Math.ceil(MAX_GENERATED_IMAGE_BYTES / 3) * 4;


function decodeGeneratedImage(value) {
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


export function createImageTools({ client, model, onArtifact }) {
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
        "Use image_generate when the user asks for an original image. Call it at most once in this request, then briefly tell the user the generated image is attached.",
      parameters: Type.Object({
        prompt: Type.String({
          minLength: 1,
          maxLength: 16_000,
          description: "A complete, detailed description of the image to create.",
        }),
      }),
      async execute(toolCallId, { prompt }) {
        if (attempted) {
          throw new Error("Only one image can be generated per request");
        }
        attempted = true;
        let response;
        try {
          response = await client.images.generate({
            model,
            prompt,
            n: 1,
            size: "1024x1024",
            quality: "medium",
            output_format: "jpeg",
            output_compression: 85,
          });
        } catch (error) {
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
      },
    }),
  ];
}
