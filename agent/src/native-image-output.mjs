import {
  decodeGeneratedImageDataUrl,
  MAX_GENERATED_IMAGE_RESPONSE_BYTES,
} from "./image-tools.mjs";


function nativeImageUrl(payload) {
  const images = payload?.choices?.[0]?.delta?.images;
  if (images === undefined || (Array.isArray(images) && images.length === 0)) {
    return null;
  }
  if (!Array.isArray(images) || images.length !== 1) {
    throw new Error("Provider returned an invalid native image response");
  }
  const imageUrl = images[0]?.image_url;
  return typeof imageUrl === "string" ? imageUrl : imageUrl?.url;
}


export function createNativeImageCaptureFetch({
  fetchImpl = globalThis.fetch,
  onImage,
}) {
  if (typeof fetchImpl !== "function") {
    throw new Error("Chat provider fetch is unavailable");
  }
  if (typeof onImage !== "function") {
    throw new Error("Native image receiver is unavailable");
  }
  return async (input, init) => {
    const response = await fetchImpl(input, init);
    if (
      !response.ok ||
      response.body === null ||
      !response.headers.get("content-type")?.includes("text/event-stream")
    ) {
      return response;
    }

    const decoder = new TextDecoder();
    let buffered = "";
    let imageSeen = false;
    let toolCallSeen = false;
    const inspectLine = (line) => {
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line.startsWith("data:")) return;
      const data = line.slice(5).trimStart();
      if (!data || data === "[DONE]") return;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch {
        return;
      }
      if (payload?.choices?.[0]?.delta?.tool_calls?.length > 0) {
        if (imageSeen) {
          throw new Error("Provider returned conflicting native image output");
        }
        toolCallSeen = true;
      }
      const url = nativeImageUrl(payload);
      if (url === null) return;
      if (imageSeen || toolCallSeen) {
        throw new Error("Provider returned conflicting native image output");
      }
      const image = decodeGeneratedImageDataUrl(url);
      if (!image) {
        throw new Error("Provider returned an invalid native image");
      }
      imageSeen = true;
      onImage({
        artifact: {
          filename: image.filename,
          mimeType: image.mimeType,
          displayAs: "image",
          data: image.encoded,
        },
        sizeBytes: image.data.length,
      });
    };
    const inspectChunk = (text) => {
      buffered += text;
      let newline = buffered.indexOf("\n");
      while (newline !== -1) {
        const line = buffered.slice(0, newline);
        buffered = buffered.slice(newline + 1);
        if (line.length > MAX_GENERATED_IMAGE_RESPONSE_BYTES) {
          throw new Error("Provider returned an oversized native image response");
        }
        inspectLine(line);
        newline = buffered.indexOf("\n");
      }
      if (buffered.length > MAX_GENERATED_IMAGE_RESPONSE_BYTES) {
        throw new Error("Provider returned an oversized native image response");
      }
    };
    const body = response.body.pipeThrough(
      new TransformStream({
        transform(chunk, controller) {
          inspectChunk(decoder.decode(chunk, { stream: true }));
          controller.enqueue(chunk);
        },
        flush() {
          inspectChunk(decoder.decode());
          if (buffered) inspectLine(buffered);
        },
      }),
    );
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}
