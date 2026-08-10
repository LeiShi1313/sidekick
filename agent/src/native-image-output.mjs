import {
  decodeGeneratedImage,
  decodeGeneratedImageDataUrl,
  MAX_GENERATED_IMAGE_RESPONSE_BYTES,
} from "./image-tools.mjs";


function chatImageCandidates(payload) {
  if (!Array.isArray(payload?.choices)) return [];
  const candidates = [];
  for (const choice of payload.choices) {
    for (const message of [choice?.delta, choice?.message]) {
      if (message?.images === undefined) continue;
      if (!Array.isArray(message.images)) {
        throw new Error("Provider returned an invalid native image response");
      }
      for (const image of message.images) {
        const imageUrl = image?.image_url;
        candidates.push({
          encoding: "data-url",
          value: typeof imageUrl === "string" ? imageUrl : imageUrl?.url,
        });
      }
    }
  }
  return candidates;
}


function responsesImageCandidates(payload) {
  if (
    payload?.type !== "response.completed" ||
    !Array.isArray(payload?.response?.output)
  ) {
    return [];
  }
  return payload.response.output
    .filter((item) => item?.type === "image_generation_call")
    .map((item) => ({ encoding: "base64", value: item.result }));
}


function geminiParts(payload) {
  if (!Array.isArray(payload?.candidates)) return [];
  return payload.candidates.flatMap((candidate) =>
    Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [],
  );
}


function geminiImageCandidates(payload) {
  return geminiParts(payload).flatMap((part) => {
    if (part?.thought === true) return [];
    const inlineData = part?.inlineData ?? part?.inline_data;
    if (inlineData === undefined) return [];
    return [
      {
        encoding: "base64",
        value: inlineData?.data,
        mimeType: inlineData?.mimeType ?? inlineData?.mime_type,
      },
    ];
  });
}


const IMAGE_CANDIDATE_EXTRACTORS = [
  chatImageCandidates,
  responsesImageCandidates,
  geminiImageCandidates,
];


function normalizeNativeImage(payload) {
  const candidates = IMAGE_CANDIDATE_EXTRACTORS.flatMap((extract) =>
    extract(payload),
  );
  if (candidates.length === 0) return null;
  if (candidates.length !== 1) {
    throw new Error("Provider returned an invalid native image response");
  }
  const candidate = candidates[0];
  const image =
    candidate.encoding === "data-url"
      ? decodeGeneratedImageDataUrl(candidate.value)
      : decodeGeneratedImage(candidate.value);
  if (
    !image ||
    (candidate.mimeType !== undefined &&
      candidate.mimeType !== image.mimeType)
  ) {
    throw new Error("Provider returned an invalid native image");
  }
  return {
    artifact: {
      filename: image.filename,
      mimeType: image.mimeType,
      displayAs: "image",
      data:
        candidate.encoding === "data-url" ? image.encoded : candidate.value,
    },
    sizeBytes: image.data.length,
  };
}


function hasToolCall(payload) {
  const choices = Array.isArray(payload?.choices) ? payload.choices : [];
  const responseOutput = Array.isArray(payload?.response?.output)
    ? payload.response.output
    : [];
  const chatToolCall = choices.some(
    (choice) =>
      choice?.delta?.tool_calls?.length > 0 ||
      choice?.message?.tool_calls?.length > 0,
  );
  const responsesToolCall = responseOutput.some(
    (item) => item?.type === "function_call",
  );
  const geminiToolCall = geminiParts(payload).some(
    (part) =>
      part?.functionCall !== undefined || part?.function_call !== undefined,
  );
  return chatToolCall || responsesToolCall || geminiToolCall;
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
      if (hasToolCall(payload)) {
        if (imageSeen) {
          throw new Error("Provider returned conflicting native image output");
        }
        toolCallSeen = true;
      }
      const output = normalizeNativeImage(payload);
      if (output === null) return;
      if (imageSeen || toolCallSeen) {
        throw new Error("Provider returned conflicting native image output");
      }
      imageSeen = true;
      onImage(output);
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
