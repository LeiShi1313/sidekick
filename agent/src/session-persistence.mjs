import { chmodSync, existsSync } from "node:fs";

import {
  sanitizeMessageInPlace,
  sanitizeSensitiveValue,
} from "./privacy-redaction.mjs";

const OMITTED_TOOL_RESULT = "[Tool result omitted after use]";
const OMITTED_CUSTOM_MESSAGE = "[Custom message omitted after use]";
const hardenedManagers = new WeakSet();

function secureSessionFile(manager) {
  const path = manager.getSessionFile();
  if (path && existsSync(path)) chmodSync(path, 0o600);
}

function safeWebEntryMetadata(data) {
  const type = data?.type === "search" || data?.type === "fetch"
    ? data.type
    : "unknown";
  const timestamp = Number.isSafeInteger(data?.timestamp)
    ? data.timestamp
    : Date.now();
  return { type, timestamp, omitted: true };
}

function safeCustomEntry(customType, data) {
  if (customType === "web-search-results") return safeWebEntryMetadata(data);
  return { omitted: true };
}

function safeContentPart(part) {
  if (!part || typeof part !== "object") return part;
  if (part.type === "thinking") return null;
  if (part.type === "toolCall") {
    return {
      type: "toolCall",
      id: String(part.id ?? ""),
      name: String(part.name ?? ""),
      arguments: {},
    };
  }
  if (part.type === "image") {
    return { type: "text", text: "[Image omitted after use]" };
  }
  if (part.type === "text") return { type: "text", text: part.text };
  return null;
}

export function sessionSafeMessage(message, state = {}) {
  const copy = structuredClone(message);
  if (copy.role === "user" && state.userMessageContent !== undefined) {
    copy.content = state.userMessageContent;
  }
  sanitizeMessageInPlace(copy, state.privacyOptions);
  if (copy.role === "assistant" && Array.isArray(copy.content)) {
    copy.content = copy.content.map(safeContentPart).filter(Boolean);
  } else if (copy.role === "user" && Array.isArray(copy.content)) {
    copy.content = copy.content.map(safeContentPart).filter(Boolean);
  } else if (copy.role === "toolResult") {
    copy.content = [{ type: "text", text: OMITTED_TOOL_RESULT }];
    if ("details" in copy) {
      copy.details = sanitizeSensitiveValue(copy.details, state.privacyOptions);
    }
  }
  return copy;
}

export function hardenSessionPersistence(manager, getState = () => ({})) {
  if (hardenedManagers.has(manager)) return manager;
  hardenedManagers.add(manager);

  const appendMessage = manager.appendMessage.bind(manager);
  manager.appendMessage = (message) => {
    const entryId = appendMessage(sessionSafeMessage(message, getState()));
    secureSessionFile(manager);
    return entryId;
  };

  const appendCustomEntry = manager.appendCustomEntry.bind(manager);
  manager.appendCustomEntry = (customType, data) => {
    const entryId = appendCustomEntry(customType, safeCustomEntry(customType, data));
    secureSessionFile(manager);
    return entryId;
  };

  const appendCustomMessageEntry = manager.appendCustomMessageEntry.bind(manager);
  manager.appendCustomMessageEntry = (customType, _content, display, _details) => {
    const entryId = appendCustomMessageEntry(
      customType,
      OMITTED_CUSTOM_MESSAGE,
      display,
      undefined,
    );
    secureSessionFile(manager);
    return entryId;
  };

  secureSessionFile(manager);
  return manager;
}
