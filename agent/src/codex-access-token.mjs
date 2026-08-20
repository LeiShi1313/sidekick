import { open } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export const CODEX_AUTH_FILE = "/run/secrets/codex-auth.json";
export const CODEX_ACCESS_TOKEN_COMMAND =
  `!node /app/src/codex-access-token.mjs ${CODEX_AUTH_FILE}`;

const DEFAULT_MINIMUM_VALIDITY_MS = 60_000;
const MAX_AUTH_FILE_BYTES = 256 * 1024;
const MAX_ACCESS_TOKEN_BYTES = 64 * 1024;
const JWT_SEGMENT_RE = /^[A-Za-z0-9_-]+$/;

function jwtPayload(token) {
  if (
    typeof token !== "string" ||
    Buffer.byteLength(token) > MAX_ACCESS_TOKEN_BYTES
  ) {
    return null;
  }
  const segments = token.split(".");
  if (
    segments.length !== 3 ||
    segments.some((segment) => !segment || !JWT_SEGMENT_RE.test(segment))
  ) {
    return null;
  }
  try {
    const payload = JSON.parse(
      Buffer.from(segments[1], "base64url").toString("utf8"),
    );
    return payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload
      : null;
  } catch {
    return null;
  }
}

export function usableCodexAccessToken(
  value,
  {
    now = Date.now(),
    minimumValidityMs = DEFAULT_MINIMUM_VALIDITY_MS,
  } = {},
) {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    value.auth_mode !== "chatgpt" ||
    !value.tokens ||
    typeof value.tokens !== "object" ||
    Array.isArray(value.tokens)
  ) {
    return null;
  }
  const token = value.tokens.access_token;
  const payload = jwtPayload(token);
  const auth = payload?.["https://api.openai.com/auth"];
  if (
    !Number.isFinite(payload?.exp) ||
    payload.exp * 1_000 <= now + minimumValidityMs ||
    !auth ||
    typeof auth !== "object" ||
    Array.isArray(auth) ||
    typeof auth.chatgpt_account_id !== "string" ||
    !auth.chatgpt_account_id.trim()
  ) {
    return null;
  }
  return token;
}

export async function readUsableCodexAccessToken(
  path = CODEX_AUTH_FILE,
  options,
) {
  let handle;
  try {
    handle = await open(path, "r");
    const { size } = await handle.stat();
    if (!Number.isSafeInteger(size) || size <= 0 || size > MAX_AUTH_FILE_BYTES) {
      return null;
    }
    const content = Buffer.allocUnsafe(size);
    let offset = 0;
    while (offset < size) {
      const { bytesRead } = await handle.read(
        content,
        offset,
        size - offset,
        offset,
      );
      if (bytesRead === 0) return null;
      offset += bytesRead;
    }
    const value = JSON.parse(content.toString("utf8"));
    return usableCodexAccessToken(value, options);
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => {});
  }
}

const invokedUrl = process.argv[1]
  ? pathToFileURL(process.argv[1]).href
  : null;
if (invokedUrl === import.meta.url) {
  const token = await readUsableCodexAccessToken(
    process.argv[2] || CODEX_AUTH_FILE,
  );
  if (token) process.stdout.write(token);
  else process.exitCode = 1;
}
