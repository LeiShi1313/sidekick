import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

import {
  readUsableCodexAccessToken,
  usableCodexAccessToken,
} from "../src/codex-access-token.mjs";

const execFileAsync = promisify(execFile);
const TOKEN_SCRIPT = new URL("../src/codex-access-token.mjs", import.meta.url);

function jwt(payload) {
  const encoded = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `header.${encoded}.signature`;
}

function authFile(accessToken) {
  return {
    auth_mode: "chatgpt",
    tokens: {
      access_token: accessToken,
      refresh_token: "refresh-token-is-never-returned",
    },
  };
}

test("accepts an unexpired ChatGPT Codex access token", () => {
  const token = jwt({
    exp: 2_000,
    "https://api.openai.com/auth": { chatgpt_account_id: "account-1" },
  });

  assert.equal(
    usableCodexAccessToken(authFile(token), {
      now: 1_000_000,
      minimumValidityMs: 60_000,
    }),
    token,
  );
});

test("rejects missing, malformed, expired, and non-ChatGPT Codex auth", () => {
  const validClaims = {
    exp: 2_000,
    "https://api.openai.com/auth": { chatgpt_account_id: "account-1" },
  };
  const expired = jwt({ ...validClaims, exp: 1_059 });
  const atMinimumValidity = jwt({ ...validClaims, exp: 1_060 });
  const noAccount = jwt({ exp: 2_000 });

  for (const value of [
    null,
    {},
    { ...authFile(jwt(validClaims)), auth_mode: "apikey" },
    authFile("not-a-jwt"),
    authFile(expired),
    authFile(atMinimumValidity),
    authFile(noAccount),
  ]) {
    assert.equal(
      usableCodexAccessToken(value, {
        now: 1_000_000,
        minimumValidityMs: 60_000,
      }),
      null,
    );
  }
});

test("reads only a usable token and fails closed for unavailable auth files", async () => {
  const directory = await mkdtemp(join(tmpdir(), "sidekick-codex-auth-"));
  const validPath = join(directory, "valid.json");
  const malformedPath = join(directory, "malformed.json");
  const token = jwt({
    exp: 2_000,
    "https://api.openai.com/auth": { chatgpt_account_id: "account-1" },
  });
  try {
    await writeFile(validPath, JSON.stringify(authFile(token)), { mode: 0o600 });
    await writeFile(malformedPath, "not json", { mode: 0o600 });

    assert.equal(
      await readUsableCodexAccessToken(validPath, {
        now: 1_000_000,
        minimumValidityMs: 60_000,
      }),
      token,
    );
    assert.equal(await readUsableCodexAccessToken(malformedPath), null);
    assert.equal(
      await readUsableCodexAccessToken(join(directory, "missing")),
      null,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("prints only the access token for credential-command use", async () => {
  const directory = await mkdtemp(join(tmpdir(), "sidekick-codex-command-"));
  const validPath = join(directory, "valid.json");
  const token = jwt({
    exp: Math.ceil(Date.now() / 1_000) + 600,
    "https://api.openai.com/auth": { chatgpt_account_id: "account-1" },
  });
  try {
    await writeFile(validPath, JSON.stringify(authFile(token)), { mode: 0o600 });
    const { stdout, stderr } = await execFileAsync(process.execPath, [
      TOKEN_SCRIPT.pathname,
      validPath,
    ]);
    assert.equal(stdout, token);
    assert.equal(stderr, "");
    await assert.rejects(
      execFileAsync(process.execPath, [
        TOKEN_SCRIPT.pathname,
        join(directory, "missing"),
      ]),
      (error) => error.code === 1 && error.stdout === "" && error.stderr === "",
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
