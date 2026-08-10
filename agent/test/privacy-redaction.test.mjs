import assert from "node:assert/strict";
import test from "node:test";

import {
  SensitiveTextStream,
  redactSensitiveText,
  sanitizeFetchedText,
  sanitizeSensitiveValue,
} from "../src/privacy-redaction.mjs";

test("pseudonymizes platform actor identifiers with a stable keyed alias", () => {
  const options = {
    identityAliasKey: "test-identity-alias-key-that-is-strong",
    identityScope: "telegram:chat:-1001",
  };
  const first = redactSensitiveText(
    "Requester telegram:user:419540347 replied to qq:user:12345678",
    options,
  );
  const repeated = redactSensitiveText(
    "Again telegram:user:419540347",
    options,
  );

  assert.match(first, /actor_[a-f0-9]{16}/);
  assert.doesNotMatch(first, /telegram:user:419540347|qq:user:12345678/);
  assert.equal(
    first.match(/actor_[a-f0-9]{16}/)?.[0],
    repeated.match(/actor_[a-f0-9]{16}/)?.[0],
  );
  assert.notEqual(
    first.match(/actor_[a-f0-9]{16}/)?.[0],
    redactSensitiveText("Again telegram:user:419540347", {
      ...options,
      identityScope: "telegram:chat:-1002",
    }).match(/actor_[a-f0-9]{16}/)?.[0],
  );
  assert.doesNotMatch(
    redactSensitiveText("Post by telegram:channel:998877", options),
    /telegram:channel:998877/,
  );
  const bridgeActorId =
    "telegram:matrix-bridge:6332621450%3A-1001%3A0123456789abcdef0123456789abcdef";
  const bridged = redactSensitiveText(`Post by ${bridgeActorId}`, options);
  assert.match(bridged, /actor_[a-f0-9]{16}/);
  assert.equal(bridged.includes(bridgeActorId), false);
});

const DOCUMENTATION_IPV4 = "203.0.113.42";
const DOCUMENTATION_IPV6 = "2001:db8::42";
const RUNTIME_PATH = "/home/example-service/private/workspace";
const TEST_SECRET = "demo-provider-secret-123";

test("redacts network, runtime, internal-service, and credential metadata", () => {
  const result = redactSensitiveText(
    `IPv4 ${DOCUMENTATION_IPV4}; IPv6 ${DOCUMENTATION_IPV6}; ` +
      `path ${RUNTIME_PATH}; service http://memory-api:8888/private; ` +
      `authorization: Bearer ${TEST_SECRET}; raw ${TEST_SECRET}`,
    { sensitiveValues: [TEST_SECRET] },
  );

  assert.match(result, /\[REDACTED_IP_ADDRESS\]/);
  assert.match(result, /\[REDACTED_RUNTIME_PATH\]/);
  assert.match(result, /\[REDACTED_INTERNAL_URL\]/);
  assert.match(result, /\[REDACTED_SECRET\]/);
  assert.doesNotMatch(
    result,
    /203\.0\.113\.42|2001:db8::42|example-service|memory-api|demo-provider-secret/,
  );
});

test("preserves a network literal that the user already supplied", () => {
  assert.equal(
    redactSensitiveText(`Check ${DOCUMENTATION_IPV4}`, {
      allowedNetworkLiterals: [DOCUMENTATION_IPV4],
    }),
    `Check ${DOCUMENTATION_IPV4}`,
  );
});

test("withholds fetched content that reflects requester metadata", () => {
  const result = sanitizeFetchedText(
    JSON.stringify({
      ip: DOCUMENTATION_IPV4,
      hostname: "runtime-node",
      city: "Example City",
      org: "Example Network",
    }),
  );

  assert.equal(
    result,
    "[Content withheld because the page reflected request or runtime metadata.]",
  );
});

test("withholds structured request metadata even without an address literal", () => {
  const result = sanitizeSensitiveValue(
    {
      hostname: "runtime-node",
      city: "Example City",
      org: "Example Network",
      timezone: "Example/Zone",
    },
    { externalText: true },
  );

  assert.equal(
    result,
    "[Content withheld because the page reflected request or runtime metadata.]",
  );
});

test("does not release a sensitive token split across stream deltas", () => {
  const stream = new SensitiveTextStream({ sensitiveValues: [TEST_SECRET] });
  const output = [
    stream.push("Address 203.0."),
    stream.push("113.42 path /home/example-"),
    stream.push("service/private key demo-provider-"),
    stream.push("secret-123"),
    stream.flush(),
  ].join("");

  assert.match(output, /\[REDACTED_IP_ADDRESS\]/);
  assert.match(output, /\[REDACTED_RUNTIME_PATH\]/);
  assert.match(output, /\[REDACTED_SECRET\]/);
  assert.doesNotMatch(
    output,
    /203\.0\.|113\.42|example-service|demo-provider|secret-123/,
  );
});

test("redacts sensitive tokens at every possible stream split", () => {
  for (const sensitive of [DOCUMENTATION_IPV4, DOCUMENTATION_IPV6, RUNTIME_PATH]) {
    for (let split = 1; split < sensitive.length; split += 1) {
      const stream = new SensitiveTextStream();
      const output =
        stream.push(`Value ${sensitive.slice(0, split)}`) +
        stream.push(`${sensitive.slice(split)} done`) +
        stream.flush();
      assert.match(output, /\[REDACTED_/);
      assert.doesNotMatch(output, new RegExp(sensitive.replaceAll(".", "\\.")));
    }
  }
});

test("preserves ordinary web content while redacting a standalone address", () => {
  const result = sanitizeFetchedText(
    `Version 1.2.3; docs https://example.com/api/v1; ` +
      `relative path agent/src/main.mjs; DNS result ${DOCUMENTATION_IPV4}.`,
  );

  assert.match(result, /Version 1\.2\.3/);
  assert.match(result, /https:\/\/example\.com\/api\/v1/);
  assert.match(result, /agent\/src\/main\.mjs/);
  assert.match(result, /DNS result \[REDACTED_IP_ADDRESS\]/);
  assert.doesNotMatch(result, /203\.0\.113\.42/);
});
