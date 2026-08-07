import assert from "node:assert/strict";
import test from "node:test";

import {
  SensitiveTextStream,
  redactSensitiveText,
  sanitizeFetchedText,
} from "../src/privacy-redaction.mjs";

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
