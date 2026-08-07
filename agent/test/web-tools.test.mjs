import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { Readable } from "node:stream";
import test from "node:test";

import {
  constrainWebTools,
  createPinnedRequester,
} from "../src/web-tools.mjs";

const PUBLIC_ADDRESS = "93.184.216.34";

async function publicLookup() {
  return [{ address: PUBLIC_ADDRESS, family: 4 }];
}

function pageResponse({
  status = 200,
  statusText = "OK",
  headers = { "content-type": "text/plain; charset=utf-8" },
  body = "page content",
} = {}) {
  return {
    status,
    statusText,
    headers: new Headers(headers),
    body: Buffer.from(body),
  };
}

async function textRequest() {
  return pageResponse();
}

function tool(name, execute) {
  return {
    name,
    label: name,
    description: name,
    parameters: {},
    execute,
  };
}

function tools(options) {
  return constrainWebTools(
    [
      tool("web_search", async () => ({ content: [], details: {} })),
      tool("fetch_content", async () => {
        throw new Error("the unpinned upstream fetcher must never run");
      }),
    ],
    options,
  );
}

test("forces raw Exa search and bounds query count", async () => {
  let received;
  const [search] = constrainWebTools([
    tool("web_search", async (_id, params) => {
      received = params;
      return { content: [{ type: "text", text: "ok" }], details: {} };
    }),
    tool("fetch_content", async () => ({ content: [], details: {} })),
  ]);

  await search.execute(
    "call-1",
    {
      queries: ["one", "two", "three", "four", "five"],
      provider: "openai",
      workflow: "summary-review",
    },
    undefined,
    undefined,
    {},
  );

  assert.deepEqual(received.queries, ["one", "two", "three", "four"]);
  assert.equal(received.provider, "exa");
  assert.equal(received.workflow, "none");
});

test("allows bounded public HTTP and HTTPS page retrieval", async () => {
  const requested = [];
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async (target) => {
      requested.push({
        url: target.url.toString(),
        address: target.address,
        family: target.family,
      });
      return pageResponse({ body: `content for ${target.url.pathname}` });
    },
  });

  const result = await fetchContent.execute(
    "call-2",
    { urls: ["https://example.com/a", "http://example.org/b"] },
    undefined,
    undefined,
    {},
  );

  assert.deepEqual(requested, [
    {
      url: "https://example.com/a",
      address: PUBLIC_ADDRESS,
      family: 4,
    },
    {
      url: "http://example.org/b",
      address: PUBLIC_ADDRESS,
      family: 4,
    },
  ]);
  assert.match(result.content[0].text, /content for \/a/);
  assert.match(result.content[0].text, /content for \/b/);
  assert.equal(result.details.successful, 2);
});

test("blocks internal hostnames, loopback, GitHub, and video retrieval", async () => {
  let requested = 0;
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async () => {
      requested += 1;
      return pageResponse();
    },
  });

  for (const url of [
    "file:///etc/passwd",
    "/etc/passwd",
    "http://hindsight:8888/v1/default/banks/",
    "http://localhost/private",
    "http://127.0.0.1/private",
    "http://192.0.2.1/reserved",
    "http://[::1]/private",
    "http://[2001:db8::1]/reserved",
    "https://github.com/owner/repo",
    "https://youtu.be/example",
  ]) {
    await assert.rejects(
      fetchContent.execute(
        "call-3",
        { url, forceClone: true, prompt: "inspect" },
        undefined,
        undefined,
        {},
      ),
      /not allowed/i,
    );
  }
  assert.equal(requested, 0);
});

test("blocks a hostname when any DNS answer is private", async () => {
  let requested = false;
  const [, fetchContent] = tools({
    lookup: async () => [
      { address: PUBLIC_ADDRESS, family: 4 },
      { address: "10.0.0.8", family: 4 },
    ],
    request: async () => {
      requested = true;
      return pageResponse();
    },
  });

  await assert.rejects(
    fetchContent.execute(
      "call-4",
      { url: "https://public-looking.example/page" },
      undefined,
      undefined,
      {},
    ),
    /not allowed/i,
  );
  assert.equal(requested, false);
});

test("uses the vetted DNS address without resolving the hostname again", async () => {
  let lookupCalls = 0;
  const targets = [];
  const [, fetchContent] = tools({
    lookup: async () => {
      lookupCalls += 1;
      return lookupCalls === 1
        ? [{ address: PUBLIC_ADDRESS, family: 4 }]
        : [{ address: "127.0.0.1", family: 4 }];
    },
    request: async (target) => {
      targets.push(target);
      return pageResponse();
    },
  });

  await fetchContent.execute(
    "call-rebinding",
    { url: "https://rebinding.example/page" },
    undefined,
    undefined,
    {},
  );

  assert.equal(lookupCalls, 1);
  assert.equal(targets.length, 1);
  assert.equal(targets[0].address, PUBLIC_ADDRESS);
});

test("pins the actual HTTP socket lookup to the vetted address", async () => {
  let requestOptions;
  const transport = (options, onResponse) => {
    requestOptions = options;
    const request = new EventEmitter();
    request.end = () => {
      const response = Readable.from([Buffer.from("socket-pinned page")]);
      response.statusCode = 200;
      response.statusMessage = "OK";
      response.headers = { "content-type": "text/plain" };
      queueMicrotask(() => onResponse(response));
    };
    return request;
  };
  const request = createPinnedRequester({
    httpRequest: transport,
    httpsRequest: transport,
  });

  const response = await request({
    url: new URL("https://rebinding.example/path?q=1"),
    address: PUBLIC_ADDRESS,
    family: 4,
  });
  const pinned = await new Promise((resolve, reject) => {
    requestOptions.lookup(
      "rebinding.example",
      {},
      (error, address, family) =>
        error ? reject(error) : resolve({ address, family }),
    );
  });

  assert.deepEqual(pinned, { address: PUBLIC_ADDRESS, family: 4 });
  assert.equal(requestOptions.hostname, "rebinding.example");
  assert.equal(requestOptions.servername, "rebinding.example");
  assert.equal(requestOptions.path, "/path?q=1");
  assert.equal(response.body.toString(), "socket-pinned page");
});

test("blocks redirects from a public URL to a private destination", async () => {
  const requested = [];
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async (target) => {
      requested.push(target.url.toString());
      return pageResponse({
        status: 302,
        statusText: "Found",
        headers: { location: "http://127.0.0.1/admin" },
        body: "",
      });
    },
  });

  await assert.rejects(
    fetchContent.execute(
      "call-5",
      { url: "https://example.com/start" },
      undefined,
      undefined,
      {},
    ),
    /not allowed/i,
  );
  assert.deepEqual(requested, ["https://example.com/start"]);
});

test("allows redirects between public destinations and pins each hop", async () => {
  const requested = [];
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async (target) => {
      requested.push({ url: target.url.toString(), address: target.address });
      if (requested.length === 1) {
        return pageResponse({
          status: 302,
          statusText: "Found",
          headers: { location: "https://www.example.org/final" },
          body: "",
        });
      }
      return pageResponse({ body: "final page" });
    },
  });

  const result = await fetchContent.execute(
    "call-6",
    { url: "https://example.com/start" },
    undefined,
    undefined,
    {},
  );

  assert.deepEqual(requested, [
    { url: "https://example.com/start", address: PUBLIC_ADDRESS },
    { url: "https://www.example.org/final", address: PUBLIC_ADDRESS },
  ]);
  assert.match(result.content[0].text, /final page/);
  assert.deepEqual(result.details.urls, ["https://www.example.org/final"]);
});

test("withholds reflected requester metadata from fetched page results", async () => {
  const reflectedAddress = "203.0.113.42";
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async () =>
      pageResponse({
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          ip: reflectedAddress,
          hostname: "runtime-node",
          city: "Example City",
          org: "Example Network",
        }),
      }),
  });

  const result = await fetchContent.execute(
    "call-7",
    { url: "https://example.com/reflect" },
    undefined,
    undefined,
    {},
  );

  assert.equal(
    result.content[0].text,
    "[Content withheld because the page reflected request or runtime metadata.]",
  );
  assert.doesNotMatch(
    JSON.stringify(result),
    /203\.0\.113\.42|runtime-node|Example City|Example Network/,
  );
});
