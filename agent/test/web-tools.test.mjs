import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { connect as netConnect, createServer } from "node:net";
import { createServer as createHttpServer } from "node:http";
import { Readable } from "node:stream";
import test from "node:test";

import { parseEgressProxy } from "../src/egress-proxy.mjs";
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

test("uses configured search routing and bounds query count", async () => {
  let received;
  const [search] = constrainWebTools([
    tool("web_search", async (_id, params) => {
      received = params;
      return { content: [{ type: "text", text: "ok" }], details: {} };
    }),
    tool("fetch_content", async () => ({ content: [], details: {} })),
  ], { hasCodexAuth: async () => true });

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
  assert.equal("provider" in received, false);
  assert.equal(received.workflow, "none");
});

test("makes Exa primary when mounted Codex auth is unavailable", async () => {
  let received;
  const [search] = constrainWebTools(
    [
      tool("web_search", async (_id, params) => {
        received = params;
        return { content: [{ type: "text", text: "ok" }], details: {} };
      }),
      tool("fetch_content", async () => ({ content: [], details: {} })),
    ],
    { hasCodexAuth: async () => false },
  );

  await search.execute("call-exa", { query: "today's news" });

  assert.equal(received.provider, "exa");
});

test("falls back to Exa when mounted Codex auth is rejected remotely", async () => {
  const attempts = [];
  const [search] = constrainWebTools(
    [
      tool("web_search", async (_id, params) => {
        attempts.push(params);
        if (attempts.length === 1) {
          throw Object.assign(new Error("OpenAI authentication failed"), {
            provider: "openai",
            kind: "auth",
          });
        }
        return { content: [{ type: "text", text: "exa" }], details: {} };
      }),
      tool("fetch_content", async () => ({ content: [], details: {} })),
    ],
    { hasCodexAuth: async () => true },
  );

  const result = await search.execute("call-auth-fallback", {
    query: "today's news",
  });

  assert.equal("provider" in attempts[0], false);
  assert.equal(attempts[1].provider, "exa");
  assert.equal(result.content[0].text, "exa");
});

test("falls back when pi-web-access returns an OpenAI auth error result", async () => {
  const attempts = [];
  const [search] = constrainWebTools(
    [
      tool("web_search", async (_id, params) => {
        attempts.push(params);
        if (attempts.length === 1) {
          return {
            content: [
              {
                type: "text",
                text: "Error: openai search failed (auth): OpenAI API error 401",
              },
            ],
            details: { successfulQueries: 0, totalResults: 0 },
          };
        }
        return { content: [{ type: "text", text: "exa" }], details: {} };
      }),
      tool("fetch_content", async () => ({ content: [], details: {} })),
    ],
    { hasCodexAuth: async () => true },
  );

  const result = await search.execute("call-result-fallback", {
    query: "today's news",
  });

  assert.equal("provider" in attempts[0], false);
  assert.equal(attempts[1].provider, "exa");
  assert.equal(result.content[0].text, "exa");
});

test("does not duplicate non-auth search failures", async () => {
  let attempts = 0;
  const failure = Object.assign(new Error("search unavailable"), {
    provider: "openai",
    kind: "network",
  });
  const [search] = constrainWebTools(
    [
      tool("web_search", async () => {
        attempts += 1;
        throw failure;
      }),
      tool("fetch_content", async () => ({ content: [], details: {} })),
    ],
    { hasCodexAuth: async () => true },
  );

  await assert.rejects(
    search.execute("call-network", { query: "today's news" }),
    failure,
  );
  assert.equal(attempts, 1);
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
    egressProxy: null,
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

test("destroys redirect and error bodies instead of draining unbounded data", async () => {
  for (const status of [302, 404]) {
    let producedBytes = 0;
    let upstreamResponse;
    const transport = (_options, onResponse) => {
      const request = new EventEmitter();
      request.end = () => {
        upstreamResponse = Readable.from(
          (function* body() {
            for (let index = 0; index < 10; index += 1) {
              producedBytes += 1024 * 1024;
              yield Buffer.alloc(1024 * 1024);
            }
          })(),
        );
        upstreamResponse.statusCode = status;
        upstreamResponse.statusMessage = status === 302 ? "Found" : "Not Found";
        upstreamResponse.headers =
          status === 302 ? { location: "https://example.org/final" } : {};
        queueMicrotask(() => onResponse(upstreamResponse));
      };
      return request;
    };
    const request = createPinnedRequester({
      httpRequest: transport,
      httpsRequest: transport,
      egressProxy: null,
    });

    const response = await request(
      {
        url: new URL("https://example.com/start"),
        address: PUBLIC_ADDRESS,
        family: 4,
      },
      { maxBytes: 1 },
    );
    await new Promise((resolve) => setImmediate(resolve));

    assert.equal(response.status, status);
    assert.equal(upstreamResponse.destroyed, true);
    assert.equal(producedBytes, 0);
  }
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

test("refuses known IP-logging hosts without contacting them", async () => {
  let requested = 0;
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async () => {
      requested += 1;
      return pageResponse();
    },
  });

  for (const url of [
    "https://urlto.me/2XUGq",
    "https://www.urlto.me/2XUGq",
    "http://grabify.link/CHECK",
    "https://iplogger.org/xyz",
    // Resolvers collapse trailing dots, so extra dots must not bypass the list.
    "http://grabify.link../CHECK",
    "https://www.urlto.me.../2XUGq",
  ]) {
    await assert.rejects(
      fetchContent.execute("call-8", { url }, undefined, undefined, {}),
      /not allowed.*IP-logging/i,
    );
  }
  assert.equal(requested, 0);
});

test("blocks redirects that land on a known IP-logging host", async () => {
  const requested = [];
  const [, fetchContent] = tools({
    lookup: publicLookup,
    request: async (target) => {
      requested.push(target.url.toString());
      return pageResponse({
        status: 302,
        statusText: "Found",
        headers: { location: "https://grabify.link/TRAP" },
        body: "",
      });
    },
  });

  await assert.rejects(
    fetchContent.execute("call-9", { url: "https://example.com/start" }, undefined, undefined, {}),
    /not allowed.*IP-logging/i,
  );
  assert.equal(requested.length, 1);
});

function listen(server) {
  return new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
}

function startConnectProxy() {
  const seen = [];
  const server = createServer((socket) => {
    let buffer = "";
    socket.on("data", function onData(chunk) {
      buffer += chunk.toString("latin1");
      const end = buffer.indexOf("\r\n\r\n");
      if (end === -1) return;
      socket.off("data", onData);
      const requestLine = buffer.slice(0, buffer.indexOf("\r\n"));
      const authority = requestLine.split(" ")[1];
      seen.push(authority);
      const [host, port] = authority.split(":");
      const upstream = netConnect(Number(port), host);
      upstream.on("connect", () => {
        socket.write("HTTP/1.1 200 Connection established\r\n\r\n");
        socket.pipe(upstream);
        upstream.pipe(socket);
      });
      upstream.on("error", () => socket.destroy());
    });
  });
  return { server, seen };
}

test("tunnels pinned requests through an HTTP CONNECT egress proxy", async () => {
  const proxy = startConnectProxy();
  await listen(proxy.server);
  const target = createHttpServer((_request, response) => {
    response.setHeader("content-type", "text/plain");
    response.end("tunnel-body");
  });
  await listen(target);

  try {
    const requester = createPinnedRequester({
      egressProxy: parseEgressProxy(`http://127.0.0.1:${proxy.server.address().port}`),
    });
    const response = await requester({
      url: new URL(`http://target.example:${target.address().port}/page`),
      address: "127.0.0.1",
      family: 4,
    });
    assert.equal(response.status, 200);
    assert.equal(response.body.toString(), "tunnel-body");
    assert.deepEqual(proxy.seen, [`127.0.0.1:${target.address().port}`]);
  } finally {
    proxy.server.close();
    target.close();
  }
});

function startSocksProxy({ auth = false } = {}) {
  const events = [];
  const server = createServer((socket) => {
    let stage = "greeting";
    let pending = Buffer.alloc(0);
    const upstream = { socket: null };
    socket.on("data", function onData(chunk) {
      pending = Buffer.concat([pending, chunk]);
      if (stage === "greeting") {
        if (pending.length < 2) return;
        const count = pending[1];
        if (pending.length < 2 + count) return;
        const methods = [...pending.subarray(2, 2 + count)];
        const chosen = auth && methods.includes(0x02) ? 0x02 : 0x00;
        events.push({ stage: "method", chosen });
        socket.write(Buffer.from([5, chosen]));
        pending = pending.subarray(2 + count);
        stage = chosen === 0x02 ? "auth" : "connect";
        if (stage === "connect") return next();
      }
      if (stage === "auth") {
        if (pending.length < 2) return;
        const userLength = pending[1];
        if (pending.length < 2 + userLength + 1) return;
        const passLength = pending[2 + userLength];
        if (pending.length < 3 + userLength + passLength) return;
        events.push({
          stage: "auth",
          user: pending.subarray(2, 2 + userLength).toString(),
          pass: pending.subarray(3 + userLength, 3 + userLength + passLength).toString(),
        });
        socket.write(Buffer.from([1, 0]));
        pending = pending.subarray(3 + userLength + passLength);
        stage = "connect";
      }
      if (stage === "connect") {
        next();
      }
      function next() {
        if (pending.length < 4) return;
        const atyp = pending[3];
        let address;
        let offset;
        if (atyp === 1) {
          if (pending.length < 10) return;
          address = [...pending.subarray(4, 8)].join(".");
          offset = 8;
        } else if (atyp === 4) {
          if (pending.length < 22) return;
          const groups = [];
          for (let index = 4; index < 20; index += 2) {
            groups.push(pending.readUInt16BE(index).toString(16));
          }
          address = groups.join(":");
          offset = 20;
        } else {
          socket.destroy();
          return;
        }
        const port = pending.readUInt16BE(offset);
        events.push({ stage: "connect", address, port });
        socket.write(Buffer.from([5, 0, 0, 1, 0, 0, 0, 0, 0, 0]));
        socket.off("data", onData);
        upstream.socket = netConnect(port, address.includes(":") ? "::1" : address);
        upstream.socket.on("connect", () => {
          upstream.socket.pipe(socket);
          socket.pipe(upstream.socket);
        });
        upstream.socket.on("error", () => socket.destroy());
      }
    });
  });
  return { server, events };
}

test("tunnels pinned requests through a SOCKS5 egress proxy", async () => {
  const proxy = startSocksProxy();
  await listen(proxy.server);
  const target = createHttpServer((_request, response) => {
    response.end("socks-body");
  });
  await listen(target);

  try {
    const requester = createPinnedRequester({
      egressProxy: parseEgressProxy(`socks5://127.0.0.1:${proxy.server.address().port}`),
    });
    const response = await requester({
      url: new URL(`http://target.example:${target.address().port}/page`),
      address: "127.0.0.1",
      family: 4,
    });
    assert.equal(response.status, 200);
    assert.equal(response.body.toString(), "socks-body");
    assert.deepEqual(proxy.events, [
      { stage: "method", chosen: 0 },
      { stage: "connect", address: "127.0.0.1", port: target.address().port },
    ]);
  } finally {
    proxy.server.close();
    target.close();
  }
});

test("negotiates SOCKS5 username and password authentication", async () => {
  const proxy = startSocksProxy({ auth: true });
  await listen(proxy.server);
  const target = createHttpServer((_request, response) => {
    response.end("socks-auth-body");
  });
  await listen(target);

  try {
    const requester = createPinnedRequester({
      egressProxy: parseEgressProxy(
        `socks5://alice:s3cret@127.0.0.1:${proxy.server.address().port}`,
      ),
    });
    const response = await requester({
      url: new URL(`http://target.example:${target.address().port}/page`),
      address: "127.0.0.1",
      family: 4,
    });
    assert.equal(response.status, 200);
    assert.equal(response.body.toString(), "socks-auth-body");
    assert.deepEqual(proxy.events, [
      { stage: "method", chosen: 2 },
      { stage: "auth", user: "alice", pass: "s3cret" },
      { stage: "connect", address: "127.0.0.1", port: target.address().port },
    ]);
  } finally {
    proxy.server.close();
    target.close();
  }
});

test("parses egress proxy configuration strictly", () => {
  assert.equal(parseEgressProxy(""), null);
  assert.equal(parseEgressProxy(null), null);
  assert.deepEqual(parseEgressProxy("socks5h://gw.internal"), {
    protocol: "socks5",
    host: "gw.internal",
    port: 1080,
    username: null,
    password: null,
  });
  assert.deepEqual(parseEgressProxy("http://bob:hunter2@10.9.8.7:3128"), {
    protocol: "http",
    host: "10.9.8.7",
    port: 3128,
    username: "bob",
    password: "hunter2",
  });
  for (const bad of [
    "ftp://proxy.example",
    "not a url",
    "http://user:@host",
    "socks5://host:notaport",
  ]) {
    assert.throws(() => parseEgressProxy(bad), /WEB_EGRESS_PROXY/);
  }
});

test("requires TLS when the egress proxy URL uses https", async () => {
  // A plaintext listener cannot complete the mandatory TLS handshake, so the
  // tunnel must fail instead of sending the CONNECT request in cleartext.
  const plainListener = createServer();
  await listen(plainListener);
  try {
    const requester = createPinnedRequester({
      egressProxy: parseEgressProxy(
        `https://127.0.0.1:${plainListener.address().port}`,
      ),
    });
    await assert.rejects(
      requester({
        url: new URL("http://target.example:80/page"),
        address: "93.184.216.34",
        family: 4,
      }),
    );
  } finally {
    plainListener.close();
  }
});
