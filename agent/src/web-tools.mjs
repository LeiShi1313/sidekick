import { Readability } from "@mozilla/readability";
import { lookup as dnsLookup } from "node:dns/promises";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { BlockList, isIP } from "node:net";
import { parseHTML } from "linkedom";
import TurndownService from "turndown";

import { readUsableCodexAccessToken } from "./codex-access-token.mjs";
import { sanitizeSensitiveValue } from "./privacy-redaction.mjs";

const FORBIDDEN_HOSTS = new Set([
  "github.com",
  "www.github.com",
  "youtu.be",
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "music.youtube.com",
]);
const RECENCY_FILTERS = new Set(["day", "week", "month", "year"]);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const MAX_REDIRECTS = 5;
const MAX_RESPONSE_BYTES = 5 * 1024 * 1024;
const MAX_INLINE_CONTENT = 30_000;
const FETCH_TIMEOUT_MS = 30_000;
const BLOCKED_HOST_SUFFIXES = [".internal", ".local", ".localhost", ".home.arpa"];
const BLOCKED_IPV4 = blockList("ipv4", [
  ["0.0.0.0", 8],
  ["10.0.0.0", 8],
  ["100.64.0.0", 10],
  ["127.0.0.0", 8],
  ["169.254.0.0", 16],
  ["172.16.0.0", 12],
  ["192.0.0.0", 24],
  ["192.0.2.0", 24],
  ["192.31.196.0", 24],
  ["192.52.193.0", 24],
  ["192.88.99.0", 24],
  ["192.168.0.0", 16],
  ["192.175.48.0", 24],
  ["198.18.0.0", 15],
  ["198.51.100.0", 24],
  ["203.0.113.0", 24],
  ["224.0.0.0", 4],
  ["240.0.0.0", 4],
]);
const PUBLIC_IPV6 = blockList("ipv6", [["2000::", 3]]);
const BLOCKED_IPV6 = blockList("ipv6", [
  ["2001::", 23],
  ["2001:db8::", 32],
  ["2002::", 16],
  ["3fff::", 20],
]);
const USER_AGENT =
  "Mozilla/5.0 (compatible; Sidekick/1.0; public-web-fetch)";
const turndown = new TurndownService({
  headingStyle: "atx",
  codeBlockStyle: "fenced",
});

function blockList(family, ranges) {
  const list = new BlockList();
  for (const [network, prefix] of ranges) {
    list.addSubnet(network, prefix, family);
  }
  return list;
}

async function defaultLookup(hostname) {
  return await dnsLookup(hostname, { all: true, verbatim: true });
}

function normalizedHostname(hostname) {
  return hostname
    .toLowerCase()
    .replace(/^\[|\]$/g, "")
    .replace(/\.$/, "");
}

function isPrivateOrReservedAddress(address) {
  if (typeof address !== "string") return true;
  const normalized = normalizedHostname(address);
  const version = isIP(normalized);
  if (version === 4) return BLOCKED_IPV4.check(normalized, "ipv4");
  if (version === 6) {
    return (
      !PUBLIC_IPV6.check(normalized, "ipv6") ||
      BLOCKED_IPV6.check(normalized, "ipv6")
    );
  }
  return true;
}

function assertPublicAddress(address) {
  if (isPrivateOrReservedAddress(address)) {
    throw new Error("URL host is not allowed");
  }
}

async function resolveAllowedTarget(raw, lookup) {
  if (typeof raw !== "string" || raw.length > 2_048) {
    throw new Error("URL is not allowed");
  }
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error("URL is not allowed");
  }
  if (!new Set(["http:", "https:"]).has(url.protocol)) {
    throw new Error("URL is not allowed");
  }
  if (url.username || url.password) {
    throw new Error("URL credentials are not allowed");
  }
  const hostname = normalizedHostname(url.hostname);
  if (
    !hostname ||
    hostname === "localhost" ||
    BLOCKED_HOST_SUFFIXES.some((suffix) => hostname.endsWith(suffix)) ||
    FORBIDDEN_HOSTS.has(hostname) ||
    (isIP(hostname) === 0 && !hostname.includes("."))
  ) {
    throw new Error("URL host is not allowed");
  }

  const literalFamily = isIP(hostname);
  if (literalFamily) {
    assertPublicAddress(hostname);
    return { url, address: hostname, family: literalFamily };
  }

  let addresses;
  try {
    addresses = await lookup(hostname);
  } catch {
    throw new Error("URL host is not allowed");
  }
  if (!Array.isArray(addresses) || addresses.length === 0) {
    throw new Error("URL host is not allowed");
  }
  const vetted = addresses.map((result) => {
    const address = normalizedHostname(result?.address ?? "");
    const family = isIP(address);
    assertPublicAddress(address);
    if (family !== 4 && family !== 6) {
      throw new Error("URL host is not allowed");
    }
    return { address, family };
  });
  const selected = vetted.find(({ family }) => family === 4) ?? vetted[0];
  return { url, ...selected };
}

function headerValue(headers, name) {
  if (typeof headers?.get === "function") return headers.get(name);
  const value = headers?.[name.toLowerCase()] ?? headers?.[name];
  return Array.isArray(value) ? value[0] : value ?? null;
}

function normalizedHeaders(headers) {
  const result = new Headers();
  for (const [name, raw] of Object.entries(headers ?? {})) {
    if (raw === undefined) continue;
    result.set(name, Array.isArray(raw) ? raw.join(", ") : String(raw));
  }
  return result;
}

function responseTooLarge() {
  const error = new Error("Response too large");
  error.code = "RESPONSE_TOO_LARGE";
  return error;
}

export function createPinnedRequester({
  httpRequest: requestHttp = httpRequest,
  httpsRequest: requestHttps = httpsRequest,
} = {}) {
  return async function requestPinned(
    target,
    { signal, maxBytes = MAX_RESPONSE_BYTES } = {},
  ) {
    const { url, address, family } = target;
    const hostname = normalizedHostname(url.hostname);
    const transport = url.protocol === "https:" ? requestHttps : requestHttp;
    const lookup = (_requestedHostname, options, callback) => {
      const done = typeof options === "function" ? options : callback;
      const wantsAll = typeof options === "object" && options?.all === true;
      if (wantsAll) {
        done(null, [{ address, family }]);
      } else {
        done(null, address, family);
      }
    };

    return await new Promise((resolve, reject) => {
      let settled = false;
      const succeed = (value) => {
        if (settled) return;
        settled = true;
        resolve(value);
      };
      const fail = (error) => {
        if (settled) return;
        settled = true;
        reject(error);
      };
      let request;
      try {
        request = transport(
          {
            protocol: url.protocol,
            hostname,
            port: url.port || undefined,
            method: "GET",
            path: `${url.pathname}${url.search}`,
            headers: {
              Accept:
                "text/html,application/xhtml+xml,text/plain,application/json,application/xml;q=0.9,*/*;q=0.1",
              "Cache-Control": "no-cache",
              "User-Agent": USER_AGENT,
            },
            family,
            lookup,
            autoSelectFamily: false,
            agent: false,
            signal,
            ...(url.protocol === "https:" && isIP(hostname) === 0
              ? { servername: hostname }
              : {}),
          },
          (response) => {
            const status = Number(response.statusCode ?? 0);
            const headers = normalizedHeaders(response.headers);
            if (REDIRECT_STATUSES.has(status) || !(status >= 200 && status < 300)) {
              response.destroy();
              succeed({
                status,
                statusText: String(response.statusMessage ?? ""),
                headers,
                body: Buffer.alloc(0),
              });
              return;
            }
            const rawLength = headerValue(headers, "content-length");
            if (rawLength !== null) {
              const length = Number(rawLength);
              if (!Number.isSafeInteger(length) || length < 0 || length > maxBytes) {
                response.destroy();
                fail(responseTooLarge());
                return;
              }
            }
            const chunks = [];
            let size = 0;
            response.on("data", (chunk) => {
              const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
              size += buffer.length;
              if (size > maxBytes) {
                response.destroy();
                fail(responseTooLarge());
                return;
              }
              chunks.push(buffer);
            });
            response.once("end", () => {
              succeed({
                status,
                statusText: String(response.statusMessage ?? ""),
                headers,
                body: Buffer.concat(chunks, size),
              });
            });
            response.once("error", fail);
          },
        );
      } catch (error) {
        fail(error);
        return;
      }
      request.once("error", fail);
      request.end();
    });
  };
}

const defaultPinnedRequest = createPinnedRequester();

function normalizeQueries(params) {
  const source = Array.isArray(params.queries)
    ? params.queries
    : params.query === undefined
      ? []
      : [params.query];
  const queries = source
    .filter((query) => typeof query === "string")
    .map((query) => query.trim().slice(0, 500))
    .filter(Boolean)
    .slice(0, 4);
  if (queries.length === 0) throw new Error("At least one search query is required");
  return queries;
}

function isOpenAIAuthFailure(error) {
  return (
    error?.provider === "openai" &&
    (error?.kind === "auth" || error?.kind === "credential")
  );
}

function isOpenAIAuthFailureResult(result) {
  if (
    result?.details?.successfulQueries !== 0 ||
    result?.details?.totalResults !== 0 ||
    !Array.isArray(result?.content)
  ) {
    return false;
  }
  const text = result.content
    .filter((item) => item?.type === "text" && typeof item.text === "string")
    .map((item) => item.text)
    .join("\n");
  return /(?:^|\n)Error: openai search failed \((?:auth|credential)\):/i.test(
    text,
  );
}

function constrainSearch(definition, hasCodexAuth) {
  return {
    ...definition,
    description:
      "Search the public web through mounted Codex authentication when available, otherwise Exa. Returns raw cited results for the agent to synthesize.",
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const queries = normalizeQueries(params ?? {});
      const safe = {
        ...(Array.isArray(params?.queries)
          ? { queries }
          : { query: queries[0] }),
        workflow: "none",
        includeContent: false,
      };
      if (Number.isInteger(params?.numResults)) {
        safe.numResults = Math.min(10, Math.max(1, params.numResults));
      }
      if (RECENCY_FILTERS.has(params?.recencyFilter)) {
        safe.recencyFilter = params.recencyFilter;
      }
      if (Array.isArray(params?.domainFilter)) {
        safe.domainFilter = params.domainFilter
          .filter((value) => typeof value === "string")
          .map((value) => value.trim().slice(0, 253))
          .filter(Boolean)
          .slice(0, 10);
      }
      let codexAvailable = false;
      try {
        codexAvailable = Boolean(await hasCodexAuth());
      } catch {
        codexAvailable = false;
      }
      if (!codexAvailable) safe.provider = "exa";
      const executeSearch = (searchParams) =>
        definition.execute(
          toolCallId,
          searchParams,
          signal,
          onUpdate,
          ctx,
        );
      try {
        const result = await executeSearch(safe);
        if (codexAvailable && isOpenAIAuthFailureResult(result)) {
          return await executeSearch({ ...safe, provider: "exa" });
        }
        return result;
      } catch (error) {
        if (!codexAvailable || !isOpenAIAuthFailure(error)) throw error;
        return await executeSearch({ ...safe, provider: "exa" });
      }
    },
  };
}

function readableText(body, contentType, url) {
  const text = new TextDecoder().decode(body);
  if (text.includes("\0")) {
    return { title: "", content: "", error: "Unsupported page content" };
  }
  const mediaType = contentType.split(";", 1)[0].trim().toLowerCase();
  const isHtml =
    mediaType === "text/html" || mediaType === "application/xhtml+xml";
  const isText =
    !mediaType ||
    mediaType.startsWith("text/") ||
    mediaType === "application/json" ||
    mediaType === "application/xml" ||
    mediaType === "application/rss+xml" ||
    mediaType === "application/atom+xml";
  if (!isHtml && !isText) {
    return { title: "", content: "", error: "Unsupported page content" };
  }
  if (!isHtml) {
    const content = text.trim();
    return content
      ? { title: url.hostname, content, error: null }
      : { title: "", content: "", error: "Page contained no readable text" };
  }

  const { document } = parseHTML(text);
  const fallbackTitle = document.title?.trim() ?? "";
  const fallbackText = document.body?.textContent?.replace(/\s+/g, " ").trim() ?? "";
  const article = new Readability(document).parse();
  const content = article?.content
    ? turndown.turndown(article.content).trim()
    : fallbackText;
  if (!content) {
    return { title: "", content: "", error: "Page contained no readable text" };
  }
  return {
    title: article?.title?.trim() || fallbackTitle || url.hostname,
    content,
    error: null,
  };
}

async function fetchPage(raw, { lookup, request, signal }) {
  let current = raw;
  for (let redirects = 0; redirects <= MAX_REDIRECTS; redirects += 1) {
    const target = await resolveAllowedTarget(current, lookup);
    let response;
    try {
      response = await request(target, {
        signal,
        maxBytes: MAX_RESPONSE_BYTES,
      });
    } catch (error) {
      return {
        url: target.url.toString(),
        title: "",
        content: "",
        error:
          error?.code === "RESPONSE_TOO_LARGE"
            ? "Response too large"
            : signal?.aborted
              ? "Page fetch cancelled"
              : "Page could not be fetched",
      };
    }

    if (REDIRECT_STATUSES.has(response.status)) {
      const location = headerValue(response.headers, "location");
      if (!location || redirects === MAX_REDIRECTS) {
        throw new Error("URL redirect is not allowed");
      }
      try {
        current = new URL(location, target.url).toString();
      } catch {
        throw new Error("URL redirect is not allowed");
      }
      continue;
    }
    if (!(response.status >= 200 && response.status < 300)) {
      return {
        url: target.url.toString(),
        title: "",
        content: "",
        error: `HTTP ${response.status}`,
      };
    }
    return {
      url: target.url.toString(),
      ...readableText(
        response.body,
        headerValue(response.headers, "content-type") ?? "",
        target.url,
      ),
    };
  }
  throw new Error("URL redirect is not allowed");
}

function formatFetchResults(results) {
  const successful = results.filter(({ error }) => !error);
  const sections = results.map((result) => {
    if (result.error) {
      return `## ${result.url}\n\nError: ${result.error}`;
    }
    return (
      `## ${result.title || result.url}\n\n` +
      `Source: ${result.url}\n\n${result.content}`
    );
  });
  const fullOutput = sections.join("\n\n---\n\n");
  const truncated = fullOutput.length > MAX_INLINE_CONTENT;
  const output = truncated
    ? `${fullOutput.slice(0, MAX_INLINE_CONTENT)}\n\n[Content truncated]`
    : fullOutput;
  return {
    content: [{ type: "text", text: output }],
    details: {
      urls: results.map(({ url }) => url),
      urlCount: results.length,
      successful: successful.length,
      totalChars: successful.reduce(
        (total, result) => total + result.content.length,
        0,
      ),
      truncated,
      ...(results.length === 1 && successful.length === 1
        ? { title: successful[0].title }
        : {}),
    },
  };
}

function constrainFetch(definition, options) {
  return {
    ...definition,
    description:
      "Fetch and extract readable content from up to three public HTTP or HTTPS pages.",
    async execute(_toolCallId, params, signal, onUpdate) {
      const supplied = Array.isArray(params?.urls)
        ? params.urls
        : params?.url === undefined
          ? []
          : [params.url];
      if (supplied.length === 0 || supplied.length > 3) {
        throw new Error("URL count is not allowed");
      }
      onUpdate?.(
        sanitizeSensitiveValue(
          {
            content: [
              {
                type: "text",
                text: `Fetching ${supplied.length} public page(s)...`,
              },
            ],
            details: { phase: "fetch" },
          },
          { externalText: true },
        ),
      );
      const timeoutSignal = AbortSignal.timeout(FETCH_TIMEOUT_MS);
      const requestSignal = signal
        ? AbortSignal.any([signal, timeoutSignal])
        : timeoutSignal;
      const results = await Promise.all(
        supplied.map((url) =>
          fetchPage(url, { ...options, signal: requestSignal }),
        ),
      );
      return sanitizeSensitiveValue(formatFetchResults(results), {
        externalText: true,
      });
    },
  };
}

export function constrainWebTools(
  definitions,
  {
    lookup = defaultLookup,
    request = defaultPinnedRequest,
    hasCodexAuth = async () =>
      Boolean(await readUsableCodexAccessToken()),
  } = {},
) {
  const byName = new Map(definitions.map((definition) => [definition.name, definition]));
  const search = byName.get("web_search");
  const fetchContent = byName.get("fetch_content");
  if (!search || !fetchContent) {
    throw new Error("pi-web-access did not register the required tools");
  }
  return [
    constrainSearch(search, hasCodexAuth),
    constrainFetch(fetchContent, { lookup, request }),
  ];
}
