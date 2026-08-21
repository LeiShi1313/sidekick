import { connect as netConnect, isIP } from "node:net";
import { connect as tlsConnect } from "node:tls";

const TUNNEL_TIMEOUT_MS = 10_000;
const MAX_PROXY_RESPONSE_BYTES = 8_192;
const SOCKS_VERSION = 0x05;
const SOCKS_AUTH_NONE = 0x00;
const SOCKS_AUTH_PASSWORD = 0x02;
const SOCKS_COMMAND_CONNECT = 0x01;
const SOCKS_ATYP_IPV4 = 0x01;
const SOCKS_ATYP_DOMAIN = 0x03;
const SOCKS_ATYP_IPV6 = 0x04;
const PROXY_PROTOCOLS = new Set(["http:", "https:", "socks5:", "socks5h:"]);

export function parseEgressProxy(raw) {
  const value = typeof raw === "string" ? raw.trim() : "";
  if (!value) return null;
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("Invalid configuration: WEB_EGRESS_PROXY");
  }
  if (!PROXY_PROTOCOLS.has(url.protocol) || !url.hostname) {
    throw new Error("Invalid configuration: WEB_EGRESS_PROXY");
  }
  let username = null;
  let password = null;
  if (url.username || url.password) {
    try {
      username = decodeURIComponent(url.username);
      password = decodeURIComponent(url.password);
    } catch {
      throw new Error("Invalid configuration: WEB_EGRESS_PROXY");
    }
    if (!username || !password) {
      throw new Error("Invalid configuration: WEB_EGRESS_PROXY");
    }
  }
  const isSocks = url.protocol.startsWith("socks");
  return {
    // socks5h is accepted but DNS stays local by design: targets are pinned to
    // pre-validated IP addresses before the tunnel is opened.
    protocol: isSocks ? "socks5" : url.protocol.replace(":", ""),
    host: url.hostname.replace(/^\[|\]$/g, ""),
    port: url.port
      ? Number(url.port)
      : isSocks
        ? 1080
        : url.protocol === "https:"
          ? 443
          : 80,
    username,
    password,
  };
}

function tunnelFailure(socket, message) {
  const error = new Error(message);
  if (socket) socket.destroy();
  return error;
}

// Reads from the socket until `parse(buffer)` returns a non-null consumed byte
// count. Resolves { chunk, consumed }; callers unshift any surplus back so the
// tunneled protocol stream starts clean.
function readProxyResponse(socket, parse) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("close", onClose);
      fn(value);
    };
    const timer = setTimeout(
      () =>
        finish(reject, tunnelFailure(socket, "Egress proxy response timed out")),
      TUNNEL_TIMEOUT_MS,
    );
    const settle = () => {
      let consumed;
      try {
        consumed = parse(Buffer.concat(chunks, size));
      } catch (error) {
        finish(reject, error);
        return;
      }
      if (consumed === null) return;
      finish(resolve, { chunk: Buffer.concat(chunks, size), consumed });
    };
    const onData = (data) => {
      chunks.push(data);
      size += data.length;
      if (size > MAX_PROXY_RESPONSE_BYTES) {
        finish(
          reject,
          tunnelFailure(socket, "Egress proxy sent an oversized response"),
        );
        return;
      }
      settle();
    };
    const onError = (error) => finish(reject, error);
    const onClose = () =>
      finish(reject, tunnelFailure(socket, "Egress proxy closed the connection"));
    socket.on("data", onData);
    socket.on("error", onError);
    socket.on("close", onClose);
    settle();
  });
}

async function establishConnectTunnel(socket, proxy, address, port) {
  const credentials =
    proxy.username !== null
      ? `Proxy-Authorization: Basic ${Buffer.from(
          `${proxy.username}:${proxy.password}`,
        ).toString("base64")}\r\n`
      : "";
  socket.write(
    `CONNECT ${address}:${port} HTTP/1.1\r\n` +
      `Host: ${address}:${port}\r\n${credentials}\r\n`,
  );
  const { chunk, consumed } = await readProxyResponse(socket, (buffer) => {
    const end = buffer.indexOf("\r\n\r\n");
    if (end === -1) return null;
    const statusLine = buffer.subarray(0, end).toString("latin1");
    const status = Number(statusLine.split(" ")[1]);
    if (!Number.isInteger(status)) {
      throw tunnelFailure(
        socket,
        "Egress proxy returned a malformed CONNECT response",
      );
    }
    if (!(status >= 200 && status < 300)) {
      throw tunnelFailure(
        socket,
        `Egress proxy refused the CONNECT request (HTTP ${status})`,
      );
    }
    return end + 4;
  });
  const leftover = chunk.subarray(consumed);
  if (leftover.length > 0) socket.unshift(leftover);
}

async function negotiateSocksPassword(socket, proxy) {
  const user = Buffer.from(proxy.username, "utf8");
  const pass = Buffer.from(proxy.password, "utf8");
  if (user.length > 255 || pass.length > 255) {
    throw tunnelFailure(socket, "Egress proxy credentials are too long");
  }
  socket.write(
    Buffer.concat([
      Buffer.from([0x01, user.length]),
      user,
      Buffer.from([pass.length]),
      pass,
    ]),
  );
  await readProxyResponse(socket, (buffer) => {
    if (buffer.length < 2) return null;
    if (buffer[0] !== 0x01 || buffer[1] !== 0x00) {
      throw tunnelFailure(socket, "Egress proxy rejected the credentials");
    }
    return 2;
  });
}

function socksAuthMethods(proxy) {
  return proxy.username !== null
    ? [SOCKS_AUTH_NONE, SOCKS_AUTH_PASSWORD]
    : [SOCKS_AUTH_NONE];
}

function ipv6Bytes(address) {
  const cleaned = address.split("%")[0];
  const halves = cleaned.split("::");
  if (halves.length > 2) {
    throw tunnelFailure(null, "Invalid IPv6 address in tunnel target");
  }
  const parseGroups = (text) => {
    if (!text) return [];
    const groups = [];
    for (const part of text.split(":")) {
      if (part.includes(".")) {
        const octets = part.split(".").map(Number);
        if (
          octets.length !== 4 ||
          octets.some((value) => !Number.isInteger(value) || value > 255)
        ) {
          throw tunnelFailure(null, "Invalid IPv6 address in tunnel target");
        }
        groups.push(((octets[0] << 8) | octets[1]).toString(16));
        groups.push(((octets[2] << 8) | octets[3]).toString(16));
        continue;
      }
      if (!/^[0-9a-fA-F]{1,4}$/.test(part)) {
        throw tunnelFailure(null, "Invalid IPv6 address in tunnel target");
      }
      groups.push(part);
    }
    return groups;
  };
  const left = parseGroups(halves[0]);
  const right = halves.length === 2 ? parseGroups(halves[1]) : null;
  let groups;
  if (right === null) {
    if (left.length !== 8) {
      throw tunnelFailure(null, "Invalid IPv6 address in tunnel target");
    }
    groups = left;
  } else {
    const missing = 8 - left.length - right.length;
    if (missing < 1) {
      throw tunnelFailure(null, "Invalid IPv6 address in tunnel target");
    }
    groups = [...left, ...Array(missing).fill("0"), ...right];
  }
  return Buffer.from(
    groups.flatMap((group) => [
      (Number.parseInt(group, 16) >> 8) & 0xff,
      Number.parseInt(group, 16) & 0xff,
    ]),
  );
}

function socksAddressChunk(address) {
  const family = isIP(address);
  if (family === 4) {
    const octets = address.split(".").map(Number);
    return Buffer.concat([Buffer.from([SOCKS_ATYP_IPV4]), Buffer.from(octets)]);
  }
  if (family === 6) {
    return Buffer.concat([Buffer.from([SOCKS_ATYP_IPV6]), ipv6Bytes(address)]);
  }
  const host = Buffer.from(address, "utf8");
  if (host.length > 255) {
    throw new Error("Egress proxy target hostname is too long");
  }
  return Buffer.concat([Buffer.from([SOCKS_ATYP_DOMAIN, host.length]), host]);
}

async function establishSocksTunnel(socket, proxy, address, port) {
  const methods = socksAuthMethods(proxy);
  socket.write(Buffer.from([SOCKS_VERSION, methods.length, ...methods]));
  const { chunk } = await readProxyResponse(socket, (buffer) => {
    if (buffer.length < 2) return null;
    if (buffer[0] !== SOCKS_VERSION) {
      throw tunnelFailure(socket, "Egress proxy does not support SOCKS5");
    }
    if (buffer[1] !== SOCKS_AUTH_NONE && buffer[1] !== SOCKS_AUTH_PASSWORD) {
      throw tunnelFailure(socket, "Egress proxy requires unsupported authentication");
    }
    return 2;
  });
  if (chunk[1] === SOCKS_AUTH_PASSWORD) {
    if (proxy.username === null) {
      throw tunnelFailure(socket, "Egress proxy requires credentials");
    }
    await negotiateSocksPassword(socket, proxy);
  }
  socket.write(
    Buffer.concat([
      Buffer.from([SOCKS_VERSION, SOCKS_COMMAND_CONNECT, 0x00]),
      socksAddressChunk(address),
      Buffer.from([(port >> 8) & 0xff, port & 0xff]),
    ]),
  );
  const { chunk: reply, consumed } = await readProxyResponse(socket, (buffer) => {
    if (buffer.length < 4) return null;
    if (buffer[0] !== SOCKS_VERSION) {
      throw tunnelFailure(socket, "Egress proxy returned a malformed SOCKS5 reply");
    }
    if (buffer[1] !== 0x00) {
      throw tunnelFailure(
        socket,
        `Egress proxy failed the SOCKS5 connect (code ${buffer[1]})`,
      );
    }
    const atyp = buffer[3];
    const boundLength =
      atyp === SOCKS_ATYP_IPV4
        ? 4
        : atyp === SOCKS_ATYP_IPV6
          ? 16
          : atyp === SOCKS_ATYP_DOMAIN
            ? buffer[4] + 1
            : null;
    if (boundLength === null) {
      throw tunnelFailure(socket, "Egress proxy returned an unknown SOCKS5 address type");
    }
    const total = 4 + boundLength + 2;
    return buffer.length < total ? null : total;
  });
  const leftover = reply.subarray(consumed);
  if (leftover.length > 0) socket.unshift(leftover);
}

export async function openEgressTunnel(proxy, { address, port }) {
  const socket = netConnect({
    host: proxy.host,
    port: proxy.port,
    autoSelectFamily: false,
  });
  socket.setNoDelay(true);
  try {
    await new Promise((resolve, reject) => {
      let settled = false;
      const finish = (fn, value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        socket.off("error", onError);
        socket.off("close", onClose);
        fn(value);
      };
      const timer = setTimeout(
        () => finish(reject, new Error("Egress proxy connection timed out")),
        TUNNEL_TIMEOUT_MS,
      );
      const onError = (error) => finish(reject, error);
      const onClose = () =>
        finish(reject, new Error("Egress proxy closed the connection"));
      socket.once("connect", () => finish(resolve, undefined));
      socket.on("error", onError);
      socket.on("close", onClose);
    });
    if (proxy.protocol === "socks5") {
      await establishSocksTunnel(socket, proxy, address, port);
    } else {
      await establishConnectTunnel(socket, proxy, address, port);
    }
    return socket;
  } catch (error) {
    socket.destroy();
    throw error;
  }
}

export function wrapTunnelSocket(socket, protocol, servername) {
  if (protocol !== "https:") return socket;
  return tlsConnect({ socket, servername, rejectUnauthorized: true });
}
