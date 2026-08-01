const state = {
  mode: "agent",
  banks: [],
  messages: [],
  sessionId: null,
  parentEntryId: null,
  runId: null,
  prepared: null,
  memory: null,
  tools: [],
  tab: "memory",
  view: "playground",
  sessions: [],
  sessionTotal: 0,
  sessionCursor: null,
  sessionQuery: "",
  selectedSessionId: null,
  sessionDetail: null,
  audits: [],
  auditTotal: 0,
  selectedAuditId: null,
  auditDetail: null,
  historyLoading: false,
  channelSnapshot: null,
  channels: [],
  channelLoading: false,
  channelError: null,
  channelConnection: "idle",
  channelEventSource: null,
  channelQuery: "",
  channelPlatform: "",
  channelStatus: "all",
};

const elements = {
  serviceStatus: document.querySelector("#service-status"),
  newChat: document.querySelector("#new-chat"),
  bank: document.querySelector("#memory-bank"),
  memoryQuery: document.querySelector("#memory-query"),
  previewMemory: document.querySelector("#preview-memory"),
  context: document.querySelector("#pasted-context"),
  systemPrompt: document.querySelector("#system-prompt"),
  transcript: document.querySelector("#transcript"),
  emptyChat: document.querySelector("#empty-chat"),
  composer: document.querySelector("#composer"),
  prompt: document.querySelector("#prompt"),
  send: document.querySelector("#send"),
  stop: document.querySelector("#stop-run"),
  runStatus: document.querySelector("#run-status"),
  playgroundView: document.querySelector("#playground-view"),
  historyView: document.querySelector("#history-view"),
  channelsView: document.querySelector("#channels-view"),
  refreshHistory: document.querySelector("#refresh-history"),
  refreshChannels: document.querySelector("#refresh-channels"),
  resumeBanner: document.querySelector("#resume-banner"),
  resumeText: document.querySelector("#resume-text"),
  clearResume: document.querySelector("#clear-resume"),
  sessionSearch: document.querySelector("#session-search"),
  sessionQuery: document.querySelector("#session-query"),
  sessionCount: document.querySelector("#session-count"),
  sessionList: document.querySelector("#session-list"),
  loadMoreSessions: document.querySelector("#load-more-sessions"),
  sessionTitle: document.querySelector("#session-title"),
  sessionMeta: document.querySelector("#session-meta"),
  sessionTree: document.querySelector("#session-tree"),
  continueLeaf: document.querySelector("#continue-leaf"),
  auditCount: document.querySelector("#audit-count"),
  auditSelect: document.querySelector("#audit-select"),
  auditSummary: document.querySelector("#audit-summary"),
  auditEvents: document.querySelector("#audit-events"),
  channelSummary: document.querySelector("#channel-summary"),
  channelLiveStatus: document.querySelector("#channel-live-status"),
  channelFilters: document.querySelector("#channel-filters"),
  channelQuery: document.querySelector("#channel-query"),
  channelPlatform: document.querySelector("#channel-platform"),
  channelStatusFilter: document.querySelector("#channel-status-filter"),
  channelSourceStatus: document.querySelector("#channel-source-status"),
  channelNotice: document.querySelector("#channel-notice"),
  channelTableWrap: document.querySelector("#channel-table-wrap"),
  channelRows: document.querySelector("#channel-rows"),
};

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined && text !== null) item.textContent = String(text);
  if (className) item.className = className;
  return item;
}

function button(text, className, onClick) {
  const item = node("button", text, className);
  item.type = "button";
  item.addEventListener("click", onClick);
  return item;
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { accept: "application/json", ...options.headers },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || `Request failed: ${response.status}`);
  return payload;
}

async function initialize() {
  try {
    const [config, banks] = await Promise.all([
      requestJson("/api/config"),
      requestJson("/api/banks"),
    ]);
    state.banks = banks.items || [];
    elements.systemPrompt.value = config.defaultSystemPrompt || "";
    renderBanks();
    elements.serviceStatus.textContent = "Connected";
  } catch {
    elements.serviceStatus.textContent = "Services unavailable";
  }
  renderMode();
  renderInspector();
  renderView();
  renderResume();
}

function renderBanks() {
  const current = elements.bank.value;
  elements.bank.replaceChildren(node("option", "Memory off"));
  elements.bank.firstChild.value = "";
  for (const bank of state.banks) {
    const option = node("option", bank.name || bank.bank_id);
    option.value = bank.bank_id;
    elements.bank.append(option);
  }
  if ([...elements.bank.options].some((option) => option.value === current)) {
    elements.bank.value = current;
  }
}

function renderMode() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  });
}

function addMessage(role, text = "") {
  const message = { role, text };
  state.messages.push(message);
  const row = node("article", null, `message ${role}`);
  row.append(node("div", role === "user" ? "You" : state.mode === "agent" ? "Pi Agent" : "LLM", "message-author"));
  const body = node("pre", text, "message-body");
  row.append(body);
  elements.transcript.append(row);
  elements.emptyChat.hidden = true;
  elements.transcript.scrollTop = elements.transcript.scrollHeight;
  return { message, body };
}

function recentConversation() {
  return state.messages
    .slice(-8)
    .map((message) => `${message.role}: ${message.text}`)
    .join("\n")
    .slice(0, 8000);
}

function setRunning(running, text) {
  elements.send.disabled = running;
  elements.newChat.disabled = running;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.disabled = running;
  });
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.disabled = running;
  });
  elements.stop.hidden = !running;
  elements.runStatus.textContent = text;
}

async function runPrompt(prompt) {
  const recallContext = recentConversation();
  addMessage("user", prompt);
  const assistant = addMessage("assistant");
  state.tools = [];
  state.prepared = null;
  state.memory = null;
  renderInspector();
  setRunning(true, "Preparing");
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/x-ndjson" },
      body: JSON.stringify({
        mode: state.mode,
        prompt,
        bankId: elements.bank.value || null,
        memoryQuery: elements.memoryQuery.value.trim() || null,
        recallContext,
        context: elements.context.value,
        systemPrompt: elements.systemPrompt.value,
        sessionId: state.sessionId,
        parentEntryId: state.parentEntryId,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error?.message || `Run failed: ${response.status}`);
    }
    await readEvents(response, (event) => handleEvent(event, assistant));
  } catch (error) {
    assistant.message.text = `Run failed: ${error.message}`;
    assistant.body.textContent = assistant.message.text;
    setRunning(false, "Failed");
  } finally {
    state.runId = null;
    setRunning(false, elements.runStatus.textContent);
  }
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function handleEvent(event, assistant) {
  if (event.type === "run_prepared") {
    state.runId = event.runId;
    state.prepared = event;
    state.memory = event.memory;
    setRunning(true, "Running");
    renderInspector();
    return;
  }
  if (event.type === "memory_snapshot") {
    state.memory = {
      bankId: event.primaryBankId,
      query: event.queries.join("\n\n"),
      queries: event.queries,
      memories: event.memories,
      managedBy: "agent",
      status: "complete",
    };
    renderInspector();
    return;
  }
  if (event.type === "run_started") return;
  if (event.type === "tool_snapshot") {
    state.tools.push(event);
    setRunning(true, event.summary || "Using tool");
    renderInspector();
    return;
  }
  if (event.type === "text_delta") {
    assistant.message.text = event.reset ? event.delta : assistant.message.text + event.delta;
    assistant.body.textContent = assistant.message.text;
    elements.transcript.scrollTop = elements.transcript.scrollHeight;
    return;
  }
  if (event.type === "run_completed") {
    assistant.message.text = event.answer;
    assistant.body.textContent = event.answer;
    state.sessionId = event.sessionId;
    state.parentEntryId = event.entryId;
    renderResume();
    setRunning(false, "Complete");
    return;
  }
  if (event.type === "run_failed") {
    assistant.message.text = event.message || "Agent run failed";
    assistant.body.textContent = assistant.message.text;
    setRunning(false, event.code === "CANCELLED" ? "Cancelled" : "Failed");
  }
}

async function previewMemory() {
  const bankId = elements.bank.value;
  const query = elements.memoryQuery.value.trim() || elements.prompt.value.trim();
  if (!bankId || !query) {
    elements.runStatus.textContent = "Select a bank and enter a query";
    return;
  }
  elements.runStatus.textContent = "Recalling";
  try {
    state.memory = await requestJson("/api/recall", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bankId, query }),
    });
    state.tab = "memory";
    elements.runStatus.textContent = "Recall complete";
    renderInspector();
  } catch (error) {
    elements.runStatus.textContent = error.message;
  }
}

function renderInspector() {
  document.querySelectorAll("[role=tab]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.tab === state.tab));
  });
  for (const tab of ["memory", "request", "tools"]) {
    document.querySelector(`#inspector-${tab}`).hidden = tab !== state.tab;
  }
  renderMemory();
  renderRequest();
  renderTools();
}

function renderMemory() {
  const panel = document.querySelector("#inspector-memory");
  if (!state.memory) {
    panel.replaceChildren(node("p", "No memory recall for this run.", "empty-inspector"));
    return;
  }
  const blocks = [
    field("Bank", state.memory.bankId),
    field(
      Array.isArray(state.memory.queries) && state.memory.queries.length > 1
        ? "Recall queries"
        : "Recall query",
      state.memory.query,
    ),
  ];
  const memories = Array.isArray(state.memory.memories) ? state.memory.memories : [];
  if (state.memory.status === "pending") {
    blocks.push(node("p", "Agent is fetching memory...", "empty-inspector"));
  } else if (memories.length === 0) {
    blocks.push(node("p", "No matching memories.", "empty-inspector"));
  }
  for (const memory of memories) {
    const item = node("section", null, "memory-item");
    item.append(node("div", memory.type || "memory", "memory-type"));
    item.append(node("p", memory.text, "memory-text"));
    const metadata = [
      ...(memory.entities || []),
      memory.occurredStart,
      memory.documentId,
    ].filter(Boolean).join(" · ");
    if (metadata) item.append(node("div", metadata, "memory-meta"));
    item.append(node("code", memory.id, "memory-id"));
    blocks.push(item);
  }
  panel.replaceChildren(...blocks);
}

function renderRequest() {
  const panel = document.querySelector("#inspector-request");
  if (!state.prepared) {
    panel.replaceChildren(node("p", "No prepared request yet.", "empty-inspector"));
    return;
  }
  panel.replaceChildren(
    field("Mode", `${state.prepared.mode} / ${state.prepared.toolPolicy}`),
    field("Run ID", state.prepared.runId),
    field("System prompt", state.prepared.request.systemPrompt),
    field("Current request", state.prepared.request.prompt),
    field("Injected context", JSON.stringify(state.prepared.request.context, null, 2)),
  );
}

function renderTools() {
  const panel = document.querySelector("#inspector-tools");
  if (state.tools.length === 0) {
    panel.replaceChildren(node("p", "No tool activity.", "empty-inspector"));
    return;
  }
  panel.replaceChildren(...state.tools.map((tool) => {
    const row = node("div", null, "tool-row");
    row.append(node("span", tool.tool || "tool", "tool-name"));
    row.append(node("span", tool.summary || tool.phase || "", "tool-summary"));
    return row;
  }));
}

function field(label, value) {
  const item = node("section", null, "inspect-field");
  item.append(node("h3", label));
  item.append(node("pre", value || "None"));
  return item;
}

function newChat() {
  state.messages = [];
  state.sessionId = null;
  state.parentEntryId = null;
  state.runId = null;
  state.prepared = null;
  state.memory = null;
  state.tools = [];
  elements.transcript.querySelectorAll(".message").forEach((message) => message.remove());
  elements.emptyChat.hidden = false;
  elements.prompt.value = "";
  setRunning(false, "Ready");
  renderInspector();
  renderResume();
  elements.prompt.focus();
}

function renderResume() {
  const continuing = Boolean(state.sessionId && state.parentEntryId);
  elements.resumeBanner.hidden = !continuing;
  elements.resumeText.textContent = continuing
    ? `Continuing ${state.sessionId} from ${state.parentEntryId}`
    : "";
}

function renderView() {
  const history = state.view === "history";
  const channels = state.view === "channels";
  elements.playgroundView.hidden = history || channels;
  elements.historyView.hidden = !history;
  elements.channelsView.hidden = !channels;
  elements.refreshHistory.hidden = !history;
  elements.refreshChannels.hidden = !channels;
  elements.newChat.hidden = history || channels;
  document.querySelectorAll("[data-view]").forEach((item) => {
    item.setAttribute("aria-pressed", String(item.dataset.view === state.view));
  });
}

function showView(view) {
  if (!new Set(["playground", "history", "channels"]).has(view)) return;
  state.view = view;
  renderView();
  if (view === "history" && state.sessions.length === 0 && !state.historyLoading) {
    void loadSessions();
  }
  if (view === "channels") {
    if (!state.channelSnapshot && !state.channelLoading) void loadChannels();
    connectChannelEvents();
  }
}

function formatDate(value) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

function short(value, maximum = 180) {
  const text = String(value ?? "").replace(/<\/?current_request>/g, "").trim();
  return text.length <= maximum ? text : `${text.slice(0, maximum - 3)}...`;
}

function currentRequest(value) {
  const text = String(value ?? "");
  const match = text.match(/<current_request>\s*([\s\S]*?)\s*<\/current_request>/);
  return (match?.[1] || text).trim();
}

function pretty(value) {
  return JSON.stringify(value, null, 2) ?? "null";
}

async function loadChannels() {
  state.channelLoading = true;
  state.channelError = null;
  renderChannels();
  try {
    applyChannelSnapshot(await loadCompleteChannelSnapshot());
  } catch (error) {
    state.channelError = error.message;
  } finally {
    state.channelLoading = false;
    renderChannels();
  }
}

async function loadCompleteChannelSnapshot() {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const first = await requestJson("/api/channels?limit=500");
    const items = [...first.items];
    let cursor = first.nextCursor;
    let consistent = true;
    const seenCursors = new Set();
    while (cursor) {
      if (seenCursors.has(cursor)) throw new Error("Channel pagination did not advance");
      seenCursors.add(cursor);
      const page = await requestJson(`/api/channels?limit=500&cursor=${encodeURIComponent(cursor)}`);
      if (page.streamId !== first.streamId || page.generation !== first.generation) {
        consistent = false;
        break;
      }
      items.push(...page.items);
      cursor = page.nextCursor;
    }
    if (consistent) {
      if (items.length !== first.total) throw new Error("Channel pagination was incomplete");
      return { ...first, items, nextCursor: null };
    }
  }
  throw new Error("Channel inventory changed while loading; try again");
}

function connectChannelEvents() {
  if (state.channelEventSource) return;
  if (!("EventSource" in window)) {
    state.channelConnection = "snapshot";
    renderChannels();
    return;
  }
  state.channelConnection = "connecting";
  renderChannels();
  const source = new EventSource("/api/channel-events");
  state.channelEventSource = source;
  source.addEventListener("open", () => {
    if (state.channelEventSource !== source) return;
    state.channelConnection = "live";
    renderChannels();
  });
  source.addEventListener("snapshot", (event) => {
    if (state.channelEventSource !== source) return;
    try {
      applyChannelSnapshot(JSON.parse(event.data));
      state.channelError = null;
    } catch {
      state.channelError = "Live channel snapshot was malformed";
      renderChannels();
    }
  });
  source.addEventListener("error", () => {
    if (state.channelEventSource !== source) return;
    state.channelConnection = "reconnecting";
    renderChannels();
  });
}

function reconnectChannelEvents() {
  state.channelEventSource?.close();
  state.channelEventSource = null;
  connectChannelEvents();
}

function applyChannelSnapshot(snapshot) {
  if (
    !snapshot
    || typeof snapshot.streamId !== "string"
    || !snapshot.streamId
    || !Number.isInteger(snapshot.generation)
    || !Array.isArray(snapshot.items)
    || !Array.isArray(snapshot.sources)
  ) {
    throw new Error("Malformed channel snapshot");
  }
  const current = state.channelSnapshot;
  const sameStream = current && snapshot.streamId === current.streamId;
  if (sameStream && snapshot.generation < current.generation) return;
  if (
    sameStream
    && snapshot.generation === current.generation
    && current.items.length > snapshot.items.length
  ) return;
  state.channelSnapshot = snapshot;
  state.channels = snapshot.items;
  renderChannelPlatforms();
  renderChannels();
}

function renderChannelPlatforms() {
  const current = state.channelPlatform;
  const all = node("option", "All platforms");
  all.value = "";
  const platforms = Array.isArray(state.channelSnapshot?.platforms)
    ? state.channelSnapshot.platforms
    : [...new Set(state.channels.map((item) => item.platform))].sort();
  const options = platforms.map((platform) => {
    const option = node("option", platform);
    option.value = platform;
    return option;
  });
  elements.channelPlatform.replaceChildren(all, ...options);
  state.channelPlatform = platforms.includes(current) ? current : "";
  elements.channelPlatform.value = state.channelPlatform;
}

function filteredChannels() {
  const query = state.channelQuery.toLocaleLowerCase();
  return state.channels.filter((item) => {
    if (state.channelPlatform && item.platform !== state.channelPlatform) return false;
    if (state.channelStatus === "attention") {
      if (!new Set(["error", "disconnected", "stale"]).has(item.status)) return false;
    } else if (state.channelStatus === "active") {
      if (!Array.isArray(item.activeRuns) || item.activeRuns.length === 0) return false;
    } else if (state.channelStatus !== "all" && item.status !== state.channelStatus) {
      return false;
    }
    if (!query) return true;
    return [
      item.displayName,
      item.scopeId,
      item.accountId,
      item.adapterInstanceId,
      item.platform,
    ].some((value) => String(value || "").toLocaleLowerCase().includes(query));
  });
}

function statusBadge(text, status = text) {
  const safeStatus = String(status || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "");
  return node("span", text, `channel-badge ${safeStatus || "unknown"}`);
}

function channelCell(className, ...children) {
  const cell = node("td", null, className);
  cell.append(...children.filter(Boolean));
  return cell;
}

function channelPrimary(text) {
  return node("div", text, "channel-primary");
}

function channelMeta(text) {
  return node("div", text, "channel-meta");
}

function renderChannelSources() {
  const sources = state.channelSnapshot?.sources || [];
  if (sources.length === 0) {
    elements.channelSourceStatus.replaceChildren();
    return;
  }
  elements.channelSourceStatus.replaceChildren(...sources.map((source) => {
    const item = node("div", null, `channel-source ${source.status}`);
    const heading = node("div", null, "channel-source-heading");
    heading.append(node("strong", source.id), statusBadge(source.status, source.status));
    item.append(heading);
    if (source.adapter) {
      item.append(channelMeta([
        source.adapter.platform,
        source.adapter.accountId || "account pending",
        source.adapter.connected ? "connected" : "disconnected",
      ].filter(Boolean).join(" · ")));
    }
    if (source.error) item.append(node("span", source.error, "channel-source-error"));
    else if (source.lastSuccessAt) {
      item.append(node("span", `Updated ${formatDate(source.lastSuccessAt)}`, "channel-source-time"));
    }
    return item;
  }));
}

function channelErrorMessages(item) {
  return [...new Set((item.errors || []).map((error) => [
    error.component,
    error.code,
    error.message,
  ].filter(Boolean).join(" · ")).filter(Boolean))];
}

function renderChannelRow(item) {
  const row = node("tr");
  const chat = channelCell(
    "channel-chat",
    channelPrimary(item.displayName || item.scopeId),
    channelMeta(`${item.chatKind || "unknown"} · ${item.scopeId}`),
    statusBadge(item.status, item.status),
  );
  const connection = item.adapter?.connected ? "connected" : "disconnected";
  const instance = channelCell(
    "channel-instance",
    channelPrimary(item.platform),
    channelMeta(item.adapterInstanceId),
    channelMeta(item.accountId || item.adapter?.accountId || "Account unavailable"),
    statusBadge(connection, connection),
  );
  const access = channelCell(
    "channel-access",
    statusBadge(item.accessMode || "unknown", item.accessMode),
  );
  const model = channelCell(
    "channel-model",
    channelPrimary(item.model || "Unavailable"),
    channelMeta(item.modelSource === "override" ? "Chat override" : item.modelSource === "default" ? "Pi default" : "Catalog unavailable"),
  );

  const memory = item.memory || {};
  const memoryCell = channelCell(
    "channel-memory",
    statusBadge(memory.effectiveMode || "unknown", memory.effectiveMode),
    channelMeta(`Continuous ${memory.continuousEnabled ? "on" : "off"} · Dream ${memory.dreamEnabled ? "on" : "off"}`),
    channelMeta(`${memory.pendingDocumentCount || 0} pending documents`),
  );
  const bank = item.bank || {};
  const bankCell = channelCell(
    "channel-bank",
    statusBadge(bank.status || "UNAVAILABLE", bank.status),
    channelMeta(bank.bankId || item.scopeId),
    channelPrimary(bank.status === "PRESENT"
      ? `${bank.factCount == null ? "Unknown" : bank.factCount} facts`
      : bank.status === "MISSING" ? "No bank yet" : "Bank service unavailable"),
  );
  const ingestion = channelCell(
    "channel-ingestion",
    channelPrimary(memory.lastIngestedAt ? formatDate(memory.lastIngestedAt) : "Never ingested"),
    bank.lastDocumentAt ? channelMeta(`Bank document ${formatDate(bank.lastDocumentAt)}`) : null,
  );

  const runCell = channelCell("channel-runs");
  const runs = item.activeRuns || [];
  if (runs.length === 0) {
    runCell.append(channelMeta("None"));
  } else {
    runCell.append(channelPrimary(`${runs.length} active`));
    for (const run of runs.slice(0, 3)) {
      const label = `${run.phase || "active"} · ${short(run.runId, 18)}`;
      if (run.sessionId) {
        const link = button(label, "channel-run-link", () => void openRunTrace(run));
        link.title = `Open session ${run.sessionId}, run ${run.runId}`;
        runCell.append(link);
      } else {
        runCell.append(node("code", label, "channel-run-id"));
      }
    }
    if (runs.length > 3) runCell.append(channelMeta(`+${runs.length - 3} more`));
  }

  const errors = channelErrorMessages(item);
  const errorCell = channelCell("channel-errors");
  if (errors.length === 0) {
    errorCell.append(channelMeta("None"));
  } else {
    errorCell.append(statusBadge(`${errors.length} issue${errors.length === 1 ? "" : "s"}`, "error"));
    for (const message of errors.slice(0, 2)) errorCell.append(node("p", message, "channel-error"));
    if (errors.length > 2) errorCell.append(channelMeta(`+${errors.length - 2} more`));
  }
  const updated = channelCell(
    "channel-updated",
    channelPrimary(formatDate(item.updatedAt)),
    item.lastObservedAt ? channelMeta(`Chat seen ${formatDate(item.lastObservedAt)}`) : channelMeta("No observed message"),
  );
  row.append(chat, instance, access, model, memoryCell, bankCell, ingestion, runCell, errorCell, updated);
  return row;
}

function renderChannels() {
  const visible = filteredChannels();
  const snapshot = state.channelSnapshot;
  const connectionLabels = {
    idle: "Not connected",
    connecting: "Connecting live updates",
    live: "Live",
    reconnecting: "Reconnecting",
    snapshot: "Snapshot only",
  };
  elements.channelLiveStatus.textContent = connectionLabels[state.channelConnection] || "Not connected";
  elements.channelLiveStatus.dataset.status = state.channelConnection;
  const knownTotal = Number.isInteger(snapshot?.total) ? snapshot.total : state.channels.length;
  elements.channelSummary.textContent = snapshot
    ? `${visible.length} shown of ${knownTotal} chats · snapshot ${snapshot.generation} · ${formatDate(snapshot.generatedAt)}`
    : "Waiting for a channel snapshot";
  renderChannelSources();

  const notices = [];
  if (state.channelLoading && !snapshot) notices.push("Loading channels...");
  if (state.channelError && !snapshot) {
    notices.push(`Channels unavailable: ${state.channelError}`);
  } else if (snapshot) {
    if (state.channels.length === 0) notices.push("No channels have been observed yet.");
    else if (visible.length === 0) notices.push("No channels match the current filters.");
    if (snapshot.stale) notices.push("Showing last-known data because one or more sources are stale.");
    else if (snapshot.degraded) notices.push("Live data is partial. Review the source errors above.");
    if (state.channelConnection === "reconnecting") {
      notices.push("Live updates disconnected. Reconnecting automatically...");
    }
    if (state.channelError) notices.push(state.channelError);
  }
  const notice = notices.join(" ");

  elements.channelNotice.textContent = notice;
  elements.channelNotice.hidden = !notice;
  elements.channelTableWrap.hidden = visible.length === 0;
  elements.channelRows.replaceChildren(...visible.map(renderChannelRow));
}

async function openRunTrace(run) {
  if (!run.sessionId) return;
  showView("history");
  await selectSession(run.sessionId, run.runId);
}

async function loadSessions({ append = false } = {}) {
  const loadToken = (state.sessionLoadToken || 0) + 1;
  state.sessionLoadToken = loadToken;
  state.historyLoading = true;
  state.historyError = null;
  renderSessionList();
  const queryAtStart = state.sessionQuery;
  const params = new URLSearchParams({ limit: "50" });
  if (queryAtStart) params.set("q", queryAtStart);
  if (append && state.sessionCursor) params.set("cursor", state.sessionCursor);
  try {
    const page = await requestJson(`/api/sessions?${params}`);
    if (loadToken !== state.sessionLoadToken || queryAtStart !== state.sessionQuery) return;
    state.sessions = append ? [...state.sessions, ...page.items] : page.items;
    state.sessionTotal = page.total;
    state.sessionCursor = page.nextCursor;
    renderSessionList();
    if (!append) {
      const retained = state.sessions.some((item) => item.id === state.selectedSessionId);
      const nextId = retained ? state.selectedSessionId : state.sessions[0]?.id;
      if (nextId) void selectSession(nextId);
      else clearSelectedSession();
    }
  } catch (error) {
    if (loadToken !== state.sessionLoadToken) return;
    state.historyError = error.message;
    renderSessionList();
  } finally {
    if (loadToken === state.sessionLoadToken) {
      state.historyLoading = false;
      renderSessionList();
    }
  }
}

function renderSessionList() {
  elements.sessionCount.textContent = state.historyLoading
    ? "Loading"
    : `${state.sessionTotal} total`;
  elements.loadMoreSessions.hidden = !state.sessionCursor;
  elements.loadMoreSessions.disabled = state.historyLoading;
  if (state.historyError) {
    elements.sessionList.replaceChildren(node("p", state.historyError, "history-empty"));
    return;
  }
  if (state.historyLoading && state.sessions.length === 0) {
    elements.sessionList.replaceChildren(node("p", "Loading sessions...", "history-empty"));
    return;
  }
  if (state.sessions.length === 0) {
    elements.sessionList.replaceChildren(node("p", "No sessions found.", "history-empty"));
    return;
  }
  const rows = state.sessions.map((session) => {
    const row = button(null, "session-row", () => void selectSession(session.id));
    row.setAttribute("role", "listitem");
    row.setAttribute("aria-current", String(session.id === state.selectedSessionId));
    row.append(node("span", session.name || short(currentRequest(session.firstMessage), 70) || session.id, "session-row-title"));
    row.append(node("span", short(currentRequest(session.firstMessage), 150) || "Empty session", "session-row-preview"));
    row.append(node("span", `${session.messageCount} messages · ${formatDate(session.modifiedAt)}`, "session-row-meta"));
    return row;
  });
  elements.sessionList.replaceChildren(...rows);
}

function clearSelectedSession() {
  state.selectedSessionId = null;
  state.sessionDetail = null;
  state.sessionDetailError = null;
  state.audits = [];
  state.auditTotal = 0;
  state.selectedAuditId = null;
  state.auditDetail = null;
  state.auditError = null;
  renderSessionList();
  renderSessionDetail();
  renderAudits();
}

async function selectSession(sessionId, preferredRunId = null) {
  state.selectedSessionId = sessionId;
  state.sessionDetail = null;
  state.sessionDetailError = null;
  state.audits = [];
  state.auditTotal = 0;
  state.selectedAuditId = null;
  state.auditDetail = null;
  state.auditError = null;
  renderSessionList();
  renderSessionDetail();
  renderAudits();
  const [detailResult, auditsResult] = await Promise.allSettled([
    requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`),
    requestJson(`/api/audits?limit=100&sessionId=${encodeURIComponent(sessionId)}`),
  ]);
  if (state.selectedSessionId !== sessionId) return;
  if (detailResult.status === "fulfilled") state.sessionDetail = detailResult.value;
  else state.sessionDetailError = detailResult.reason.message;
  if (auditsResult.status === "fulfilled") {
    state.audits = auditsResult.value.items;
    state.auditTotal = auditsResult.value.total;
  } else {
    state.auditError = auditsResult.reason.message;
  }
  renderSessionDetail();
  renderAudits();
  if (state.audits.length > 0) {
    const selected = state.audits.some((audit) => audit.runId === preferredRunId)
      ? preferredRunId
      : state.audits[0].runId;
    void selectAudit(selected);
  }
}

function entryDepth(entry, byId, cache, trail = new Set()) {
  if (cache.has(entry.id)) return cache.get(entry.id);
  if (!entry.parentId || trail.has(entry.id)) return 0;
  const parent = byId.get(entry.parentId);
  if (!parent) return 0;
  const nextTrail = new Set(trail);
  nextTrail.add(entry.id);
  const depth = Math.min(32, entryDepth(parent, byId, cache, nextTrail) + 1);
  cache.set(entry.id, depth);
  return depth;
}

function contentPart(part) {
  if (!part || typeof part !== "object") {
    return node("pre", String(part ?? ""), "entry-text");
  }
  if (part.type === "text") {
    const promptParts = structuredPrompt(part.text || "");
    if (promptParts.length === 0) return node("pre", part.text || "", "entry-text");
    const block = node("div");
    block.append(...promptParts);
    return block;
  }
  if (part.type === "thinking") {
    const details = node("details", null, "entry-part thinking");
    details.append(node("summary", "Thinking"), node("pre", part.thinking || "", "entry-text"));
    return details;
  }
  if (part.type === "toolCall") {
    const details = node("details", null, "entry-part tool-call");
    details.open = true;
    details.append(
      node("summary", `Tool call · ${part.name || "unknown"}`),
      node("pre", pretty(part.arguments ?? {}), "entry-json"),
    );
    return details;
  }
  if (part.type === "image") {
    return node("p", `Image · ${part.mimeType || "unknown type"} · ${part.sizeBytes || "unknown"} bytes`, "entry-part");
  }
  const details = node("details", null, "entry-part");
  details.append(node("summary", part.type || "Content"), node("pre", pretty(part), "entry-json"));
  return details;
}

function structuredPrompt(text) {
  const pattern = /<(untrusted_memory_context|untrusted_reference_context|current_request)>\s*([\s\S]*?)\s*<\/\1>/g;
  const parts = [];
  for (const match of text.matchAll(pattern)) {
    if (match[1] === "current_request") {
      const block = node("section", null, "entry-current-request");
      block.append(node("div", "Current request", "entry-section-label"));
      block.append(node("pre", match[2].trim(), "entry-text"));
      parts.push(block);
    } else {
      const details = node("details", null, "entry-part context");
      const label = match[1] === "untrusted_memory_context" ? "Memory context" : "Reference context";
      details.append(node("summary", label), node("pre", match[2].trim(), "entry-text"));
      parts.push(details);
    }
  }
  return parts;
}

function renderSessionEntry(entry, depth, isLeaf) {
  const message = entry.message && typeof entry.message === "object" ? entry.message : null;
  const role = String(message?.role || entry.type || "entry");
  const article = node("article", null, `session-entry${isLeaf ? " is-leaf" : ""}`);
  article.style.setProperty("--depth", String(depth));
  const header = node("div", null, "entry-header");
  const roleClass = role.toLowerCase().replace(/[^a-z0-9]/g, "");
  header.append(node("span", role, `entry-role ${roleClass}`));
  header.append(node("code", entry.id, "entry-id"));
  header.append(node("time", formatDate(entry.timestamp || message?.timestamp), "entry-time"));
  header.append(button("Continue", "entry-continue", () => openContinuation(entry.id)));
  article.append(header);
  const content = node("div", null, "entry-content");
  if (typeof message?.content === "string") {
    const promptParts = structuredPrompt(message.content);
    if (promptParts.length > 0) content.append(...promptParts);
    else content.append(node("pre", message.content, "entry-text"));
  } else if (Array.isArray(message?.content)) {
    for (const part of message.content) content.append(contentPart(part));
  } else if (message) {
    content.append(node("pre", pretty(message.content ?? message), "entry-text"));
  } else {
    content.append(node("pre", short(entry.summary || entry.name || entry.type, 1_000), "entry-text"));
  }
  if (message?.usage) content.append(node("div", `Usage · ${pretty(message.usage)}`, "entry-usage"));
  const raw = node("details", null, "raw-details");
  raw.append(node("summary", "Stored entry JSON"), node("pre", pretty(entry), "entry-json"));
  content.append(raw);
  article.append(content);
  return article;
}

function renderSessionDetail() {
  const detail = state.sessionDetail;
  elements.continueLeaf.hidden = !detail?.leafId;
  if (state.sessionDetailError) {
    elements.sessionTitle.textContent = "Session unavailable";
    elements.sessionMeta.textContent = state.sessionDetailError;
    elements.sessionTree.replaceChildren(node("p", state.sessionDetailError, "history-empty"));
    return;
  }
  if (!state.selectedSessionId) {
    elements.sessionTitle.textContent = "Select a session";
    elements.sessionMeta.textContent = "No session selected";
    elements.sessionTree.replaceChildren(node("p", "Select a session to inspect its stored entries.", "history-empty"));
    return;
  }
  if (!detail) {
    elements.sessionTitle.textContent = "Loading session";
    elements.sessionMeta.textContent = state.selectedSessionId;
    elements.sessionTree.replaceChildren(node("p", "Loading stored entries...", "history-empty"));
    return;
  }
  elements.sessionTitle.textContent = detail.name || short(currentRequest(detail.firstMessage), 90) || detail.id;
  elements.sessionMeta.textContent = `${detail.messageCount} messages · ${formatDate(detail.createdAt)} · ${detail.id}`;
  const byId = new Map(detail.entries.map((entry) => [entry.id, entry]));
  const cache = new Map();
  const entries = detail.entries.map((entry) =>
    renderSessionEntry(entry, entryDepth(entry, byId, cache), entry.id === detail.leafId),
  );
  const header = node("details", null, "session-entry");
  header.append(node("summary", "Session header JSON"), node("pre", pretty(detail.header), "entry-json"));
  elements.sessionTree.replaceChildren(header, ...entries);
}

function openContinuation(entryId) {
  const sessionId = state.selectedSessionId;
  if (!sessionId || !entryId) return;
  newChat();
  state.sessionId = sessionId;
  state.parentEntryId = entryId;
  renderResume();
  showView("playground");
  elements.prompt.focus();
}

function auditOptionLabel(audit) {
  return `${audit.status} · ${formatDate(audit.startedAt)} · ${short(audit.prompt, 70) || audit.runId}`;
}

function formatDuration(value) {
  if (!Number.isFinite(value) || value < 0) return "Pending";
  if (value < 1_000) return `${Math.round(value)} ms`;
  if (value < 60_000) return `${(value / 1_000).toFixed(value < 10_000 ? 1 : 0)} s`;
  return `${Math.floor(value / 60_000)}m ${Math.round((value % 60_000) / 1_000)}s`;
}

function routeLabel(route) {
  return ({
    off: "Memory off",
    current_bank_only: "Current bank only",
    source_discovery_only: "Source discovery only",
    cross_bank_attempted: "Cross-bank query attempted",
    cross_bank_failed: "Cross-bank query failed",
    cross_bank_queried: "Cross-bank queried",
  })[route] || "Unknown memory path";
}

function traceFact(label, value, className = "") {
  const item = node("div", null, "trace-fact");
  item.append(node("dt", label));
  item.append(node("dd", value, className));
  return item;
}

function inspectAuditEvent(sequence) {
  const target = document.querySelector(`#audit-event-${sequence}`);
  if (!target) return;
  target.focus({ preventScroll: true });
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
}

function traceStep({ label, detail, status, eventSequence }) {
  const item = button(null, `trace-step ${status}`, () => inspectAuditEvent(eventSequence));
  item.setAttribute("aria-label", `Inspect raw event #${eventSequence}: ${label}`);
  const header = node("span", null, "trace-step-header");
  header.append(node("span", label, "trace-step-label"));
  header.append(node("span", status.replaceAll("_", " "), "trace-step-status"));
  item.append(header);
  if (detail) item.append(node("span", detail, "trace-step-detail"));
  item.append(node("span", `Event #${eventSequence}`, "trace-step-event"));
  return item;
}

function sourceLabel(source) {
  return [...new Set([
    source?.displayName,
    source?.handle,
    source?.bankId,
  ].filter(Boolean))].join(" · ");
}

function toolStep(tool) {
  const labels = {
    memory_query_current: "Current-bank query",
    memory_find_sources: "Model-requested source discovery",
    memory_query_source: "Cross-bank source query",
  };
  const source = tool.source ? sourceLabel(tool.source) : null;
  const detail = [
    source,
    tool.query && `Query: ${short(tool.query, 240)}`,
    tool.durationMs != null && formatDuration(tool.durationMs),
  ].filter(Boolean).join("\n");
  return {
    label: labels[tool.name] || `Tool · ${tool.name}`,
    detail,
    status: tool.status,
    eventSequence: tool.eventSequence,
  };
}

function renderAuditDiagnosis(summary, audit) {
  if (!audit) {
    elements.auditSummary.replaceChildren(
      node("p", "Select a session and run to inspect its diagnosis.", "history-empty"),
    );
    return;
  }
  if (!summary) {
    const loading = state.auditError ? "Run diagnosis unavailable." : "Loading run diagnosis...";
    elements.auditSummary.replaceChildren(
      node("p", `${audit.status} · ${audit.eventCount} events · ${audit.memoryScopeId || "memory off"}`, "trace-loading"),
      node("p", loading, "trace-loading-detail"),
    );
    return;
  }

  const diagnosis = node("article", null, "trace-diagnosis");
  const header = node("header", null, "trace-diagnosis-header");
  const status = node("span", summary.status.replaceAll("_", " "), `trace-status ${summary.status}`);
  header.append(status);
  header.append(node("span", formatDuration(summary.durationMs), "trace-duration"));
  header.append(node("span", `${summary.eventCount} events`, "trace-event-count"));
  diagnosis.append(header);
  if (summary.prompt) diagnosis.append(node("p", summary.prompt, "trace-prompt"));
  diagnosis.append(node("code", audit.runId, "trace-run-id"));

  const session = summary.session || {};
  const sessionText = session.kind === "continuation"
    ? `Continuation · ${session.id || "unknown session"}\nParent ${session.parentEntryId || "unknown"}`
    : `Root run · ${session.id || "session pending"}`;
  const modelText = summary.model
    ? [
        [summary.model.id, summary.model.provider].filter(Boolean).join(" · ") || "Unknown model",
        `Thinking ${summary.model.thinkingLevel || "unknown"}`,
      ].join("\n")
    : "Model pending";
  const queriedSources = summary.tools
    .filter((tool) => tool.name === "memory_query_source" && tool.source)
    .map((tool) => sourceLabel(tool.source))
    .filter(Boolean);
  const routeText = [
    routeLabel(summary.memory.route),
    queriedSources.length > 0 ? `Sources: ${[...new Set(queriedSources)].join(", ")}` : null,
  ].filter(Boolean).join("\n");
  const facts = node("dl", null, "trace-facts");
  facts.append(traceFact("Session", sessionText));
  facts.append(traceFact("Model", modelText));
  facts.append(traceFact("Primary memory", summary.memory.primaryBankId || "Off"));
  facts.append(traceFact("Memory path", routeText, `trace-route ${summary.memory.route}`));
  diagnosis.append(facts);

  const steps = [];
  const initial = summary.memory.initialRecall;
  if (summary.memory.route !== "off" && initial) {
    const queryDetail = initial.queries.length > 0
      ? ` · ${short(initial.queries[0], 160)}${initial.queries.length > 1 ? ` (+${initial.queries.length - 1} more)` : ""}`
      : "";
    steps.push({
      label: "Automatic primary recall",
      detail: `${initial.memoryCount} memories from ${initial.queries.length} queries${queryDetail}`,
      status: initial.status,
      eventSequence: initial.eventSequence,
    });
  }
  const directory = summary.memory.directory;
  if (summary.memory.route !== "off" && directory) {
    steps.push({
      label: "Automatic directory discovery",
      detail: [
        `${directory.sourceCount} source handles offered`,
        directory.query && `Query: ${short(directory.query, 200)}`,
      ].filter(Boolean).join("\n"),
      status: directory.status === "available" ? "completed" : directory.status,
      eventSequence: directory.eventSequence,
    });
  }
  steps.push(...summary.tools.map(toolStep));

  const trail = node("section", null, "trace-section");
  trail.append(node("h3", "Decision trail"));
  if (steps.length === 0) {
    trail.append(node("p", summary.memory.route === "off" ? "Memory was disabled and no tools ran." : "No diagnostic steps were recorded.", "trace-empty"));
  } else {
    const list = node("ol", null, "trace-steps");
    for (const step of steps) {
      const listItem = node("li");
      listItem.append(traceStep(step));
      list.append(listItem);
    }
    trail.append(list);
  }
  diagnosis.append(trail);

  const notices = [
    ...(summary.failure ? [{
      label: `${summary.failure.code} · ${summary.failure.message}`,
      eventSequence: summary.failure.eventSequence,
    }] : []),
    ...summary.warnings.map((warning) => {
      const noun = warning.unavailableBankCount === 1 ? "bank" : "banks";
      return {
        label: `Access warning · ${warning.unavailableBankCount} earlier source ${noun} unavailable`,
        eventSequence: warning.eventSequence,
      };
    }),
  ];
  if (notices.length > 0) {
    const warningSection = node("section", null, "trace-section trace-warnings");
    warningSection.append(node("h3", "Warnings"));
    for (const notice of notices) {
      const warning = button(notice.label, "trace-warning", () => inspectAuditEvent(notice.eventSequence));
      warning.setAttribute("aria-label", `Inspect raw event #${notice.eventSequence}: ${notice.label}`);
      warningSection.append(warning);
    }
    diagnosis.append(warningSection);
  }
  elements.auditSummary.replaceChildren(diagnosis);
}

function renderAudits() {
  elements.auditCount.textContent = `${state.auditTotal} runs`;
  elements.auditSelect.disabled = state.audits.length === 0;
  if (state.audits.length === 0) {
    const option = node("option", "No audited runs");
    option.value = "";
    elements.auditSelect.replaceChildren(option);
  } else {
    elements.auditSelect.replaceChildren(...state.audits.map((audit) => {
      const option = node("option", auditOptionLabel(audit));
      option.value = audit.runId;
      return option;
    }));
    elements.auditSelect.value = state.selectedAuditId || state.audits[0].runId;
  }
  const summary = state.audits.find((audit) => audit.runId === state.selectedAuditId);
  renderAuditDiagnosis(state.auditDetail?.summary, summary);
  if (state.auditError) {
    elements.auditEvents.replaceChildren(node("p", state.auditError, "history-empty"));
  } else if (!state.selectedSessionId) {
    elements.auditEvents.replaceChildren(node("p", "Detailed audit is unavailable until a session is selected.", "history-empty"));
  } else if (state.audits.length === 0) {
    elements.auditEvents.replaceChildren(node("p", "This session predates detailed run auditing, or was created outside the audited service.", "history-empty"));
  } else if (!state.auditDetail) {
    elements.auditEvents.replaceChildren(node("p", "Loading run events...", "history-empty"));
  } else {
    renderAuditEvents();
  }
}

async function selectAudit(runId) {
  state.selectedAuditId = runId;
  state.auditDetail = null;
  state.auditError = null;
  renderAudits();
  try {
    const detail = await requestJson(`/api/audits/${encodeURIComponent(runId)}`);
    if (state.selectedAuditId !== runId) return;
    state.auditDetail = detail;
  } catch (error) {
    if (state.selectedAuditId !== runId) return;
    state.auditError = error.message;
  }
  renderAudits();
}

function auditDescription(event) {
  const data = event.data || {};
  if (event.type === "memory.http.request") {
    const request = data.request || {};
    return `${data.operation || "memory"}${data.variant ? ` · ${data.variant}` : ""}\n${request.method || "GET"} ${request.url || ""}`;
  }
  if (event.type === "memory.http.response") {
    const response = data.response || {};
    const outcome = response.usable === false
      ? ` · unusable (${(response.failureReason || "invalid response").replaceAll("_", " ")})`
      : "";
    return `${data.operation || "memory"} · HTTP ${response.status ?? "?"}${outcome} · ${response.durationMs ?? "?"} ms · ${response.bodyBytes ?? "?"} bytes`;
  }
  if (event.type === "memory.http.error") return `${data.operation || "memory"} · ${data.error?.message || "request failed"}`;
  if (event.type === "memory.directory.policy") {
    return `${data.requester?.owner ? "Owner" : "Delegated"} directory policy · ${(data.allowedBankIds || []).length || "unrestricted"} allowed banks · ${(data.participants || []).length} participants`;
  }
  if (event.type === "memory.directory.result") {
    return `${data.status || "unknown"} · ${(data.references || []).length} validated directory references`;
  }
  if (event.type === "memory.capabilities.issued") {
    return `${(data.sources || []).length} opaque source handles issued · ${data.stopReason || "complete"}`;
  }
  if (event.type === "memory.access.warning") {
    return `Access warning · ${data.unavailableBankIds?.length || 0} prior source banks are no longer available to this requester. Earlier branch evidence remains in the shared session; the non-disclosure safeguard is advisory in this version.`;
  }
  if (event.type === "tool.started") return `${data.toolName || "tool"} started\n${short(pretty(data.args), 300)}`;
  if (event.type === "tool.completed") return `${data.toolName || "tool"} ${data.isError ? "failed" : "completed"} · ${data.durationMs ?? "?"} ms`;
  if (event.type === "model.input") return `${data.model?.id || "model"} · ${(data.tools || []).length} tools\n${short(currentRequest(data.prompt), 300)}`;
  if (event.type === "model.turn.started") return `Model turn ${data.turn ?? "?"} started`;
  if (event.type === "model.turn.completed") return `Model turn ${data.turn ?? "?"} completed · ${data.durationMs ?? "?"} ms`;
  if (event.type === "memory.context") return `${(data.memories || []).length} memories merged from ${(data.queries || []).length} recall queries`;
  if (event.type === "run.request") return short(data.prompt, 300);
  if (event.type === "run.completed") return short(data.answer, 300);
  if (event.type === "run.failed") return `${data.code || "FAILED"} · ${data.message || data.error?.message || "Run failed"}`;
  if (event.type === "session.opened") return `${data.sessionId || "session"} from ${data.parentEntryId || "root"}`;
  return short(pretty(data), 300);
}

function auditCorrelation(data) {
  return [
    data.exchangeId && `exchange ${data.exchangeId}`,
    data.toolCallId && `tool ${data.toolCallId}`,
    data.sessionId && `session ${data.sessionId}`,
    data.entryId && `entry ${data.entryId}`,
  ].filter(Boolean).join(" · ");
}

function renderAuditEvents() {
  const events = state.auditDetail?.events || [];
  if (events.length === 0) {
    elements.auditEvents.replaceChildren(node("p", "No events recorded.", "history-empty"));
    return;
  }
  const heading = node("div", null, "raw-events-header");
  heading.append(node("h3", "Raw events"), node("span", `${events.length} recorded`, "panel-count"));
  elements.auditEvents.replaceChildren(heading, ...events.map((event) => {
    const category = event.type.split(".")[0];
    const warning = event.type === "memory.access.warning" ? " warning" : "";
    const item = node("article", null, `audit-event ${category}${warning}`);
    item.id = `audit-event-${event.sequence}`;
    item.tabIndex = -1;
    const header = node("div", null, "audit-event-header");
    header.append(node("span", `#${event.sequence}`, "audit-sequence"));
    header.append(node("span", event.type, "audit-type"));
    header.append(node("time", formatDate(event.timestamp), "audit-time"));
    item.append(header, node("p", auditDescription(event), "audit-description"));
    const correlation = auditCorrelation(event.data || {});
    if (correlation) item.append(node("div", correlation, "audit-correlation"));
    const details = node("details");
    details.append(node("summary", "Event JSON"), node("pre", pretty(event), "audit-json"));
    item.append(details);
    return item;
  }));
}

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.mode !== state.mode) {
      state.mode = button.dataset.mode;
      newChat();
      renderMode();
    }
  });
});

document.querySelectorAll("[data-view]").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

document.querySelectorAll("[role=tab]").forEach((button) => {
  button.addEventListener("click", () => {
    state.tab = button.dataset.tab;
    renderInspector();
  });
});

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = elements.prompt.value.trim();
  if (!prompt || state.runId) return;
  elements.prompt.value = "";
  void runPrompt(prompt);
});
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.previewMemory.addEventListener("click", () => void previewMemory());
elements.newChat.addEventListener("click", newChat);
elements.clearResume.addEventListener("click", newChat);
elements.refreshHistory.addEventListener("click", () => {
  state.sessions = [];
  state.sessionCursor = null;
  void loadSessions();
});
elements.refreshChannels.addEventListener("click", () => {
  void loadChannels();
  reconnectChannelEvents();
});
elements.channelFilters.addEventListener("submit", (event) => event.preventDefault());
elements.channelQuery.addEventListener("input", () => {
  state.channelQuery = elements.channelQuery.value.trim();
  renderChannels();
});
elements.channelPlatform.addEventListener("change", () => {
  state.channelPlatform = elements.channelPlatform.value;
  renderChannels();
});
elements.channelStatusFilter.addEventListener("change", () => {
  state.channelStatus = elements.channelStatusFilter.value;
  renderChannels();
});
elements.sessionSearch.addEventListener("submit", (event) => {
  event.preventDefault();
  state.sessionQuery = elements.sessionQuery.value.trim();
  state.sessions = [];
  state.sessionCursor = null;
  void loadSessions();
});
elements.loadMoreSessions.addEventListener("click", () => void loadSessions({ append: true }));
elements.continueLeaf.addEventListener("click", () => {
  if (state.sessionDetail?.leafId) openContinuation(state.sessionDetail.leafId);
});
elements.auditSelect.addEventListener("change", () => {
  if (elements.auditSelect.value) void selectAudit(elements.auditSelect.value);
});
elements.stop.addEventListener("click", async () => {
  if (!state.runId) return;
  elements.runStatus.textContent = "Cancelling";
  await requestJson(`/api/runs/${encodeURIComponent(state.runId)}/cancel`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  }).catch(() => {});
});

void initialize();
