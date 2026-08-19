"use strict";

const state = {
  catalog: null,
  group: null,
  run: null,
  index: null,
  episode: null,
  stepId: null,
  step: null,
  conversation: null,
  renderedConversationKey: null,
  selectionRevision: 0,
  mode: "snapshot",
  inspectorTab: "request",
};

const elements = {
  runName: document.querySelector("#run-name"),
  promptVersion: document.querySelector("#prompt-version"),
  versionButton: document.querySelector("#version-button"),
  versionDialog: document.querySelector("#version-dialog"),
  dialogVersion: document.querySelector("#dialog-version"),
  closeVersionDialog: document.querySelector("#close-version-dialog"),
  groupSelect: document.querySelector("#group-select"),
  runSelect: document.querySelector("#run-select"),
  episodeSelect: document.querySelector("#episode-select"),
  episodeStatus: document.querySelector("#episode-status"),
  stepFilter: document.querySelector("#step-filter"),
  stepList: document.querySelector("#step-list"),
  snapshotMode: document.querySelector("#snapshot-mode"),
  conversationMode: document.querySelector("#conversation-mode"),
  previousStep: document.querySelector("#previous-step"),
  nextStep: document.querySelector("#next-step"),
  stepPosition: document.querySelector("#step-position"),
  requestMeta: document.querySelector("#request-meta"),
  chat: document.querySelector("#chat"),
  inspectorTabs: document.querySelectorAll("[data-inspector-tab]"),
  jsonView: document.querySelector("#json-view"),
  copyJson: document.querySelector("#copy-json"),
  sidebar: document.querySelector("#sidebar"),
  inspector: document.querySelector("#inspector"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  inspectorToggle: document.querySelector("#inspector-toggle"),
  toast: document.querySelector("#toast"),
};

async function requestJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function currentEpisode() {
  return state.index.episodes.find((episode) => episode.id === state.episode);
}

function stepEntries() {
  return currentEpisode()?.step_entries || [];
}

function setLoading(label) {
  elements.chat.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading";
  loading.textContent = label;
  elements.chat.append(loading);
}

async function initialize() {
  try {
    state.catalog = await requestJson("/api/index");
    for (const group of state.catalog.groups) {
      const option = document.createElement("option");
      option.value = group.id;
      option.textContent = `${group.name} (${group.runs.length})`;
      elements.groupSelect.append(option);
    }
    const route = readLocationState();
    const routeGroup = groupContainingRun(route.run);
    state.mode = route.mode === "conversation" ? "conversation" : "snapshot";
    updateModeButtons();
    await selectGroup(
      routeGroup?.id || state.catalog.default_group_id,
      routeGroup ? route.run : state.catalog.default_run_id,
      {
        episodeId: routeGroup ? route.episode : null,
        stepId: routeGroup ? route.stepId : null,
        historyMode: "replace",
      },
    );
  } catch (error) {
    showFatal(error);
  }
}

function currentGroup() {
  return state.catalog.groups.find((group) => group.id === state.group);
}

function currentRun() {
  return currentGroup()?.runs.find((run) => run.id === state.run);
}

function groupContainingRun(runId) {
  if (!runId) {
    return null;
  }
  return state.catalog.groups.find((group) =>
    group.runs.some((run) => run.id === runId)
  ) || null;
}

async function selectGroup(groupId, preferredRunId = null, options = {}) {
  state.group = groupId;
  elements.groupSelect.value = groupId;
  elements.runSelect.replaceChildren();
  const group = currentGroup();
  if (!group || group.runs.length === 0) {
    throw new Error("This experiment group has no reviewable runs.");
  }
  for (const run of group.runs) {
    const option = document.createElement("option");
    option.value = run.id;
    const metrics = run.aggregate_metrics;
    const success =
      metrics && typeof metrics.success_rate === "number"
        ? ` | ${(metrics.success_rate * 100).toFixed(0)}%`
        : "";
    const config = [run.model, run.reasoning_effort, run.api]
      .filter(Boolean)
      .join(" / ");
    option.textContent =
      `${run.name} | ${run.episode_count} episodes${success}` +
      (config ? ` | ${config}` : "");
    elements.runSelect.append(option);
  }
  const runId = group.runs.some((run) => run.id === preferredRunId)
    ? preferredRunId
    : group.runs[0].id;
  await selectRun(runId, options);
}

async function selectRun(runId, options = {}) {
  state.run = runId;
  state.index = null;
  state.episode = null;
  state.stepId = null;
  state.step = null;
  state.conversation = null;
  state.renderedConversationKey = null;
  elements.runSelect.value = runId;
  elements.episodeSelect.replaceChildren();
  setLoading("Loading run");
  state.index = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
  const run = currentRun();
  elements.runName.textContent =
    `${state.catalog.name} / ${currentGroup().name} / ${run.name}`;
  for (const episode of state.index.episodes) {
    const option = document.createElement("option");
    option.value = episode.id;
    option.textContent = `${episode.id} - ${episode.termination_reason}`;
    elements.episodeSelect.append(option);
  }
  if (state.index.episodes.length === 0) {
    throw new Error("This run has no reviewable episodes.");
  }
  const episodeId = state.index.episodes.some(
    (episode) => episode.id === options.episodeId,
  )
    ? options.episodeId
    : state.index.episodes[0].id;
  await selectEpisode(episodeId, options);
}

async function selectEpisode(episodeId, options = {}) {
  state.episode = episodeId;
  state.stepId = null;
  state.step = null;
  state.conversation = null;
  state.renderedConversationKey = null;
  elements.episodeSelect.value = episodeId;
  elements.stepFilter.value = "";
  elements.stepList.scrollTop = 0;
  const episode = currentEpisode();
  elements.episodeStatus.textContent =
    `${episode.steps} agent steps | ${episode.termination_reason} | ` +
    `${episode.success ? "success" : "not successful"}`;
  elements.promptVersion.textContent = episode.prompt_version || "unknown";
  elements.dialogVersion.textContent = episode.prompt_version || "Unknown prompt version";
  renderStepList();
  const selectedStep = episode.step_ids.includes(options.stepId)
    ? options.stepId
    : episode.step_ids[0];
  if (selectedStep === undefined) {
    showFatal(new Error("This episode has no model request artifacts."));
    return;
  }
  await selectStep(selectedStep, {
    historyMode: options.historyMode === undefined ? "push" : options.historyMode,
  });
}

function renderStepList() {
  const filter = elements.stepFilter.value.trim();
  const scrollTop = elements.stepList.scrollTop;
  elements.stepList.replaceChildren();
  for (const entry of stepEntries()) {
    if (filter && !String(entry.id).includes(filter)) {
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `step-button${entry.id === state.stepId ? " active" : ""}`;
    button.dataset.stepId = String(entry.id);

    const dot = document.createElement("span");
    dot.className = `status-dot ${entry.status || "unknown"}`;
    const title = document.createElement("span");
    title.textContent = `Step ${entry.id}`;
    const detail = document.createElement("small");
    detail.textContent =
      entry.status === "ok" && entry.valid ? "valid" : entry.status || "unknown";
    button.append(dot, title, detail);
    button.addEventListener("click", () => selectStep(entry.id));
    elements.stepList.append(button);
  }
  elements.stepList.scrollTop = scrollTop;
  updateStepListSelection(false);
}

async function selectStep(stepId, options = {}) {
  const revision = ++state.selectionRevision;
  const selectedRun = state.run;
  const selectedEpisode = state.episode;
  state.stepId = stepId;
  updateStepListSelection();
  updateStepNavigation();
  if (state.mode === "snapshot") {
    setLoading(`Loading step ${stepId}`);
  }
  try {
    const step = await requestJson(
      `${runApiPrefix()}/episodes/${encodeURIComponent(state.episode)}/steps/${stepId}`,
    );
    if (!selectionIsCurrent(revision, selectedRun, selectedEpisode, stepId)) {
      return;
    }
    state.step = step;
    if (state.mode === "conversation") {
      await renderConversation(revision);
    } else {
      renderSnapshot();
      renderInspector();
    }
    if (!selectionIsCurrent(revision, selectedRun, selectedEpisode, stepId)) {
      return;
    }
    writeLocationState(
      options.historyMode === undefined ? "push" : options.historyMode,
    );
    closeCompactPanels();
  } catch (error) {
    if (selectionIsCurrent(revision, selectedRun, selectedEpisode, stepId)) {
      showFatal(error);
    }
  }
}

function selectionIsCurrent(revision, runId, episodeId, stepId) {
  return revision === state.selectionRevision
    && runId === state.run
    && episodeId === state.episode
    && stepId === state.stepId;
}

function updateStepListSelection(ensureVisible = true) {
  let selected = null;
  for (const button of elements.stepList.querySelectorAll(".step-button")) {
    const active = Number(button.dataset.stepId) === state.stepId;
    button.classList.toggle("active", active);
    if (active) {
      selected = button;
    }
  }
  if (ensureVisible && selected) {
    requestAnimationFrame(() => keepStepButtonVisible(selected));
  }
}

function keepStepButtonVisible(button) {
  if (!button.isConnected) {
    return;
  }
  const listBounds = elements.stepList.getBoundingClientRect();
  const buttonBounds = button.getBoundingClientRect();
  if (buttonBounds.top < listBounds.top) {
    elements.stepList.scrollTop += buttonBounds.top - listBounds.top;
  } else if (buttonBounds.bottom > listBounds.bottom) {
    elements.stepList.scrollTop += buttonBounds.bottom - listBounds.bottom;
  }
}

function updateStepNavigation() {
  const ids = currentEpisode().step_ids;
  const index = ids.indexOf(state.stepId);
  elements.stepPosition.textContent = `Step ${state.stepId}`;
  elements.previousStep.disabled = index <= 0;
  elements.nextStep.disabled = index < 0 || index >= ids.length - 1;
}

function renderSnapshot() {
  const step = state.step;
  elements.chat.replaceChildren();
  const imageLookup = buildImageLookup(step.images);
  const messages = step.display_messages;
  messages.forEach((message, index) => {
    const isCurrent = index === messages.length - 1;
    elements.chat.append(renderMessage(message, imageLookup, {
      current: isCurrent,
      label: isCurrent ? "Current observation" : null,
    }));
  });
  elements.chat.append(renderModelResult(step));
  renderRequestMeta(step);
  elements.chat.scrollTop = elements.chat.scrollHeight;
}

function renderMessage(message, imageLookup, options = {}) {
  const role = message.role || "unknown";
  const section = document.createElement("article");
  section.className = `message ${role}${options.current ? " current" : ""}`;
  if (options.stepId !== undefined) {
    section.classList.add("conversation-turn");
    section.dataset.stepId = String(options.stepId);
  }
  const inner = document.createElement("div");
  inner.className = "message-inner";
  const header = document.createElement("div");
  header.className = "message-header";
  const title = document.createElement("strong");
  title.textContent = roleLabel(role, options);
  const meta = document.createElement("span");
  meta.textContent = options.label || "";
  header.append(title, meta);
  inner.append(header);
  appendContent(inner, message.content, imageLookup);
  section.append(inner);
  return section;
}

function roleLabel(role, options) {
  if (options.result) {
    return options.error ? "Agent error" : "Model response";
  }
  if (role === "assistant") {
    return "Assistant history";
  }
  if (role === "system") {
    return "System";
  }
  return role === "user" ? "User" : role;
}

function appendContent(container, content, imageLookup) {
  if (typeof content === "string") {
    const pre = document.createElement("pre");
    pre.textContent = content;
    container.append(pre);
    return;
  }
  if (!Array.isArray(content)) {
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(content, null, 2);
    container.append(pre);
    return;
  }
  for (const item of content) {
    if (!item || typeof item !== "object") {
      continue;
    }
    if (item.type === "text") {
      const text = String(item.text || "");
      const node = document.createElement(text.includes("[image omitted") ? "span" : "p");
      node.className = text.includes("[image omitted") ? "omitted" : "";
      node.textContent = text;
      container.append(node);
      continue;
    }
    if (item.type === "image_url") {
      const url = item.image_url?.url || "";
      const digest = url.includes("sha256=") ? url.split("sha256=")[1] : null;
      const image = imageLookup.get(`artifact:${item.artifact}`)
        || imageLookup.get(`sha256:${digest}`);
      const figure = document.createElement("figure");
      figure.className = "image-block";
      if (image) {
        const imageNode = document.createElement("img");
        imageNode.src = image.url;
        imageNode.alt = `${image.kind || "rgb"} ${image.camera} observation at step ${image.step_id}`;
        imageNode.loading = "lazy";
        if (image.width && image.height) {
          imageNode.width = image.width;
          imageNode.height = image.height;
        }
        const caption = document.createElement("figcaption");
        const source = image.kind === "magnifier"
          ? ` | source ${image.source_rgb_key} | bbox ${JSON.stringify(image.bbox_xyxy)}`
          : "";
        caption.textContent =
          `${image.camera} | ${image.kind || "rgb"}${source} | step ${image.step_id} | sha256 ${image.sha256}`;
        figure.append(imageNode, caption);
      } else {
        const missing = document.createElement("span");
        missing.className = "omitted";
        missing.textContent = `Image artifact unavailable: ${digest || "unknown hash"}`;
        figure.append(missing);
      }
      container.append(figure);
    }
  }
}

function renderModelResult(step) {
  const hasError = Boolean(step.error);
  const section = document.createElement("article");
  section.className = `message ${hasError ? "error" : "assistant"} current`;
  const inner = document.createElement("div");
  inner.className = "message-inner";
  const header = document.createElement("div");
  header.className = "message-header";
  const title = document.createElement("strong");
  title.textContent = hasError ? "Agent error" : "Model response";
  const meta = document.createElement("span");
  meta.textContent = "Raw provider content";
  header.append(title, meta);
  inner.append(header);
  const pre = document.createElement("pre");
  pre.className = "raw-response";
  pre.textContent = hasError
    ? JSON.stringify(step.error, null, 2)
    : rawContent(step.response_content);
  inner.append(pre);
  if (!hasError && hasNextStep()) {
    const history = document.createElement("p");
    history.className = "follow-up";
    history.textContent =
      step.api === "responses"
        ? `Retained by response ${step.public_trace?.response_id || "unknown"} for step ${nextStepId()}.`
        : `Included as assistant history in step ${nextStepId()}.`;
    inner.append(history);
  }
  section.append(inner);
  return section;
}

function rawContent(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function renderRequestMeta(step) {
  const messages = step.display_messages.length;
  elements.requestMeta.replaceChildren();
  for (const text of [
    `${step.api || "unknown"} API`,
    `${messages} displayed request items`,
    `${step.images.length} included image artifacts`,
    `${step.common_prefix_bytes} common prefix bytes`,
    `model ${step.request.model || "unknown"}`,
    ...(step.api === "responses"
      ? [
          `previous ${step.public_trace?.previous_response_id || "root"}`,
          `response ${step.public_trace?.response_id || "unavailable"}`,
        ]
      : []),
  ]) {
    const span = document.createElement("span");
    span.textContent = text;
    elements.requestMeta.append(span);
  }
}

async function renderConversation(revision = state.selectionRevision) {
  const key = conversationKey();
  if (state.renderedConversationKey !== key) {
    setLoading("Loading episode conversation");
  }
  try {
    if (!state.conversation) {
      const conversation = await requestJson(
        `${runApiPrefix()}/episodes/${encodeURIComponent(state.episode)}/conversation`,
      );
      if (revision !== state.selectionRevision || key !== conversationKey()) {
        return;
      }
      state.conversation = conversation;
    }
    if (revision !== state.selectionRevision || key !== conversationKey()) {
      return;
    }
    if (state.renderedConversationKey !== key) {
      renderConversationDocument();
    }
    updateConversationSelection();
    renderRequestMeta(state.step);
    renderInspector();
    scrollToConversationStep(revision);
    closeCompactPanels();
  } catch (error) {
    if (revision === state.selectionRevision && key === conversationKey()) {
      showFatal(error);
    }
  }
}

function conversationKey() {
  return `${state.run}\n${state.episode}`;
}

function renderConversationDocument() {
  elements.chat.replaceChildren();
  for (const message of state.conversation.static_messages) {
    elements.chat.append(renderMessage(message, new Map()));
  }
  for (const turn of state.conversation.turns) {
    const imageLookup = buildImageLookup(turn.images);
    elements.chat.append(
      renderMessage(turn.observation, imageLookup, {
        current: false,
        label: `Step ${turn.step_id}`,
        stepId: turn.step_id,
      }),
    );
    elements.chat.append(renderConversationResponse(turn));
  }
  state.renderedConversationKey = conversationKey();
}

function updateConversationSelection() {
  for (const turn of elements.chat.querySelectorAll(".conversation-turn")) {
    turn.classList.toggle(
      "current",
      Number(turn.dataset.stepId) === state.stepId,
    );
  }
}

function scrollToConversationStep(revision) {
  requestAnimationFrame(() => {
    if (revision !== state.selectionRevision || state.mode !== "conversation") {
      return;
    }
    const selected = elements.chat.querySelector(
      `.conversation-turn[data-step-id="${state.stepId}"]`,
    );
    if (!selected) {
      return;
    }
    const chatBounds = elements.chat.getBoundingClientRect();
    const selectedBounds = selected.getBoundingClientRect();
    elements.chat.scrollTo({
      top: elements.chat.scrollTop + selectedBounds.top - chatBounds.top,
      behavior: "auto",
    });
  });
}

function buildImageLookup(images) {
  const lookup = new Map();
  for (const image of images) {
    lookup.set(`artifact:${image.artifact}`, image);
    if (!lookup.has(`sha256:${image.sha256}`)) {
      lookup.set(`sha256:${image.sha256}`, image);
    }
  }
  return lookup;
}

function renderConversationResponse(turn) {
  const hasError = Boolean(turn.error);
  const section = document.createElement("article");
  section.className = `message ${hasError ? "error" : "assistant"}`;
  const inner = document.createElement("div");
  inner.className = "message-inner";
  const header = document.createElement("div");
  header.className = "message-header";
  const title = document.createElement("strong");
  title.textContent = hasError ? "Agent error" : "Model response";
  const meta = document.createElement("span");
  meta.textContent = `Step ${turn.step_id}`;
  header.append(title, meta);
  const pre = document.createElement("pre");
  pre.textContent = hasError
    ? JSON.stringify(turn.error, null, 2)
    : rawContent(turn.response_content);
  inner.append(header, pre);
  section.append(inner);
  return section;
}

function renderInspector() {
  if (!state.step) {
    return;
  }
  let value;
  if (state.inspectorTab === "request") {
    value = state.step.request;
  } else if (state.inspectorTab === "response") {
    value = state.step.response || state.step.error;
  } else {
    value = {
      prompt_version: state.step.prompt_version,
      common_prefix_bytes: state.step.common_prefix_bytes,
      public_trace: state.step.public_trace,
      action: state.step.action,
    };
  }
  elements.jsonView.textContent = JSON.stringify(value, null, 2);
}

function hasNextStep() {
  return nextStepId() !== null;
}

function nextStepId() {
  const ids = currentEpisode().step_ids;
  const index = ids.indexOf(state.stepId);
  return index >= 0 && index < ids.length - 1 ? ids[index + 1] : null;
}

function adjacentStep(offset) {
  const ids = currentEpisode().step_ids;
  const index = ids.indexOf(state.stepId);
  const target = ids[index + offset];
  if (target !== undefined) {
    selectStep(target);
  }
}

function setMode(mode, options = {}) {
  state.mode = mode;
  updateModeButtons();
  selectStep(state.stepId, {
    historyMode: options.historyMode === undefined ? "push" : options.historyMode,
  });
}

function updateModeButtons() {
  elements.snapshotMode.classList.toggle("active", state.mode === "snapshot");
  elements.conversationMode.classList.toggle("active", state.mode === "conversation");
}

function readLocationState() {
  const params = new URLSearchParams(window.location.search);
  const rawStep = params.get("step");
  const parsedStep = rawStep === null ? null : Number(rawStep);
  return {
    run: params.get("run"),
    episode: params.get("episode"),
    stepId: Number.isInteger(parsedStep) ? parsedStep : null,
    mode: params.get("mode"),
  };
}

function writeLocationState(historyMode) {
  if (!historyMode || !state.run || !state.episode || state.stepId === null) {
    return;
  }
  const url = new URL(window.location.href);
  url.searchParams.set("run", state.run);
  url.searchParams.set("episode", state.episode);
  url.searchParams.set("step", String(state.stepId));
  url.searchParams.set("mode", state.mode);
  if (historyMode === "replace") {
    window.history.replaceState(null, "", url);
  } else {
    window.history.pushState(null, "", url);
  }
}

async function restoreLocationState() {
  const route = readLocationState();
  const group = groupContainingRun(route.run);
  if (!group) {
    return;
  }
  state.mode = route.mode === "conversation" ? "conversation" : "snapshot";
  updateModeButtons();
  if (route.run !== state.run) {
    await selectGroup(group.id, route.run, {
      episodeId: route.episode,
      stepId: route.stepId,
      historyMode: null,
    });
    return;
  }
  if (
    route.episode !== state.episode
    && state.index.episodes.some((episode) => episode.id === route.episode)
  ) {
    await selectEpisode(route.episode, {
      stepId: route.stepId,
      historyMode: null,
    });
    return;
  }
  const stepId = currentEpisode().step_ids.includes(route.stepId)
    ? route.stepId
    : currentEpisode().step_ids[0];
  await selectStep(stepId, { historyMode: null });
}

function setInspectorTab(tab) {
  state.inspectorTab = tab;
  for (const button of elements.inspectorTabs) {
    button.classList.toggle("active", button.dataset.inspectorTab === tab);
  }
  renderInspector();
}

function runApiPrefix() {
  return `/api/runs/${encodeURIComponent(state.run)}`;
}

function showFatal(error) {
  elements.chat.replaceChildren();
  const message = document.createElement("div");
  message.className = "empty-state";
  message.textContent = error.message || String(error);
  elements.chat.append(message);
}

function showToast(text) {
  elements.toast.textContent = text;
  elements.toast.classList.add("visible");
  window.setTimeout(() => elements.toast.classList.remove("visible"), 1800);
}

function closeCompactPanels() {
  elements.sidebar.classList.remove("open");
  elements.inspector.classList.remove("open");
}

elements.episodeSelect.addEventListener("change", () => {
  selectEpisode(elements.episodeSelect.value).catch(showFatal);
});
elements.groupSelect.addEventListener("change", () => {
  selectGroup(elements.groupSelect.value).catch(showFatal);
});
elements.runSelect.addEventListener("change", () => {
  selectRun(elements.runSelect.value).catch(showFatal);
});
elements.stepFilter.addEventListener("input", renderStepList);
elements.previousStep.addEventListener("click", () => adjacentStep(-1));
elements.nextStep.addEventListener("click", () => adjacentStep(1));
elements.snapshotMode.addEventListener("click", () => setMode("snapshot"));
elements.conversationMode.addEventListener("click", () => setMode("conversation"));
elements.inspectorTabs.forEach((button) => {
  button.addEventListener("click", () => setInspectorTab(button.dataset.inspectorTab));
});
elements.copyJson.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.jsonView.textContent);
  showToast("JSON copied");
});
elements.versionButton.addEventListener("click", () => elements.versionDialog.showModal());
elements.closeVersionDialog.addEventListener("click", () => elements.versionDialog.close());
elements.sidebarToggle.addEventListener("click", () => elements.sidebar.classList.toggle("open"));
elements.inspectorToggle.addEventListener("click", () => elements.inspector.classList.toggle("open"));
window.addEventListener("popstate", () => restoreLocationState().catch(showFatal));

initialize();
