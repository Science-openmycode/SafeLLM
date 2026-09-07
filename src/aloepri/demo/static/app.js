const form = document.querySelector("#prompt-form");
const submit = document.querySelector("#submit");
const state = document.querySelector("#run-state");
const stateMessage = document.querySelector("#run-message");
const answerBox = document.querySelector("#answer");
const promptBox = document.querySelector("#prompt");
const traceToggle = document.querySelector("#trace-toggle");
const privateTrace = document.querySelector("#private-trace");
const newChat = document.querySelector("#new-chat");
const historyList = document.querySelector("#history-list");
const savedChatList = document.querySelector("#saved-chat-list");
const savedChatCount = document.querySelector("#saved-chat-count");

const CHAT_STORAGE_KEY = "aloepri-saved-conversations-v1";
const ACTIVE_CHAT_STORAGE_KEY = "aloepri-active-conversation-v1";
const MAX_SAVED_CONVERSATIONS = 30;
const MAX_SAVED_MESSAGES = 100;
const MAX_CONTEXT_MESSAGES = 32;
const isQuickTunnel = window.location.hostname.endsWith(".trycloudflare.com");

let privateOutputIds = [];
let answerTarget = "";
let typedAnswer = "";
let typingPromise = null;
let conversationHistory = [];
let completedTurn = null;
let activePrompt = "";
let currentConversationId = null;
let savedConversations = [];
let currentSecurityMode = document.documentElement.dataset.securityMode || "permutation";

function applySecurityMode(mode) {
  currentSecurityMode = mode === "tee_gm" ? "tee_gm" : "permutation";
  document.documentElement.dataset.securityMode = currentSecurityMode;
  const tee = currentSecurityMode === "tee_gm";
  text("trace-lock-mark", tee ? "国密" : "τ");
  text("private-input-heading", tee ? "加密后的提示词" : "混淆后的提示词");
  text("private-output-heading", tee ? "加密后的回答" : "混淆后的回答");
  text("input-transform-label", tee ? "GM-TLS(input)" : "τ(input)");
  text("output-transform-label", tee ? "GM-TLS(output)" : "τ(output)");
  text("input-boundary-caption", tee ? "国密TLS密文仅在TEE内解密" : "模型服务器实际接收");
  text("output-boundary-caption", tee ? "普通Token仅在TEE内采样" : "客户端逐 token 执行 τ⁻¹");
  text("boundary-title", tee ? "TEE 国密可信边界" : "本地可信客户端");
  text("boundary-detail", tee ? "普通Token仅进入软件模拟TEE，不进入GPU主体" : "明文不会进入模型服务请求");
  text(
    "welcome-boundary",
    tee
      ? "问题通过国密安全通道进入TEE；GPU模型主体只看到P/Q私有坐标。"
      : "问题在本地完成分词和置换，模型服务只会看到混淆 token。",
  );
  const mapping = document.querySelector("#tau-mapping");
  if (mapping) mapping.hidden = tee;
}
window.applySecurityMode = applySecurityMode;

function newConversationId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normaliseStoredMessages(value) {
  if (!Array.isArray(value)) return [];
  const messages = [];
  for (let index = 0; index + 1 < value.length; index += 2) {
    const user = value[index];
    const assistant = value[index + 1];
    if (
      user?.role !== "user" || typeof user.content !== "string" ||
      assistant?.role !== "assistant" || typeof assistant.content !== "string"
    ) break;
    messages.push(
      {role: "user", content: user.content},
      {role: "assistant", content: assistant.content},
    );
  }
  return messages.slice(-MAX_SAVED_MESSAGES);
}

function readSavedConversations() {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(CHAT_STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => ({
        id: typeof item?.id === "string" ? item.id : "",
        title: typeof item?.title === "string" ? item.title : "",
        createdAt: typeof item?.createdAt === "string" ? item.createdAt : "",
        updatedAt: typeof item?.updatedAt === "string" ? item.updatedAt : "",
        messages: normaliseStoredMessages(item?.messages),
      }))
      .filter((item) => item.id && item.messages.length >= 2)
      .slice(0, MAX_SAVED_CONVERSATIONS);
  } catch (_error) {
    return [];
  }
}

function writeSavedConversations(nextConversations) {
  try {
    window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(nextConversations));
    savedConversations = nextConversations;
    return true;
  } catch (_error) {
    return false;
  }
}

function conversationTitle(messages) {
  const firstQuestion = messages.find((message) => message.role === "user")?.content || "新对话";
  const compact = firstQuestion.replace(/\s+/g, " ").trim();
  return compact.length > 28 ? `${compact.slice(0, 28)}…` : compact;
}

function formatConversationTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "刚刚";
  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) return "刚刚";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)} 分钟前`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)} 小时前`;
  return new Intl.DateTimeFormat("zh-CN", {month: "numeric", day: "numeric"}).format(date);
}

function renderSavedConversations() {
  savedChatList.replaceChildren();
  savedChatCount.textContent = String(savedConversations.length);
  if (savedConversations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "saved-chat-empty";
    empty.textContent = "回答完成后自动保存在此浏览器";
    savedChatList.appendChild(empty);
    return;
  }
  savedConversations.forEach((conversation) => {
    const entry = document.createElement("div");
    entry.className = "saved-chat-entry";
    if (conversation.id === currentConversationId) entry.classList.add("active");

    const open = document.createElement("button");
    open.type = "button";
    open.className = "saved-chat-open";
    open.title = conversation.title;
    const title = document.createElement("span");
    title.className = "saved-chat-title";
    title.textContent = conversation.title;
    const time = document.createElement("small");
    time.className = "saved-chat-time";
    time.textContent = formatConversationTime(conversation.updatedAt);
    open.append(title, time);
    open.addEventListener("click", () => restoreConversation(conversation.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "saved-chat-delete";
    remove.title = "删除这段本地历史";
    remove.setAttribute("aria-label", `删除对话：${conversation.title}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => deleteConversation(conversation.id));
    entry.append(open, remove);
    savedChatList.appendChild(entry);
  });
}

function text(id, value) {
  const element = document.querySelector(`#${id}`);
  element.textContent = value;
  element.classList.remove("placeholder");
}

function tokenRibbon(id, values) {
  const root = document.querySelector(`#${id}`);
  root.replaceChildren();
  values.slice(-18).forEach((value, index, visible) => {
    const token = document.createElement("span");
    token.textContent = String(value);
    if (index === visible.length - 1) token.classList.add("new-token");
    root.appendChild(token);
  });
}

function renderMapping(items) {
  const body = document.querySelector("#mapping-body");
  body.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement("tr");
    [item.position, item.plain_piece, item.plain_id, item.private_id, item.private_piece].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = String(value).replaceAll("\n", "↵");
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
}

function setState(kind, message) {
  state.className = `run-state ${kind}`;
  stateMessage.textContent = message;
}

function resizePrompt() {
  promptBox.style.height = "auto";
  promptBox.style.height = `${Math.min(promptBox.scrollHeight, 160)}px`;
}

function setTraceVisibility(visible) {
  traceToggle.checked = visible;
  privateTrace.hidden = !visible;
  try {
    window.localStorage.setItem("aloepri-show-private-trace", String(visible));
  } catch (_error) {
    // Privacy trace visibility remains usable without persistent storage.
  }
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function commonPrefix(left, right) {
  let index = 0;
  while (index < left.length && index < right.length && left[index] === right[index]) index += 1;
  return left.slice(0, index);
}

function startTyping() {
  if (typingPromise) return typingPromise;
  const run = async () => {
    while (typedAnswer !== answerTarget) {
      if (!answerTarget.startsWith(typedAnswer)) {
        typedAnswer = commonPrefix(typedAnswer, answerTarget);
      }
      const remaining = Array.from(answerTarget.slice(typedAnswer.length));
      if (remaining.length === 0) continue;
      typedAnswer += remaining[0];
      answerBox.textContent = typedAnswer;
      answerBox.scrollTop = answerBox.scrollHeight;
      await sleep(22);
    }
  };
  // The extra microtask is intentional: when the first decoded token is empty,
  // run() can otherwise finish before typingPromise receives its value, leaving
  // a resolved Promise installed forever and blocking the next submission.
  typingPromise = Promise.resolve()
    .then(run)
    .finally(() => {
      typingPromise = null;
      if (typedAnswer !== answerTarget) startTyping();
    });
  return typingPromise;
}

function queueTypedAnswer(nextText) {
  answerTarget = nextText;
  if (!answerTarget.startsWith(typedAnswer)) {
    typedAnswer = commonPrefix(typedAnswer, answerTarget);
    answerBox.textContent = typedAnswer;
  }
  answerBox.classList.remove("placeholder");
  answerBox.classList.add("streaming");
  return startTyping();
}

async function finishTyping() {
  while (typingPromise || typedAnswer !== answerTarget) {
    if (!typingPromise) startTyping();
    await typingPromise;
  }
  answerBox.classList.remove("streaming");
}

function resetRun() {
  privateOutputIds = [];
  answerTarget = "";
  typedAnswer = "";
  typingPromise = null;
  ["plain-prompt", "private-input-text", "private-output-text", "answer"].forEach((id) => {
    const element = document.querySelector(`#${id}`);
    element.textContent = id === "answer" ? "等待首个模型 token…" : "准备中…";
    element.classList.add("placeholder");
    element.classList.remove("streaming");
  });
  tokenRibbon("private-input-ids", []);
  tokenRibbon("private-output-ids", []);
  text("plain-count", "—");
  text("input-count", "—");
  text("output-count", "0 private IDs");
  text("request-id", "等待 request ID");
  text("ttft", "—");
  text("tpot", "—");
  text("roundtrip", "—");
  text("throughput", "—");
}

function clearConversation() {
  if (submit.disabled) return;
  conversationHistory = [];
  completedTurn = null;
  activePrompt = "";
  currentConversationId = null;
  try {
    window.localStorage.removeItem(ACTIVE_CHAT_STORAGE_KEY);
  } catch (_error) {
    // The chat still works when browser storage is unavailable.
  }
  historyList.replaceChildren();
  document.body.classList.remove("has-run");
  promptBox.value = "";
  resizePrompt();
  setState("idle", "准备就绪");
  text("context-status", "0 条历史消息");
  renderSavedConversations();
  promptBox.focus();
}

function archivedMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "assistant-avatar"}`;
  avatar.textContent = role === "user" ? "你" : "A";
  const body = document.createElement("div");
  body.className = "message-content";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "AloePri";
  const message = document.createElement("pre");
  message.className = "message-text";
  message.textContent = content;
  body.append(label, message);
  article.append(avatar, body);
  return article;
}

function showStoredTraceNotice() {
  text("private-input-text", "本地历史仅保存明文问答，不保存混淆 token。再次提问时会重新执行隐私变换。");
  text("private-output-text", "该回答已在可信浏览器中恢复；原始服务端混淆输出未被持久化。");
  text("input-count", "未保存");
  text("output-count", "未保存");
  text("identity", "浏览器本地历史");
  tokenRibbon("private-input-ids", []);
  tokenRibbon("private-output-ids", []);
  renderMapping([]);
  const mappingBody = document.querySelector("#mapping-body");
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.textContent = "历史会话不持久化密钥、映射或 token ID";
  row.appendChild(cell);
  mappingBody.appendChild(row);
}

function saveCurrentConversation() {
  if (conversationHistory.length < 2) return false;
  const now = new Date().toISOString();
  const existing = savedConversations.find((item) => item.id === currentConversationId);
  if (!currentConversationId) currentConversationId = newConversationId();
  const record = {
    id: currentConversationId,
    title: existing?.title || conversationTitle(conversationHistory),
    createdAt: existing?.createdAt || now,
    updatedAt: now,
    messages: normaliseStoredMessages(conversationHistory),
  };
  const next = [record, ...savedConversations.filter((item) => item.id !== record.id)]
    .slice(0, MAX_SAVED_CONVERSATIONS);
  if (!writeSavedConversations(next)) return false;
  try {
    window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, record.id);
  } catch (_error) {
    // The conversation list is already stored; active selection is optional.
  }
  renderSavedConversations();
  return true;
}

function restoreConversation(conversationId) {
  if (submit.disabled) return;
  const conversation = savedConversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  const messages = normaliseStoredMessages(conversation.messages);
  if (messages.length < 2) return;

  currentConversationId = conversation.id;
  conversationHistory = messages;
  activePrompt = messages.at(-2).content;
  completedTurn = {prompt: activePrompt, answer: messages.at(-1).content};
  historyList.replaceChildren();
  messages.slice(0, -2).forEach((message) => {
    historyList.appendChild(archivedMessage(message.role, message.content));
  });

  resetRun();
  document.body.classList.add("has-run");
  text("plain-prompt", activePrompt);
  text("plain-count", "从浏览器本地历史恢复");
  text("answer", completedTurn.answer);
  text("request-id", "本地历史 · 可继续追问");
  text("context-status", `${messages.length} 条已保存消息`);
  showStoredTraceNotice();
  promptBox.value = "";
  resizePrompt();
  setState("success", "已恢复这段对话，可以从原上下文继续提问。");
  try {
    window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, conversation.id);
  } catch (_error) {
    // Active selection is a convenience only.
  }
  renderSavedConversations();
  promptBox.focus();
  window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"});
}

function deleteConversation(conversationId) {
  if (submit.disabled) return;
  const next = savedConversations.filter((item) => item.id !== conversationId);
  if (!writeSavedConversations(next)) {
    setState("error", "浏览器拒绝更新本地历史，请检查站点存储权限。");
    return;
  }
  if (currentConversationId === conversationId) {
    clearConversation();
    return;
  }
  renderSavedConversations();
}

function archiveCompletedTurn() {
  if (!completedTurn) return;
  historyList.append(
    archivedMessage("user", completedTurn.prompt),
    archivedMessage("assistant", completedTurn.answer),
  );
  completedTurn = null;
}

function renderStart(event) {
  applySecurityMode(event.security_mode || currentSecurityMode);
  text("plain-prompt", event.prompt);
  text("private-input-text", event.private_input_text || "[请查看 token IDs]");
  text("plain-count", `${event.plain_input_ids.length} plain IDs`);
  text(
    "input-count",
    currentSecurityMode === "tee_gm"
      ? `${event.input_tokens} Token · 国密加密传输`
      : `${event.input_tokens} private IDs`,
  );
  text("identity", `${event.model_id} · ${event.key_id}`);
  tokenRibbon("private-input-ids", event.private_input_ids);
  renderMapping(event.input_trace);
  const retainedHistory = Math.max(0, event.context_messages - 1);
  const dropped = event.dropped_history_messages || 0;
  text(
    "context-status",
    dropped
      ? `${retainedHistory} 条历史消息 · 已裁剪 ${dropped} 条`
      : `${retainedHistory} 条历史消息`,
  );
  setState(
    "running",
    currentSecurityMode === "tee_gm"
      ? "加密提示词已进入TEE，GPU主体正在生成私有隐藏状态……"
      : "混淆输入已发送，正在等待模型生成首个私有 token……",
  );
}

function renderToken(event) {
  applySecurityMode(event.security_mode || currentSecurityMode);
  if (currentSecurityMode !== "tee_gm" && event.private_output_id >= 0) {
    privateOutputIds.push(event.private_output_id);
  }
  tokenRibbon("private-output-ids", privateOutputIds);
  text("private-output-text", event.private_output_text || "[该 token 无可视字符]");
  text(
    "output-count",
    currentSecurityMode === "tee_gm"
      ? `${event.output_tokens} Token · 国密加密返回`
      : `${event.output_tokens} private IDs`,
  );
  text("request-id", event.request_id);
  text("ttft", event.ttft_ms.toFixed(1));
  text("tpot", event.tpot_ms.toFixed(1));
  text("throughput", event.output_tokens > 1 ? (1000 / Math.max(event.tpot_ms, 0.001)).toFixed(1) : "—");
  queueTypedAnswer(event.answer);
  setState(
    "running",
    currentSecurityMode === "tee_gm"
      ? `实时生成中：TEE已采样并加密返回 ${event.output_tokens} 个 token。`
      : `实时生成中：已接收并恢复 ${event.output_tokens} 个 token。`,
  );
}

async function renderDone(event) {
  queueTypedAnswer(event.answer);
  await finishTyping();
  text("ttft", event.ttft_ms.toFixed(1));
  text("tpot", event.tpot_ms.toFixed(1));
  text("roundtrip", event.client_roundtrip_ms.toFixed(1));
  text("throughput", event.output_tokens > 1 ? (1000 / Math.max(event.tpot_ms, 0.001)).toFixed(1) : "—");
  completedTurn = {prompt: activePrompt, answer: event.answer};
  conversationHistory.push(
    {role: "user", content: activePrompt},
    {role: "assistant", content: event.answer},
  );
  if (conversationHistory.length > MAX_SAVED_MESSAGES) {
    conversationHistory = conversationHistory.slice(-MAX_SAVED_MESSAGES);
  }
  const saved = saveCurrentConversation();
  setState(
    "success",
    saved
      ? currentSecurityMode === "tee_gm"
        ? `完成：${event.output_tokens} 个Token已由TEE安全生成；对话已保存在本机。`
        : `完成：${event.output_tokens} 个私有 token 已恢复；对话已保存在此浏览器。`
      : currentSecurityMode === "tee_gm"
        ? `完成：${event.output_tokens} 个Token已由TEE安全生成；本机未允许保存历史。`
        : `完成：${event.output_tokens} 个私有 token 已恢复；浏览器未允许保存历史。`,
  );
}

async function consumeEvent(event) {
  if (event.type === "start") renderStart(event);
  else if (event.type === "token") renderToken(event);
  else if (event.type === "done") await renderDone(event);
  else if (event.type === "error") throw new Error(event.detail || "流式推理失败");
}

async function readEventStream(response) {
  if (!response.body) throw new Error("浏览器没有提供流式响应体");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const data = frame.split("\n").find((line) => line.startsWith("data: "));
      if (data) await consumeEvent(JSON.parse(data.slice(6)));
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const data = buffer.split("\n").find((line) => line.startsWith("data: "));
    if (data) await consumeEvent(JSON.parse(data.slice(6)));
  }
}

async function readPollingStream(payload) {
  const created = await fetch("/api/generate/jobs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!created.ok) throw new Error(`${created.status} ${await created.text()}`);
  const job = await created.json();
  setState(
    "running",
    job.queue_position > 1
      ? `公网推理队列：前方还有 ${job.queue_position - 1} 个请求。`
      : "公网安全通道已建立，正在等待模型生成……",
  );
  let after = 0;
  while (true) {
    await sleep(220);
    const response = await fetch(`/api/generate/jobs/${encodeURIComponent(job.job_id)}?after=${after}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
    const update = await response.json();
    for (const event of update.events) await consumeEvent(event);
    after = update.next;
    if (update.done) break;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptBox.value.trim();
  if (!prompt) {
    setState("error", "请输入问题后再运行。");
    return;
  }
  submit.disabled = true;
  newChat.disabled = true;
  archiveCompletedTurn();
  activePrompt = prompt;
  promptBox.value = "";
  resizePrompt();
  document.body.classList.add("has-run");
  resetRun();
  setState(
    "running",
    currentSecurityMode === "tee_gm"
      ? "正在本地分词并建立TEE国密安全通道……"
      : "正在本地应用 Chat Template 并置换输入 token……",
  );
  try {
    const payload = {
      prompt,
      history: conversationHistory.slice(-MAX_CONTEXT_MESSAGES),
      max_new_tokens: Number(document.querySelector("#max-tokens").value),
      temperature: 0.0,
    };
    if (isQuickTunnel) {
      await readPollingStream(payload);
    } else {
      const response = await fetch("/api/generate/stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      await readEventStream(response);
    }
  } catch (error) {
    answerBox.classList.remove("streaming");
    setState("error", `运行失败：${error.message}`);
  } finally {
    submit.disabled = false;
    newChat.disabled = false;
    promptBox.focus();
  }
});

traceToggle.addEventListener("change", () => setTraceVisibility(traceToggle.checked));
newChat.addEventListener("click", clearConversation);

promptBox.addEventListener("input", resizePrompt);
promptBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    promptBox.value = button.dataset.prompt;
    resizePrompt();
    form.requestSubmit();
  });
});

savedConversations = readSavedConversations();
renderSavedConversations();
if (isQuickTunnel) {
  document.querySelector("#boundary-title").textContent = "HTTPS 可信演示网关";
  document.querySelector("#boundary-detail").textContent = "网关接收明文；模型只接收混淆 token";
  document.querySelector("#welcome-boundary").textContent =
    "问题经 HTTPS 进入可信演示网关完成分词和置换，模型服务只会看到混淆 token。";
}
let showPrivateTrace = false;
let activeConversationId = null;
try {
  showPrivateTrace = window.localStorage.getItem("aloepri-show-private-trace") === "true";
  activeConversationId = window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
} catch (_error) {
  // The app still runs when the browser blocks local storage.
}
setTraceVisibility(showPrivateTrace);
applySecurityMode(currentSecurityMode);
if (activeConversationId && savedConversations.some((item) => item.id === activeConversationId)) {
  restoreConversation(activeConversationId);
}
resizePrompt();
