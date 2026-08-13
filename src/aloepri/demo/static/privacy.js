const tabs = document.querySelectorAll("[data-step]");
const panels = document.querySelectorAll("[data-panel]");
const labForm = document.querySelector("#lab-form");
const labPrompt = document.querySelector("#lab-prompt");
const labSubmit = document.querySelector("#lab-submit");
const labStatus = document.querySelector("#lab-status");
const labStatusText = labStatus.querySelector("span");
const isQuickTunnel = window.location.hostname.endsWith(".trycloudflare.com");

let outputIds = [];
let recoveredIds = [];

if (isQuickTunnel) document.querySelector("#public-boundary-notice").hidden = false;

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const selected = tab.dataset.step;
    tabs.forEach((item) => item.classList.toggle("active", item === tab));
    panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === selected));
  });
});

function setText(id, value) {
  const element = document.querySelector(`#${id}`);
  element.textContent = value;
  element.classList.remove("placeholder");
}

function setStatus(kind, message) {
  labStatus.className = `lab-status ${kind}`;
  labStatusText.textContent = message;
}

function setStage(name, state) {
  const stage = document.querySelector(`[data-live-stage="${name}"]`);
  stage.classList.remove("active", "complete");
  if (state) stage.classList.add(state);
}

function tokenChips(id, values, {animateLast = false} = {}) {
  const root = document.querySelector(`#${id}`);
  root.replaceChildren();
  values.slice(0, 96).forEach((value, index, visible) => {
    const chip = document.createElement("span");
    chip.textContent = String(value);
    if (animateLast && index === visible.length - 1) chip.classList.add("arrive");
    root.appendChild(chip);
  });
  if (values.length > 96) {
    const more = document.createElement("span");
    more.textContent = `+${values.length - 96}`;
    root.appendChild(more);
  }
}

function renderMapping(items) {
  const root = document.querySelector("#lab-mapping");
  root.replaceChildren();
  items.slice(0, 64).forEach((item) => {
    const row = document.createElement("div");
    row.className = "map-row";
    const values = [
      `#${item.position}`,
      item.plain_piece.replaceAll("\n", "↵") || "∅",
      item.plain_id,
      "→",
      item.private_id,
      item.private_piece.replaceAll("\n", "↵") || "∅",
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 3 ? "b" : "span");
      if (index === 1 || index === 5) cell.className = "map-piece";
      cell.textContent = String(value);
      row.appendChild(cell);
    });
    root.appendChild(row);
  });
  if (items.length > 64) {
    const note = document.createElement("p");
    note.textContent = `为了便于阅读，仅展示前 64 个映射；本次共有 ${items.length} 个 token。`;
    root.appendChild(note);
  }
}

function resetLab(prompt) {
  outputIds = [];
  recoveredIds = [];
  ["plain", "tokenize", "obfuscate", "server", "restore", "answer"].forEach((name) => setStage(name, null));
  setStage("plain", "active");
  setText("lab-plain-text", prompt);
  ["lab-private-text", "lab-output-text", "lab-answer"].forEach((id) => {
    const element = document.querySelector(`#${id}`);
    element.textContent = "等待上一步完成";
    element.classList.add("placeholder");
    element.classList.remove("streaming");
  });
  ["lab-plain-ids", "lab-private-ids", "lab-output-ids"].forEach((id) => document.querySelector(`#${id}`).replaceChildren());
  setText("lab-plain-count", "正在本地分词");
  setText("lab-private-count", "—");
  setText("lab-output-count", "—");
  setText("lab-request-id", "—");
  setText("lab-ttft", "—");
  setText("lab-tpot", "—");
  document.querySelector("#lab-payload").textContent = "等待生成私有请求载荷";
  document.querySelector("#lab-mapping").innerHTML = "<p>等待执行 τ</p>";
}

function renderStart(event) {
  setStage("plain", "complete");
  setStage("tokenize", "complete");
  setStage("obfuscate", "complete");
  setStage("server", "active");
  tokenChips("lab-plain-ids", event.plain_input_ids);
  tokenChips("lab-private-ids", event.private_input_ids);
  setText("lab-private-text", event.private_input_text || "[私有 ID 没有可视字符]");
  setText("lab-plain-count", `${event.plain_input_ids.length} plain IDs`);
  setText("lab-private-count", `${event.private_input_ids.length} private IDs`);
  renderMapping(event.input_trace);
  const payload = {
    model_id: event.model_id,
    key_id: event.key_id,
    input_ids: event.private_input_ids,
    max_new_tokens: Number(document.querySelector("#lab-max-tokens").value),
    temperature: 0.0,
    top_k: 0,
    top_p: 1.0,
  };
  document.querySelector("#lab-payload").textContent = JSON.stringify(payload, null, 2);
  setStatus("running", "私有请求已发送，等待模型首 token");
}

function renderToken(event) {
  outputIds.push(event.private_output_id);
  recoveredIds.push(event.recovered_output_id);
  setStage("server", "complete");
  setStage("restore", "active");
  setStage("answer", "active");
  tokenChips("lab-output-ids", outputIds, {animateLast: true});
  setText("lab-output-text", event.private_output_text || "[当前私有 token 没有可视字符]");
  setText("lab-output-count", `${event.output_tokens} private IDs`);
  setText("lab-request-id", event.request_id);
  setText("lab-ttft", event.ttft_ms.toFixed(1));
  setText("lab-tpot", event.tpot_ms.toFixed(1));
  setText("lab-answer", event.answer || "[等待可见字符]");
  document.querySelector("#lab-answer").classList.add("streaming");
  setStatus("running", `已接收 ${event.output_tokens} 个私有 token，正在本地恢复`);
}

function renderDone(event) {
  setStage("restore", "complete");
  setStage("answer", "complete");
  setText("lab-answer", event.answer || "[模型返回空回答]");
  document.querySelector("#lab-answer").classList.remove("streaming");
  setText("lab-ttft", event.ttft_ms.toFixed(1));
  setText("lab-tpot", event.tpot_ms.toFixed(1));
  setStatus("done", `完成 · ${event.output_tokens} 个 token · ${event.client_roundtrip_ms.toFixed(0)} ms`);
}

async function consumeEvent(event) {
  if (event.type === "start") renderStart(event);
  else if (event.type === "token") renderToken(event);
  else if (event.type === "done") renderDone(event);
  else if (event.type === "error") throw new Error(event.detail || "隐私推理失败");
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
      const line = frame.split("\n").find((item) => item.startsWith("data: "));
      if (line) await consumeEvent(JSON.parse(line.slice(6)));
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const line = buffer.split("\n").find((item) => item.startsWith("data: "));
    if (line) await consumeEvent(JSON.parse(line.slice(6)));
  }
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function readPollingStream(payload) {
  const created = await fetch("/api/generate/jobs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!created.ok) throw new Error(`${created.status} ${await created.text()}`);
  const job = await created.json();
  setStatus(
    "running",
    job.queue_position > 1 ? `公网队列前方 ${job.queue_position - 1} 个请求` : "公网任务已进入本地模型",
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

labForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = labPrompt.value.trim();
  if (!prompt) {
    setStatus("error", "请先输入一句话");
    labPrompt.focus();
    return;
  }
  labSubmit.disabled = true;
  resetLab(prompt);
  setStatus("running", "正在可信客户端应用 Chat Template");
  try {
    const payload = {
      prompt,
      history: [],
      max_new_tokens: Number(document.querySelector("#lab-max-tokens").value),
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
    setStatus("error", `运行失败：${error.message}`);
    document.querySelector("#lab-answer").classList.remove("streaming");
  } finally {
    labSubmit.disabled = false;
    labPrompt.focus();
  }
});

labPrompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    labForm.requestSubmit();
  }
});
