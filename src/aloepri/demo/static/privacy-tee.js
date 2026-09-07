const teeForm = document.querySelector("#tee-lab-form");
const teeSubmit = document.querySelector("#tee-lab-submit");
const teePrompt = document.querySelector("#tee-lab-prompt");
const liveSection = document.querySelector("#run");

const stageEvents = new Map();
let activeStartEvent = null;
let loopCaptured = false;

const stageConfig = {
  entry: {root: "stage-entry", value: "stage-entry-value"},
  embedding: {root: "stage-embedding", value: "stage-embedding-value"},
  body: {root: "stage-body", value: "stage-body-value"},
  head: {root: "stage-head", value: "stage-head-value"},
  loop: {root: "stage-loop", value: "stage-loop-value"},
};

function node(id) {
  return document.querySelector(`#${id}`);
}

function setText(id, value) {
  const target = node(id);
  if (!target) return;
  target.textContent = value;
  target.classList.toggle("placeholder", !value || value.startsWith("等待"));
}

function setStage(name, value) {
  const config = stageConfig[name];
  if (!config) return;
  setText(config.value, value);
  node(config.root)?.classList.add("ready");
  updateProgress();
}

function resetStages() {
  stageEvents.clear();
  activeStartEvent = null;
  loopCaptured = false;
  const empty = {
    entry: "等待运行：这里显示普通 Token 与加密后的提示词。",
    embedding: "等待运行：这里显示 z₀ 的形状和真实数值样本。",
    body: "等待运行：这里显示 GPU 输入 z₀ 与输出 zL。",
    head: "等待运行：这里显示 h × B、u × W 与最终 Token。",
    loop: "等待运行：这里显示一个 Token 的完整返回与下一轮路径。",
  };
  for (const [name, config] of Object.entries(stageConfig)) {
    node(config.root)?.classList.remove("ready");
    setText(config.value, empty[name]);
  }
  setText("tee-encrypted-output", "等待首个 Token");
  setText("tee-answer", "等待模型回答");
  setText("tee-token-count", "—");
  setText("tee-latency", "—");
  node("tee-loop-sample").textContent = "";
  liveSection.classList.remove("complete");
  updateProgress();
}

function updateProgress() {
  const completed = Object.values(stageConfig).filter(config => node(config.root)?.classList.contains("ready")).length;
  setText("runner-progress-text", `${completed} / 5`);
  node("runner-progress-bar").style.width = `${completed * 20}%`;
}

function teeState(message, error = false) {
  const state = node("tee-lab-state");
  state.textContent = message;
  state.classList.toggle("error", error);
}

function compactArray(values, headCount = 5, tailCount = 4) {
  if (!Array.isArray(values) || values.length === 0) return "[]";
  if (values.length <= headCount + tailCount) return `[${values.join(", ")}]`;
  return `[${values.slice(0, headCount).join(", ")}, …, ${values.slice(-tailCount).join(", ")}]`;
}

function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return String(value);
  if (Number.isInteger(Number(value))) return String(value);
  return Number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
}

function tensorSample(tensor) {
  if (!tensor) return "尚未产生";
  const shape = Array.isArray(tensor.shape) ? `[${tensor.shape.join(", ")}]` : "[未知]";
  const head = (tensor.sample_head || []).map(formatNumber);
  const tail = (tensor.sample_tail || []).map(formatNumber);
  const values = head.length
    ? `[${head.join(", ")}${tail.length ? `, …, ${tail.join(", ")}` : ""}]`
    : "本步骤未返回数值切片";
  return `形状 ${shape}\n数值 ${values}`;
}

function shortTensor(tensor) {
  if (!tensor) return "尚未产生";
  const shape = Array.isArray(tensor.shape) ? `[${tensor.shape.join(", ")}]` : "[未知]";
  const head = (tensor.sample_head || []).slice(0, 3).map(formatNumber);
  const tail = (tensor.sample_tail || []).slice(-3).map(formatNumber);
  const values = head.length ? `[${head.join(", ")}${tail.length ? `, …, ${tail.join(", ")}` : ""}]` : "";
  return `${shape}  ${values}`.trim();
}

function cipherSummary(record) {
  if (!record) return "加密记录尚未产生";
  const lines = String(record).split(/\r?\n/);
  const nonce = lines.find(line => line.toLowerCase().startsWith("nonce:"));
  const cipher = lines.find(line => line.toLowerCase().startsWith("ciphertext:"));
  const tag = lines.find(line => line.toLowerCase().startsWith("tag:"));
  const shortened = cipher && cipher.length > 150 ? `${cipher.slice(0, 150)}…` : cipher;
  return [nonce, shortened, tag].filter(Boolean).join("\n") || "已生成加密记录";
}

function rememberTrace(trace = []) {
  for (const event of trace) {
    // The presentation shows one coherent sample: the first completed path.
    // Later decode events must not overwrite its Embedding, body or Head values.
    if (event?.stage && !stageEvents.has(event.stage)) stageEvents.set(event.stage, event);
  }
  renderCoreStages();
}

function evidence(stage) {
  return stageEvents.get(stage)?.evidence || {};
}

function renderCoreStages() {
  if (!activeStartEvent) return;

  const encrypt = evidence("client_encrypt_request");
  setStage("entry", [
    `普通 Token：${compactArray(activeStartEvent.plain_input_ids)}`,
    `共 ${activeStartEvent.input_tokens ?? activeStartEvent.plain_input_ids?.length ?? "—"} 个 Token`,
    `网络只见密文：${encrypt.ciphertext_bytes ?? "—"} 字节`,
    cipherSummary(activeStartEvent.private_input_text),
  ].join("\n"));

  const embedding = evidence("tee_embedding_lookup");
  if (embedding.private_embedding) {
    setStage("embedding", [
      `TEE 解密得到普通 Token：${compactArray(activeStartEvent.plain_input_ids)}`,
      "z₀ = (E · P)[Token]",
      `输出给 GPU：${tensorSample(embedding.private_embedding)}`,
    ].join("\n"));
  }

  const body = evidence("gpu_transformer_body");
  if (body.input && body.last_hidden_state) {
    setStage("body", [
      `GPU 输入 z₀：${shortTensor(body.input)}`,
      `经过 ${body.model_layers ?? "—"} 层 Transformer`,
      `GPU 输出 zL：${shortTensor(body.last_hidden_state)}`,
      `KV Cache：${body.kv_cache?.sequence_length ?? "—"} Token`,
    ].join("\n"));
  }

  const hidden = evidence("tee_recover_hidden").plain_hidden;
  const coordinate = evidence("masked_head_coordinate");
  const outbound = evidence("masked_head_outbound").u;
  const worker = evidence("masked_head_worker");
  const check = evidence("masked_head_freivalds");
  const sampling = evidence("tee_sampling");
  if (hidden && outbound && worker.y && sampling.token_id !== undefined) {
    const hWidth = hidden.shape?.at(-1) ?? "—";
    const uWidth = outbound.shape?.at(-1) ?? "—";
    const vocab = worker.y.shape?.at(-1) ?? "—";
    setStage("head", [
      `TEE 恢复：h = zL × QL  →  [1 × ${hWidth}]`,
      `TEE 换坐标并加掩码：u = Quantize(h × B) + ρ  →  [1 × ${uWidth}]`,
      `GPU 外包计算：y = u × WB,qᵀ  →  [1 × ${vocab}]`,
      `TEE 去掩码并校验：${check.verified ? `Freivalds ${check.rounds ?? 2} 轮通过` : "等待校验"}`,
      `TEE 采样普通 Token ID：${sampling.token_id}`,
    ].join("\n"));
  } else if (coordinate.hidden && outbound) {
    node("runner-current-stage").textContent = "LM Head 正在执行外包矩阵计算";
  }
}

function renderLoopSample(event) {
  if (loopCaptured) return;
  loopCaptured = true;
  const tokenId = event.recovered_output_id;
  const route = event.head_route || {};
  const text = [
    `TEE 得到普通 Token ID：${tokenId}`,
    `加密返回客户端：${cipherSummary(event.private_output_text).split("\n")[0] || "已生成密文"}`,
    `客户端解密并显示：${event.answer || "正在解码"}`,
    `下一轮：Token ${tokenId} → TEE Embedding → znext`,
    `znext + KV Cache → GPU 模型主体 → 下一个 Token`,
    `本次 Head：${route.head_used_fallback ? "TEE 精确回退" : "安全外包通过"}`,
  ].join("\n");
  setStage("loop", text);
  node("tee-loop-sample").textContent = text;
}

function stageMessage(event) {
  const labels = {
    client_encrypt_request: "客户端正在发送加密 Token",
    tee_embedding_lookup: "TEE 正在查询私有 Embedding",
    gpu_transformer_body: "GPU 正在运行混淆模型主体",
    masked_head_worker: "GPU 正在执行 LM Head 大矩阵乘法",
    masked_head_freivalds: "TEE 正在校验外包结果",
    tee_sampling: "TEE 正在采样普通 Token",
  };
  return labels[event.stage] || event.title || "正在执行隐私推理";
}

async function loadActiveDeployment() {
  const label = node("tee-active-deployment");
  try {
    const response = await fetch("/api/desktop/deployments");
    if (!response.ok) throw new Error("desktop API unavailable");
    const deployments = await response.json();
    const selectedId = window.localStorage.getItem("yinbian-selected-deployment");
    const selected = deployments.find(item => item.deployment_id === selectedId)
      || deployments.find(item => item.status === "HEALTHY" && item.metadata?.security_mode === "tee_gm");
    if (!selected) {
      label.textContent = "尚未选择健康的 TEE 部署";
      return;
    }
    const target = selected.metadata?.target_type === "local" ? "本机" : "远程";
    label.textContent = `TEE国密 · ${target} · ${selected.model_id} · ${selected.status}`;
  } catch (_error) {
    label.textContent = "当前真实推理演示";
  }
}

async function consumeTeeStream(response) {
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.split("\n").find(item => item.startsWith("data: "));
      if (!line) continue;
      const event = JSON.parse(line.slice(6));
      if (event.type === "start") {
        if (event.security_mode !== "tee_gm") throw new Error("当前不是TEE国密部署，请先在对话页切换模型");
        activeStartEvent = event;
        rememberTrace(event.execution_trace || []);
        teeState("安全通道已建立，模型正在生成回答");
        node("runner-current-stage").textContent = "首个 Token 已完成完整隐私推理路径";
      } else if (event.type === "trace") {
        rememberTrace([event]);
        node("runner-current-stage").textContent = stageMessage(event);
      } else if (event.type === "token") {
        rememberTrace(event.execution_trace || []);
        setText("tee-encrypted-output", cipherSummary(event.private_output_text));
        setText("tee-answer", event.answer || "");
        setText("tee-token-count", `${event.output_tokens} Token`);
        setText("tee-latency", `TTFT ${Number(event.ttft_ms).toFixed(1)} ms · TPOT ${Number(event.tpot_ms).toFixed(1)} ms`);
        renderLoopSample(event);
        teeState(`已安全返回 ${event.output_tokens} 个 Token`);
      } else if (event.type === "done") {
        setText("tee-answer", event.answer || "");
        teeState("完成：回答已经通过加密通道返回客户端");
        node("runner-current-stage").textContent = "五个关键数据均来自本次运行";
        liveSection.classList.remove("running");
        liveSection.classList.add("complete");
      } else if (event.type === "error") {
        throw new Error(event.detail || "TEE推理失败");
      }
    }
    if (done) break;
  }
}

teeForm.addEventListener("submit", async event => {
  event.preventDefault();
  const prompt = teePrompt.value.trim();
  if (!prompt) return;
  teeSubmit.disabled = true;
  teeSubmit.textContent = "正在推理…";
  resetStages();
  liveSection.classList.add("running");
  teeState("正在建立安全通道");
  node("runner-current-stage").textContent = "准备本次真实请求";
  try {
    const response = await fetch("/api/generate/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt, history: [], max_new_tokens: 24, temperature: 0.0}),
    });
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
    await consumeTeeStream(response);
  } catch (error) {
    teeState(`运行失败：${error.message}`, true);
    node("runner-current-stage").textContent = "本次运行已停止";
    liveSection.classList.remove("running");
  } finally {
    teeSubmit.disabled = false;
    teeSubmit.textContent = "重新运行";
  }
});

const indexedSections = [...document.querySelectorAll("#overview, #tee-route, #run, #security")];
const indexLinks = [...document.querySelectorAll(".story-link")];
if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    indexLinks.forEach(link => link.classList.toggle("active", link.hash === `#${visible.target.id}`));
  }, {rootMargin: "-20% 0px -65%", threshold: [0, .2, .5]});
  indexedSections.forEach(section => sectionObserver.observe(section));
}

resetStages();
loadActiveDeployment();
