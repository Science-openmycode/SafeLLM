const pages = {
  dashboard: ["部署总览", "选择模型、完成本地改造并部署到您的服务器"],
  models: ["模型目录", "推荐模型和经过结构识别的高级模型"],
  jobs: ["转换任务", "查看下载、改造、校验和上传进度"],
  servers: ["服务器", "保存并检查您的 Ubuntu GPU 服务器"],
  deployments: ["已部署模型", "选择健康模型启动对话，或管理模型服务"],
  keys: ["密钥与备份", "管理本地在线密钥和可迁移备份"],
};
const sessionServerSecrets = new Map();
let catalogModels = [];
let modelCatalogRendered = false;
let activeCatalogFamily = "";
let refreshing = false;
let editingServerId = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(value) {
  if (!value) return "未知体积";
  const gib = Number(value) / (1024 ** 3);
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
}

function estimateLocalPreparationBytes(model) {
  const source = Number(model?.expected_bytes || 0);
  if (!source) return 0;
  const ratio = Number(model?.conversion?.estimated_output_ratio || 1.15);
  const tile = Number(model?.conversion?.tile_mib || 256) * 1024 ** 2;
  return Math.ceil((source + source * ratio + tile) * 1.2);
}

function estimateExistingModelPreparationBytes(model) {
  const source = Number(model?.expected_bytes || 0);
  if (!source) return 0;
  const ratio = Number(model?.conversion?.estimated_output_ratio || 1.15);
  const tile = Number(model?.conversion?.tile_mib || 256) * 1024 ** 2;
  return Math.ceil((source * ratio + tile) * 1.2);
}

function explainDeploymentError(message) {
  const text = String(message || "");
  if (text.includes("server disk is insufficient before upload")) {
    const required = Number(text.match(/required=(\d+)/)?.[1] || 0);
    const free = Number(text.match(/free=(\d+)/)?.[1] || 0);
    return `部署器错误地按全新部署重复计算了已上传文件。当前服务器可用 ${formatBytes(free)}，旧预检要求 ${formatBytes(required)}；新版会只计算尚未上传的字节和未安装的运行环境。请重启部署程序后点击“继续部署”。`;
  }
  if (text.includes("libtorchaudio") || text.includes("import torchaudio")) {
    return "服务器预装的torchaudio与独立PyTorch环境冲突。新版模型服务使用隔离Python模式，只加载隐变智模自己的依赖，不再读取服务器预装的视觉或音频组件。请重启部署程序后点击“继续部署”。";
  }
  if (text.includes("torchvision::nms") || text.includes("DeepseekV3ForCausalLM")) {
    return "服务器预装的torchvision与独立PyTorch运行环境冲突。新版已固定兼容的torchvision 0.20.1 + CUDA 12.1，并会在启动模型前完成一致性检查。请重启部署程序后点击“继续部署”。";
  }
  if (text.includes("System has not been booted with systemd")) {
    return "该GPU租赁环境是容器，不提供systemd。部署器将改用Python原生运行方式，请再次点击“安装运行环境并继续部署”。";
  }
  if (text.includes("ensurepip") || text.includes("native runtime installation failed at python3")) {
    return "该GPU租赁镜像不支持标准Python venv。部署器已切换为独立依赖目录模式，请再次点击“安装运行环境并继续部署”。";
  }
  if (text.includes("NVIDIA driver on your system is too old") || text.includes("CUDA initialization")) {
    return "推理依赖安装了与服务器驱动不兼容的PyTorch。部署器已固定为PyTorch 2.5.1 + CUDA 12.1，请再次点击“安装运行环境并继续部署”。";
  }
  if (text.includes("Docker is not installed") || text.includes("NVIDIA Container Toolkit is not configured")) {
    return "服务器尚未安装Docker或尚未配置NVIDIA Container Toolkit。请点击“安装运行环境并继续部署”。";
  }
  return text;
}

function latestServerOperation(serverId) {
  return [...(window.dashboard?.server_operations || [])].reverse().find(
    (operation) => operation.server_id === serverId,
  );
}

function latestJobOperation(jobId) {
  return [...(window.dashboard?.server_operations || [])].reverse().find(
    (operation) => operation.job_id === jobId,
  );
}

function operationProgress(operation) {
  if (!operation) return "";
  const percent = Math.max(0, Math.min(100, Number(operation.percent || 0)));
  const statusText = operation.status === "FAILED"
    ? `失败：${escapeHtml(explainDeploymentError(operation.error || operation.message))}`
    : escapeHtml(operation.message || "正在处理");
  return `
    <div class="server-operation ${operation.status.toLowerCase()}">
      <div class="operation-head"><strong>${statusText}</strong><span>${percent.toFixed(0)}%</span></div>
      <div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${percent}">
        <div class="progress-fill" style="width:${percent}%"></div>
      </div>
      <small>${escapeHtml(operation.stage)}</small>
    </div>`;
}

function jobProgress(job) {
  const progress = job.progress || {};
  const completed = job.status === "COMPLETED";
  const byteTotal = Number(progress.bytes_total || 0);
  const byteDone = Number(progress.bytes_completed || 0);
  const conversion = progress.conversion || {};
  let percent = null;
  let label = "";

  if (completed) {
    percent = 100;
    label = progress.download_skipped
      ? "本地已有模型的改造和校验已完成"
      : "本地下载、改造和校验已完成";
  } else if (byteTotal > 0) {
    percent = Math.max(0, Math.min(100, 100 * byteDone / byteTotal));
    label = `正在下载 ${escapeHtml(progress.item || "模型权重")} · ${formatBytes(byteDone)} / ${formatBytes(byteTotal)}`;
  } else if (job.phase === "CONVERTING") {
    const measured = Number(conversion.percent);
    if (Number.isFinite(measured)) percent = Math.max(0, Math.min(99, measured));
    const tensor = conversion.tensor || progress.item;
    const tile = Number.isFinite(Number(conversion.tile)) ? ` · tile ${conversion.tile}` : "";
    label = `下载已完成，正在改造模型权重${tensor ? ` · ${escapeHtml(tensor)}` : ""}${tile}`;
  } else {
    const labels = {
      METADATA: "正在读取模型元数据",
      SOURCE_VERIFY: progress.source?.download_skipped
        ? "已跳过下载，正在校验本地模型"
        : "下载完成，正在校验并规范化源模型",
      PRIVATE_VERIFY: "改造完成，正在校验私有模型",
      UPLOADING: "正在上传私有模型",
      REMOTE_VERIFY: "正在校验远端文件",
      FINALIZING: "正在生成最终清单",
    };
    label = labels[job.phase] || progress.item || "正在处理";
  }

  const determinate = percent !== null;
  return `
    <div class="job-progress">
      <div class="operation-head"><strong>${label}</strong><span>${determinate ? `${percent.toFixed(1)}%` : "处理中"}</span></div>
      <div class="progress-track${determinate ? "" : " indeterminate"}" role="progressbar" aria-valuemin="0" aria-valuemax="100" ${determinate ? `aria-valuenow="${percent}"` : `aria-valuetext="${escapeHtml(label)}"`}>
        <div class="progress-fill" ${determinate ? `style="width:${percent}%"` : ""}></div>
      </div>
    </div>`;
}

function parseSshCommand(command) {
  const source = String(command || "").trim();
  if (!source) return null;
  const match = source.match(/^ssh(?:\s+-p\s+(\d+))?\s+([^\s@]+)@([^\s]+)$/i);
  if (!match) throw new Error("SSH命令格式应为：ssh -p 51838 root@gpu.example.com");
  return { port: Number(match[1] || 22), username: match[2], host: match[3] };
}

function modelCanRun(model) {
  return model.conversion_ready === true;
}

function groupedModels(models) {
  const families = new Map();
  [...models]
    .sort((left, right) => {
      const familyOrder = String(left.family_name || left.family_id || "其他模型")
        .localeCompare(String(right.family_name || right.family_id || "其他模型"), "zh-CN");
      if (familyOrder !== 0) return familyOrder;
      return Number(left.expected_bytes || Number.MAX_SAFE_INTEGER) - Number(right.expected_bytes || Number.MAX_SAFE_INTEGER);
    })
    .forEach((model) => {
      const family = model.family_name || model.family_id || "其他模型";
      if (!families.has(family)) families.set(family, []);
      families.get(family).push(model);
    });
  return families;
}

function renderFamilyOptions(models, selectedFamily = "") {
  const families = groupedModels(models);
  const options = ['<option value="">请选择模型族</option>'];
  for (const [family, entries] of families.entries()) {
    const runnable = entries.filter(modelCanRun).length;
    options.push(`<option value="${escapeHtml(family)}">${escapeHtml(family)} · ${entries.length} 个型号 · ${runnable} 个可用</option>`);
  }
  const select = document.querySelector("#wizard-family");
  select.innerHTML = options.join("");
  if (families.has(selectedFamily)) select.value = selectedFamily;
}

function renderModelOptions(models, family, selectedModel = "") {
  const select = document.querySelector("#wizard-model");
  const entries = groupedModels(models).get(family) || [];
  if (!family) {
    select.innerHTML = '<option value="">请先选择模型族</option>';
    select.disabled = true;
    return;
  }
  const rows = entries.map((model) => {
    const ready = modelCanRun(model);
    const state = model.deployment_ready ? "可部署" : ready ? "可转换" : "开发中";
    return `<option value="${escapeHtml(model.catalog_id)}" ${ready ? "" : "disabled"}>${escapeHtml(model.display_name)} · ${escapeHtml(model.parameter_summary)} · ${formatBytes(model.expected_bytes)} · ${state}</option>`;
  });
  select.innerHTML = ['<option value="">请选择参数量与版本</option>', ...rows].join("");
  select.disabled = false;
  if (entries.some((model) => model.catalog_id === selectedModel && modelCanRun(model))) {
    select.value = selectedModel;
  }
}

function renderModelCatalog(models, requestedFamily = activeCatalogFamily) {
  const families = groupedModels(models);
  const familyNames = [...families.keys()];
  if (!familyNames.length) return '<div class="empty">暂无可展示的模型</div>';

  activeCatalogFamily = families.has(requestedFamily) ? requestedFamily : familyNames[0];
  const entries = families.get(activeCatalogFamily) || [];
  const runnable = entries.filter(modelCanRun).length;
  const familyTabs = [...families.entries()].map(([family, familyEntries]) => {
    const selected = family === activeCatalogFamily;
    return `
      <button class="family-tab${selected ? " active" : ""}" type="button" role="tab"
        aria-selected="${selected}" data-catalog-family="${escapeHtml(family)}">
        <span>${escapeHtml(family)}</span><small>${familyEntries.length} 个型号</small>
      </button>`;
  }).join("");
  const cards = entries.map((model) => `
    <article class="card model-card">
      <div class="model-card-head"><div><small>${escapeHtml(model.parameter_summary)}</small><h3>${escapeHtml(model.display_name)}</h3></div><strong>${formatBytes(model.expected_bytes)}</strong></div>
      <p>${escapeHtml(model.adapter_id)} · ${escapeHtml(model.license)}<br>${escapeHtml(model.support_note)}</p>
      <div class="model-card-actions">
        <span class="badge ${escapeHtml(model.status)}">${model.deployment_ready ? "可转换、可部署" : model.conversion_ready ? "可转换" : "结构已识别"}</span>
        ${modelCanRun(model) ? `<button class="primary model-deploy" data-family="${escapeHtml(activeCatalogFamily)}" data-model="${escapeHtml(model.catalog_id)}">选择这个型号</button>` : ""}
      </div>
    </article>`).join("");
  return `
    <div class="family-tabs" role="tablist" aria-label="模型族">${familyTabs}</div>
    <div class="family-catalog-head">
      <div><small>当前模型族</small><h3>${escapeHtml(activeCatalogFamily)}</h3></div>
      <span>${entries.length} 个型号 · ${runnable} 个可用</span>
    </div>
    <div class="family-model-grid" role="tabpanel">${cards}</div>`;
}

function updateModelNote() {
  const selected = catalogModels.find((item) => item.catalog_id === document.querySelector("#wizard-model").value);
  const note = document.querySelector("#wizard-model-note");
  if (!selected) {
    note.className = "model-note full";
    note.textContent = "请选择模型。已实测与同架构可执行模型会分别标识。";
    return;
  }
  const ram = selected.conversion?.minimum_host_ram_gib;
  const verified = selected.status === "supported";
  const localExisting = document.querySelector("#wizard-source-mode")?.value === "local-existing";
  const requiredDisk = localExisting
    ? estimateExistingModelPreparationBytes(selected)
    : estimateLocalPreparationBytes(selected);
  const freeDisk = Number(window.dashboard?.resources?.disk_free_bytes || 0);
  const diskText = requiredDisk
    ? `；本地准备峰值约 ${formatBytes(requiredDisk)}${freeDisk ? `，当前目录可用 ${formatBytes(freeDisk)}` : ""}`
    : "";
  note.className = `model-note full${verified ? "" : " warning"}`;
  const sourceText = localExisting ? "；将跳过模型下载" : "";
  note.textContent = `${selected.display_name}：${selected.support_note || "执行前会重新检查checkpoint"}${sourceText}${ram ? `；建议主机内存至少 ${ram} GiB` : ""}${diskText}。`;
}

function updateModelSource() {
  const localExisting = document.querySelector("#wizard-source-mode").value === "local-existing";
  const field = document.querySelector("#wizard-local-path-field");
  const input = document.querySelector("#wizard-local-model-path");
  const check = document.querySelector("#wizard-local-check");
  const endpoint = document.querySelector("#wizard-download-endpoint");
  field.hidden = !localExisting;
  input.disabled = !localExisting;
  input.required = localExisting;
  check.hidden = !localExisting;
  endpoint.hidden = localExisting;
  endpoint.disabled = localExisting;
  if (!localExisting) {
    check.textContent = "请选择包含 config.json、Tokenizer和Safetensors权重的模型目录；系统会跳过下载并先检查架构。";
    check.className = "model-note full";
  }
  updateModelNote();
}

function updateDeploymentMode() {
  const direct = document.querySelector("#wizard-mode").value === "direct-deploy";
  const server = document.querySelector("#wizard-server");
  const password = document.querySelector("#wizard-password");
  server.hidden = !direct;
  server.disabled = !direct;
  password.hidden = !direct;
  password.disabled = !direct;
  if (!direct) {
    server.value = "";
    password.value = "";
  }
  updateModelNote();
}

function updateSecurityMode() {
  const tee = document.querySelector("#wizard-security-mode").value === "tee_gm";
  const field = document.querySelector("#wizard-tee-backend-field");
  const backend = document.querySelector("#wizard-tee-backend");
  const note = document.querySelector("#wizard-tee-note");
  field.hidden = !tee;
  backend.disabled = !tee;
  note.hidden = !tee;
  if (!tee) backend.value = "intel_tdx";
}

function toast(message) {
  const node = document.querySelector("#toast");
  node.textContent = message;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 3000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || response.statusText);
  }
  return response.json();
}

function switchPage(name) {
  document.querySelectorAll(".page").forEach((node) => node.classList.toggle("active", node.id === name));
  document.querySelectorAll("nav button").forEach((node) => node.classList.toggle("active", node.dataset.page === name));
  document.querySelector("#title").textContent = pages[name][0];
  document.querySelector("#subtitle").textContent = pages[name][1];
}

function jobRows(jobs) {
  if (!jobs.length) return '<div class="empty">暂无任务</div>';
  return jobs.map((job) => {
    const operation = latestJobOperation(job.job_id);
    const mode = job.plan?.conversion?.output?.deployment_mode;
    const modeLabel = mode === "local-deploy"
      ? "改造后自动部署到本机"
      : mode === "direct-deploy"
      ? "改造后自动部署到云服务器"
      : "仅改造并保存在本地";
    return `
    <article class="card">
      <h3>${job.job_id}</h3>
      <p>${job.status} · ${job.phase}<br>${modeLabel}</p>
      ${job.error?.message ? `<div class="job-error"><b>失败原因</b><br>${escapeHtml(job.error.message)}</div>` : `<span class="badge ${["CANCELLED", "COMPLETED"].includes(job.status) ? "terminal" : ""}">${escapeHtml(job.status === "CANCELLED" ? "已取消" : job.status === "COMPLETED" ? "已完成" : job.status === "PAUSED" ? "已暂停" : job.progress?.bytes_total ? `${job.progress.item} · ${(100 * job.progress.bytes_completed / job.progress.bytes_total).toFixed(1)}% · ${formatBytes(job.progress.bytes_completed)} / ${formatBytes(job.progress.bytes_total)}` : job.progress?.item || "正在初始化")}</span>`}
      ${job.status === "CANCELLED" || job.status === "FAILED" ? "" : jobProgress(job)}
      ${operationProgress(operation)}
      <div class="card-actions">
        ${job.status === "RUNNING" ? `<button data-job-action="pause" data-id="${job.job_id}">暂停</button>` : ""}
        ${["PAUSED", "FAILED"].includes(job.status) ? `<button data-job-action="resume" data-id="${job.job_id}">恢复</button>` : ""}
        ${!["COMPLETED", "CANCELLED"].includes(job.status) ? `<button data-job-action="cancel" data-id="${job.job_id}">取消</button>` : ""}
        ${job.status === "COMPLETED" && job.plan?.conversion?.output?.deployment_mode === "direct-deploy" ? `<button class="primary" data-job-deploy="${job.job_id}">继续部署</button>` : ""}
        ${job.status === "COMPLETED" && job.plan?.conversion?.output?.deployment_mode === "local-only" ? `<button class="primary" data-machine-job-deploy="${job.job_id}">部署到本机</button><button data-local-job-deploy="${job.job_id}">选择服务器部署</button>` : ""}
      </div>
    </article>`;
  }).join("");
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const dashboard = await api("/api/dashboard");
    const models = catalogModels.length ? catalogModels : await api("/api/models");
    window.dashboard = dashboard;
    catalogModels = models;
    const destination = document.querySelector("#wizard-destination");
    if (destination && !destination.value) {
      destination.placeholder = `自动保存到 ${dashboard.paths.local_private}`;
    }
    const activeJobs = dashboard.jobs.filter((job) => !["CANCELLED", "COMPLETED"].includes(job.status));
    const terminalJobs = dashboard.jobs.filter((job) => ["CANCELLED", "COMPLETED"].includes(job.status));
    if (!modelCatalogRendered) {
      const selectedFamily = document.querySelector("#wizard-family").value;
      const selectedModel = document.querySelector("#wizard-model").value;
      renderFamilyOptions(models, selectedFamily);
      renderModelOptions(models, document.querySelector("#wizard-family").value, selectedModel);
      document.querySelector("#model-grid").innerHTML = renderModelCatalog(models);
      modelCatalogRendered = true;
    }
    updateModelNote();
    const metrics = [
      ["可用模型", dashboard.models],
      ["当前任务", activeJobs.length],
      ["服务器", dashboard.servers.length],
      ["健康部署", dashboard.deployments.filter((item) => item.status === "HEALTHY").length],
      ["本地可用空间", formatBytes(dashboard.resources.disk_free_bytes)],
    ];
    document.querySelector("#metrics").innerHTML = metrics.map((item) => `<div class="metric"><small>${item[0]}</small><strong>${item[1]}</strong></div>`).join("");
    document.querySelector("#recent-jobs").innerHTML = activeJobs.length ? jobRows(activeJobs.slice(0, 4)) : '<div class="empty">暂无需要处理的任务，请点击“新建任务”开始</div>';
    document.querySelector("#job-list").innerHTML = activeJobs.length ? jobRows(activeJobs) : '<div class="empty">暂无进行中的任务</div>';
    document.querySelector("#terminal-job-list").innerHTML = terminalJobs.length ? jobRows(terminalJobs) : '<div class="empty">暂无已结束任务</div>';
    document.querySelector("#wizard-server").innerHTML = '<option value="">请选择服务器</option>' + dashboard.servers.map((server) => `<option value="${escapeHtml(server.server_id)}">${escapeHtml(server.display_name)} · ${escapeHtml(server.username)}@${escapeHtml(server.host)}:${server.port}</option>`).join("");
    document.querySelector("#server-list").innerHTML = dashboard.servers.length ? dashboard.servers.map((server) => {
      const operation = latestServerOperation(server.server_id);
      const operationRunning = operation?.status === "RUNNING";
      return `
      <article class="card">
        <h3>${escapeHtml(server.display_name)}</h3>
        <p>${escapeHtml(server.username)}@${escapeHtml(server.host)}:${server.port}<br>${server.auth_type === "password" ? (server.has_saved_password ? "密码认证（本机已加密记住）" : "密码认证（尚未保存密码）") : "SSH私钥认证"}<br>${server.host_key_fingerprint ? "主机指纹已固定" : "等待确认主机指纹"}</p>
        <span class="badge ${server.host_key_fingerprint ? "" : "experimental"}">${server.host_key_fingerprint ? "已保存" : "待检查"}</span>
        ${operationProgress(operation)}
        <div class="card-actions"><button data-server-edit="${server.server_id}">更新连接信息</button><button data-server-check="${server.server_id}">检查连接</button><button data-server-bootstrap="${server.server_id}" ${operationRunning ? "disabled" : ""}>${operationRunning ? "安装中…" : "安装运行环境"}</button></div>
      </article>`;
    }).join("") : '<div class="empty">尚未添加服务器</div>';
    const deploymentJobIds = new Set(dashboard.deployments.map((item) => item.job_id));
    const waitingDeployments = dashboard.jobs.filter((job) => job.status === "COMPLETED" && job.plan?.conversion?.output?.deployment_mode === "direct-deploy" && !deploymentJobIds.has(job.job_id));
    const localPackages = dashboard.jobs.filter((job) => job.status === "COMPLETED" && job.plan?.conversion?.output?.deployment_mode === "local-only" && job.progress?.output && !deploymentJobIds.has(job.job_id));
    const orderedDeployments = [...dashboard.deployments].sort((left, right) => {
      const leftLocal = left.metadata?.target_type === "local" ? 1 : 0;
      const rightLocal = right.metadata?.target_type === "local" ? 1 : 0;
      return leftLocal - rightLocal;
    });
    const deploymentCards = orderedDeployments.map((deployment) => {
      const backingJob = dashboard.jobs.find((item) => item.job_id === deployment.job_id);
      const localOnly = backingJob?.plan?.conversion?.output?.deployment_mode === "local-only";
      const onMachine = deployment.metadata?.target_type === "local";
      const linkedServer = onMachine ? null : dashboard.servers.find((item) => item.server_id === deployment.server_id);
      const operation = onMachine
        ? latestJobOperation(deployment.job_id)
        : latestServerOperation(deployment.server_id);
      const operationRunning = operation?.status === "RUNNING";
      const resumeButton = onMachine
        ? `<button class="primary" data-deploy-action="start" data-id="${deployment.deployment_id}">重新启动本机服务</button>`
        : localOnly
        ? `<button class="primary" data-local-job-deploy="${deployment.job_id}" data-server-id="${deployment.server_id}" ${operationRunning ? "disabled" : ""}>${operationRunning ? "正在继续部署…" : "继续部署"}</button>`
        : `<button class="primary" data-job-deploy="${deployment.job_id}" ${operationRunning ? "disabled" : ""}>${operationRunning ? "正在继续部署…" : "继续部署"}</button>`;
      return `
      <article class="card">
        <h3>${deployment.model_id}</h3>
        <p>${deployment.deployment_id}<br>${deployment.model_version} · ${deployment.key_id}<br>${onMachine ? "本机 · 127.0.0.1 · 无需SSH" : `远程 · ${escapeHtml(linkedServer?.display_name || deployment.server_id)} · ${escapeHtml(linkedServer?.username || "root")}@${escapeHtml(linkedServer?.host || "未知主机")}:${linkedServer?.port || "?"}`}</p>
        ${deployment.metadata?.last_error ? `<div class="job-error"><b>部署失败原因</b><br>${escapeHtml(explainDeploymentError(deployment.metadata.last_error))}</div>` : ""}
        ${operationProgress(operation)}
        <span class="badge ${deployment.status === "HEALTHY" ? "" : "experimental"}">${deployment.status}</span>
        <div class="card-actions">
          ${onMachine ? "" : `<button data-deployment-server-edit="${deployment.server_id}">更新此部署的服务器登录信息</button>`}
          ${deployment.status === "HEALTHY" ? `<button class="primary" data-launch-chat="${deployment.deployment_id}">启动对话</button><button data-deploy-migrate="${deployment.deployment_id}">${onMachine ? "复制到云服务器" : "复制到另一台服务器"}</button><button data-deploy-action="stop" data-id="${deployment.deployment_id}">停止服务</button>${onMachine ? "" : `<button data-deploy-action="rollback" data-id="${deployment.deployment_id}">回滚</button>`}` : deployment.status === "STOPPED" ? `<button class="primary" data-deploy-action="start" data-id="${deployment.deployment_id}">启动原远程服务</button>${onMachine ? "" : `<button data-deploy-action="rollback" data-id="${deployment.deployment_id}">回滚</button>`}` : `${resumeButton}${deployment.status === "FAILED" && !onMachine ? `<button data-server-bootstrap="${deployment.server_id}" data-retry-job="${deployment.job_id}" ${operationRunning ? "disabled" : ""}>安装运行环境并继续部署</button>` : ""}`}
        </div>
      </article>`;
    });
    deploymentCards.push(...localPackages.map((job) => {
      const operation = [...(dashboard.server_operations || [])].reverse().find(
        (item) => item.job_id === job.job_id && item.kind === "LOCAL_PACKAGE_DEPLOY",
      );
      const running = operation?.status === "RUNNING";
      return `
      <article class="card">
        <h3>${escapeHtml(job.plan?.conversion?.output?.model_id || "本地私有模型")}</h3>
        <p>已在本地完成改造和校验<br>${escapeHtml(job.progress.output)}</p>
        ${operationProgress(operation)}
        <span class="badge">LOCAL_READY</span>
        <div class="card-actions"><button class="primary" data-machine-job-deploy="${job.job_id}">部署到本机</button><button data-local-job-deploy="${job.job_id}" ${running ? "disabled" : ""}>${running ? "正在上传部署…" : "选择服务器部署"}</button></div>
      </article>`;
    }));
    deploymentCards.push(...waitingDeployments.map((job) => `
      <article class="card">
        <h3>${escapeHtml(job.plan?.conversion?.output?.model_id || "私有模型")}</h3>
        <p>模型已改造并上传，等待启动云端服务</p>
        <span class="badge experimental">READY_TO_DEPLOY</span>
        <div class="card-actions"><button class="primary" data-job-deploy="${job.job_id}">继续部署</button></div>
      </article>`));
    document.querySelector("#deployment-list").innerHTML = deploymentCards.length ? deploymentCards.join("") : '<div class="empty">尚无可用部署；完成模型上传后会在这里出现</div>';
  } catch (error) {
    toast(error.message);
  } finally {
    refreshing = false;
  }
}

document.querySelectorAll("nav button").forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
document.querySelector("#refresh").addEventListener("click", refresh);
function resetServerForm() {
  const form = document.querySelector("#server-form");
  editingServerId = null;
  form.reset();
  document.querySelector("#server-form-title").textContent = "添加Ubuntu GPU服务器";
  document.querySelector("#save-server-button").textContent = "保存并检查";
  document.querySelector("#server-port").value = "22";
  document.querySelector("#server-username").value = "root";
  form.elements.model_root.value = "/opt/yinbian";
  form.elements.private_key_path.value = window.dashboard?.paths?.default_ssh_private_key || "";
  document.querySelector("#server-auth").dispatchEvent(new Event("change"));
}

function beginServerEdit(serverId) {
  const server = window.dashboard.servers.find((item) => item.server_id === serverId);
  if (!server) {
    toast("原部署关联的服务器记录不存在");
    return;
  }
  const form = document.querySelector("#server-form");
  editingServerId = server.server_id;
  form.elements.display_name.value = server.display_name || "";
  form.elements.ssh_command.value = `ssh -p ${server.port} ${server.username}@${server.host}`;
  form.elements.host.value = server.host || "";
  form.elements.port.value = server.port || 22;
  form.elements.username.value = server.username || "root";
  form.elements.auth_type.value = server.auth_type || "password";
  form.elements.password.value = "";
  form.elements.private_key_path.value = server.private_key_path_valid
    ? server.private_key_path
    : window.dashboard?.paths?.default_ssh_private_key || "";
  form.elements.model_root.value = server.model_root || "/opt/yinbian";
  form.elements.remember_password.checked = server.auth_type === "password";
  form.elements.check_now.checked = true;
  document.querySelector("#server-form-title").textContent = "更新原部署的服务器连接信息";
  document.querySelector("#save-server-button").textContent = "更新并重新连接原部署";
  document.querySelector("#server-auth").dispatchEvent(new Event("change"));
  form.hidden = false;
  switchPage("servers");
  form.scrollIntoView({ behavior: "smooth", block: "start" });
}

document.querySelector("#show-server-form").addEventListener("click", () => {
  resetServerForm();
  document.querySelector("#server-form").hidden = false;
});
document.querySelector("#cancel-server-form").addEventListener("click", () => {
  document.querySelector("#server-form").hidden = true;
  resetServerForm();
});
document.querySelector("#new-job").addEventListener("click", () => {
  document.querySelector("#wizard-mode").value = "local-deploy";
  document.querySelector("#wizard-source-mode").value = "catalog-download";
  updateModelSource();
  updateDeploymentMode();
  document.querySelector("#wizard-security-mode").value = "permutation";
  updateSecurityMode();
  document.querySelector("#deploy-wizard").hidden = false;
});
document.querySelector("#cancel-wizard").addEventListener("click", () => { document.querySelector("#deploy-wizard").hidden = true; });
document.querySelector("#wizard-family").addEventListener("change", (event) => {
  renderModelOptions(catalogModels, event.target.value);
  updateModelNote();
});
document.querySelector("#wizard-model").addEventListener("change", updateModelNote);
document.querySelector("#wizard-source-mode").addEventListener("change", updateModelSource);
document.querySelector("#wizard-local-model-path").addEventListener("input", () => {
  const check = document.querySelector("#wizard-local-check");
  check.className = "model-note full";
  check.textContent = "提交前会检查 config.json、Tokenizer、Safetensors权重及所选架构族。";
});
document.querySelector("#wizard-mode").addEventListener("change", updateDeploymentMode);
document.querySelector("#wizard-security-mode").addEventListener("change", updateSecurityMode);
document.querySelector("#server-auth").addEventListener("change", (event) => {
  const form = document.querySelector("#server-form");
  const password = event.target.value === "password";
  const passwordField = document.querySelector("#server-password-field");
  const keyField = document.querySelector("#server-key-field");
  passwordField.hidden = !password;
  keyField.hidden = password;
  form.elements.password.disabled = !password;
  form.elements.private_key_path.disabled = password;
  form.elements.remember_password.disabled = !password;
  if (!password) {
    form.elements.remember_password.checked = false;
    if (!form.elements.private_key_path.value) {
      form.elements.private_key_path.value = window.dashboard?.paths?.default_ssh_private_key || "";
    }
  }
});
document.querySelector("#ssh-command").addEventListener("change", (event) => {
  try {
    const parsed = parseSshCommand(event.target.value);
    if (!parsed) return;
    document.querySelector("#server-host").value = parsed.host;
    document.querySelector("#server-port").value = parsed.port;
    document.querySelector("#server-username").value = parsed.username;
  } catch (error) { toast(error.message); }
});

document.addEventListener("click", async (event) => {
  const familyTab = event.target.closest("[data-catalog-family]");
  if (familyTab) {
    document.querySelector("#model-grid").innerHTML = renderModelCatalog(
      catalogModels,
      familyTab.dataset.catalogFamily,
    );
    return;
  }
  const chat = event.target.closest("[data-launch-chat]");
  if (chat) {
    if (chat.disabled) return;
    chat.disabled = true;
    const label = chat.textContent;
    chat.textContent = "正在打开…";
    try {
      const result = await api(`/api/deployments/${encodeURIComponent(chat.dataset.launchChat)}/chat`, { method: "POST", body: "{}" });
      toast(result.already_running ? "对话程序已经打开" : "正在启动对话程序");
    } catch (error) {
      toast(error.message);
    } finally {
      chat.disabled = false;
      chat.textContent = label;
    }
    return;
  }
  const model = event.target.closest(".model-deploy");
  if (model) {
    document.querySelector("#wizard-family").value = model.dataset.family;
    renderModelOptions(catalogModels, model.dataset.family, model.dataset.model);
    document.querySelector("#wizard-model").value = model.dataset.model;
    document.querySelector("#wizard-mode").value = "local-deploy";
    document.querySelector("#wizard-source-mode").value = "catalog-download";
    updateModelSource();
    updateDeploymentMode();
    updateModelNote();
    document.querySelector("#deploy-wizard").hidden = false;
    switchPage("jobs");
  }
  const job = event.target.closest("[data-job-action]");
  if (job) {
    if (job.disabled) return;
    job.disabled = true;
    try {
      let body = {};
      if (job.dataset.jobAction === "resume") {
        const current = window.dashboard.jobs.find((item) => item.job_id === job.dataset.id);
        const serverId = current?.plan?.conversion?.output?.server_id;
        const saved = window.dashboard.servers.find((item) => item.server_id === serverId)?.has_saved_password;
        const cached = serverId ? sessionServerSecrets.get(serverId) : null;
        const offlineKeyPassword = window.prompt("\u8bf7\u8f93\u5165\u79bb\u7ebf\u5bc6\u94a5\u5907\u4efd\u53e3\u4ee4\uff08\u81f3\u5c1112\u4f4d\uff0c\u4e0d\u4fdd\u5b58\uff09");
        if (!offlineKeyPassword || offlineKeyPassword.length < 12) throw new Error("\u79bb\u7ebf\u5bc6\u94a5\u5907\u4efd\u53e3\u4ee4\u81f3\u5c11\u9700\u898112\u4f4d");
        const password = serverId
          ? (saved ? null : (cached ?? window.prompt("请输入SSH密码；验证成功后会加密保存在本机")))
          : null;
        body = {
          password: password || null,
          offline_key_password: offlineKeyPassword,
          accept_license: Boolean(current?.plan?.desktop?.accept_license),
        };
        if (serverId && password) sessionServerSecrets.set(serverId, password);
      }
      await api(`/api/jobs/${job.dataset.id}/${job.dataset.jobAction}`, { method: "POST", body: JSON.stringify(body) });
      toast("任务状态已更新");
      refresh();
    } catch (error) { toast(error.message); }
    finally { job.disabled = false; }
  }
  const check = event.target.closest("[data-server-check]");
  if (check) {
    const saved = window.dashboard.servers.find((item) => item.server_id === check.dataset.serverCheck)?.has_saved_password;
    const cached = sessionServerSecrets.get(check.dataset.serverCheck);
    const password = saved ? null : (cached ?? window.prompt("请输入SSH密码；验证成功后会加密保存在本机"));
    try {
      const result = await api(`/api/servers/${check.dataset.serverCheck}/check`, { method: "POST", body: JSON.stringify({ password, trust_host_key: true }) });
      toast(result.pass ? "服务器检查通过" : "服务器检查未通过");
      refresh();
    } catch (error) {
      const authenticationFailed = /permission denied|authentication|password/i.test(String(error.message || ""));
      if (!saved || !authenticationFailed) {
        toast(error.message);
      } else {
        const replacement = window.prompt("本机保存的SSH密码已经失效，请输入当前服务器的新密码");
        if (!replacement) return;
        try {
          const first = await api(`/api/servers/${check.dataset.serverCheck}/check`, { method: "POST", body: JSON.stringify({ password: replacement, trust_host_key: false }) });
          if (!first.pass) throw new Error(first.hard_failures?.join("；") || "服务器预检未通过");
          if (!first.trusted) {
            const accepted = window.confirm(`服务器当前指纹为：\n${first.host_key_fingerprint}\n\n确认这是您的服务器吗？`);
            if (!accepted) return;
            await api(`/api/servers/${check.dataset.serverCheck}/check`, { method: "POST", body: JSON.stringify({ password: replacement, trust_host_key: true }) });
          }
          sessionServerSecrets.set(check.dataset.serverCheck, replacement);
          toast("新密码已在本机加密保存，服务器检查通过");
          refresh();
        } catch (replacementError) { toast(replacementError.message); }
      }
    }
  }
  const edit = event.target.closest("[data-server-edit]");
  if (edit) {
    beginServerEdit(edit.dataset.serverEdit);
  }
  const deploymentServerEdit = event.target.closest("[data-deployment-server-edit]");
  if (deploymentServerEdit) {
    beginServerEdit(deploymentServerEdit.dataset.deploymentServerEdit);
  }
  const pendingDeployment = event.target.closest("[data-job-deploy]");
  if (pendingDeployment) {
    const current = window.dashboard.jobs.find((item) => item.job_id === pendingDeployment.dataset.jobDeploy);
    const serverId = current?.plan?.conversion?.output?.server_id;
    const saved = window.dashboard.servers.find((item) => item.server_id === serverId)?.has_saved_password;
    const password = saved ? null : window.prompt("请输入SSH密码；验证成功后会加密保存在本机");
    if (password === null) return;
    pendingDeployment.disabled = true;
    try {
      await api(`/api/jobs/${pendingDeployment.dataset.jobDeploy}/deploy`, { method: "POST", body: JSON.stringify({ password }) });
      toast("云端模型服务已经启动");
      refresh();
    } catch (error) { toast(error.message); refresh(); }
    finally { pendingDeployment.disabled = false; }
  }
  const localDeployment = event.target.closest("[data-local-job-deploy]");
  if (localDeployment) {
    const targets = window.dashboard.servers;
    if (!targets.length) {
      toast("本地私有模型已经保存；添加服务器后即可上传部署");
      return;
    }
    const pinnedServerId = localDeployment.dataset.serverId;
    let target = pinnedServerId ? targets.find((item) => item.server_id === pinnedServerId) : null;
    if (!target) {
      const choices = targets.map((item, index) => `${index + 1}. ${item.display_name} (${item.username}@${item.host}:${item.port})`).join("\n");
      const selectedIndex = Number(window.prompt(`选择目标服务器编号：\n${choices}`, "1")) - 1;
      target = targets[selectedIndex];
    }
    if (!target) return;
    if (!window.confirm("将上传已经校验的私有模型并安装或启动远端运行环境，不会重新下载或改造权重。确认继续吗？")) return;
    const password = target.has_saved_password ? null : (sessionServerSecrets.get(target.server_id) ?? window.prompt("请输入目标服务器SSH密码"));
    if (!target.has_saved_password && !password) return;
    localDeployment.disabled = true;
    try {
      const operation = await api(`/api/jobs/${encodeURIComponent(localDeployment.dataset.localJobDeploy)}/deploy-local`, {
        method: "POST",
        body: JSON.stringify({ server_id: target.server_id, password, confirmed: true }),
      });
      toast(`上传部署已开始：${operation.stage}`);
      refresh();
    } catch (error) { toast(error.message); refresh(); }
    finally { localDeployment.disabled = false; }
  }
  const machineDeployment = event.target.closest("[data-machine-job-deploy]");
  if (machineDeployment) {
    if (!window.confirm("将在本机启动真实私有模型服务，完成健康检查后可直接进入对话。不会改变现有云端部署。确认继续吗？")) return;
    machineDeployment.disabled = true;
    try {
      const operation = await api(`/api/jobs/${encodeURIComponent(machineDeployment.dataset.machineJobDeploy)}/deploy-to-machine`, {
        method: "POST",
        body: JSON.stringify({ device: "auto", port: 0, confirmed: true }),
      });
      toast(`本机部署已开始：${operation.stage}`);
      switchPage("deployments");
      refresh();
    } catch (error) { toast(error.message); refresh(); }
    finally { machineDeployment.disabled = false; }
  }
  const deployment = event.target.closest("[data-deploy-action]");
  if (deployment) {
    const current = window.dashboard.deployments.find((item) => item.deployment_id === deployment.dataset.id);
    const onMachine = current?.metadata?.target_type === "local";
    const saved = onMachine || window.dashboard.servers.find((item) => item.server_id === current?.server_id)?.has_saved_password;
    const password = saved ? null : window.prompt("请输入SSH密码；验证成功后会加密保存在本机");
    try {
      await api(`/api/deployments/${deployment.dataset.id}/${deployment.dataset.deployAction}`, { method: "POST", body: JSON.stringify({ password }) });
      toast("部署状态已更新");
      refresh();
    } catch (error) {
      const connectionChanged = /permission denied|authentication|connection|refused|timed out|host key/i.test(String(error.message || ""));
      if (!onMachine && connectionChanged && window.confirm("原部署仍然保留，但服务器登录信息已经失效。是否现在更新SSH地址、端口或密码？")) {
        beginServerEdit(current.server_id);
      } else {
        toast(error.message);
      }
    }
  }
  const migration = event.target.closest("[data-deploy-migrate]");
  if (migration) {
    const current = window.dashboard.deployments.find((item) => item.deployment_id === migration.dataset.deployMigrate);
    const targets = window.dashboard.servers.filter((item) => item.server_id !== current?.server_id);
    if (!targets.length) {
      toast("请先在“服务器”页面添加另一台服务器");
      return;
    }
    const choices = targets.map((item, index) => `${index + 1}. ${item.display_name} (${item.username}@${item.host}:${item.port})`).join("\n");
    const selectedIndex = Number(window.prompt(`选择目标服务器编号：\n${choices}`, "1")) - 1;
    const target = targets[selectedIndex];
    if (!target) return;
    const password = target.has_saved_password ? null : (sessionServerSecrets.get(target.server_id) ?? window.prompt("请输入目标服务器SSH密码"));
    if (!target.has_saved_password && !password) return;
    migration.disabled = true;
    try {
      await api(`/api/deployments/${encodeURIComponent(current.deployment_id)}/migrate`, {
        method: "POST",
        body: JSON.stringify({ target_server_id: target.server_id, password }),
      });
      toast("私有模型已复制到新服务器并启动；没有重新改造权重");
      refresh();
    } catch (error) { toast(error.message); }
    finally { migration.disabled = false; }
  }
  const bootstrap = event.target.closest("[data-server-bootstrap]");
  if (bootstrap) {
    if (bootstrap.disabled) return;
    if (!window.confirm("将检查服务器类型，并自动安装Docker运行环境或GPU租赁容器所需的Python运行环境。确认继续吗？")) return;
    const saved = window.dashboard.servers.find((item) => item.server_id === bootstrap.dataset.serverBootstrap)?.has_saved_password;
    const password = saved ? null : window.prompt("请输入SSH密码；验证成功后会加密保存在本机");
    if (password === null) return;
    bootstrap.disabled = true;
    const originalLabel = bootstrap.textContent;
    bootstrap.textContent = "正在启动…";
    try {
      const operation = await api(`/api/servers/${bootstrap.dataset.serverBootstrap}/bootstrap`, {
        method: "POST",
        body: JSON.stringify({
          password,
          confirmed: true,
          retry_job_id: bootstrap.dataset.retryJob || null,
        }),
      });
      toast(operation.kind === "BOOTSTRAP_AND_DEPLOY" ? "安装与部署已开始，请查看进度条" : "运行环境安装已开始，请查看进度条");
      await refresh();
    } catch (error) {
      toast(error.message);
      bootstrap.disabled = false;
      bootstrap.textContent = originalLabel;
    }
  }
});

document.querySelector("#deploy-wizard").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const password = data.password || sessionServerSecrets.get(data.server_id) || null;
    if (data.mode === "direct-deploy" && !data.server_id) throw new Error("直接部署必须选择一台服务器");
    if (data.security_mode === "tee_gm" && data.mode === "direct-deploy" && data.tee_backend !== "intel_tdx") throw new Error("云端TEE部署必须选择Intel TDX，软件模拟不能作为生产部署");
    const selected = catalogModels.find((item) => item.catalog_id === data.model);
    if (!selected?.conversion_ready) throw new Error(`${selected?.display_name || "该模型"}目前只支持结构检查，尚不能执行权重改造`);
    if (data.security_mode === "tee_gm" && selected.adapter_id !== "qwen2") throw new Error("TEE 1.0 首轮仅支持 Qwen2/Qwen2.5 模型族");
    if (data.security_mode === "tee_gm" && data.mode === "direct-deploy") {
      const teeCheck = await api(`/api/servers/${data.server_id}/tee-check`, {
        method: "POST",
        body: JSON.stringify({ password, private_key_passphrase: null, trust_host_key: false }),
      });
      if (!teeCheck.pass) throw new Error(`服务器未通过 Intel TDX 预检：${(teeCheck.failures || []).join("；")}`);
    }
    if (["direct-deploy", "local-deploy"].includes(data.mode) && !selected.deployment_ready) {
      throw new Error(`${selected.display_name}尚未通过产品部署验收，请先选择“仅本地改造”`);
    }
    if (selected.status !== "supported" && !window.confirm(`${selected.display_name} 的转换代码已接入，但尚未完成该检查点的产品验收。系统会在读取模型后严格检查全部张量，确认继续吗？`)) return;
    let planModel = data.model;
    let catalogModel = null;
    if (data.source_mode === "local-existing") {
      planModel = String(data.local_model_path || "").trim();
      if (!planModel) throw new Error("请选择本地已经下载好的模型目录");
      const localCheck = document.querySelector("#wizard-local-check");
      localCheck.className = "model-note full";
      localCheck.textContent = "正在检查本地模型结构和权重清单…";
      const inspection = await api("/api/models/inspect", {
        method: "POST",
        body: JSON.stringify({ model: planModel, catalog_model: data.model }),
      });
      if (inspection.resolved_path) {
        planModel = inspection.resolved_path;
        document.querySelector("#wizard-local-model-path").value = planModel;
      }
      if (String(inspection.status || "").toUpperCase() !== "SUPPORTED") {
        throw new Error(`本地模型检查未通过：${(inspection.reasons || []).join("；") || inspection.status}`);
      }
      if (inspection.adapter_id !== selected.adapter_id) {
        throw new Error(`本地目录属于 ${inspection.adapter_id} 架构，但当前选择的是 ${selected.adapter_id} 架构`);
      }
      localCheck.className = "model-note full success";
      localCheck.textContent = `${inspection.input_was_parent ? "已自动定位具体模型目录；" : ""}检查通过：${inspection.adapter_id}，共 ${inspection.tensor_count} 个张量；将直接改造，不再下载。`;
      catalogModel = data.model;
    }
    const plan = await api("/api/plans", { method: "POST", body: JSON.stringify({ model: planModel, catalog_model: catalogModel, destination: data.destination, mode: data.mode, device: data.device, download_endpoint: data.download_endpoint || "auto", server_id: data.server_id || null, security_mode: data.security_mode || "permutation", tee_backend: data.security_mode === "tee_gm" ? data.tee_backend : null }) });
    const offlineKeyPassword = window.prompt("\u8bf7\u8bbe\u7f6e\u79bb\u7ebf\u5bc6\u94a5\u5907\u4efd\u53e3\u4ee4\uff08\u81f3\u5c1112\u4f4d\uff0c\u4e0d\u4fdd\u5b58\uff09");
    if (!offlineKeyPassword || offlineKeyPassword.length < 12) throw new Error("\u79bb\u7ebf\u5bc6\u94a5\u5907\u4efd\u53e3\u4ee4\u81f3\u5c11\u9700\u898112\u4f4d");
    await api("/api/jobs", { method: "POST", body: JSON.stringify({ plan_path: plan.path, password, offline_key_password: offlineKeyPassword, accept_license: data.accept_license === "on" }) });
    event.target.hidden = true;
    toast("任务已开始，可随时暂停或恢复");
    refresh();
  } catch (error) { toast(error.message); }
});

document.querySelector("#server-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const parsed = parseSshCommand(data.ssh_command);
    if (parsed) {
      data.host = parsed.host;
      data.port = parsed.port;
      data.username = parsed.username;
    }
    data.port = Number(data.port || 22);
    const password = data.password || null;
    data.password = password;
    data.remember_password = data.remember_password === "on";
    const checkNow = data.check_now === "on";
    if (!data.remember_password) delete data.password;
    delete data.check_now;
    delete data.ssh_command;
    if (data.auth_type === "password") {
      data.private_key_path = null;
    } else {
      data.password = null;
      data.remember_password = false;
      data.private_key_path = data.private_key_path
        || window.dashboard?.paths?.default_ssh_private_key
        || null;
      if (!data.private_key_path) {
        throw new Error("请选择SSH私钥文件，或将认证方式切换为“密码”");
      }
    }
    const endpoint = editingServerId
      ? `/api/servers/${encodeURIComponent(editingServerId)}`
      : "/api/servers";
    const method = editingServerId ? "PUT" : "POST";
    const server = await api(endpoint, { method, body: JSON.stringify(data) });
    if (password) sessionServerSecrets.set(server.server_id, password);
    if (checkNow) {
      const first = await api(`/api/servers/${server.server_id}/check`, { method: "POST", body: JSON.stringify({ password: server.has_saved_password ? null : password, trust_host_key: false }) });
      if (!first.pass) throw new Error(first.hard_failures.join("；") || "服务器预检未通过");
      if (!first.trusted) {
        const accepted = window.confirm(`首次连接的服务器指纹为：\n${first.host_key_fingerprint}\n\n确认这是您的服务器吗？`);
        if (!accepted) throw new Error("服务器已保存，但主机指纹尚未确认");
        await api(`/api/servers/${server.server_id}/check`, { method: "POST", body: JSON.stringify({ password: server.has_saved_password ? null : password, trust_host_key: true }) });
      }
    }
    event.target.hidden = true;
    resetServerForm();
    toast(checkNow ? "服务器连接信息已保存并通过检查" : "服务器连接信息已保存");
    refresh();
  } catch (error) { toast(error.message); }
});

document.querySelector("#key-backup-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const result = await api("/api/keys/backup", { method: "POST", body: JSON.stringify(data) });
    event.target.reset();
    toast(`备份已创建：${result.output}`);
  } catch (error) { toast(error.message); }
});

document.querySelector("#key-restore-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const result = await api("/api/keys/restore", { method: "POST", body: JSON.stringify(data) });
    event.target.reset();
    toast(`密钥已恢复：${result.destination}`);
  } catch (error) { toast(error.message); }
});

function autoRefresh() {
  // Forms on inactive pages remain in the DOM. Only pause polling while the
  // user is actively editing a control on the page that is currently shown.
  const activeElement = document.activeElement;
  const userIsEditing = activeElement
    && ["INPUT", "SELECT", "TEXTAREA"].includes(activeElement.tagName)
    && activeElement.closest(".page.active");
  if (!userIsEditing) refresh();
}

updateModelSource();
updateDeploymentMode();
refresh();
setInterval(autoRefresh, 2000);
