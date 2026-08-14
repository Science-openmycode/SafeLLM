const pages = {
  dashboard: ["部署总览", "选择模型、完成本地改造并部署到您的服务器"],
  models: ["模型目录", "推荐模型和经过结构识别的高级模型"],
  jobs: ["转换任务", "查看下载、改造、校验和上传进度"],
  servers: ["服务器", "保存并检查您的 Ubuntu GPU 服务器"],
  deployments: ["云端部署", "启动、停止、升级和回滚模型服务"],
  keys: ["密钥与备份", "管理本地在线密钥和可迁移备份"],
};

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
  return jobs.map((job) => `
    <article class="card">
      <h3>${job.job_id}</h3>
      <p>${job.status} · ${job.phase}</p>
      <span class="badge">${job.progress?.item || "等待下一阶段"}</span>
      <div class="card-actions">
        ${job.status === "RUNNING" ? `<button data-job-action="pause" data-id="${job.job_id}">暂停</button>` : ""}
        ${["PAUSED", "FAILED"].includes(job.status) ? `<button data-job-action="resume" data-id="${job.job_id}">恢复</button>` : ""}
        ${!["COMPLETED", "CANCELLED"].includes(job.status) ? `<button data-job-action="cancel" data-id="${job.job_id}">取消</button>` : ""}
      </div>
    </article>`).join("");
}

async function refresh() {
  try {
    const [dashboard, models] = await Promise.all([api("/api/dashboard"), api("/api/models")]);
    window.dashboard = dashboard;
    const metrics = [
      ["可用模型", dashboard.models],
      ["转换任务", dashboard.jobs.length],
      ["服务器", dashboard.servers.length],
      ["健康部署", dashboard.deployments.filter((item) => item.status === "HEALTHY").length],
    ];
    document.querySelector("#metrics").innerHTML = metrics.map((item) => `<div class="metric"><small>${item[0]}</small><strong>${item[1]}</strong></div>`).join("");
    document.querySelector("#recent-jobs").innerHTML = jobRows(dashboard.jobs.slice(0, 4));
    document.querySelector("#job-list").innerHTML = jobRows(dashboard.jobs);
    document.querySelector("#model-grid").innerHTML = models.map((model) => `
      <article class="card">
        <h3>${model.display_name}</h3>
        <p>${model.parameter_summary}<br>${model.adapter_id} · ${model.license}<br>最高阶段：${model.max_stage}</p>
        <span class="badge ${model.status}">${model.status}</span>
        ${["supported", "validated-conversion"].includes(model.status) ? `<button class="primary model-deploy" data-model="${model.catalog_id}">选择并部署</button>` : ""}
      </article>`).join("");
    document.querySelector("#wizard-server").innerHTML = '<option value="">不选择服务器</option>' + dashboard.servers.map((server) => `<option value="${server.server_id}">${server.display_name}</option>`).join("");
    document.querySelector("#server-list").innerHTML = dashboard.servers.length ? dashboard.servers.map((server) => `
      <article class="card">
        <h3>${server.display_name}</h3>
        <p>${server.username}@${server.host}:${server.port}<br>${server.host_key_fingerprint ? "主机指纹已固定" : "等待确认主机指纹"}</p>
        <span class="badge ${server.host_key_fingerprint ? "" : "experimental"}">${server.host_key_fingerprint ? "已保存" : "待检查"}</span>
        <div class="card-actions"><button data-server-check="${server.server_id}">检查连接</button><button data-server-bootstrap="${server.server_id}">安装运行环境</button></div>
      </article>`).join("") : '<div class="empty">尚未添加服务器</div>';
    document.querySelector("#deployment-list").innerHTML = dashboard.deployments.length ? dashboard.deployments.map((deployment) => `
      <article class="card">
        <h3>${deployment.model_id}</h3>
        <p>${deployment.deployment_id}<br>${deployment.model_version} · ${deployment.key_id}</p>
        <span class="badge ${deployment.status === "HEALTHY" ? "" : "experimental"}">${deployment.status}</span>
        <div class="card-actions">
          <button data-deploy-action="start" data-id="${deployment.deployment_id}">启动</button>
          <button data-deploy-action="stop" data-id="${deployment.deployment_id}">停止</button>
          <button data-deploy-action="rollback" data-id="${deployment.deployment_id}">回滚</button>
        </div>
      </article>`).join("") : '<div class="empty">只有健康部署会进入对话程序</div>';
  } catch (error) {
    toast(error.message);
  }
}

document.querySelectorAll("nav button").forEach((button) => button.addEventListener("click", () => switchPage(button.dataset.page)));
document.querySelector("#refresh").addEventListener("click", refresh);
document.querySelector("#show-server-form").addEventListener("click", () => { document.querySelector("#server-form").hidden = false; });
document.querySelector("#new-job").addEventListener("click", () => { document.querySelector("#deploy-wizard").hidden = false; });
document.querySelector("#cancel-wizard").addEventListener("click", () => { document.querySelector("#deploy-wizard").hidden = true; });

document.addEventListener("click", async (event) => {
  const model = event.target.closest(".model-deploy");
  if (model) {
    document.querySelector("#wizard-model").value = model.dataset.model;
    document.querySelector("#deploy-wizard").hidden = false;
    switchPage("jobs");
  }
  const job = event.target.closest("[data-job-action]");
  if (job) {
    try {
      await api(`/api/jobs/${job.dataset.id}/${job.dataset.jobAction}`, { method: "POST", body: "{}" });
      toast("任务状态已更新");
      refresh();
    } catch (error) { toast(error.message); }
  }
  const check = event.target.closest("[data-server-check]");
  if (check) {
    const password = window.prompt("如使用密码认证，请输入 SSH 密码；私钥认证可留空");
    try {
      const result = await api(`/api/servers/${check.dataset.serverCheck}/check`, { method: "POST", body: JSON.stringify({ password, trust_host_key: true }) });
      toast(result.pass ? "服务器检查通过" : "服务器检查未通过");
      refresh();
    } catch (error) { toast(error.message); }
  }
  const deployment = event.target.closest("[data-deploy-action]");
  if (deployment) {
    const password = window.prompt("如使用密码认证，请输入 SSH 密码；私钥认证可留空");
    try {
      await api(`/api/deployments/${deployment.dataset.id}/${deployment.dataset.deployAction}`, { method: "POST", body: JSON.stringify({ password }) });
      toast("部署状态已更新");
      refresh();
    } catch (error) { toast(error.message); }
  }
  const bootstrap = event.target.closest("[data-server-bootstrap]");
  if (bootstrap) {
    if (!window.confirm("将通过 apt 安装 Docker 和 NVIDIA Container Toolkit，并重启 Docker。确认继续吗？")) return;
    const password = window.prompt("如使用密码认证，请输入 SSH 密码；私钥认证可留空");
    try {
      await api(`/api/servers/${bootstrap.dataset.serverBootstrap}/bootstrap`, { method: "POST", body: JSON.stringify({ password, confirmed: true }) });
      toast("服务器运行环境已安装");
    } catch (error) { toast(error.message); }
  }
});

document.querySelector("#deploy-wizard").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  try {
    const plan = await api("/api/plans", { method: "POST", body: JSON.stringify({ model: data.model, destination: data.destination, mode: data.mode, device: data.device, server_id: data.server_id || null }) });
    await api("/api/jobs", { method: "POST", body: JSON.stringify({ plan_path: plan.path, password: data.password || null, accept_license: data.accept_license === "on" }) });
    event.target.hidden = true;
    toast("任务已开始，可随时暂停或恢复");
    refresh();
  } catch (error) { toast(error.message); }
});

document.querySelector("#server-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(event.target));
  data.port = Number(data.port || 22);
  data.auth_type = data.private_key_path ? "private_key" : "password";
  try {
    await api("/api/servers", { method: "POST", body: JSON.stringify(data) });
    event.target.hidden = true;
    event.target.reset();
    toast("服务器已保存，下一步请执行连接检查");
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

refresh();
