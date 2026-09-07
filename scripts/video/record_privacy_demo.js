const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "../..");
const outputRoot = path.join(root, "artifacts/video/yinbian-privacy-demo");
const generated = JSON.parse(fs.readFileSync(path.join(outputRoot, "audio/scenes.generated.json"), "utf8").replace(/^\uFEFF/, ""));
const byId = Object.fromEntries(generated.map(item => [item.id, item]));
const base = process.env.YINBIAN_DEMO_URL || "http://127.0.0.1:55330";
const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";

fs.mkdirSync(path.join(outputRoot, "raw"), { recursive: true });

function seconds(id) {
  return Math.max(1, Number(byId[id].duration_seconds) + 0.65);
}

async function pause(page, id) {
  await page.waitForTimeout(seconds(id) * 1000);
}

async function installOverlay(page) {
  await page.evaluate(() => {
    document.querySelector("#video-scene-overlay")?.remove();
    const overlay = document.createElement("div");
    overlay.id = "video-scene-overlay";
    overlay.innerHTML = '<b></b><span></span>';
    Object.assign(overlay.style, {
      position: "fixed", top: "22px", right: "24px", zIndex: "2147483646",
      width: "430px", padding: "14px 18px", borderRadius: "15px",
      color: "#063d31", background: "rgba(239,250,246,.96)",
      border: "1px solid #7bcbb5", boxShadow: "0 14px 38px rgba(4,74,58,.15)",
      fontFamily: '"Microsoft YaHei", sans-serif', pointerEvents: "none"
    });
    Object.assign(overlay.querySelector("b").style, {display:"block",fontSize:"19px",lineHeight:"1.35"});
    Object.assign(overlay.querySelector("span").style, {display:"block",marginTop:"4px",fontSize:"13px",color:"#4e6c64"});
    document.body.appendChild(overlay);
  });
}

async function scene(page, id) {
  if (!(await page.locator("#video-scene-overlay").count())) await installOverlay(page);
  await page.evaluate(({title, caption}) => {
    const overlay = document.querySelector("#video-scene-overlay");
    overlay.querySelector("b").textContent = title;
    overlay.querySelector("span").textContent = caption;
    overlay.animate([{opacity:0,transform:"translateY(-8px)"},{opacity:1,transform:"translateY(0)"}], {duration:350,fill:"both"});
  }, byId[id]);
}

async function scrollTo(page, selector) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  await page.waitForTimeout(900);
}

async function titleCard(page, ending = false) {
  const item = byId[ending ? "finish" : "title"];
  await page.setContent(`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}body{margin:0;width:100vw;height:100vh;overflow:hidden;font-family:"Microsoft YaHei",sans-serif;color:#10251f;background:radial-gradient(circle at 78% 15%,#c9f3e7 0,transparent 31%),linear-gradient(145deg,#f9fcfb,#eaf7f2)}
    main{height:100%;display:grid;align-content:center;padding:110px 150px;position:relative}main:before{content:"隐";position:absolute;right:140px;top:115px;width:150px;height:150px;border-radius:40px;display:grid;place-items:center;color:white;background:#10a37f;font-size:76px;font-weight:900;box-shadow:0 24px 60px #0c8a6a44}
    small{font:700 18px/1.2 ui-monospace,monospace;letter-spacing:.22em;color:#0d9877}h1{max-width:1050px;margin:24px 0 20px;font-size:62px;line-height:1.13;letter-spacing:-.04em}p{max-width:930px;margin:0;color:#527068;font-size:25px;line-height:1.65}.pills{display:flex;gap:14px;margin-top:38px}.pills span{padding:11px 17px;border:1px solid #9ddbc9;border-radius:999px;background:#f5fffb;color:#08775c;font-weight:700;font-size:16px}.foot{position:absolute;left:150px;bottom:70px;color:#6e817b;font-size:14px}
  </style></head><body><main><small>YINBIAN PRIVACY INFERENCE</small><h1>${item.title}</h1><p>${item.caption}</p><div class="pills"><span>真实Qwen模型</span><span>TEE可信边界</span><span>GPU混淆推理</span><span>流式多轮对话</span></div><div class="foot">隐变智模 · 隐私保护大模型部署与推理框架</div></main></body></html>`);
}

async function waitForRun(page) {
  await page.waitForFunction(() => {
    const button = document.querySelector("#tee-lab-submit");
    return button && !button.disabled && document.querySelector("#tee-answer")?.textContent && !document.querySelector("#tee-answer")?.classList.contains("placeholder");
  }, null, { timeout: 120000 });
}

async function waitForChat(page) {
  await page.waitForFunction(() => {
    const button = document.querySelector("#submit");
    const answer = document.querySelector("#answer");
    return button && !button.disabled && answer && !answer.classList.contains("placeholder") && answer.textContent.trim().length > 1;
  }, null, { timeout: 150000 });
}

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: edge, args:["--disable-gpu-sandbox"]});
  const context = await browser.newContext({
    viewport: {width: 1600, height: 900},
    deviceScaleFactor: 1,
    recordVideo: {dir: path.join(outputRoot, "raw"), size: {width:1600,height:900}},
    locale: "zh-CN",
  });
  const page = await context.newPage();
  const video = page.video();

  await titleCard(page);
  await pause(page, "title");

  await page.goto(`${base}/privacy/tee?video=20260831`, {waitUntil:"networkidle", timeout:120000});
  await installOverlay(page);
  await scene(page, "entry");
  await scrollTo(page, "#entry");
  await pause(page, "entry");

  await scene(page, "body");
  await scrollTo(page, "#body");
  await pause(page, "body");

  await scene(page, "head");
  await scrollTo(page, "#head");
  await pause(page, "head");

  await scene(page, "run");
  await scrollTo(page, "#run");
  await page.locator("#tee-lab-prompt").fill("请用一句话说明北京是什么地方。");
  await page.waitForTimeout(900);
  await page.locator("#tee-lab-submit").click();
  const runPause = pause(page, "run");
  await Promise.all([waitForRun(page), runPause]);

  await scene(page, "encrypted");
  await scrollTo(page, "#ingress-detail");
  await pause(page, "encrypted");

  await scene(page, "body_result");
  await scrollTo(page, "#body");
  await pause(page, "body_result");

  await scene(page, "head_result");
  await scrollTo(page, "#head");
  await pause(page, "head_result");

  await scene(page, "loop");
  await scrollTo(page, "#return-loop");
  await pause(page, "loop");

  await page.goto(`${base}/?video=20260831`, {waitUntil:"networkidle", timeout:120000});
  await page.locator("#new-chat").click();
  await page.waitForTimeout(700);
  await installOverlay(page);
  await scene(page, "chat_one");
  await page.locator("#prompt").fill("请用两句话介绍北京，并说明它最有代表性的历史建筑。");
  await page.locator("#submit").click();
  const chatOnePause = pause(page, "chat_one");
  await Promise.all([waitForChat(page), chatOnePause]);

  await scene(page, "chat_two");
  await page.locator("#prompt").fill("刚才提到的那座建筑有什么历史作用？");
  await page.locator("#submit").click();
  const chatTwoPause = pause(page, "chat_two");
  await Promise.all([waitForChat(page), chatTwoPause]);

  await titleCard(page, true);
  await pause(page, "finish");

  await page.close();
  const rawVideo = await video.path();
  await context.close();
  await browser.close();
  fs.copyFileSync(rawVideo, path.join(outputRoot, "raw", "screen.webm"));
  console.log(path.join(outputRoot, "raw", "screen.webm"));
})().catch(error => {
  console.error(error.stack || error);
  process.exit(1);
});
