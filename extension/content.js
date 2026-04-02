/**
 * PREVERVECT Chrome 外掛：1 FPS 擷取影片影格，POST 至本機 FastAPI，更新懸浮 UI。
 * 有臉時一律顯示 SpecXNet 假分數與綠／黃／紅進度條；is_reliable 僅影響狀態說明（生理訊號）。
 * 連線失敗時暫停擷取。
 */
(() => {
  // 使用 127.0.0.1 可避免部分環境下 localhost 解析異常；須與 uvicorn 監聽埠一致
  const API_URL = "http://127.0.0.1:8000/detect";
  const CAPTURE_MS = 1000; // 1 FPS

  // 進度條（Fake 分數）：<45% 綠、45%~65% 黃、>65% 紅；僅無臉時為灰
  const COLOR_GREEN = "#22c55e";
  const COLOR_YELLOW = "#eab308";
  const COLOR_RED = "#ef4444";
  const COLOR_GRAY = "#64748b";

  let stopped = false;
  let capturePaused = false; // 伺服器連線失敗後暫停，不繼續 setInterval 觸發
  let timerId = null;
  let currentVideo = null;
  let panelEl = null;
  let statusEl = null;
  let barEl = null;
  let textEl = null;
  let canvas = null;
  let ctx = null;

  function createPanel() {
    if (panelEl) return;

    panelEl = document.createElement("div");
    panelEl.style.position = "fixed";
    panelEl.style.top = "16px";
    panelEl.style.right = "16px";
    panelEl.style.zIndex = "2147483647";
    panelEl.style.background = "rgba(15, 23, 42, 0.82)";
    panelEl.style.backdropFilter = "blur(4px)";
    panelEl.style.border = "1px solid rgba(148, 163, 184, 0.4)";
    panelEl.style.borderRadius = "10px";
    panelEl.style.padding = "10px 12px";
    panelEl.style.width = "280px";
    panelEl.style.color = "#e2e8f0";
    panelEl.style.fontFamily = "system-ui, Arial, sans-serif";
    panelEl.style.boxShadow = "0 8px 24px rgba(2, 6, 23, 0.45)";

    const title = document.createElement("div");
    title.textContent = "PREVERVECT 即時偵測";
    title.style.fontSize = "14px";
    title.style.fontWeight = "700";
    title.style.marginBottom = "8px";

    statusEl = document.createElement("div");
    statusEl.textContent = "狀態：初始化中...";
    statusEl.style.fontSize = "12px";
    statusEl.style.marginBottom = "8px";
    statusEl.style.color = "#bfdbfe";

    const label = document.createElement("div");
    label.textContent = "Fake Score";
    label.style.fontSize = "12px";
    label.style.marginBottom = "6px";

    const barWrap = document.createElement("div");
    barWrap.style.height = "12px";
    barWrap.style.background = "rgba(148, 163, 184, 0.25)";
    barWrap.style.borderRadius = "99px";
    barWrap.style.overflow = "hidden";
    barWrap.style.marginBottom = "6px";

    barEl = document.createElement("div");
    barEl.style.height = "100%";
    barEl.style.width = "0%";
    barEl.style.background = COLOR_GRAY;
    barEl.style.transition = "width 0.25s ease, background 0.2s ease";

    barWrap.appendChild(barEl);

    textEl = document.createElement("div");
    textEl.textContent = "—";
    textEl.style.fontSize = "12px";
    textEl.style.color = "#e2e8f0";
    textEl.style.marginBottom = "10px";

    const stopBtn = document.createElement("button");
    stopBtn.textContent = "中斷偵測 (Stop)";
    stopBtn.style.width = "100%";
    stopBtn.style.padding = "7px 8px";
    stopBtn.style.background = "#ef4444";
    stopBtn.style.border = "none";
    stopBtn.style.color = "#fff";
    stopBtn.style.borderRadius = "8px";
    stopBtn.style.cursor = "pointer";
    stopBtn.style.fontSize = "12px";
    stopBtn.onclick = () => stopDetection("使用者已中斷");

    panelEl.appendChild(title);
    panelEl.appendChild(statusEl);
    panelEl.appendChild(label);
    panelEl.appendChild(barWrap);
    panelEl.appendChild(textEl);
    panelEl.appendChild(stopBtn);

    document.body.appendChild(panelEl);
  }

  /** 無臉：灰色進度條歸零 */
  function setBarGray(widthPercent) {
    const p = Math.max(0, Math.min(100, Number(widthPercent) || 0));
    if (barEl) {
      barEl.style.width = `${p}%`;
      barEl.style.background = COLOR_GRAY;
    }
  }

  /** 依假分數百分比著色：綠 / 黃 / 紅（有臉推論時一律使用） */
  function setBarByFakeScore(fakeScore) {
    const pct = Math.max(0, Math.min(100, Number(fakeScore) * 100));
    if (barEl) {
      barEl.style.width = `${pct.toFixed(1)}%`;
      if (pct < 45) barEl.style.background = COLOR_GREEN;
      else if (pct <= 65) barEl.style.background = COLOR_YELLOW;
      else barEl.style.background = COLOR_RED;
    }
  }

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = `狀態：${msg}`;
  }

  /** 停止擷取迴圈（使用者 Stop 或伺服器錯誤暫停） */
  function stopDetection(reason) {
    stopped = true;
    capturePaused = true;
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
    setStatus(reason || "已停止");
  }

  function setScoreTextHidden() {
    if (textEl) {
      textEl.textContent = "—";
      textEl.style.color = "#94a3b8";
    }
  }

  function getPrimaryVideo() {
    const videos = [...document.querySelectorAll("video")].filter(
      (v) => v.videoWidth > 0 && v.videoHeight > 0
    );
    if (!videos.length) return null;
    return videos.sort(
      (a, b) => b.clientWidth * b.clientHeight - a.clientWidth * a.clientHeight
    )[0];
  }

  /**
   * 非同步：擷取一幀 → POST /detect → 依 JSON 更新 UI。
   * 連線失敗時暫停擷取並顯示「伺服器連線失敗」。
   */
  async function captureAndSend() {
    if (stopped || capturePaused || !currentVideo) return;
    if (currentVideo.readyState < 2 || currentVideo.paused) return;

    try {
      if (!canvas) {
        canvas = document.createElement("canvas");
        ctx = canvas.getContext("2d", { willReadFrequently: true });
      }
      canvas.width = currentVideo.videoWidth;
      canvas.height = currentVideo.videoHeight;
      ctx.drawImage(currentVideo, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.82);

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl }),
      });

      if (!res.ok) {
        setStatus(`後端錯誤 HTTP ${res.status}（請看終端機 uvicorn 日誌）`);
        setBarGray(0);
        setScoreTextHidden();
        return;
      }

      const data = await res.json();
      const msg = data.message;
      const reliable = data.is_reliable === true;

      // 未偵測到人臉：不顯示分數，灰條歸零
      if (msg === "No face detected") {
        setStatus("未偵測到人臉，等待中...");
        setBarGray(0);
        setScoreTextHidden();
        return;
      }

      // 有臉且已推論：一律顯示 Fake 分數與彩色進度條（1 FPS 下 rPPG 多半不可靠，不應隱藏主分數）
      // 使用與桌面版一致的 avg_fake_score（已校準/deque(15)/rPPG保守回拉）
      const fs = Math.max(0, Math.min(1, Number(data.avg_fake_score ?? data.fake_score) || 0));
      const pctStr = `${(fs * 100).toFixed(1)}%`;
      if (textEl) {
        textEl.textContent = pctStr;
        textEl.style.color = "#e2e8f0";
      }
      setBarByFakeScore(fs);
      if (reliable) {
        setStatus("即時偵測中（影像＋生理訊號）");
      } else {
        setStatus("即時偵測中（影像分數；生理訊號採樣中）");
      }
    } catch (err) {
      console.warn("[PREVERVECT] fetch 失敗:", err);
      setStatus("伺服器連線失敗");
      setBarGray(0);
      setScoreTextHidden();
      // 暫停擷取，避免持續噴錯誤
      if (timerId) {
        clearInterval(timerId);
        timerId = null;
      }
      capturePaused = true;
    }
  }

  function startDetection() {
    createPanel();
    currentVideo = getPrimaryVideo();
    if (!currentVideo) {
      setStatus("找不到影片標籤，等待中...");
      return;
    }
    capturePaused = false;
    setStatus("即時偵測中...");
    if (timerId) clearInterval(timerId);
    timerId = setInterval(() => {
      captureAndSend();
    }, CAPTURE_MS);
  }

  createPanel();
  const bootstrap = setInterval(() => {
    if (stopped) {
      clearInterval(bootstrap);
      return;
    }
    currentVideo = getPrimaryVideo();
    if (currentVideo) {
      startDetection();
      clearInterval(bootstrap);
    }
  }, 1000);
})();
