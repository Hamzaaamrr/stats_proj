(() => {
  const defaults = JSON.parse(document.body.dataset.defaults || "{}");
  const byId = (id) => document.getElementById(id);
  const ui = {
    status: byId("connectionStatus"),
    session: byId("sessionId"),
    stage: byId("videoStage"),
    video: byId("videoFeed"),
    empty: byId("videoEmpty"),
    liveMessage: byId("liveMessage"),
    fps: byId("fpsValue"),
    behaviorPanel: byId("behaviorPanel"),
    behavior: byId("behaviorValue"),
    detail: byId("assessmentDetail"),
    confidence: byId("confidenceValue"),
    confidenceBar: byId("confidenceBar"),
    ear: byId("earValue"),
    blink: byId("blinkValue"),
    perclos: byId("perclosValue"),
    headPose: byId("headPoseValue"),
    headAngles: byId("headAngles"),
    riskBanner: byId("riskBanner"),
    riskText: byId("riskText"),
    start: byId("startButton"),
    stop: byId("stopButton"),
    recalibrate: byId("recalibrateButton"),
    mute: byId("muteButton"),
    form: byId("settingsForm"),
    driver: byId("driverId"),
    camera: byId("cameraIndex"),
    lighting: byId("lighting"),
    position: byId("cameraPosition"),
    glasses: byId("glasses"),
    file: byId("videoFile"),
    uploadName: byId("uploadName"),
    testMode: byId("testMode"),
    objects: byId("objectDetection"),
    audio: byId("audioAlerts"),
    trueLabelField: byId("trueLabelField"),
    trueLabel: byId("trueLabel"),
    formError: byId("formError"),
    historyCount: byId("historyCount"),
    historyRows: byId("historyRows"),
    historySearch: byId("historySearch"),
    selectAllSessions: byId("selectAllSessions"),
    deleteSelected: byId("deleteSelected"),
    refreshHistory: byId("refreshHistory"),
    previousPage: byId("previousPage"),
    nextPage: byId("nextPage"),
    pageLabel: byId("pageLabel"),
    historyError: byId("historyError"),
    historyDetail: byId("historyDetail"),
    detailTitle: byId("detailTitle"),
    detailSummary: byId("detailSummary"),
    detailEvents: byId("detailEvents"),
    closeDetails: byId("closeDetails"),
  };

  let running = false;
  let pendingStart = false;
  let streamSession = "";
  let polling = false;
  let refreshedFinishedSession = "";
  const historyLimit = 25;
  let historyOffset = 0;
  let historyTotal = 0;
  let historyLoading = false;
  const maxSelectedSessions = 100;
  const selectedSessions = new Set();

  ui.driver.value = defaults.driver || "D01";
  ui.camera.value = Number.isInteger(defaults.camera) ? defaults.camera : 0;
  ui.lighting.value = defaults.lighting || "";
  ui.position.value = defaults.camera_position || "";
  ui.glasses.value = defaults.glasses || "";
  ui.testMode.checked = Boolean(defaults.test_mode);
  ui.objects.checked = defaults.objects !== false;
  ui.audio.checked = defaults.audio !== false;
  if (defaults.video) {
    ui.uploadName.textContent = `Command-line video: ${defaults.video.split(/[\\/]/).pop()}`;
  }

  function setStatus(label, state = "ready") {
    ui.status.dataset.state = state;
    ui.status.lastElementChild.textContent = label;
  }

  function setError(message) {
    ui.formError.textContent = message;
    ui.formError.hidden = !message;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, {
      cache: "no-store",
      ...options,
      headers: { ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Request failed (${response.status}).`);
    }
    return payload;
  }

  function setControls(active) {
    running = active;
    ui.start.disabled = active;
    ui.stop.disabled = !active;
    ui.recalibrate.disabled = !active;
    ui.mute.disabled = !active;
    ui.form.querySelectorAll("input, select, button").forEach((element) => {
      element.disabled = active;
    });
    ui.trueLabel.disabled = active || !ui.testMode.checked;
  }

  function updateMetrics(metrics = {}) {
    const behavior = metrics.behavior || "Ready";
    const confidence = Number(metrics.confidence || 0);
    const risk = !["Ready", "Starting", "Calibrating", "Normal"].includes(behavior);

    ui.behavior.textContent = behavior;
    ui.behaviorPanel.dataset.risk = String(risk);
    ui.confidence.textContent = behavior === "Calibrating"
      ? "CALIBRATING"
      : behavior === "Ready" || behavior === "Starting"
        ? "—"
        : `${Math.round(confidence * 100)}%`;
    ui.confidenceBar.value = Math.max(0, Math.min(100, confidence * 100));
    if (behavior === "Calibrating") {
      ui.detail.textContent = `Look ahead with eyes open · ${Math.round((metrics.calibration_progress || 0) * 100)}%`;
    } else if (behavior === "Ready" || behavior === "Starting") {
      ui.detail.textContent = "No active session";
    } else {
      ui.detail.textContent = metrics.face_detected ? "Driver detected" : "Face not detected";
    }

    ui.ear.textContent = Number.isFinite(metrics.ear) ? metrics.ear.toFixed(3) : "—";
    ui.blink.textContent = Number.isFinite(metrics.blink_rate) ? Math.round(metrics.blink_rate) : "—";
    ui.perclos.textContent = Number.isFinite(metrics.perclos) ? `${Math.round(metrics.perclos * 100)}%` : "—";
    ui.headPose.textContent = metrics.head_pose || "—";
    const yaw = Number.isFinite(metrics.yaw) ? `${metrics.yaw >= 0 ? "+" : ""}${Math.round(metrics.yaw)}°` : "—";
    const pitch = Number.isFinite(metrics.pitch) ? `${metrics.pitch >= 0 ? "+" : ""}${Math.round(metrics.pitch)}°` : "—";
    ui.headAngles.textContent = `Yaw ${yaw} · Pitch ${pitch}`;
    ui.fps.textContent = Number.isFinite(metrics.fps) ? `${Math.round(metrics.fps)} FPS` : "— FPS";
    ui.mute.textContent = metrics.audio_enabled === false ? "Enable alerts" : "Mute alerts";
    ui.trueLabel.disabled = running || !ui.testMode.checked;

    ui.riskBanner.hidden = !risk;
    if (risk) {
      ui.riskText.textContent = `${behavior} detected · ${Math.round(confidence * 100)}% confidence`;
    }
  }

  function updateStatus(data) {
    if (!pendingStart && data.running !== running) {
      setControls(data.running);
    }
    ui.deleteSelected.disabled = running || selectedSessions.size === 0;
    if (data.running && data.session_id) {
      ui.session.textContent = `SESSION ${data.session_id}`;
    } else {
      ui.session.textContent = "NO ACTIVE SESSION";
    }
    ui.session.parentElement.dataset.active = String(data.running);
    ui.liveMessage.textContent = data.message || "Ready when you are.";

    if (data.has_frame && data.session_id && streamSession !== data.session_id) {
      streamSession = data.session_id;
      ui.video.src = `/video-feed?session=${encodeURIComponent(streamSession)}`;
      ui.video.hidden = false;
      ui.empty.hidden = true;
    }
    ui.stage.dataset.live = String(Boolean(data.running && data.has_frame));
    if (data.finished) {
      ui.video.removeAttribute("src");
      ui.video.hidden = true;
      ui.empty.hidden = false;
      ui.stage.dataset.live = "false";
      streamSession = "";
      if (data.saved && data.session_id && data.session_id !== refreshedFinishedSession) {
        refreshedFinishedSession = data.session_id;
        loadHistory();
      }
    }
    if (!data.has_frame && !streamSession) {
      ui.video.hidden = true;
      ui.empty.hidden = false;
    }

    if (data.error) {
      setControls(false);
      setStatus("ERROR", "error");
      setError(data.error);
    } else if (data.running) {
      const behavior = data.metrics && data.metrics.behavior;
      setStatus(behavior === "Calibrating" ? "CALIBRATING" : "MONITORING",
        behavior === "Calibrating" ? "calibrating" : "ready");
      setError("");
    } else {
      setStatus(data.finished ? (data.saved ? "SAVED" : "NOT SAVED") : "READY", "ready");
      if (data.finished) {
        setControls(false);
        updateMetrics({});
      }
    }
    updateMetrics(data.finished ? {} : data.metrics || {});
  }

  function formatDate(value) {
    if (!value) return "—";
    const date = new Date(value.replace(" ", "T"));
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
  }

  function formatDuration(value) {
    const total = Math.max(0, Math.round(Number(value) || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    if (hours) return `${hours}h ${minutes}m`;
    if (minutes) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
  }

  function appendCell(row, value, className = "") {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    cell.textContent = value == null || value === "" ? "—" : String(value);
    row.appendChild(cell);
    return cell;
  }

  function applyHistoryFilter() {
    const query = ui.historySearch.value.trim().toLowerCase();
    ui.historyRows.querySelectorAll("tr[data-search]").forEach((row) => {
      row.hidden = Boolean(query) && !row.dataset.search.includes(query);
    });
  }

  function updateDeleteButton() {
    const count = selectedSessions.size;
    ui.deleteSelected.textContent = count ? `Delete selected (${count})` : "Delete selected";
    ui.deleteSelected.disabled = running || count === 0;
  }

  function renderHistory(items) {
    ui.historyRows.replaceChildren();
    ui.selectAllSessions.checked = false;
    ui.selectAllSessions.indeterminate = false;
    if (!items.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.className = "history-empty";
      cell.colSpan = 9;
      cell.textContent = historyTotal ? "No sessions on this page." : "No saved sessions yet.";
      row.appendChild(cell);
      ui.historyRows.appendChild(row);
      updateDeleteButton();
      return;
    }

    for (const session of items) {
      const sessionId = session.Session_ID || "";
      const row = document.createElement("tr");
      row.dataset.session = sessionId;
      row.dataset.search = `${sessionId} ${session.Driver_ID || ""}`.toLowerCase();
      if (selectedSessions.has(sessionId)) row.classList.add("selected-row");

      const selectCell = document.createElement("td");
      selectCell.className = "select-column";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selectedSessions.has(sessionId);
      checkbox.setAttribute("aria-label", `Select session ${sessionId}`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked && !selectedSessions.has(sessionId)
            && selectedSessions.size >= maxSelectedSessions) {
          checkbox.checked = false;
          ui.historyError.textContent = `Select no more than ${maxSelectedSessions} sessions at a time.`;
          ui.historyError.hidden = false;
          return;
        }
        ui.historyError.hidden = true;
        if (checkbox.checked) selectedSessions.add(sessionId);
        else selectedSessions.delete(sessionId);
        row.classList.toggle("selected-row", checkbox.checked);
        updateDeleteButton();
        updateSelectAll(items);
      });
      selectCell.appendChild(checkbox);
      row.appendChild(selectCell);

      appendCell(row, sessionId, "session-id");
      appendCell(row, session.Driver_ID);
      appendCell(row, formatDate(session.Start_Time));
      appendCell(row, formatDuration(session.Duration_s));
      appendCell(row, session.Events || "0");
      appendCell(row, session.Risky_Events || "0", "risk-count");
      appendCell(row, session.Calibrated === "1" ? "Calibrated" : "Fallback",
        session.Calibrated === "1" ? "calibration-ok" : "calibration-failed");

      const detailCell = document.createElement("td");
      const detailButton = document.createElement("button");
      detailButton.type = "button";
      detailButton.className = "table-detail-button";
      detailButton.textContent = "View";
      detailButton.addEventListener("click", () => showSessionDetails(sessionId));
      detailCell.appendChild(detailButton);
      row.appendChild(detailCell);
      ui.historyRows.appendChild(row);
    }
    applyHistoryFilter();
    updateSelectAll(items);
    updateDeleteButton();
  }

  function updateSelectAll(items) {
    const ids = items.map((item) => item.Session_ID);
    const selectedCount = ids.filter((id) => selectedSessions.has(id)).length;
    ui.selectAllSessions.checked = ids.length > 0 && selectedCount === ids.length;
    ui.selectAllSessions.indeterminate = selectedCount > 0 && selectedCount < ids.length;
  }

  async function loadHistory() {
    if (historyLoading) return;
    historyLoading = true;
    ui.historyError.hidden = true;
    try {
      const data = await request(`/api/history?offset=${historyOffset}&limit=${historyLimit}`);
      historyTotal = data.total;
      if (historyOffset >= historyTotal && historyTotal > 0) {
        historyOffset = Math.floor((historyTotal - 1) / historyLimit) * historyLimit;
        historyLoading = false;
        return loadHistory();
      }
      if (historyTotal === 0) historyOffset = 0;
      const first = historyTotal ? historyOffset + 1 : 0;
      const last = Math.min(historyOffset + historyLimit, historyTotal);
      ui.historyCount.textContent = historyTotal
        ? `Showing ${first}–${last} of ${historyTotal}`
        : "0 sessions";
      ui.pageLabel.textContent = historyTotal
        ? `Page ${Math.floor(historyOffset / historyLimit) + 1} of ${Math.ceil(historyTotal / historyLimit)}`
        : "Page 1";
      ui.previousPage.disabled = historyOffset === 0;
      ui.nextPage.disabled = historyOffset + historyLimit >= historyTotal;
      renderHistory(data.items);
    } catch (error) {
      ui.historyCount.textContent = "History unavailable";
      ui.historyRows.replaceChildren();
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.className = "history-empty";
      cell.colSpan = 9;
      cell.textContent = error.message;
      row.appendChild(cell);
      ui.historyRows.appendChild(row);
    } finally {
      historyLoading = false;
    }
  }

  function summaryItem(label, value) {
    const item = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = `${label}: `;
    item.append(title, document.createTextNode(value || "—"));
    ui.detailSummary.appendChild(item);
  }

  async function showSessionDetails(sessionId) {
    ui.historyError.hidden = true;
    try {
      const result = await request(`/api/history/${encodeURIComponent(sessionId)}`);
      const session = result.session;
      ui.detailTitle.textContent = `Session ${session.Session_ID}`;
      ui.detailSummary.replaceChildren();
      summaryItem("Driver", session.Driver_ID);
      summaryItem("Started", formatDate(session.Start_Time));
      summaryItem("Duration", formatDuration(session.Duration_s));
      summaryItem("Source", session.Source);
      summaryItem("Lighting", session.Lighting);
      summaryItem("Camera position", session.Camera_Position);
      summaryItem("Glasses", session.Glasses);
      ui.detailEvents.replaceChildren();
      if (!result.events.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 5;
        cell.className = "history-empty";
        cell.textContent = "No event records for this session.";
        row.appendChild(cell);
        ui.detailEvents.appendChild(row);
      } else {
        for (const event of result.events) {
          const row = document.createElement("tr");
          appendCell(row, formatDate(event.Timestamp));
          appendCell(row, event.Behavior);
          appendCell(row, formatDuration(event.Duration_s));
          appendCell(row, event.Trigger);
          appendCell(row, event.Alert, event.Alert === "ON" ? "alert-on" : "");
          ui.detailEvents.appendChild(row);
        }
      }
      ui.historyDetail.hidden = false;
      ui.historyDetail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (error) {
      ui.historyError.textContent = error.message;
      ui.historyError.hidden = false;
    }
  }

  async function deleteSelectedSessions() {
    const ids = Array.from(selectedSessions);
    if (!ids.length || running) return;
    const names = ids.slice(0, 5).join(", ");
    const suffix = ids.length > 5 ? ` and ${ids.length - 5} more` : "";
    if (!window.confirm(`Permanently delete ${ids.length} session(s) (${names}${suffix}) and their event/sample records?`)) {
      return;
    }
    ui.historyError.hidden = true;
    ui.deleteSelected.disabled = true;
    try {
      const result = await request("/api/history", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_ids: ids }),
      });
      for (const id of result.deleted_sessions) selectedSessions.delete(id);
      ui.historyDetail.hidden = true;
      await loadHistory();
    } catch (error) {
      ui.historyError.textContent = error.message;
      ui.historyError.hidden = false;
      updateDeleteButton();
    }
  }

  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const data = await request("/api/status");
      if (!pendingStart || data.running) updateStatus(data);
    } catch (error) {
      setStatus("OFFLINE", "error");
      ui.liveMessage.textContent = error.message;
    } finally {
      polling = false;
      window.setTimeout(poll, 500);
    }
  }

  async function startSession(event) {
    event.preventDefault();
    setError("");
    if (!ui.form.reportValidity()) return;

    pendingStart = true;
    setControls(true);
    ui.stop.disabled = true;
    ui.liveMessage.textContent = "Preparing session...";
    let uploadId = "";
    const file = ui.file.files[0];
    try {
      if (file) {
        ui.liveMessage.textContent = "Uploading video...";
        const formData = new FormData();
        formData.append("video", file);
        const upload = await request("/api/upload-video", { method: "POST", body: formData });
        uploadId = upload.upload_id;
      }
      const settings = {
        driver: ui.driver.value.trim(),
        camera: Number(ui.camera.value),
        lighting: ui.lighting.value,
        camera_position: ui.position.value,
        glasses: ui.glasses.value,
        test_mode: ui.testMode.checked,
        objects: ui.objects.checked,
        audio: ui.audio.checked,
        video_upload: uploadId,
      };
      streamSession = "";
      ui.video.hidden = true;
      ui.empty.hidden = false;
      setStatus("STARTING", "calibrating");
      ui.behavior.textContent = "Starting";
      ui.detail.textContent = "Loading models and opening camera";
      await request("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      pendingStart = false;
      setControls(true);
      poll();
    } catch (error) {
      pendingStart = false;
      setControls(false);
      setStatus("READY", "error");
      setError(error.message);
      ui.liveMessage.textContent = "Could not start this session.";
    }
  }

  async function postCommand(action, label) {
    try {
      await request("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, label }),
      });
    } catch (error) {
      setError(error.message);
    }
  }

  ui.form.addEventListener("submit", startSession);
  ui.start.addEventListener("click", startSession);
  ui.stop.addEventListener("click", async () => {
    try {
      await request("/api/stop", { method: "POST" });
      ui.stop.disabled = true;
      ui.liveMessage.textContent = "Stopping safely and saving the session...";
    } catch (error) {
      setError(error.message);
    }
  });
  ui.recalibrate.addEventListener("click", () => postCommand("recalibrate"));
  ui.mute.addEventListener("click", () => postCommand("toggle_audio"));
  ui.file.addEventListener("change", () => {
    ui.uploadName.textContent = ui.file.files[0]
      ? ui.file.files[0].name
      : defaults.video
        ? `Command-line video: ${defaults.video.split(/[\\/]/).pop()}`
        : "Optional · MP4, MOV, AVI or MKV · max 512 MB";
  });
  ui.testMode.addEventListener("change", () => {
    ui.trueLabelField.hidden = !ui.testMode.checked;
    ui.trueLabel.disabled = running || !ui.testMode.checked;
  });
  ui.trueLabel.addEventListener("change", () => {
    if (running && ui.testMode.checked) postCommand("true_label", ui.trueLabel.value);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (event.key === "Escape" || event.key.toLowerCase() === "q") {
      if (running) ui.stop.click();
    } else if (event.key.toLowerCase() === "c" && running) {
      postCommand("recalibrate");
    } else if (event.key.toLowerCase() === "m" && running) {
      postCommand("toggle_audio");
    } else if (/^[0-5]$/.test(event.key) && running && ui.testMode.checked) {
      ui.trueLabel.value = ["Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking"][Number(event.key)];
      postCommand("true_label", ui.trueLabel.value);
    }
  });

  ui.trueLabelField.hidden = !ui.testMode.checked;
  ui.refreshHistory.addEventListener("click", () => loadHistory());
  ui.deleteSelected.addEventListener("click", deleteSelectedSessions);
  ui.previousPage.addEventListener("click", () => {
    historyOffset = Math.max(0, historyOffset - historyLimit);
    loadHistory();
  });
  ui.nextPage.addEventListener("click", () => {
    historyOffset += historyLimit;
    loadHistory();
  });
  ui.selectAllSessions.addEventListener("change", () => {
    ui.selectAllSessions.indeterminate = false;
    let limitReached = false;
    ui.historyRows.querySelectorAll("tr[data-session]").forEach((row) => {
      const checkbox = row.querySelector('input[type="checkbox"]');
      const shouldSelect = ui.selectAllSessions.checked;
      if (shouldSelect && !selectedSessions.has(row.dataset.session)
          && selectedSessions.size >= maxSelectedSessions) {
        checkbox.checked = false;
        limitReached = true;
      } else {
        checkbox.checked = shouldSelect;
      }
      if (checkbox.checked) selectedSessions.add(row.dataset.session);
      else selectedSessions.delete(row.dataset.session);
      row.classList.toggle("selected-row", checkbox.checked);
    });
    const pageItems = Array.from(
      ui.historyRows.querySelectorAll("tr[data-session]"),
      (row) => ({ Session_ID: row.dataset.session }),
    );
    updateSelectAll(pageItems);
    updateDeleteButton();
    if (limitReached) {
      ui.historyError.textContent = `Select no more than ${maxSelectedSessions} sessions at a time.`;
      ui.historyError.hidden = false;
    } else {
      ui.historyError.hidden = true;
    }
  });
  ui.historySearch.addEventListener("input", applyHistoryFilter);
  ui.closeDetails.addEventListener("click", () => {
    ui.historyDetail.hidden = true;
  });
  poll();
  loadHistory();
})();
