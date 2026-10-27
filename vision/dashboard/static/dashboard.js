// Deskbot Vision dashboard: polls the camera worker, sends button presses,
// filter toggles and settings to the server, and handles dragging the desk box.

const $ = (id) => document.getElementById(id);
const POLL_MS = 500;

let status = null;
let selecting = false;
let stopped = false;
let settingsLoaded = false;
let lastLogKey = "";
let toastTimer = null;

// ----- server calls --------------------------------------------------------

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

async function action(name, body) {
  try {
    return await post(`/api/action/${name}`, body);
  } catch (e) {
    toast(e.message);
    return null;
  }
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3000);
}

// ----- status polling ------------------------------------------------------

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

function render(s) {
  const cal = s.calibration;
  const corners = cal.calibrated ? "desk calibrated" : `corners ${cal.visible.length}/4`;
  const pico = s.pico ? (s.pico.connected ? "Pico connected" : "Pico not connected") : "";
  $("summary").textContent = [`${Math.round(s.stats.fps)} fps`, corners, pico]
    .filter(Boolean).join(" · ");

  $("warning").hidden = !s.warning;
  $("warning").textContent = s.warning || "";

  $("btn-pause").textContent = s.paused ? "Resume" : "Pause";
  $("learn-text").textContent = s.learning
    ? `Learning ${s.learning.done}/${s.learning.total}`
    : s.desk ? "" : "Desk not learned";

  // Filters follow the server (another tab may have changed them).
  document.querySelectorAll(".toggle").forEach((box) => {
    box.checked = !!s.toggles[box.value];
  });

  $("bot").textContent = s.bot
    ? `x ${s.bot.x} · y ${s.bot.y} · ${s.bot.hd}°`
    : cal.calibrated ? "Marker not visible" : "Waiting for the corner markers";
  $("bot").classList.toggle("muted", !s.bot);

  const rows = s.objects.map((o) => {
    const tr = document.createElement("tr");
    tr.append(cell(o.id), cell(o.x), cell(o.y), cell(`${o.w} × ${o.h}`), cell(o.a));
    return tr;
  });
  $("objects").querySelector("tbody").replaceChildren(...rows);
  $("objects").hidden = rows.length === 0;
  $("no-objects").hidden = rows.length > 0;
  $("no-objects").textContent = s.desk ? "None" : "Learn the desk first";
  $("units").textContent = s.objects.length ? `(${s.objects[0].units === "mm" ? "mm, mm²" : "pixels"})` : "";

  renderPico(s.pico);
  if (!settingsLoaded) {
    fillSettings(s.settings);
    settingsLoaded = true;
  }
  renderLog(s.log);
}

function renderPico(p) {
  const rows = p ? [
    ["Status", p.connected ? `Connected on ${p.port}` : (p.error || "Not connected")],
    ["Port setting", p.port_setting],
    ["Sent", `${p.sent} messages at ${p.send_hz}/s`],
    ["Last sent", p.last_message],
    ["Last reply", p.last_reply],
  ] : [["Status", "No link"]];
  $("pico").replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    if (k === "Last sent" && v) {
      const code = document.createElement("code");
      code.textContent = v;
      dd.append(code);
    } else {
      dd.textContent = v ?? "–";
    }
    return [dt, dd];
  }));
}

function renderLog(entries) {
  const last = entries[entries.length - 1];
  const key = `${entries.length}|${last ? last.time + last.message : ""}`;
  if (key === lastLogKey) return;
  lastLogKey = key;
  $("log").replaceChildren(...entries.slice().reverse().map((e) => {
    const li = document.createElement("li");
    li.className = e.level;
    const time = document.createElement("time");
    time.textContent = e.time;
    li.append(time, e.message);
    return li;
  }));
}

async function poll() {
  if (stopped) return;
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) throw new Error();
    status = await res.json();
    $("dot").classList.add("on");
    render(status);
    if (!status.running) showStopped();
  } catch {
    $("dot").classList.remove("on");
  }
  setTimeout(poll, POLL_MS);
}

function showStopped() {
  stopped = true;
  $("dot").classList.remove("on");
  cancelSelecting();
  $("stopped").hidden = false;
  document.querySelectorAll("button, input").forEach((el) => { el.disabled = true; });
}

// ----- buttons and keys ----------------------------------------------------

function toggleLearn() {
  if (selecting) cancelSelecting();
  else startSelecting();
}

function togglePause() {
  if (status) action(status.paused ? "resume" : "pause");
}

async function snapshot() {
  const result = await action("snapshot");
  if (result) toast(`Saved ${result.file}`);
}

async function quit() {
  const result = await action("quit");
  if (result) showStopped();
}

$("btn-learn").addEventListener("click", toggleLearn);
$("btn-pause").addEventListener("click", togglePause);
$("btn-snapshot").addEventListener("click", snapshot);
$("btn-quit").addEventListener("click", quit);
$("btn-clear").addEventListener("click", () => action("clear_desk"));
$("btn-recal").addEventListener("click", () => action("recalibrate"));

document.addEventListener("keydown", (e) => {
  if (stopped || e.ctrlKey || e.metaKey || e.altKey) return;
  const tag = document.activeElement && document.activeElement.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;

  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (key === "Escape") {
    // Esc first cancels a desk selection in progress; otherwise it quits.
    if (selecting) cancelSelecting();
    else quit();
  } else if (key === "Enter" && selecting) {
    cancelSelecting();
    action("learn", { default: true });
  } else if (key === "q") quit();
  else if (key === "l") toggleLearn();
  else if (key === "p") togglePause();
  else if (key === "s") snapshot();
  else return;
  e.preventDefault();
});

// ----- filters -------------------------------------------------------------

document.querySelectorAll(".toggle").forEach((box) => {
  box.addEventListener("change", () => {
    post("/api/view", { toggles: { [box.value]: box.checked } }).catch((err) => toast(err.message));
  });
});

// ----- dragging the desk box -----------------------------------------------

const canvas = $("selector");
const ctx = canvas.getContext("2d");
let dragStart = null;
let dragRect = null;

function startSelecting() {
  if (!status || !status.stats.frame_size) return toast("Waiting for the first camera frame");
  if (status.paused) return toast("Resume the camera before learning the desk");
  selecting = true;
  $("frame").classList.add("selecting");
  $("hint").hidden = false;
  $("btn-learn").classList.add("active");
  const box = canvas.getBoundingClientRect();
  canvas.width = Math.round(box.width * devicePixelRatio);
  canvas.height = Math.round(box.height * devicePixelRatio);
}

function cancelSelecting() {
  selecting = false;
  dragStart = dragRect = null;
  $("frame").classList.remove("selecting");
  $("hint").hidden = true;
  $("btn-learn").classList.remove("active");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function pointerPos(e) {
  const box = canvas.getBoundingClientRect();
  return {
    x: Math.min(Math.max(e.clientX - box.left, 0), box.width),
    y: Math.min(Math.max(e.clientY - box.top, 0), box.height),
  };
}

function drawRect() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!dragRect) return;
  const d = devicePixelRatio;
  const r = dragRect;
  ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.clearRect(r.x * d, r.y * d, r.w * d, r.h * d);
  ctx.strokeStyle = "#ffc800";
  ctx.lineWidth = 2 * d;
  ctx.strokeRect(r.x * d, r.y * d, r.w * d, r.h * d);
}

canvas.addEventListener("pointerdown", (e) => {
  if (!selecting) return;
  canvas.setPointerCapture(e.pointerId);
  dragStart = pointerPos(e);
  dragRect = { ...dragStart, w: 0, h: 0 };
});

canvas.addEventListener("pointermove", (e) => {
  if (!dragStart) return;
  const p = pointerPos(e);
  dragRect = {
    x: Math.min(dragStart.x, p.x),
    y: Math.min(dragStart.y, p.y),
    w: Math.abs(p.x - dragStart.x),
    h: Math.abs(p.y - dragStart.y),
  };
  drawRect();
});

canvas.addEventListener("pointerup", async () => {
  if (!dragStart) return;
  const r = dragRect;
  dragStart = null;
  if (!r || r.w < 6 || r.h < 6) {
    dragRect = null;
    drawRect();
    toast("Drag a larger box, or press Enter for the default area");
    return;
  }
  // Convert from on-screen pixels to camera frame pixels.
  const box = canvas.getBoundingClientRect();
  const [fw, fh] = status.stats.frame_size;
  const sx = fw / box.width;
  const sy = fh / box.height;
  cancelSelecting();
  await action("learn", {
    x: Math.round(r.x * sx),
    y: Math.round(r.y * sy),
    w: Math.round(r.w * sx),
    h: Math.round(r.h * sy),
  });
});

// ----- settings ------------------------------------------------------------

const form = $("settings");
const NUMBER_FIELDS = [
  "learn_frames", "blur_kernel", "l_weight", "diff_threshold", "open_ksize", "close_ksize",
  "min_object_area", "desk_width_mm", "desk_height_mm", "send_hz", "jpeg_quality", "max_fps",
];

function fillSettings(s) {
  for (const name of NUMBER_FIELDS) form.elements[name].value = s[name];
  s.std_floor.forEach((v, i) => { form.elements[`std_floor_${i}`].value = v; });
  form.elements.pico_port.value = s.pico_port;
}

function readSettings() {
  const values = {};
  for (const name of NUMBER_FIELDS) values[name] = Number(form.elements[name].value);
  values.std_floor = [0, 1, 2].map((i) => Number(form.elements[`std_floor_${i}`].value));
  values.pico_port = form.elements.pico_port.value.trim() || "auto";
  return values;
}

function settingsMessage(text, kind) {
  $("settings-msg").textContent = text;
  $("settings-msg").className = `msg ${kind || ""}`;
}

async function applySettings(save) {
  try {
    await post(`/api/settings${save ? "?save=1" : ""}`, readSettings());
    settingsMessage(save ? "Saved." : "Applied.", "ok");
  } catch (e) {
    settingsMessage(e.message, "error");
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  applySettings(false);
});
$("btn-save").addEventListener("click", () => applySettings(true));
$("btn-revert").addEventListener("click", () => {
  if (status) fillSettings(status.settings);
  settingsMessage("Reverted.");
});

poll();
