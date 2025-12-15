let selected = null;
let lastJobId = null;

const $ = (id) => document.getElementById(id);
function setStatus(t) { $("status").textContent = t || ""; }

async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
}

function openModal(id, on) { $(id).classList.toggle("on", !!on); }

function renderSources(list) {
    const wrap = $("sources");
    wrap.innerHTML = "";
    list.forEach((s) => {
        const div = document.createElement("div");
        div.className = "item";
        div.dataset.id = s.id;
        div.innerHTML = `
      <div style="font-weight:700">${s.name}</div>
      <div class="muted">${s.type === "video" ? s.path : "RTSP"}</div>
    `;
        div.onclick = () => selectSource(s, div);
        wrap.appendChild(div);
    });
}

function resetResultsUI() {
    $("links").innerHTML = "";
    $("heatImg").style.display = "none";
    $("heatImg").src = "";
    $("overlayImg").style.display = "none";
    $("overlayImg").src = "";
    lastJobId = null;
}

/* =========================
   ZONES EDITOR
========================= */
let editingZones = false;
let zonesData = null;     // payload tal cual de /api/zones
let drawZones = [];       // zonas en coord del video actual (native)
let baseW = 0, baseH = 0; // tamaño real del preview actual (videoWidth/Height o img natural)
let drag = null;          // {z, p}
const canvas = $("zoneCanvas");
const ctx = canvas.getContext("2d");

function getActiveMedia() {
    const vid = $("vid");
    const img = $("mjpeg");
    if (vid.style.display !== "none") return { type: "video", el: vid };
    return { type: "img", el: img };
}

function resizeCanvasToMedia() {
    const { el } = getActiveMedia();
    const rect = el.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
    drawZoneCanvas();
}

function toCanvas(pt) {
    if (!baseW || !baseH) return { x: pt[0], y: pt[1] };
    return {
        x: (pt[0] / baseW) * canvas.width,
        y: (pt[1] / baseH) * canvas.height
    };
}

function toNative(x, y) {
    if (!baseW || !baseH) return [Math.round(x), Math.round(y)];
    return [
        Math.round((x / canvas.width) * baseW),
        Math.round((y / canvas.height) * baseH)
    ];
}

function drawZoneCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!editingZones || !drawZones.length) return;

    ctx.lineWidth = 2;

    drawZones.forEach((z, zi) => {
        // polígono
        ctx.beginPath();
        z.points.forEach((p, i) => {
            const c = toCanvas(p);
            if (i === 0) ctx.moveTo(c.x, c.y);
            else ctx.lineTo(c.x, c.y);
        });
        ctx.closePath();
        ctx.strokeStyle = "rgba(96,165,250,0.95)";
        ctx.stroke();
        ctx.fillStyle = "rgba(96,165,250,0.12)";
        ctx.fill();

        // puntos (vértices)
        z.points.forEach((p) => {
            const c = toCanvas(p);
            ctx.beginPath();
            ctx.arc(c.x, c.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(245,158,11,0.95)";
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.35)";
            ctx.stroke();
        });

        // label
        const c0 = toCanvas(z.points[0]);
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.font = "12px system-ui";
        ctx.fillText(`Z${zi}`, c0.x + 8, c0.y - 8);
    });
}

function hitTest(mx, my) {
    const thr = 10; // px en canvas
    for (let zi = 0; zi < drawZones.length; zi++) {
        const pts = drawZones[zi].points;
        for (let pi = 0; pi < pts.length; pi++) {
            const c = toCanvas(pts[pi]);
            const dx = mx - c.x;
            const dy = my - c.y;
            if (Math.sqrt(dx * dx + dy * dy) <= thr) return { z: zi, p: pi };
        }
    }
    return null;
}

function zoneMouseDown(ev) {
    if (!editingZones) return;
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    const h = hitTest(mx, my);
    if (h) drag = h;
}

function zoneMouseMove(ev) {
    if (!editingZones || !drag) return;
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left;
    const my = ev.clientY - r.top;
    const nat = toNative(mx, my);
    drawZones[drag.z].points[drag.p] = nat;
    drawZoneCanvas();
}

function zoneMouseUp() { drag = null; }

async function loadZonesFromServer() {
    zonesData = await fetchJSON("/api/zones");
}

function ensureBaseDims() {
    const { type, el } = getActiveMedia();
    if (type === "video") {
        baseW = el.videoWidth || baseW;
        baseH = el.videoHeight || baseH;
    } else {
        baseW = el.naturalWidth || baseW;
        baseH = el.naturalHeight || baseH;
    }
}

function rescaleZonesToCurrentMedia() {
    ensureBaseDims();
    const zW = (zonesData && zonesData.video_width) || baseW || 1;
    const zH = (zonesData && zonesData.video_height) || baseH || 1;
    const sx = (baseW || 1) / (zW || 1);
    const sy = (baseH || 1) / (zH || 1);

    drawZones = (zonesData.zones || []).map((z, idx) => ({
        id: z.id ?? idx,
        name: z.name ?? `Zona ${idx}`,
        points: (z.points || []).map(p => [Math.round(p[0] * sx), Math.round(p[1] * sy)])
    }));
}

async function startZoneEdit() {
    if (!selected) { setStatus("Seleccione una fuente"); return; }

    // cargar zonas actuales
    await loadZonesFromServer();

    // esperar a tener dimensiones reales del preview
    ensureBaseDims();
    if (!baseW || !baseH) {
        setStatus("Cargando dimensiones del preview...");
        return;
    }

    rescaleZonesToCurrentMedia();
    editingZones = true;
    canvas.style.pointerEvents = "auto";
    $("zoneTools").classList.add("on");
    resizeCanvasToMedia();
    setStatus("Editando zonas (arrastre vértices).");
}

async function saveZoneEdit() {
    if (!editingZones || !baseW || !baseH) return;

    const payload = {
        video_width: baseW,
        video_height: baseH,
        zones: drawZones.map((z, idx) => ({
            id: idx,
            name: z.name || `Zona ${idx}`,
            points: z.points
        }))
    };

    await fetchJSON("/api/zones", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    // refrescar cache local
    zonesData = payload;
    setStatus("Zonas guardadas. El próximo análisis y heatmap usarán estas zonas.");
    stopZoneEdit();
}

function stopZoneEdit() {
    editingZones = false;
    drag = null;
    canvas.style.pointerEvents = "none";
    $("zoneTools").classList.remove("on");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

window.addEventListener("resize", () => {
    if (editingZones) resizeCanvasToMedia();
});
canvas.addEventListener("mousedown", zoneMouseDown);
canvas.addEventListener("mousemove", zoneMouseMove);
window.addEventListener("mouseup", zoneMouseUp);

/* =========================
   APP FLOW
========================= */

function selectSource(src, el) {
    selected = src;
    [...document.querySelectorAll(".item")].forEach((x) => x.classList.remove("active"));
    el.classList.add("active");

    $("selInfo").textContent = `${src.name} (${src.type})`;
    resetResultsUI();
    stopZoneEdit();

    const vid = $("vid");
    const mjpeg = $("mjpeg");

    if (src.type === "video") {
        mjpeg.style.display = "none";
        mjpeg.src = "";

        vid.style.display = "block";
        vid.src = `/media/${src.path}`;
        vid.load();

        vid.onloadedmetadata = () => {
            baseW = vid.videoWidth || 0;
            baseH = vid.videoHeight || 0;
            if (editingZones) resizeCanvasToMedia();
        };
    } else {
        vid.style.display = "none";
        vid.src = "";

        mjpeg.style.display = "block";
        mjpeg.src = `/stream/${src.id}`;

        mjpeg.onload = () => {
            baseW = mjpeg.naturalWidth || 0;
            baseH = mjpeg.naturalHeight || 0;
            if (editingZones) resizeCanvasToMedia();
        };
    }

    setStatus("");
}

async function loadSources() {
    const data = await fetchJSON("/api/sources");
    renderSources(data.sources || []);
}

async function startAnalyze(mode) {
    if (!selected) { setStatus("Seleccione una fuente"); return; }
    openModal("modal", false);

    setStatus("Creando job...");
    const j = await fetchJSON("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_id: selected.id, mode })
    });

    lastJobId = j.job_id;
    setStatus(`Job ${lastJobId}: en proceso...`);
    pollJob(lastJobId);
}

async function pollJob(jobId) {
    while (true) {
        const st = await fetchJSON(`/api/jobs/${jobId}`);

        if (st.status === "running" || st.status === "queued") {
            const total = st.frames_total ?? 0;
            const done = st.frames_done ?? 0;
            const pct = st.progress_pct ?? 0;
            if (total > 0) setStatus(`Job ${jobId}: ${st.status}... ${pct.toFixed(2)}% (${done}/${total})`);
            else setStatus(`Job ${jobId}: ${st.status}... frames=${done}`);
            await new Promise(r => setTimeout(r, 1500));
            continue;
        }

        if (st.status === "error") {
            setStatus(`Job ${jobId}: ERROR`);
            $("links").textContent = st.error || "error";
            return;
        }

        if (st.status === "done") {
            setStatus(`Job ${jobId}: listo`);
            const art = await fetchJSON(`/api/jobs/${jobId}/artifacts`);
            renderArtifacts(jobId, art);
            return;
        }
    }
}

function renderArtifacts(jobId, art) {
    const lines = [];
    if (art.csv_positions) lines.push(`CSV posiciones: <a href="/api/jobs/${jobId}/download/positions">descargar</a>`);
    if (art.csv_counts) lines.push(`CSV conteos: <a href="/api/jobs/${jobId}/download/counts">descargar</a>`);
    if (art.out_video) lines.push(`Video analizado: <a href="/api/jobs/${jobId}/download/video">descargar</a>`);
    $("links").innerHTML = lines.join(" · ");
}

async function generateHeatmap() {
    if (!lastJobId) { setStatus("Primero ejecuta un análisis"); return; }
    setStatus("Generando heatmap...");

    const res = await fetchJSON("/api/heatmap", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: lastJobId })
    });

    const art = res.artifacts || {};
    if (art.heatmap) {
        $("heatImg").style.display = "block";
        $("heatImg").src = `/${art.heatmap}?t=${Date.now()}`;
    }
    if (art.overlay) {
        $("overlayImg").style.display = "block";
        $("overlayImg").src = `/${art.overlay}?t=${Date.now()}`;
    }

    setStatus("Heatmap listo");
}

function openCsvModal() {
    if (!lastJobId) { setStatus("Primero ejecuta un análisis"); return; }
    openModal("csvModal", true);
}
function downloadPositions() { openModal("csvModal", false); window.location.href = `/api/jobs/${lastJobId}/download/positions`; }
function downloadCounts() { openModal("csvModal", false); window.location.href = `/api/jobs/${lastJobId}/download/counts`; }

document.addEventListener("DOMContentLoaded", () => {
    $("btnAnalyze").onclick = () => openModal("modal", true);
    $("mFast").onclick = () => startAnalyze("fast");
    $("mSlow").onclick = () => startAnalyze("slow");
    $("mClose").onclick = () => openModal("modal", false);

    $("btnCsv").onclick = openCsvModal;
    $("csvPos").onclick = downloadPositions;
    $("csvCnt").onclick = downloadCounts;
    $("csvClose").onclick = () => openModal("csvModal", false);

    $("btnHeat").onclick = generateHeatmap;

    $("btnZones").onclick = () => {
        if (!editingZones) startZoneEdit().catch(e => setStatus("Error editor zonas: " + e.message));
        else stopZoneEdit();
    };
    $("btnZonesSave").onclick = () => saveZoneEdit().catch(e => setStatus("Error guardando zonas: " + e.message));
    $("btnZonesCancel").onclick = stopZoneEdit;

    loadSources().catch(e => setStatus("Error cargando fuentes: " + e.message));
});
