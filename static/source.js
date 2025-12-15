let selected = null;
let lastJobId = null;

const $ = (id) => document.getElementById(id);

function setStatus(t) { $("status").textContent = t || ""; }

async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    if (!r.ok) throw new Error(await r.text());
    return await r.json();
}

function openModal(id, on) {
    $(id).classList.toggle("on", !!on);
}

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

function selectSource(src, el) {
    selected = src;
    [...document.querySelectorAll(".item")].forEach((x) => x.classList.remove("active"));
    el.classList.add("active");

    $("selInfo").textContent = `${src.name} (${src.type})`;
    resetResultsUI();

    if (src.type === "video") {
        $("mjpeg").style.display = "none";
        $("mjpeg").src = "";

        $("vid").style.display = "block";
        $("vid").src = `/media/${src.path}`;
        $("vid").load();
    } else {
        $("vid").style.display = "none";
        $("vid").src = "";

        $("mjpeg").style.display = "block";
        $("mjpeg").src = `/stream/${src.id}`;
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

            if (total > 0) {
                setStatus(`Job ${jobId}: ${st.status}... ${pct.toFixed(2)}% (${done}/${total})`);
            } else {
                setStatus(`Job ${jobId}: ${st.status}... frames=${done}`);
            }

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

function downloadPositions() {
    openModal("csvModal", false);
    window.location.href = `/api/jobs/${lastJobId}/download/positions`;
}

function downloadCounts() {
    openModal("csvModal", false);
    window.location.href = `/api/jobs/${lastJobId}/download/counts`;
}

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

    loadSources().catch(e => setStatus("Error cargando fuentes: " + e.message));
});
