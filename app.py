import json
import uuid
import time
import threading
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, Any, List

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

(BASE_DIR / "static").mkdir(exist_ok=True)
(BASE_DIR / "media").mkdir(exist_ok=True)
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

SOURCES_FILE = BASE_DIR / "sources.json"

app = FastAPI(title="People Counter Dashboard")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(BASE_DIR / "media")), name="media")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


def load_sources() -> Dict[str, Any]:
    if not SOURCES_FILE.exists():
        return {"sources": []}
    return json.loads(SOURCES_FILE.read_text(encoding="utf-8"))


def get_source(source_id: str) -> Dict[str, Any]:
    data = load_sources()
    for s in data.get("sources", []):
        if s.get("id") == source_id:
            return s
    raise KeyError(source_id)


JOBS: Dict[str, Dict[str, Any]] = {}


class AnalyzeReq(BaseModel):
    source_id: str
    mode: str  # fast | slow


class HeatmapReq(BaseModel):
    job_id: str
    colormap: str = "JET"
    gamma: float = 2.6
    clip_quantile: float = 0.95
    log_scale: bool = True
    legend: bool = True
    radius: int = 8
    blur: int = 31
    alpha: float = 0.6


def run_cmd(cmd, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/sources")
def api_sources():
    return load_sources()


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    mode = req.mode.lower().strip()
    if mode not in ("fast", "slow"):
        raise HTTPException(400, "mode debe ser fast o slow")

    try:
        _ = get_source(req.source_id)
    except KeyError:
        raise HTTPException(404, "source_id no existe")

    job_id = uuid.uuid4().hex[:10]
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    JOBS[job_id] = {
        "job_id": job_id,
        "source_id": req.source_id,
        "mode": mode,
        "status": "queued",
        "created_at": time.time(),
        "run_dir": str(run_dir),
        "error": None,
        "artifacts": {},
        # progreso
        "frames_done": 0,
        "frames_total": 0,
        "progress_pct": 0.0,
    }

    threading.Thread(target=run_analysis_job,
                     args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job_id no existe")
    j = JOBS[job_id].copy()
    j.pop("run_dir", None)
    return j


@app.get("/api/jobs/{job_id}/artifacts")
def api_job_artifacts(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job_id no existe")
    return JOBS[job_id].get("artifacts", {})


@app.get("/api/jobs/{job_id}/download/{kind}")
def api_download(job_id: str, kind: str):
    if job_id not in JOBS:
        raise HTTPException(404, "job_id no existe")

    art = JOBS[job_id].get("artifacts", {})
    keymap = {
        "positions": "csv_positions",
        "counts": "csv_counts",
        "video": "out_video",
        "heatmap": "heatmap",
        "overlay": "overlay",
    }
    if kind not in keymap:
        raise HTTPException(400, "kind inválido")

    rel = art.get(keymap[kind])
    if not rel:
        raise HTTPException(404, "archivo no disponible")

    path = BASE_DIR / rel
    if not path.exists():
        raise HTTPException(404, "archivo no encontrado")

    return FileResponse(str(path), filename=path.name)


_re_total = re.compile(r"\[TOTAL_FRAMES\]\s+(\d+)")
_re_prog = re.compile(r"\[PROGRESS\]\s+frames=(\d+)\s+total=(\d+)")


def run_analysis_job(job_id: str):
    job = JOBS[job_id]
    job["status"] = "running"

    run_dir = Path(job["run_dir"])

    try:
        src = get_source(job["source_id"])
        mode = job["mode"]

        if src["type"] == "video":
            p = (BASE_DIR / "media" / src["path"]).resolve()
            if not p.exists():
                raise RuntimeError(f"Video no existe: {p}")
            input_arg = str(p)
        elif src["type"] == "rtsp":
            input_arg = src["url"]
        else:
            raise RuntimeError("type de source no soportado")

        out_video = run_dir / "out.mp4"
        csv_prefix = "people"
        session = job_id

        w_fast = (BASE_DIR / "yolov8n.pt").resolve()
        w_slow = (BASE_DIR / "yolov8s.pt").resolve()
        if mode == "fast" and not w_fast.exists():
            raise RuntimeError(f"No existe yolov8n.pt en: {w_fast}")
        if mode == "slow" and not w_slow.exists():
            raise RuntimeError(f"No existe yolov8s.pt en: {w_slow}")

        script = (BASE_DIR / "conteo_unificado.py").resolve()
        if not script.exists():
            raise RuntimeError(f"No existe conteo_unificado.py en: {script}")

        if mode == "fast":
            cmd = [
                sys.executable, str(script),
                "-i", input_arg,
                "-o", str(out_video),
                "--model", str(w_fast),
                "--imgsz", "480",
                "--frame-step", "5",
                "--log-dir", str(run_dir),
                "--csv-prefix", csv_prefix,
                "--session", session,
                "--track",
            ]
        else:
            cmd = [
                sys.executable, str(script),
                "-i", input_arg,
                "-o", str(out_video),
                "--model", str(w_slow),
                "--imgsz", "1280",
                "--frame-step", "1",
                "--log-dir", str(run_dir),
                "--csv-prefix", csv_prefix,
                "--session", session,
                "--track",
            ]

        # ejecutar con stdout en vivo
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        out_lines: List[str] = []
        assert proc.stdout is not None

        for line in proc.stdout:
            s = line.rstrip("\n")
            out_lines.append(s)

            m1 = _re_total.search(s)
            if m1:
                job["frames_total"] = int(m1.group(1))
                job["progress_pct"] = 0.0

            m2 = _re_prog.search(s)
            if m2:
                done = int(m2.group(1))
                total = int(m2.group(2))
                job["frames_done"] = done
                job["frames_total"] = total
                if total > 0:
                    job["progress_pct"] = round((done / total) * 100.0, 2)
                else:
                    job["progress_pct"] = 0.0

        code = proc.wait()

        # log completo
        (run_dir / "analysis_cmd.log").write_text(
            "CMD:\n" + " ".join(cmd) + "\n\nOUTPUT:\n" + "\n".join(out_lines),
            encoding="utf-8",
            errors="replace",
        )

        if code != 0:
            tail = "\n".join(out_lines[-60:]) if out_lines else ""
            raise RuntimeError(
                f"subprocess error: exit status {code}\n\nLAST OUTPUT:\n{tail}")

        csv_positions = run_dir / f"{csv_prefix}_positions_{session}.csv"
        csv_counts = run_dir / f"{csv_prefix}_counts_{session}.csv"

        job["artifacts"] = {
            "out_video": f"runs/{job_id}/out.mp4",
            "csv_positions": f"runs/{job_id}/{csv_positions.name}" if csv_positions.exists() else None,
            "csv_counts": f"runs/{job_id}/{csv_counts.name}" if csv_counts.exists() else None,
            "heatmap": None,
            "overlay": None,
        }

        job["frames_done"] = job["frames_total"] or job["frames_done"]
        job["progress_pct"] = 100.0 if (
            job["frames_total"] or 0) > 0 else job["progress_pct"]
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/heatmap")
def api_heatmap(req: HeatmapReq):
    if req.job_id not in JOBS:
        raise HTTPException(404, "job_id no existe")

    job = JOBS[req.job_id]
    if job["status"] != "done":
        raise HTTPException(400, "job no está listo")

    run_dir = RUNS_DIR / req.job_id
    if not run_dir.exists():
        raise HTTPException(404, "run_dir no existe")

    src = get_source(job["source_id"])
    video_arg = ""
    if src["type"] == "video":
        p = (BASE_DIR / "media" / src["path"]).resolve()
        if p.exists():
            video_arg = str(p)

    script = (BASE_DIR / "mapa_calor.py").resolve()
    if not script.exists():
        raise HTTPException(500, f"No existe mapa_calor.py en: {script}")

    cmd = [
        sys.executable, str(script),
        "--logs-dir", str(run_dir),
        "--pattern", "people_positions*.csv",
        "--video", video_arg,
        "--out-prefix", "hist",
        "--colormap", req.colormap,
        "--radius", str(req.radius),
        "--blur", str(req.blur),
        "--clip-quantile", str(req.clip_quantile),
        "--gamma", str(req.gamma),
        "--alpha", str(req.alpha),
    ]

    if req.legend:
        cmd.append("--legend")
    if req.log_scale:
        cmd.append("--log-scale")

    try:
        cp = run_cmd(cmd, cwd=str(run_dir))
        (run_dir / "heatmap_cmd.log").write_text(
            "CMD:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" +
            (cp.stdout or "") + "\n\nSTDERR:\n" + (cp.stderr or ""),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            500,
            f"heatmap subprocess error:\nCMD: {' '.join(e.cmd)}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}",
        )

    heatmap = run_dir / "hist_heatmap.png"
    overlay = run_dir / "hist_heatmap_overlay.png"

    job["artifacts"]["heatmap"] = f"runs/{req.job_id}/{heatmap.name}" if heatmap.exists(
    ) else None
    job["artifacts"]["overlay"] = f"runs/{req.job_id}/{overlay.name}" if overlay.exists(
    ) else None

    return {"ok": True, "artifacts": job["artifacts"]}


def mjpeg_generator(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok2, jpg = cv2.imencode(".jpg", frame)
            if not ok2:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/stream/{source_id}")
def stream_rtsp(source_id: str):
    try:
        src = get_source(source_id)
    except KeyError:
        raise HTTPException(404, "source_id no existe")

    if src.get("type") != "rtsp":
        raise HTTPException(400, "source_id no es RTSP")

    return StreamingResponse(
        mjpeg_generator(src["url"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
