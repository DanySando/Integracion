# make_heatmap_historical.py
import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2


def main():
    p = argparse.ArgumentParser("Heatmap histórico desde múltiples CSV")
    p.add_argument("--logs-dir", dest="logs_dir",
                   default="logs", help="Carpeta con CSVs")
    p.add_argument("--pattern", default="people_positions*.csv",
                   help="Patrón de archivos CSV")
    p.add_argument("--video", default="demo2.mp4",
                   help="Para tamaño y 1er frame (opcional)")
    p.add_argument("--out-prefix", default="hist")
    p.add_argument("--scale", type=float, default=0.75)
    p.add_argument("--blur", type=int, default=51)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--zone", type=int, default=None, help="Filtrar por zona")
    p.add_argument("--from-date", dest="from_date",
                   default=None, help="YYYY-MM-DD")
    p.add_argument("--to-date", dest="to_date",
                   default=None, help="YYYY-MM-DD")
    p.add_argument("--radius", type=int, default=12,
                   help="Radio del punto por detección")
    p.add_argument("--colormap", default="INFERNO",
                   help="INFERNO|TURBO|JET|PLASMA")
    # NUEVO: control de intensidad para evitar saturación en históricos
    p.add_argument("--clip-quantile", type=float, default=0.98,
                   help="Recorta picos (0.90–0.999)")
    p.add_argument("--log-scale", action="store_true",
                   help="Aplica log1p antes de normalizar")
    p.add_argument("--gamma", type=float, default=1.6,
                   help=">1 atenúa picos tras normalizar")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs_dir, args.pattern)))
    if not paths:
        raise SystemExit(
            f"No se encontraron CSVs en {args.logs_dir} con patrón {args.pattern}")

    dfs = [pd.read_csv(pth) for pth in paths]
    df = pd.concat(dfs, ignore_index=True)

    if args.zone is not None and "zone" in df.columns:
        df = df[df["zone"] == args.zone]

    if "datetime" in df.columns and (args.from_date or args.to_date):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if args.from_date:
            df = df[df["datetime"] >= pd.to_datetime(args.from_date)]
        if args.to_date:
            df = df[df["datetime"] <= pd.to_datetime(args.to_date)]

    if df.empty:
        raise SystemExit("No quedó data tras los filtros.")

    xs = df["cx"].to_numpy(dtype=np.float32)
    ys = df["cy"].to_numpy(dtype=np.float32)

    # Tamaño base desde video o desde datos
    cap = cv2.VideoCapture(args.video)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            first_frame = None
    else:
        first_frame = None
        w = h = None

    if not w or not h:
        w = int(np.nanmax(xs)) + 1
        h = int(np.nanmax(ys)) + 1

    # Acumulación
    scale = max(0.1, min(1.0, args.scale))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    xi = np.clip((xs * scale).astype(np.int32), 0, sw - 1)
    yi = np.clip((ys * scale).astype(np.int32), 0, sh - 1)

    heat = np.zeros((sh, sw), dtype=np.float32)
    for x, y in zip(xi, yi):
        cv2.circle(heat, (int(x), int(y)), args.radius, 1.0, -1)

    # ---- Control de intensidad (anti-saturación) ----
    heat = heat.astype(np.float32)

    q = max(0.0, min(0.9999, args.clip_quantile))
    if q < 1.0 and np.any(heat > 0):
        vmax = float(np.quantile(heat[heat > 0], q))
        if vmax > 0:
            heat = np.minimum(heat, vmax)

    if args.log_scale:
        heat = np.log1p(heat)

    if args.blur and args.blur % 2 == 1:
        heat = cv2.GaussianBlur(heat, (args.blur, args.blur), 0)

    heat_norm = cv2.normalize(heat, None, 0.0, 1.0, cv2.NORM_MINMAX)
    if args.gamma and args.gamma > 0:
        heat_norm = np.power(heat_norm, args.gamma)
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    # -----------------------------------------------

    cmap_map = {
        "INFERNO": cv2.COLORMAP_INFERNO,
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
        "PLASMA": cv2.COLORMAP_PLASMA,
    }
    cmap = cmap_map.get(args.colormap.upper(), cv2.COLORMAP_INFERNO)
    heat_color = cv2.applyColorMap(heat_u8, cmap)
    heat_color = cv2.resize(heat_color, (w, h), interpolation=cv2.INTER_CUBIC)

    hm = f"{args.out_prefix}_heatmap.png"
    cv2.imwrite(hm, heat_color)

    ov = f"{args.out_prefix}_heatmap_overlay.png"
    if first_frame is not None:
        if first_frame.ndim == 3 and first_frame.shape[2] == 3:
            overlay = cv2.addWeighted(
                first_frame, 1.0 - args.alpha, heat_color, args.alpha, 0.0)
        else:
            first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_BGRA2BGR)
            overlay = cv2.addWeighted(
                first_frame_bgr, 1.0 - args.alpha, heat_color, args.alpha, 0.0)
        cv2.imwrite(ov, overlay)
    else:
        cv2.imwrite(ov, heat_color)

    print(f"[OK] Heatmap: {hm}")
    print(f"[OK] Overlay: {ov}")
    print(f"Archivos fusionados: {len(paths)}  |  puntos: {len(df)}")


if __name__ == "__main__":
    main()


# python mapa_calor.py --logs-dir logs --pattern "people_positions*.csv" --video demo2.mp4 --out-prefix hist

  # es el comando para generar el heatmap mas legible.
