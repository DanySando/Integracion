# mapa_calor.py
import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2


def _draw_legend_box(img_bgr, lines, x=18, y=18, font_scale=0.65, thickness=2, pad=10, alpha=0.55):
    """
    Dibuja un recuadro semitransparente con texto (leyenda) en img_bgr.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[
        0] for line in lines]
    max_w = max([s[0] for s in sizes]) if sizes else 0
    line_h = max([s[1] for s in sizes]) if sizes else int(18 * font_scale)
    box_w = max_w + pad * 2
    box_h = (line_h + 6) * len(lines) + pad * 2

    x2 = min(img_bgr.shape[1] - 1, x + box_w)
    y2 = min(img_bgr.shape[0] - 1, y + box_h)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, alpha, img_bgr, 1.0 - alpha, 0.0, img_bgr)

    ty = y + pad + line_h
    for line in lines:
        cv2.putText(img_bgr, line, (x + pad, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        ty += (line_h + 6)


def main():
    p = argparse.ArgumentParser("Heatmap histórico desde múltiples CSV")
    p.add_argument("--logs-dir", dest="logs_dir",
                   default="logs", help="Carpeta con CSVs")
    p.add_argument("--pattern", default="people_positions*.csv",
                   help="Patrón de archivos CSV")
    p.add_argument("--video", default="",
                   help="Ruta a video (para tamaño y 1er frame, opcional)")
    p.add_argument("--out-prefix", default="hist")
    p.add_argument("--scale", type=float, default=0.75)
    p.add_argument("--blur", type=int, default=31)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--zone", type=int, default=None, help="Filtrar por zona")
    p.add_argument("--from-date", dest="from_date",
                   default=None, help="YYYY-MM-DD")
    p.add_argument("--to-date", dest="to_date",
                   default=None, help="YYYY-MM-DD")
    p.add_argument("--radius", type=int, default=8,
                   help="Radio del punto por detección")
    p.add_argument("--colormap", default="JET",
                   help="INFERNO|TURBO|JET|PLASMA")
    p.add_argument("--clip-quantile", type=float, default=0.95,
                   help="Recorta picos (0.90–0.999)")
    p.add_argument("--log-scale", action="store_true",
                   help="Aplica log1p antes de normalizar")
    p.add_argument("--gamma", type=float, default=2.6,
                   help=">1 atenúa medios / resalta contraste")
    p.add_argument("--legend", action="store_true",
                   help="Dibuja % por intensidad (aprox) y % por zona")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.logs_dir, args.pattern)))
    if not paths:
        raise SystemExit(
            f"No se encontraron CSVs en {args.logs_dir} con patrón {args.pattern}")

    dfs = [pd.read_csv(pth) for pth in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Filtros
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

    # Coordenadas (detecciones)
    xs = df["cx"].to_numpy(dtype=np.float32)
    ys = df["cy"].to_numpy(dtype=np.float32)

    # Tamaño base desde video (si existe) o desde datos
    first_frame = None
    w = h = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, ff = cap.read()
            cap.release()
            first_frame = ff if ret else None

    if not w or not h:
        w = int(np.nanmax(xs)) + 1
        h = int(np.nanmax(ys)) + 1

    # Acumulación (heatmap)
    scale = max(0.1, min(1.0, float(args.scale)))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    xi = np.clip((xs * scale).astype(np.int32), 0, sw - 1)
    yi = np.clip((ys * scale).astype(np.int32), 0, sh - 1)

    heat = np.zeros((sh, sw), dtype=np.float32)
    r = max(1, int(args.radius))
    for x, y in zip(xi, yi):
        cv2.circle(heat, (int(x), int(y)), r, 1.0, -1)

    # Anti-saturación
    q = max(0.0, min(0.9999, float(args.clip_quantile)))
    if q < 1.0 and np.any(heat > 0):
        vmax = float(np.quantile(heat[heat > 0], q))
        if vmax > 0:
            heat = np.minimum(heat, vmax)

    if args.log_scale:
        heat = np.log1p(heat)

    if args.blur and int(args.blur) % 2 == 1:
        k = int(args.blur)
        heat = cv2.GaussianBlur(heat, (k, k), 0)

    heat_norm = cv2.normalize(heat, None, 0.0, 1.0, cv2.NORM_MINMAX)
    if args.gamma and float(args.gamma) > 0:
        heat_norm = np.power(heat_norm, float(args.gamma))

    heat_u8 = (heat_norm * 255).astype(np.uint8)

    cmap_map = {
        "INFERNO": cv2.COLORMAP_INFERNO,
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
        "PLASMA": cv2.COLORMAP_PLASMA,
    }
    cmap = cmap_map.get(args.colormap.upper(), cv2.COLORMAP_JET)
    heat_color = cv2.applyColorMap(heat_u8, cmap)
    heat_color = cv2.resize(heat_color, (w, h), interpolation=cv2.INTER_CUBIC)

    hm = f"{args.out_prefix}_heatmap.png"
    cv2.imwrite(hm, heat_color)

    # Overlay
    ov = f"{args.out_prefix}_heatmap_overlay.png"
    if first_frame is not None:
        if first_frame.ndim == 3 and first_frame.shape[2] == 3:
            overlay = cv2.addWeighted(
                first_frame, 1.0 - float(args.alpha),
                heat_color, float(args.alpha), 0.0
            )
        else:
            first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_BGRA2BGR)
            overlay = cv2.addWeighted(
                first_frame_bgr, 1.0 - float(args.alpha),
                heat_color, float(args.alpha), 0.0
            )
    else:
        overlay = heat_color.copy()

    # ---- LEYENDA con porcentajes ----
    if args.legend:
        # Intensidad por detección (aprox) usando heat_norm en grilla pequeña (antes de resize)
        vals_det = heat_norm[yi, xi] if len(
            xi) else np.array([], dtype=np.float32)

        bins = [
            ("Muy bajo (azul)", 0.00, 0.20),
            ("Bajo (cian)", 0.20, 0.40),
            ("Medio (verde)", 0.40, 0.60),
            ("Alto (amarillo)", 0.60, 0.80),
            ("Muy alto (rojo)", 0.80, 1.01),
        ]

        # ¿Hay tracking válido para contar personas únicas?
        has_id = "id" in df.columns
        has_valid_ids = False
        if has_id:
            # coerción defensiva
            try:
                ids = pd.to_numeric(df["id"], errors="coerce")
                has_valid_ids = bool((ids >= 0).fillna(False).any())
            except Exception:
                has_valid_ids = False

        lines = []

        if has_valid_ids:
            # Construir df mínimo con id, zone y val (val por detección)
            tmp = df.copy()
            tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce")
            tmp = tmp[tmp["id"].notna() & (tmp["id"] >= 0)].copy()

            # Alinear valores de intensidad por fila
            # (vals_det está alineado con df original por orden)
            tmp["__val"] = vals_det[tmp.index.to_numpy()]

            # Tomar un valor por persona: máximo (representa el tramo más intenso por el que pasó)
            per_person_val = tmp.groupby(
                "id")["__val"].max().to_numpy(dtype=np.float32)
            n_people = int(len(per_person_val))

            lines.append(
                "Distribución por intensidad (personas únicas, aprox):")
            if n_people > 0:
                for name, a, b in bins:
                    c = int(np.sum((per_person_val >= a) & (per_person_val < b)))
                    pct = (c / n_people) * 100.0
                    lines.append(f"{name}: {pct:5.2f}% ({c})")
            else:
                lines.append("Sin datos para intensidad (IDs).")

            # Distribución por zona en personas únicas (si existe zone)
            if "zone" in tmp.columns:
                lines.append("")
                lines.append("Distribución por zona (personas únicas):")
                per_zone_people = tmp.groupby(
                    "zone")["id"].nunique().sort_index()
                total_people = int(tmp["id"].nunique())
                for z, c in per_zone_people.items():
                    try:
                        z_int = int(z)
                    except Exception:
                        z_int = z
                    pct = (int(c) / total_people) * \
                        100.0 if total_people > 0 else 0.0
                    lines.append(f"Zona {z_int}: {pct:5.2f}% ({int(c)})")

        else:
            # Fallback a detecciones (lo que tenías antes), pero lo deja claro
            n = int(len(vals_det))
            lines.append("Distribución por intensidad (detecciones, aprox):")
            if n > 0:
                for name, a, b in bins:
                    c = int(np.sum((vals_det >= a) & (vals_det < b)))
                    pct = (c / n) * 100.0
                    lines.append(f"{name}: {pct:5.2f}% ({c})")
            else:
                lines.append("Sin datos para intensidad.")

            if "zone" in df.columns:
                lines.append("")
                lines.append("Distribución por zona (detecciones):")
                zc = df["zone"].value_counts(dropna=False).sort_index()
                total = int(zc.sum())
                for z, c in zc.items():
                    try:
                        z_int = int(z)
                    except Exception:
                        z_int = z
                    pct = (int(c) / total) * 100.0 if total > 0 else 0.0
                    lines.append(f"Zona {z_int}: {pct:5.2f}% ({int(c)})")

        # Escala de texto según resolución
        fs = 0.60
        if w >= 1600:
            fs = 0.70
        if w >= 2200:
            fs = 0.80

        _draw_legend_box(overlay, lines, x=18, y=18,
                         font_scale=fs, thickness=2, pad=10, alpha=0.55)

    cv2.imwrite(ov, overlay)

    print(f"[OK] Heatmap: {hm}")
    print(f"[OK] Overlay: {ov}")
    print(f"Archivos fusionados: {len(paths)}  |  puntos: {len(df)}")


if __name__ == "__main__":
    main()
