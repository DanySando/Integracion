import os
import csv
import argparse
import datetime as dt

import numpy as np
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser("Detecta/trackea y guarda histórico")
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)

parser.add_argument("--model", required=True,
                    help="Ruta a yolov8n.pt / yolov8s.pt")
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--track", action="store_true")
parser.add_argument("--frame-step", type=int, default=1,
                    help="Procesar 1 de cada N frames")
parser.add_argument("--gpu", action="store_true", help="Intentar usar CUDA")

parser.add_argument("--csv-prefix", default="people")
parser.add_argument("--log-dir", default="logs")
parser.add_argument("--append", action="store_true")
parser.add_argument("--session", default=None)
parser.add_argument("--start-datetime", default=None)

args = parser.parse_args()


class CountObject:
    def __init__(self, input_video_path: str, output_video_path: str) -> None:
        print("[INFO] Cargando modelo YOLO...", flush=True)
        self.model = YOLO(args.model)

        if args.gpu:
            try:
                self.model.to("cuda")
                print("[INFO] Usando GPU (cuda)", flush=True)
            except Exception:
                print("[INFO] GPU no disponible, usando CPU", flush=True)

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        os.makedirs(args.log_dir, exist_ok=True)
        self.session = args.session or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[INFO] Sesión: {self.session}", flush=True)

        self.polygons = [
            np.array([[540, 985], [1620, 985], [2160, 1920], [
                     1620, 2855], [540, 2855], [0, 1920]], np.int32),
            np.array([[0, 1920], [540, 985], [0, 0]], np.int32),
            np.array([[1620, 985], [2160, 1920], [2160, 0]], np.int32),
            np.array([[540, 985], [0, 0], [2160, 0], [1620, 985]], np.int32),
            np.array([[0, 1920], [0, 3840], [540, 2855]], np.int32),
            np.array([[2160, 1920], [1620, 2855], [2160, 3840]], np.int32),
            np.array([[1620, 2855], [540, 2855], [
                     0, 3840], [2160, 3840]], np.int32),
        ]

        cap_tmp = cv2.VideoCapture(input_video_path)
        if not cap_tmp.isOpened():
            print("[ERROR] No se pudo abrir el input para resolución/FPS", flush=True)
            self.frame_width = 1920
            self.frame_height = 1080
            self.fps = 25.0
        else:
            self.frame_width = int(cap_tmp.get(
                cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            self.frame_height = int(cap_tmp.get(
                cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            self.fps = float(cap_tmp.get(cv2.CAP_PROP_FPS) or 25.0)
            cap_tmp.release()

        if self.fps <= 0:
            self.fps = 25.0

        print(
            f"[INFO] Resolución detectada: {self.frame_width}x{self.frame_height}", flush=True)
        print(f"[INFO] FPS detectados: {self.fps}", flush=True)
        print(f"[INFO] Frame-step: {args.frame_step}", flush=True)

        self.start_dt = None
        if args.start_datetime:
            try:
                self.start_dt = dt.datetime.fromisoformat(args.start_datetime)
            except Exception:
                self.start_dt = None

        self.zone_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (128, 128, 255),
        ]
        if len(self.zone_colors) < len(self.polygons):
            extra = len(self.polygons) - len(self.zone_colors)
            self.zone_colors.extend(self.zone_colors[:extra])

        self.logs_positions = []
        self.logs_counts = []

        base = args.csv_prefix.rstrip("_")
        if args.append:
            self.csv_positions = os.path.join(
                args.log_dir, f"{base}_positions_master.csv")
            self.csv_counts = os.path.join(
                args.log_dir, f"{base}_counts_master.csv")
        else:
            self.csv_positions = os.path.join(
                args.log_dir, f"{base}_positions_{self.session}.csv")
            self.csv_counts = os.path.join(
                args.log_dir, f"{base}_counts_{self.session}.csv")

        print(f"[INFO] CSV posiciones: {self.csv_positions}", flush=True)
        print(f"[INFO] CSV conteos: {self.csv_counts}", flush=True)

    def _export_csv(self):
        print("[INFO] Exportando CSV...", flush=True)
        mode_pos = "a" if (args.append and os.path.exists(
            self.csv_positions)) else "w"
        mode_cnt = "a" if (args.append and os.path.exists(
            self.csv_counts)) else "w"

        with open(self.csv_positions, mode_pos, newline="") as f:
            fields = ["session", "frame", "ts",
                      "zone", "id", "cx", "cy", "fps"]
            if self.start_dt:
                fields.insert(2, "datetime")
            w = csv.DictWriter(f, fieldnames=fields)
            if mode_pos == "w":
                w.writeheader()
            for r in self.logs_positions:
                w.writerow(r)

        with open(self.csv_counts, mode_cnt, newline="") as f:
            fields = ["session", "frame", "ts", "zone", "count", "fps"]
            if self.start_dt:
                fields.insert(2, "datetime")
            w = csv.DictWriter(f, fieldnames=fields)
            if mode_cnt == "w":
                w.writeheader()
            for r in self.logs_counts:
                w.writerow(r)

        print("[INFO] CSV exportados correctamente", flush=True)

    def _get_detections_from_result(self, res):
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
            )

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        mask = (cls == 0) & (conf >= args.conf)
        xyxy = xyxy[mask]
        conf = conf[mask]
        cls = cls[mask]

        if args.track and getattr(boxes, "id", None) is not None:
            all_ids = boxes.id.cpu().numpy().astype(int)
            ids = all_ids[mask]
        else:
            ids = np.full(len(xyxy), -1, dtype=int)

        return xyxy, conf, cls, ids

    def process_frame(self, frame: np.ndarray, i: int) -> np.ndarray:
        if args.track:
            res = self.model.track(
                frame, imgsz=args.imgsz, persist=True, verbose=False)[0]
        else:
            res = self.model(frame, imgsz=args.imgsz, verbose=False)[0]

        xyxy, conf, cls, ids = self._get_detections_from_result(res)

        ts = float(i) / float(self.fps or 25.0)
        dt_iso = None
        if self.start_dt:
            dt_iso = (self.start_dt + dt.timedelta(seconds=ts)).isoformat()

        zone_counts = [0] * len(self.polygons)

        for z_idx, poly in enumerate(self.polygons):
            cv2.polylines(frame, [poly], isClosed=True,
                          color=self.zone_colors[z_idx], thickness=4)

        if xyxy.shape[0] == 0:
            for z_idx in range(len(self.polygons)):
                rec_cnt = {
                    "session": self.session,
                    "frame": int(i),
                    "ts": ts,
                    "zone": int(z_idx),
                    "count": 0,
                    "fps": self.fps,
                }
                if dt_iso:
                    rec_cnt["datetime"] = dt_iso
                self.logs_counts.append(rec_cnt)
            return frame

        cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0).astype(int)
        cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0).astype(int)

        for k in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[k].astype(int)
            center = (int(cx[k]), int(cy[k]))

            zone_idx = None
            for z_idx, poly in enumerate(self.polygons):
                poly_for_cv = poly.reshape((-1, 1, 2))
                inside_flag = cv2.pointPolygonTest(poly_for_cv, center, False)
                if inside_flag >= 0:
                    zone_idx = z_idx
                    break

            color = (0, 255, 0)
            if zone_idx is not None:
                color = self.zone_colors[zone_idx]
                zone_counts[zone_idx] += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, center, 4, color, -1)

            track_id = int(ids[k]) if k < len(ids) else -1

            rec_pos = {
                "session": self.session,
                "frame": int(i),
                "ts": ts,
                "zone": int(zone_idx) if zone_idx is not None else -1,
                "id": track_id,
                "cx": int(center[0]),
                "cy": int(center[1]),
                "fps": self.fps,
            }
            if dt_iso:
                rec_pos["datetime"] = dt_iso
            self.logs_positions.append(rec_pos)

        for z_idx, c in enumerate(zone_counts):
            rec_cnt = {
                "session": self.session,
                "frame": int(i),
                "ts": ts,
                "zone": int(z_idx),
                "count": int(c),
                "fps": self.fps,
            }
            if dt_iso:
                rec_cnt["datetime"] = dt_iso
            self.logs_counts.append(rec_cnt)

        for z_idx, poly in enumerate(self.polygons):
            center_poly = np.mean(poly, axis=0).astype(int)
            label = f"Z{z_idx}: {zone_counts[z_idx]}"
            cv2.putText(
                frame,
                label,
                (int(center_poly[0]), int(center_poly[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.zone_colors[z_idx],
                2,
                cv2.LINE_AA,
            )
        return frame

    def process_video(self):
        print(f"[INFO] Abriendo input: {self.input_video_path}", flush=True)
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir el input", flush=True)
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[TOTAL_FRAMES] {total_frames}", flush=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_video_path, fourcc,
                              self.fps, (self.frame_width, self.frame_height))

        i = 0
        print("[INFO] Procesando frames...", flush=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del input", flush=True)
                break

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            if args.frame_step <= 1 or (i % args.frame_step == 0):
                frame = self.process_frame(frame, i)

            out.write(frame)

            i += 1
            if i % 60 == 0:
                print(
                    f"[PROGRESS] frames={i} total={total_frames}", flush=True)

        cap.release()
        out.release()
        print("[INFO] Video generado", flush=True)
        self._export_csv()


if __name__ == "__main__":
    CountObject(args.input, args.output).process_video()
