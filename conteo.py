# conteo.py
import os
import csv
import argparse
import datetime as dt

import numpy as np
import cv2
from ultralytics import YOLO


parser = argparse.ArgumentParser("Detecta/trackea y guarda histórico")
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('--csv-prefix', default='people')
parser.add_argument('--imgsz', type=int, default=1280)
parser.add_argument('--conf', type=float, default=0.5)
parser.add_argument('--track', action='store_true')
# histórico
parser.add_argument(
    '--log-dir',
    default='logs',
    help='Carpeta para CSVs históricos'
)
parser.add_argument(
    '--append',
    action='store_true',
    help='Append a archivos *_master.csv'
)
parser.add_argument(
    '--session',
    default=None,
    help='ID de sesión (auto si None)'
)
parser.add_argument(
    '--start-datetime',
    default=None,
    help='ISO 8601 para timestampear (opcional)'
)
args = parser.parse_args()


class CountObject:
    def __init__(self, input_video_path, output_video_path) -> None:
        print("[INFO] Cargando modelo YOLO...")
        self.model = YOLO('yolov8n.pt')

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # sesión y carpeta de logs
        os.makedirs(args.log_dir, exist_ok=True)
        self.session = args.session or dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"[INFO] Sesión: {self.session}")

        # polígonos (ajusta a tu escena si quieres)
        # coordenadas en formato (x, y)
        self.polygons = [
            np.array([[540, 985], [1620, 985], [2160, 1920],
                      [1620, 2855], [540, 2855], [0, 1920]], np.int32),
            np.array([[0, 1920], [540, 985], [0, 0]], np.int32),
            np.array([[1620, 985], [2160, 1920], [2160, 0]], np.int32),
            np.array([[540, 985], [0, 0], [2160, 0], [1620, 985]], np.int32),
            np.array([[0, 1920], [0, 3840], [540, 2855]], np.int32),
            np.array([[2160, 1920], [1620, 2855], [2160, 3840]], np.int32),
            np.array([[1620, 2855], [540, 2855],
                      [0, 3840], [2160, 3840]], np.int32),
        ]

        # abrir video para extraer resolución y fps
        cap_tmp = cv2.VideoCapture(input_video_path)
        if not cap_tmp.isOpened():
            print("[ERROR] No se pudo abrir el video para leer resolución/FPS")
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
            f"[INFO] Resolución detectada: {self.frame_width}x{self.frame_height}")
        print(f"[INFO] FPS detectados: {self.fps}")

        # datetime de inicio (opcional)
        self.start_dt = None
        if args.start_datetime:
            try:
                self.start_dt = dt.datetime.fromisoformat(args.start_datetime)
            except Exception:
                self.start_dt = None

        # colores BGR por zona
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
            # repetir colores si faltan
            extra = len(self.polygons) - len(self.zone_colors)
            self.zone_colors.extend(self.zone_colors[:extra])

        # buffers
        self.logs_positions = []
        self.logs_counts = []

        # rutas CSV (histórico)
        base = args.csv_prefix.rstrip('_')
        if args.append:
            self.csv_positions = os.path.join(
                args.log_dir,
                f'{base}_positions_master.csv'
            )
            self.csv_counts = os.path.join(
                args.log_dir,
                f'{base}_counts_master.csv'
            )
        else:
            self.csv_positions = os.path.join(
                args.log_dir,
                f'{base}_positions_{self.session}.csv'
            )
            self.csv_counts = os.path.join(
                args.log_dir,
                f'{base}_counts_{self.session}.csv'
            )

        print(f"[INFO] CSV posiciones: {self.csv_positions}")
        print(f"[INFO] CSV conteos: {self.csv_counts}")

    def _export_csv(self):
        print("[INFO] Exportando CSV...")
        # append o write
        mode_pos = 'a' if (args.append and os.path.exists(
            self.csv_positions)) else 'w'
        mode_cnt = 'a' if (args.append and os.path.exists(
            self.csv_counts)) else 'w'

        # POSICIONES
        with open(self.csv_positions, mode_pos, newline='') as f:
            fields = ['session', 'frame', 'ts',
                      'zone', 'id', 'cx', 'cy', 'fps']
            if self.start_dt:
                fields.insert(2, 'datetime')  # opcional
            w = csv.DictWriter(f, fieldnames=fields)
            if mode_pos == 'w':
                w.writeheader()
            for r in self.logs_positions:
                w.writerow(r)

        # CONTEOS
        with open(self.csv_counts, mode_cnt, newline='') as f:
            fields = ['session', 'frame', 'ts', 'zone', 'count', 'fps']
            if self.start_dt:
                fields.insert(2, 'datetime')
            w = csv.DictWriter(f, fieldnames=fields)
            if mode_cnt == 'w':
                w.writeheader()
            for r in self.logs_counts:
                w.writerow(r)

        print("[INFO] CSV exportados correctamente")

    def _get_detections_from_result(self, res):
        """
        Extrae xyxy, conf, cls, ids (opcional) desde resultado YOLOv8.
        """
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            xyxy = np.empty((0, 4), dtype=float)
            conf = np.empty((0,), dtype=float)
            cls = np.empty((0,), dtype=int)
            ids = np.empty((0,), dtype=int)
            return xyxy, conf, cls, ids

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        # filtrar personas (class_id == 0) y umbral de confianza
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
        # detección o tracking
        if args.track:
            res = self.model.track(
                frame,
                imgsz=args.imgsz,
                persist=True,
                verbose=False
            )[0]
        else:
            res = self.model(
                frame,
                imgsz=args.imgsz,
                verbose=False
            )[0]

        xyxy, conf, cls, ids = self._get_detections_from_result(res)

        # timestamps
        ts = float(i) / float(self.fps or 25.0)
        dt_iso = None
        if self.start_dt:
            dt_iso = (self.start_dt + dt.timedelta(seconds=ts)).isoformat()

        # inicializar conteos por zona
        zone_counts = [0] * len(self.polygons)

        # dibujar polígonos primero
        for z_idx, poly in enumerate(self.polygons):
            cv2.polylines(
                frame,
                [poly],
                isClosed=True,
                color=self.zone_colors[z_idx],
                thickness=4
            )

        # si no hay detecciones, solo registrar conteos 0 y devolver frame
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

        # centros de cada detección
        cx = ((xyxy[:, 0] + xyxy[:, 2]) / 2.0).astype(int)
        cy = ((xyxy[:, 1] + xyxy[:, 3]) / 2.0).astype(int)

        # procesar cada detección
        for k in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[k].astype(int)
            center = (int(cx[k]), int(cy[k]))

            # determinar zona (primera en la que caiga el centro)
            zone_idx = None
            for z_idx, poly in enumerate(self.polygons):
                # pointPolygonTest requiere polígono en forma Nx1x2
                poly_for_cv = poly.reshape((-1, 1, 2))
                inside_flag = cv2.pointPolygonTest(poly_for_cv, center, False)
                if inside_flag >= 0:
                    zone_idx = z_idx
                    break

            # color para esta detección
            color = (0, 255, 0)
            if zone_idx is not None:
                color = self.zone_colors[zone_idx]
                zone_counts[zone_idx] += 1

            # dibujar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # dibujar centro
            cv2.circle(frame, center, 4, color, -1)

            # id de tracking (si existe)
            track_id = int(ids[k]) if k < len(ids) else -1

            # registrar posición en CSV
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

        # registrar conteos por zona
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

        # mostrar conteo sobre cada zona
        for z_idx, poly in enumerate(self.polygons):
            # centro aproximado de polígono (promedio de vértices)
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
        print(f"[INFO] Abriendo video: {self.input_video_path}")
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("[ERROR] No se pudo abrir el video de entrada")
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        i = 0
        print("[INFO] Iniciando procesamiento de frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fin del video (no más frames)")
                break

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            frame = self.process_frame(frame, i)
            out.write(frame)

            i += 1
            if i % 30 == 0:
                print(f"[INFO] Procesados {i} frames")

        cap.release()
        out.release()
        print("[INFO] Procesamiento de video terminado")
        self._export_csv()


if __name__ == "__main__":
    CountObject(args.input, args.output).process_video()

# python conteo.py -i "demo2.mp4" -o "demo2_out.pm4"
