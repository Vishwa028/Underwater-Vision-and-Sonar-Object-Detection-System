import csv
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import serial
from serial.tools import list_ports
import numpy as np

from PySide6.QtCore import QMutex, QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
EVENT_LOG_PATH = LOG_DIR / "events.csv"
MODEL_DIR = APP_DIR / "models"
PROTOTXT_PATH = MODEL_DIR / "deploy.prototxt"
CAFFE_MODEL_PATH = MODEL_DIR / "mobilenet_iter_73000.caffemodel"
MARINE_CLASSIFIER_PATH = MODEL_DIR / "marine_classifier.keras"
MARINE_LABELS_PATH = MODEL_DIR / "marine_labels.json"
VISION_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VISION_COLORS = np.random.default_rng(42).integers(80, 255, size=(len(VISION_CLASSES), 3))
LAND_ANIMAL_LABELS = {"bird", "cat", "cow", "dog", "horse", "sheep"}


@dataclass
class SerialPacket:
    distance: int
    raw_line: str
    received_at: float


@dataclass
class FramePacket:
    image: QImage
    human_detected: bool
    aquatic_detected: bool
    detected_label: str
    detected_confidence: float
    detected_count: int
    water_clarity_score: float
    enhancement_mode: str
    marine_hint: str
    marine_hint_confidence: float
    classifier_active: bool
    received_at: float


class MarineClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.labels: list[str] = []
        self.available = False
        self._image_size = (224, 224)
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists() or not self.labels_path.exists():
            return

        try:
            import tensorflow as tf
        except ImportError:
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            self.available = bool(self.labels)
        except Exception:
            self.model = None
            self.labels = []
            self.available = False

    def predict(self, frame) -> tuple[str, float]:
        if not self.available or self.model is None:
            return "", 0.0

        resized = cv2.resize(frame, self._image_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype("float32") / 255.0
        batch = np.expand_dims(normalized, axis=0)

        predictions = self.model.predict(batch, verbose=0)[0]
        index = int(np.argmax(predictions))
        confidence = float(predictions[index])
        if index >= len(self.labels):
            return "", 0.0
        return self.labels[index].upper(), confidence * 100.0


class EventLogger:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["timestamp", "state", "target", "distance_cm", "confidence", "source", "visual_label"]
                )

    def log_event(
        self,
        state: str,
        target: str,
        distance: int,
        confidence: float,
        source: str,
        visual_label: str,
    ) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    state,
                    target,
                    distance,
                    f"{confidence:.1f}",
                    source,
                    visual_label,
                ]
            )


class HistoryChart(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.values = deque([0] * 40, maxlen=40)
        self.setMinimumHeight(170)

    def add_value(self, value: int) -> None:
        self.values.append(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#08131d"))
        painter.setPen(QPen(QColor("#173042"), 1))

        for step in range(5):
            y = rect.top() + int(rect.height() * step / 4)
            painter.drawLine(rect.left(), y, rect.right(), y)

        values = list(self.values)
        if not values:
            return

        max_value = max(max(values), 100)
        min_value = 0
        span = max(max_value - min_value, 1)

        points = []
        for index, value in enumerate(values):
            x = rect.left() + int(index * rect.width() / max(len(values) - 1, 1))
            normalized = (value - min_value) / span
            y = rect.bottom() - int(normalized * rect.height())
            points.append(QPoint(x, y))

        painter.setPen(QPen(QColor("#00d4ff"), 2))
        for start, end in zip(points, points[1:]):
            painter.drawLine(start, end)

        painter.setPen(QColor("#9bb8cb"))
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(QRect(rect.left() + 8, rect.top() + 8, 120, 20), Qt.AlignLeft, "Range History")
        painter.drawText(QRect(rect.right() - 110, rect.top() + 8, 100, 20), Qt.AlignRight, f"Max {max_value} cm")


class MetricCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setStyleSheet(
            f"""
            QFrame#metricCard {{
                background-color: #091824;
                border: 1px solid #173042;
                border-radius: 14px;
            }}
            QLabel#metricTitle {{
                color: #84a6bd;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            QLabel#metricValue {{
                color: {accent};
                font-size: 26px;
                font-weight: 700;
            }}
            """
        )

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricTitle")

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class SerialWorker(QThread):
    packet_received = Signal(object)
    status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._port = "COM9"
        self._baudrate = 9600
        self._mutex = QMutex()

    def configure(self, port: str, baudrate: int = 9600) -> None:
        self._port = port
        self._baudrate = baudrate

    def run(self) -> None:
        self._running = True
        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=0.2)
            self.status_changed.emit(f"Connected to {self._port}")
        except serial.SerialException as error:
            self.status_changed.emit(f"Serial error: {error}")
            self._running = False
            return

        while self._running:
            try:
                line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                packet = self._parse_line(line)
                if packet:
                    self.packet_received.emit(packet)
            except serial.SerialException as error:
                self.status_changed.emit(f"Serial disconnected: {error}")
                break
            except Exception as error:  # noqa: BLE001
                self.status_changed.emit(f"Serial read warning: {error}")

        if self._serial and self._serial.is_open:
            self._serial.close()
        self.status_changed.emit("Serial offline")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def send_command(self, command: str) -> None:
        if not self._serial or not self._serial.is_open:
            return

        self._mutex.lock()
        try:
            self._serial.write(command.encode("utf-8"))
        except serial.SerialException:
            pass
        finally:
            self._mutex.unlock()

    @staticmethod
    def _parse_line(line: str) -> Optional[SerialPacket]:
        if line.startswith("Distance:"):
            try:
                value = int(line.split(":", 1)[1].strip())
                return SerialPacket(distance=value, raw_line=line, received_at=time.time())
            except ValueError:
                return None

        if "DIST:" in line:
            try:
                fragment = line.split("DIST:", 1)[1].split(",", 1)[0]
                value = int(fragment.strip())
                return SerialPacket(distance=value, raw_line=line, received_at=time.time())
            except ValueError:
                return None

        return None


class CameraWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self.camera_index = camera_index
        self._running = False
        self.enhancement_mode = "UNDERWATER RESEARCH"
        self.confidence_threshold = 0.38
        self.net = self._load_model()
        self.marine_classifier = MarineClassifier(MARINE_CLASSIFIER_PATH, MARINE_LABELS_PATH)

    def configure(self, enhancement_mode: str, confidence_threshold: float) -> None:
        self.enhancement_mode = enhancement_mode
        self.confidence_threshold = confidence_threshold

    def _load_model(self):
        if not PROTOTXT_PATH.exists() or not CAFFE_MODEL_PATH.exists():
            self.status_changed.emit("Vision model files missing")
            return None

        try:
            return cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFE_MODEL_PATH))
        except cv2.error:
            self.status_changed.emit("Vision model could not be loaded")
            return None

    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            self.status_changed.emit("No camera detected - simulation mode")
            return

        self.status_changed.emit("Camera online")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.msleep(50)
                continue

            processed_frame = self._enhance_frame(frame)
            analyzed_frame, human_detected, aquatic_detected, detected_label, detected_confidence, detected_count = (
                self._detect_targets(processed_frame)
            )
            clarity_score = self._estimate_clarity(processed_frame)

            rgb = cv2.cvtColor(analyzed_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            bytes_per_line = channels * width
            image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            packet = FramePacket(
                image=image,
                human_detected=human_detected,
                aquatic_detected=aquatic_detected,
                detected_label=detected_label,
                detected_confidence=detected_confidence,
                detected_count=detected_count,
                water_clarity_score=clarity_score,
                enhancement_mode=self.enhancement_mode,
                marine_hint=detected_label,
                marine_hint_confidence=detected_confidence,
                classifier_active=self.marine_classifier.available,
                received_at=time.time(),
            )
            self.frame_ready.emit(packet)
            self.msleep(33)

        cap.release()
        self.status_changed.emit("Camera offline")

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def _enhance_frame(self, frame):
        if self.enhancement_mode == "STANDARD":
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        if self.enhancement_mode == "LOW LIGHT WATER":
            gamma_table = np.array([((index / 255.0) ** 0.8) * 255 for index in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, gamma_table)
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 4, 4, 7, 21)

        # Simple gray-world color balancing helps reduce underwater blue-green cast.
        channels = cv2.split(enhanced.astype(np.float32))
        channel_means = [float(np.mean(channel)) for channel in channels]
        gray_mean = sum(channel_means) / len(channel_means)
        balanced_channels = []
        for channel, mean_value in zip(channels, channel_means):
            scale = gray_mean / max(mean_value, 1.0)
            balanced_channels.append(np.clip(channel * scale, 0, 255).astype(np.uint8))

        balanced = cv2.merge(balanced_channels)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(balanced, -1, sharpen_kernel)

    @staticmethod
    def _estimate_clarity(frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(100.0, variance / 4.5)

    def _detect_targets(self, frame) -> tuple[object, bool, bool, str, float, int]:
        annotated = frame.copy()
        best_label = "SCANNING"
        best_confidence = 0.0
        detected_count = 0
        human_detected = False
        aquatic_detected = False
        marine_bias = self._estimate_marine_bias(frame)

        if self.net is not None:
            height, width = annotated.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(annotated, (300, 300)),
                scalefactor=0.007843,
                size=(300, 300),
                mean=127.5,
            )
            self.net.setInput(blob)
            detections = self.net.forward()

            for index in range(detections.shape[2]):
                confidence = float(detections[0, 0, index, 2])
                if confidence < self.confidence_threshold:
                    continue

                class_id = int(detections[0, 0, index, 1])
                if class_id < 0 or class_id >= len(VISION_CLASSES):
                    continue

                detected_count += 1
                label = VISION_CLASSES[class_id]
                adjusted_label, adjusted_confidence = self._adjust_underwater_label(label, confidence, marine_bias)
                if adjusted_confidence > best_confidence:
                    best_confidence = adjusted_confidence
                    best_label = adjusted_label

                if label == "person":
                    human_detected = True
                elif label in LAND_ANIMAL_LABELS:
                    aquatic_detected = True

                box = detections[0, 0, index, 3:7] * np.array([width, height, width, height])
                start_x, start_y, end_x, end_y = box.astype("int")
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(width - 1, end_x)
                end_y = min(height - 1, end_y)

                color = tuple(int(channel) for channel in VISION_COLORS[class_id])
                cv2.rectangle(annotated, (start_x, start_y), (end_x, end_y), color, 2)
                overlay = f"{adjusted_label} {adjusted_confidence * 100:.1f}%"
                anchor_y = start_y - 10 if start_y > 20 else start_y + 22
                cv2.putText(
                    annotated,
                    overlay,
                    (start_x, anchor_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 8 and not human_detected and best_label == "SCANNING":
            aquatic_detected = True
            best_label = "MOVING OBJECT"
            best_confidence = max(best_confidence, 0.62)

        heuristic_label, heuristic_confidence, heuristic_aquatic = self._classify_underwater_shape(frame, contours)
        if heuristic_label and (best_label in {"SCANNING", "MOVING OBJECT"} or heuristic_confidence > best_confidence):
            best_label = heuristic_label
            best_confidence = heuristic_confidence
            aquatic_detected = aquatic_detected or heuristic_aquatic

        if marine_bias >= 0.55 and best_label in {"DOG", "CAT", "HORSE", "SHEEP", "COW", "BIRD"}:
            aquatic_detected = True
            best_label = "FISH"
            best_confidence = max(best_confidence, 0.66)

        if best_label == "SCANNING" and marine_bias >= 0.5:
            best_label = "MARINE TARGET"
            best_confidence = max(best_confidence, 0.54)
            aquatic_detected = True

        classifier_label, classifier_confidence = self.marine_classifier.predict(frame)
        if classifier_label and classifier_confidence >= max(best_confidence * 100.0, 54.0):
            best_label = classifier_label
            best_confidence = classifier_confidence / 100.0
            aquatic_detected = classifier_label in {
                "FISH",
                "TURTLE",
                "CORAL",
                "ROCK",
                "DEBRIS",
                "MINE_LIKE_OBJECT",
                "SEAWEED",
            }

        if best_label in {"FISH", "TURTLE", "CORAL", "SEAWEED / DEBRIS", "REEF / LARGE STRUCTURE"}:
            aquatic_detected = True

        cv2.putText(
            annotated,
            f"VISION: {best_label}",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 212, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"DETECTIONS: {detected_count}",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (126, 242, 160),
            2,
        )
        cv2.putText(
            annotated,
            f"MODE: {self.enhancement_mode}",
            (18, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 182, 72),
            2,
        )
        cv2.putText(
            annotated,
            f"MARINE BIAS: {marine_bias * 100:.0f}%",
            (18, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (120, 225, 255),
            2,
        )
        if classifier_label:
            cv2.putText(
                annotated,
                f"SLM: {classifier_label} {classifier_confidence:.1f}%",
                (18, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (166, 255, 180),
                2,
            )
        return annotated, human_detected, aquatic_detected, best_label, best_confidence * 100.0, detected_count

    def _adjust_underwater_label(self, label: str, confidence: float, marine_bias: float) -> tuple[str, float]:
        if self.enhancement_mode == "STANDARD":
            return label.upper(), confidence

        if label in LAND_ANIMAL_LABELS and marine_bias >= 0.45:
            relabeled = "FISH" if label in {"dog", "cat", "horse", "sheep"} else "MARINE LIFE"
            damped_confidence = max(0.42, confidence * (0.58 + marine_bias * 0.15))
            return relabeled, damped_confidence

        if label == "boat" and marine_bias >= 0.45:
            return "SURFACE OBJECT", confidence * 0.92

        return label.upper(), confidence

    @staticmethod
    def _estimate_marine_bias(frame) -> float:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        blue_green_mask = ((hue >= 70) & (hue <= 120) & (saturation >= 35)).mean()
        bright_orange_mask = ((hue >= 5) & (hue <= 28) & (saturation >= 80)).mean()
        low_contrast = 1.0 - min(1.0, float(np.std(value)) / 85.0)
        marine_bias = (blue_green_mask * 0.55) + (bright_orange_mask * 0.2) + (low_contrast * 0.25)
        return float(max(0.0, min(1.0, marine_bias)))

    def _classify_underwater_shape(self, frame, contours) -> tuple[str, float, bool]:
        if not contours:
            return "", 0.0, False

        height, width = frame.shape[:2]
        frame_area = float(height * width)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < frame_area * 0.015:
            return "", 0.0, False

        perimeter = cv2.arcLength(largest, True)
        if perimeter <= 0:
            return "", 0.0, False

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / max(h, 1)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return "", 0.0, False

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        saturation = float(np.mean(hsv_roi[:, :, 1]))
        brightness = float(np.mean(hsv_roi[:, :, 2]))

        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 40, 120)
        edge_density = float(np.count_nonzero(edges)) / max(roi.shape[0] * roi.shape[1], 1)

        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)
        extent = area / max(w * h, 1)

        if 1.2 <= aspect_ratio <= 3.8 and 0.18 <= circularity <= 0.62 and solidity <= 0.9:
            return "FISH", 0.68, True
        if 0.75 <= aspect_ratio <= 1.8 and 0.35 <= circularity <= 0.72 and solidity >= 0.78:
            return "TURTLE", 0.61, True
        if edge_density >= 0.11 and saturation >= 52 and extent <= 0.58:
            return "CORAL", 0.57, False
        if brightness <= 85 and solidity >= 0.86 and circularity >= 0.48:
            return "ROCK", 0.56, False
        if edge_density >= 0.08 and aspect_ratio >= 2.4:
            return "SEAWEED / DEBRIS", 0.53, False
        if circularity >= 0.68 and area > frame_area * 0.04:
            return "MINE-LIKE OBJECT", 0.58, False
        if area > frame_area * 0.2:
            return "REEF / LARGE STRUCTURE", 0.52, False
        return "", 0.0, False


class DetectionEngine:
    def __init__(self) -> None:
        self.last_target = "SCANNING"
        self.last_detection_at = 0.0
        self.last_command = ""
        self.last_logged_state = ""

    def evaluate(
        self,
        distance: int,
        human_detected: bool,
        aquatic_detected: bool,
        visual_label: str,
        visual_confidence: float,
        water_clarity_score: float,
        mode: str,
    ) -> dict:
        now = time.time()
        state = "SAFE"
        target = "SCANNING"
        confidence = max(visual_confidence, 76.0)
        source = mode

        if human_detected:
            state = "TRACKING"
            target = visual_label if visual_label != "SCANNING" else "PERSON"
            confidence = max(visual_confidence, 93.0)
            self.last_detection_at = now
        elif aquatic_detected:
            state = "TRACKING"
            target = visual_label if visual_label != "SCANNING" else "AQUATIC"
            confidence = max(visual_confidence, 88.0)
            self.last_detection_at = now
        elif visual_label not in {"SCANNING", "MOVING OBJECT"}:
            state = "TRACKING" if visual_confidence >= 60 else "WARNING"
            target = visual_label
            confidence = max(visual_confidence, 82.0)
            self.last_detection_at = now
        elif 2 < distance < 25:
            state = "DANGER"
            target = "MINE"
            confidence = 97.5
            self.last_detection_at = now
        elif 25 <= distance < 45:
            state = "WARNING"
            target = "UNKNOWN"
            confidence = 84.0
        elif distance >= 45:
            state = "SAFE"
            target = "CLEAR"
            confidence = 91.0

        if now - self.last_detection_at < 2.0 and target == "SCANNING":
            target = self.last_target

        self.last_target = target
        command = "H" if state == "DANGER" else "S"
        water_state = "CLEAR" if water_clarity_score >= 55 else "MODERATE" if water_clarity_score >= 28 else "TURBID"

        return {
            "state": state,
            "target": target,
            "confidence": confidence,
            "command": command,
            "source": source,
            "water_state": water_state,
        }


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DAG-1 Desktop Control")
        self.resize(1380, 820)

        self.serial_worker: Optional[SerialWorker] = None
        self.camera_worker: Optional[CameraWorker] = None
        self.logger = EventLogger(EVENT_LOG_PATH)
        self.engine = DetectionEngine()

        self.current_distance = 0
        self.human_detected = False
        self.aquatic_detected = False
        self.visual_label = "SCANNING"
        self.visual_confidence = 0.0
        self.detected_count = 0
        self.water_clarity_score = 0.0
        self.enhancement_mode = "UNDERWATER RESEARCH"
        self.marine_hint = "SCANNING"
        self.marine_hint_confidence = 0.0
        self.classifier_active = False
        self.current_frame = QPixmap()
        self.last_frame_at = 0.0
        self.serial_connected = False
        self.camera_online = False

        self._build_ui()
        self._apply_styles()
        self.refresh_serial_ports()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.refresh_dashboard)
        self.update_timer.start(120)
        QApplication.instance().aboutToQuit.connect(self.shutdown_workers)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(18)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 18, 18, 18)
        sidebar_layout.setSpacing(16)

        title = QLabel("DAG-1 DESKTOP CONTROL")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Standalone sensor fusion application for Arduino, camera, and local prediction.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subTitle")

        form = QFormLayout()
        form.setSpacing(10)

        self.port_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["LIVE ARDUINO", "MANUAL SIMULATION"])
        self.vision_mode_combo = QComboBox()
        self.vision_mode_combo.addItems(["UNDERWATER RESEARCH", "LOW LIGHT WATER", "STANDARD"])
        self.vision_threshold_slider = QSlider(Qt.Horizontal)
        self.vision_threshold_slider.setRange(25, 70)
        self.vision_threshold_slider.setValue(38)
        self.vision_threshold_label = QLabel("38%")
        self.vision_threshold_slider.valueChanged.connect(self._update_threshold_label)
        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(0, 100)
        self.distance_slider.setValue(30)
        self.distance_slider.valueChanged.connect(self._update_manual_distance_label)
        self.manual_distance_label = QLabel("30 cm")

        form.addRow("COM Port", self.port_combo)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Vision Mode", self.vision_mode_combo)
        form.addRow("Detect Threshold", self.vision_threshold_slider)
        form.addRow("Threshold", self.vision_threshold_label)
        form.addRow("Manual Range", self.distance_slider)
        form.addRow("Distance", self.manual_distance_label)

        button_row = QHBoxLayout()
        self.refresh_ports_button = QPushButton("Refresh Ports")
        self.connect_button = QPushButton("Connect Arduino")
        self.camera_button = QPushButton("Start Camera")
        button_row.addWidget(self.refresh_ports_button)
        button_row.addWidget(self.connect_button)
        button_row.addWidget(self.camera_button)

        self.refresh_ports_button.clicked.connect(self.refresh_serial_ports)
        self.connect_button.clicked.connect(self.toggle_serial)
        self.camera_button.clicked.connect(self.toggle_camera)

        self.serial_status_label = QLabel("Serial offline")
        self.camera_status_label = QLabel("Camera offline")
        self.pipeline_status_label = QLabel("Prediction engine ready")
        self.log_hint_label = QLabel(f"Events log: {EVENT_LOG_PATH.name}")
        self.log_hint_label.setWordWrap(True)

        sidebar_layout.addWidget(title)
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addLayout(form)
        sidebar_layout.addLayout(button_row)
        sidebar_layout.addWidget(self.serial_status_label)
        sidebar_layout.addWidget(self.camera_status_label)
        sidebar_layout.addWidget(self.pipeline_status_label)
        sidebar_layout.addWidget(self.log_hint_label)
        sidebar_layout.addStretch(1)

        content = QVBoxLayout()
        content.setSpacing(16)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)
        self.range_card = MetricCard("Range", "#00d4ff")
        self.state_card = MetricCard("State", "#ff8360")
        self.target_card = MetricCard("Target", "#7ef2a0")
        self.conf_card = MetricCard("Vision", "#fddc6c")
        self.water_card = MetricCard("Water", "#7cd5ff")
        self.marine_card = MetricCard("Marine", "#8dffb0")
        for column in range(3):
            metrics_grid.setColumnStretch(column, 1)
        metrics_grid.addWidget(self.range_card, 0, 0)
        metrics_grid.addWidget(self.state_card, 0, 1)
        metrics_grid.addWidget(self.target_card, 0, 2)
        metrics_grid.addWidget(self.conf_card, 1, 0)
        metrics_grid.addWidget(self.water_card, 1, 1)
        metrics_grid.addWidget(self.marine_card, 1, 2)

        panels_row = QHBoxLayout()
        panels_row.setSpacing(16)

        video_panel = QFrame()
        video_panel.setObjectName("panel")
        video_layout = QVBoxLayout(video_panel)
        video_title = QLabel("Live Visual Feed")
        video_title.setObjectName("panelTitle")
        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setObjectName("videoSurface")
        video_layout.addWidget(video_title)
        video_layout.addWidget(self.video_label)

        insight_panel = QFrame()
        insight_panel.setObjectName("panel")
        insight_panel.setMinimumWidth(320)
        insight_layout = QVBoxLayout(insight_panel)
        insight_title = QLabel("System Timeline")
        insight_title.setObjectName("panelTitle")
        self.chart = HistoryChart()
        self.events_list = QListWidget()
        insight_layout.addWidget(insight_title)
        insight_layout.addWidget(self.chart)
        insight_layout.addWidget(self.events_list)

        panels_row.addWidget(video_panel, 5)
        panels_row.addWidget(insight_panel, 2)

        content.addLayout(metrics_grid)
        content.addLayout(panels_row)

        main_layout.addWidget(sidebar, 1)
        main_layout.addLayout(content, 3)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #041018;
                color: #e9f6ff;
                font-family: 'Segoe UI';
            }
            QFrame#sidebar, QFrame#panel {
                background-color: #071521;
                border: 1px solid #123041;
                border-radius: 18px;
            }
            QLabel#mainTitle {
                font-size: 24px;
                font-weight: 700;
                color: #00d4ff;
            }
            QLabel#subTitle, QLabel, QListWidget {
                color: #b5cad8;
                font-size: 12px;
            }
            QLabel#panelTitle {
                color: #eff9ff;
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#videoSurface {
                background-color: #02070c;
                border: 1px solid #173042;
                border-radius: 14px;
                color: #84a6bd;
            }
            QPushButton {
                background-color: #0e3142;
                color: #eff9ff;
                border-radius: 10px;
                padding: 10px 12px;
                border: 1px solid #16445b;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #134860;
            }
            QComboBox, QSlider, QListWidget {
                background-color: #091824;
                border: 1px solid #173042;
                border-radius: 10px;
                padding: 6px;
            }
            QListWidget {
                min-width: 280px;
            }
            """
        )

    def refresh_serial_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [port.device for port in list_ports.comports()]
        if not ports:
            ports = ["COM9"]
        self.port_combo.addItems(ports)
        if current and current in ports:
            self.port_combo.setCurrentText(current)

    def toggle_serial(self) -> None:
        if self.serial_worker and self.serial_worker.isRunning():
            self.serial_worker.stop()
            self.serial_worker = None
            self.serial_connected = False
            self.connect_button.setText("Connect Arduino")
            self.serial_status_label.setText("Serial offline")
            return

        self.serial_worker = SerialWorker()
        self.serial_worker.configure(self.port_combo.currentText())
        self.serial_worker.packet_received.connect(self.handle_serial_packet)
        self.serial_worker.status_changed.connect(self.handle_serial_status)
        self.serial_worker.start()
        self.connect_button.setText("Disconnect Arduino")

    def toggle_camera(self) -> None:
        if self.camera_worker and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker = None
            self.camera_online = False
            self.camera_button.setText("Start Camera")
            self.camera_status_label.setText("Camera offline")
            self.video_label.setText("Camera not started")
            self.video_label.setPixmap(QPixmap())
            return

        self.camera_worker = CameraWorker(camera_index=0)
        self.camera_worker.configure(
            enhancement_mode=self.vision_mode_combo.currentText(),
            confidence_threshold=self.vision_threshold_slider.value() / 100.0,
        )
        self.camera_worker.frame_ready.connect(self.handle_frame_packet)
        self.camera_worker.status_changed.connect(self.handle_camera_status)
        self.camera_worker.start()
        self.camera_button.setText("Stop Camera")

    def handle_serial_packet(self, packet: SerialPacket) -> None:
        self.current_distance = packet.distance
        self.chart.add_value(packet.distance)

    def handle_serial_status(self, status: str) -> None:
        self.serial_status_label.setText(status)
        self.serial_connected = status.startswith("Connected")
        if not self.serial_connected and self.serial_worker and not self.serial_worker.isRunning():
            self.connect_button.setText("Connect Arduino")

    def handle_camera_status(self, status: str) -> None:
        self.camera_status_label.setText(status)
        self.camera_online = status == "Camera online"
        if status != "Camera online" and self.camera_worker and not self.camera_worker.isRunning():
            self.camera_button.setText("Start Camera")

    def handle_frame_packet(self, packet: FramePacket) -> None:
        self.human_detected = packet.human_detected
        self.aquatic_detected = packet.aquatic_detected
        self.visual_label = packet.detected_label
        self.visual_confidence = packet.detected_confidence
        self.detected_count = packet.detected_count
        self.water_clarity_score = packet.water_clarity_score
        self.enhancement_mode = packet.enhancement_mode
        self.marine_hint = packet.marine_hint
        self.marine_hint_confidence = packet.marine_hint_confidence
        self.classifier_active = packet.classifier_active
        self.last_frame_at = packet.received_at
        pixmap = QPixmap.fromImage(packet.image).scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.current_frame = pixmap
        self.video_label.setPixmap(pixmap)

    def refresh_dashboard(self) -> None:
        mode = self.mode_combo.currentText()
        if mode == "MANUAL SIMULATION":
            self.current_distance = self.distance_slider.value()
            self.chart.add_value(self.current_distance)

        result = self.engine.evaluate(
            distance=self.current_distance,
            human_detected=self.human_detected,
            aquatic_detected=self.aquatic_detected,
            visual_label=self.visual_label,
            visual_confidence=self.visual_confidence,
            water_clarity_score=self.water_clarity_score,
            mode=mode,
        )

        self.range_card.set_value(f"{self.current_distance} cm")
        self.state_card.set_value(result["state"])
        self.target_card.set_value(result["target"])
        self.conf_card.set_value(f"{self.visual_label} {self.visual_confidence:.1f}%")
        self.water_card.set_value(f'{result["water_state"]} {self.water_clarity_score:.0f}')
        self.marine_card.set_value(f"{self.marine_hint} {self.marine_hint_confidence:.1f}%")
        self.pipeline_status_label.setText(
            f'{result["source"]} pipeline active | Mode: {self.enhancement_mode} | Vision: {self.visual_label} | Marine: {self.marine_hint} | Water: {result["water_state"]} | SLM: {"READY" if self.classifier_active else "NOT LOADED"}'
        )

        if self.serial_worker and self.serial_worker.isRunning() and result["command"] != self.engine.last_command:
            self.serial_worker.send_command(result["command"])
            self.engine.last_command = result["command"]

        signature = f'{result["state"]}:{result["target"]}:{self.current_distance}'
        if signature != self.engine.last_logged_state:
            self.engine.last_logged_state = signature
            self.logger.log_event(
                state=result["state"],
                target=result["target"],
                distance=self.current_distance,
                confidence=result["confidence"],
                source=result["source"],
                visual_label=self.visual_label,
            )
            self._push_event(
                f'{time.strftime("%H:%M:%S")} | {result["state"]} | {result["target"]} | {self.visual_label} | {self.current_distance} cm'
            )

    def _push_event(self, text: str) -> None:
        self.events_list.insertItem(0, QListWidgetItem(text))
        while self.events_list.count() > 15:
            self.events_list.takeItem(self.events_list.count() - 1)

    def _update_manual_distance_label(self, value: int) -> None:
        self.manual_distance_label.setText(f"{value} cm")

    def _update_threshold_label(self, value: int) -> None:
        self.vision_threshold_label.setText(f"{value}%")

    def shutdown_workers(self) -> None:
        self.update_timer.stop()

        if self.serial_worker is not None:
            if self.serial_worker.isRunning():
                self.serial_worker.stop()
            self.serial_worker = None

        if self.camera_worker is not None:
            if self.camera_worker.isRunning():
                self.camera_worker.stop()
            self.camera_worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        self.shutdown_workers()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
