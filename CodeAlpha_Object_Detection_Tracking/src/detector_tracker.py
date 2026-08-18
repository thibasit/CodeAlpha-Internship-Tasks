"""
CodeAlpha Task 4 - Object Detection and Tracking
=================================================

Core pipeline module.

Combines:
    - YOLO (Ultralytics)      -> object detection (boxes, classes, confidence)
    - Deep SORT (deep-sort-realtime) -> multi-object tracking (persistent IDs)
    - OpenCV                  -> video I/O, drawing, display

This module is imported by:
    - Object_Detection_Tracking.ipynb  (notebook walkthrough / demos)
    - webcam_tracker.py                (standalone live webcam script)

Author: CodeAlpha AI Internship - Task 4
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
# Central place for every tunable value used across the project.
# Nothing below this section should be hard-coded elsewhere.

@dataclass
class Config:
    # YOLO
    MODEL_PATH: str = "yolo26n.pt"       # CPU-friendly nano model
    CONFIDENCE_THRESHOLD: float = 0.40   # minimum detection confidence to keep
    DEVICE: str = "cpu"                  # this environment is CPU-only

    # Deep SORT
    MAX_AGE: int = 30                    # frames to keep a lost track alive
    N_INIT: int = 3                      # detections needed to confirm a track
    MAX_COSINE_DISTANCE: float = 0.3     # appearance-matching threshold

    # Optional: restrict detection to specific COCO classes.
    # None = detect every class the model knows.
    # Example: {0, 1, 2, 3, 5, 7} -> person, bicycle, car, motorcycle, bus, truck
    CLASSES_OF_INTEREST: Optional[set] = None

    # Video I/O
    VIDEO_INPUT_PATH: str = "videos/input.mp4"
    VIDEO_OUTPUT_PATH: str = "outputs/tracked_output.mp4"

    # Drawing
    BOX_THICKNESS: int = 2
    FONT_SCALE: float = 0.55
    FONT_THICKNESS: int = 2


# --------------------------------------------------------------------------
# 2. SMALL HELPERS
# --------------------------------------------------------------------------

def id_to_color(track_id: int) -> tuple:
    """Deterministic BGR color per track ID, so the same ID keeps the same
    box color across frames (helps visually confirm identity persistence)."""
    np.random.seed(int(track_id) * 37 + 7)
    color = tuple(int(c) for c in np.random.randint(60, 255, size=3))
    return color


def clamp_box(x1, y1, x2, y2, width, height):
    """Keep a bounding box inside the frame and ensure it has positive area."""
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1:
        x2 = min(x1 + 1, width - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, height - 1)
    return x1, y1, x2, y2


# --------------------------------------------------------------------------
# 3. STATISTICS CONTAINER
# --------------------------------------------------------------------------

@dataclass
class RunStats:
    """Everything measured during a run. Only real, computed numbers ever
    go in here -- nothing is estimated or invented."""

    total_frames_in_video: int = 0
    frames_processed: int = 0
    total_detections: int = 0
    confidence_sum: float = 0.0

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    unique_track_ids: set = field(default_factory=set)
    # class_name -> set of track_ids ever seen for that class
    unique_ids_per_class: dict = field(default_factory=lambda: defaultdict(set))
    # class_name -> count of currently active (confirmed) tracks, updated per frame
    last_frame_class_counts: dict = field(default_factory=dict)

    def record_detection(self, confidence: float):
        self.total_detections += 1
        self.confidence_sum += confidence

    def record_track(self, track_id, class_name: str):
        self.unique_track_ids.add(track_id)
        self.unique_ids_per_class[class_name].add(track_id)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.time()
        return max(end - self.start_time, 1e-9)

    @property
    def processing_fps(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds

    @property
    def average_confidence(self) -> float:
        if self.total_detections == 0:
            return 0.0
        return self.confidence_sum / self.total_detections

    def summary(self, video_fps: float) -> str:
        lines = [
            "=" * 50,
            "PROCESSING SUMMARY",
            "=" * 50,
            f"Frames in source video : {self.total_frames_in_video}",
            f"Frames processed       : {self.frames_processed}",
            f"Total detections       : {self.total_detections}",
            f"Average confidence     : {self.average_confidence:.3f}",
            f"Processing time        : {self.elapsed_seconds:.2f} sec",
            f"Video FPS (source)     : {video_fps:.2f}",
            f"Processing FPS (actual): {self.processing_fps:.2f}",
            f"Unique track IDs total : {len(self.unique_track_ids)}",
        ]
        if self.unique_ids_per_class:
            lines.append("-" * 50)
            lines.append("Unique IDs observed per class:")
            for cls, ids in sorted(self.unique_ids_per_class.items()):
                lines.append(f"  {cls:<15}: {len(ids)}")
        is_realtime = self.processing_fps >= video_fps
        lines.append("-" * 50)
        lines.append(
            f"Real-time capable on this hardware: "
            f"{'YES' if is_realtime else 'NO'} "
            f"(processing FPS {'>=' if is_realtime else '<'} video FPS)"
        )
        lines.append("=" * 50)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. MAIN PIPELINE CLASS
# --------------------------------------------------------------------------

class ObjectDetectorTracker:
    """
    Wraps a YOLO detector + Deep SORT tracker into a single reusable object.

    Usage:
        pipeline = ObjectDetectorTracker(config)
        annotated_frame, tracks_info = pipeline.process_frame(frame)
    """

    def __init__(self, config: Config = None):
        self.cfg = config or Config()

        if not os.path.exists(self.cfg.MODEL_PATH):
            # Ultralytics will auto-download official models by name (e.g.
            # "yolo26n.pt"), so this is just informational, not a hard stop.
            print(f"[INFO] '{self.cfg.MODEL_PATH}' not found locally - "
                  f"Ultralytics will attempt to download it automatically.")

        print(f"[INFO] Loading YOLO model: {self.cfg.MODEL_PATH} (device={self.cfg.DEVICE}) ...")
        self.model = YOLO(self.cfg.MODEL_PATH)
        self.class_names = self.model.names  # {class_id: class_name}
        print("[INFO] YOLO model loaded successfully.")

        print("[INFO] Initializing Deep SORT tracker ...")
        self.tracker = DeepSort(
            max_age=self.cfg.MAX_AGE,
            n_init=self.cfg.N_INIT,
            max_cosine_distance=self.cfg.MAX_COSINE_DISTANCE,
        )
        print("[INFO] Deep SORT initialized successfully!")

        self.stats = RunStats()

    # ---------------------------------------------------------------- #
    def detect(self, frame: np.ndarray):
        """
        Run YOLO on a single frame.

        Returns a list of raw detections in the format Deep SORT expects:
            [ ([left, top, w, h], confidence, class_name), ... ]
        """
        results = self.model.predict(
            frame,
            conf=self.cfg.CONFIDENCE_THRESHOLD,
            device=self.cfg.DEVICE,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = self.class_names.get(cls_id, str(cls_id))

            if self.cfg.CLASSES_OF_INTEREST is not None and cls_id not in self.cfg.CLASSES_OF_INTEREST:
                continue

            # YOLO gives x1,y1,x2,y2 -> Deep SORT wants [left, top, width, height]
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, class_name))

            self.stats.record_detection(conf)

        return detections

    # ---------------------------------------------------------------- #
    def track(self, raw_detections, frame: np.ndarray):
        """Update Deep SORT with this frame's detections and return confirmed tracks."""
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        return tracks

    # ---------------------------------------------------------------- #
    def draw_tracks(self, frame: np.ndarray, tracks):
        """Draw boxes, class labels and persistent IDs. Returns (frame, per-class counts)."""
        h, w = frame.shape[:2]
        current_class_counts = defaultdict(int)

        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            class_name = t.get_det_class() or "object"
            ltrb = t.to_ltrb()
            if ltrb is None:
                continue
            x1, y1, x2, y2 = clamp_box(*ltrb, w, h)

            self.stats.record_track(track_id, class_name)
            current_class_counts[class_name] += 1

            color = id_to_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.cfg.BOX_THICKNESS)

            label = f"{class_name} | ID:{track_id}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.cfg.FONT_SCALE, self.cfg.FONT_THICKNESS
            )
            label_y1 = max(0, y1 - th - 8)
            cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, self.cfg.FONT_SCALE, (255, 255, 255),
                self.cfg.FONT_THICKNESS, cv2.LINE_AA,
            )

        self.stats.last_frame_class_counts = dict(current_class_counts)
        return frame, current_class_counts

    # ---------------------------------------------------------------- #
    def process_frame(self, frame: np.ndarray):
        """Full pipeline for one frame: detect -> track -> draw. Returns (frame, counts)."""
        raw_detections = self.detect(frame)
        tracks = self.track(raw_detections, frame)
        annotated, counts = self.draw_tracks(frame, tracks)
        self.stats.frames_processed += 1
        return annotated, counts

    # ---------------------------------------------------------------- #
    def draw_hud(self, frame, extra_lines=None):
        """Small heads-up overlay: live FPS + current object counts. Used by webcam mode."""
        y = 25
        cv2.putText(frame, f"Processing FPS: {self.stats.processing_fps:.1f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 25
        for cls, count in self.stats.last_frame_class_counts.items():
            cv2.putText(frame, f"{cls}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y += 22
        if extra_lines:
            for line in extra_lines:
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 20
        return frame


# --------------------------------------------------------------------------
# 5. VIDEO-FILE PROCESSING (used by the notebook)
# --------------------------------------------------------------------------

def process_video(config: Config = None, max_frames: Optional[int] = None,
                   show_progress_every: int = 30):
    """
    Run the full detection + tracking pipeline over a saved video file and
    write the annotated result to disk.

    Returns (pipeline, video_fps) so the caller (notebook) can inspect stats.
    """
    cfg = config or Config()

    if not os.path.exists(cfg.VIDEO_INPUT_PATH):
        raise FileNotFoundError(
            f"Input video not found at '{cfg.VIDEO_INPUT_PATH}'.\n"
            f"Place a video file there (any objects: people, cars, buses, "
            f"bicycles, dogs, etc.) and re-run this cell."
        )

    cap = cv2.VideoCapture(cfg.VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open '{cfg.VIDEO_INPUT_PATH}'. "
                       f"The file may be corrupted or use an unsupported codec.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        print("[WARN] Video FPS unavailable/invalid from metadata - defaulting to 25.0")
        video_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video opened: {width}x{height} @ {video_fps:.2f} FPS, "
          f"{total_frames} total frames")

    os.makedirs(os.path.dirname(cfg.VIDEO_OUTPUT_PATH) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg.VIDEO_OUTPUT_PATH, fourcc, video_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"VideoWriter failed to open output path "
                       f"'{cfg.VIDEO_OUTPUT_PATH}'. Check that the 'outputs/' "
                       f"folder exists and is writable.")

    pipeline = ObjectDetectorTracker(cfg)
    pipeline.stats.total_frames_in_video = total_frames
    pipeline.stats.start_time = time.time()

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video, or unreadable frame

            annotated, _ = pipeline.process_frame(frame)
            writer.write(annotated)

            frame_idx += 1
            if show_progress_every and frame_idx % show_progress_every == 0:
                print(f"  processed {frame_idx}/{total_frames or '?'} frames "
                      f"(current FPS: {pipeline.stats.processing_fps:.1f})")

            if max_frames is not None and frame_idx >= max_frames:
                print(f"[INFO] Reached max_frames={max_frames} test limit, stopping early.")
                break
    finally:
        pipeline.stats.end_time = time.time()
        cap.release()
        writer.release()

    if pipeline.stats.frames_processed == 0:
        print("[WARN] No frames were processed - output video will be empty. "
              "Check that the input video actually contains readable frames.")
    else:
        print(f"[INFO] Output video saved to: {cfg.VIDEO_OUTPUT_PATH}")

    return pipeline, video_fps


if __name__ == "__main__":
    # Allows: python src/detector_tracker.py  (quick manual test from a terminal)
    pipeline, vfps = process_video()
    print(pipeline.stats.summary(vfps))
