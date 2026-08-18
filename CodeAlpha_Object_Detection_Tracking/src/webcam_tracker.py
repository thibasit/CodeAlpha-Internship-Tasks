"""
CodeAlpha Task 4 - Object Detection and Tracking
=================================================

Standalone webcam script.

WHY A SEPARATE SCRIPT INSTEAD OF A NOTEBOOK CELL?
--------------------------------------------------
cv2.imshow() opens a native OS window and needs its own event loop
(cv2.waitKey in a tight loop). Inside Jupyter on Windows this frequently:
    - freezes the kernel
    - leaves a window that never closes
    - blocks you from running any other cell until you restart the kernel

Running it as a plain .py script from Anaconda Prompt avoids all of that,
and is the standard, recommended approach for live cv2.imshow() apps.

HOW TO RUN
-----------
1. Open "Anaconda Prompt".
2. Activate the environment:
       conda activate codealpha_cv
3. Move into the project folder, e.g.:
       cd path\\to\\CodeAlpha_Object_Detection_Tracking
4. Run:
       python src\\webcam_tracker.py

CONTROLS
--------
    q  -> quit and close the window safely

If no webcam is available, the script prints a clear error message and
exits instead of hanging.
"""

import sys
import time

import cv2

# Allow running this file directly (python src/webcam_tracker.py) as well as
# importing it as part of the `src` package.
try:
    from .detector_tracker import Config, ObjectDetectorTracker
except ImportError:
    from detector_tracker import Config, ObjectDetectorTracker


def run_webcam(camera_index: int = 0, confidence_threshold: float = 0.40):
    cfg = Config()
    cfg.CONFIDENCE_THRESHOLD = confidence_threshold

    print("[INFO] Starting webcam tracker. Press 'q' in the video window to quit.")
    pipeline = ObjectDetectorTracker(cfg)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW = fast, reliable open on Windows
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {camera_index}.")
        print("        - Check that a webcam is connected.")
        print("        - Check that no other application (Zoom, Teams, etc.) is using it.")
        print("        - Try a different camera_index (0, 1, 2 ...).")
        sys.exit(1)

    window_name = "CodeAlpha Task 4 - Live Object Detection & Tracking"
    pipeline.stats.start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame from webcam. Stopping.")
                break

            annotated, counts = pipeline.process_frame(frame)
            annotated = pipeline.draw_hud(
                annotated,
                extra_lines=[f"Unique IDs so far: {len(pipeline.stats.unique_track_ids)}",
                             "Press 'q' to quit"],
            )

            cv2.imshow(window_name, annotated)

            # waitKey(1) is required for the OpenCV window to actually repaint.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] 'q' pressed - exiting.")
                break

            # Safety net: if the user closes the window with the [X] button
            # instead of pressing 'q', getWindowProperty drops below 1.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed - exiting.")
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user (Ctrl+C).")
    finally:
        pipeline.stats.end_time = time.time()
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + pipeline.stats.summary(video_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0))


if __name__ == "__main__":
    run_webcam()
