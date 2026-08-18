This folder is a convenient place to keep downloaded YOLO weight files
(e.g. yolo26n.pt) if you want them stored inside the project instead of
wherever Ultralytics' default cache/working directory is.

By default this project uses:
    MODEL_PATH = "yolo26n.pt"
which matches the exact model name already verified working in this
project. Ultralytics auto-downloads it on first use and caches it.

If you'd rather keep the weights file here explicitly, download/move it
into this folder and update Config.MODEL_PATH in src/detector_tracker.py
(or the notebook's configuration cell) to "models/yolo26n.pt".
