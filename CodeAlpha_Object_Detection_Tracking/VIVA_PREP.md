# Viva Preparation — CodeAlpha Task 4: Object Detection and Tracking

Short, clear answers you can say out loud. Expand in your own words as needed.

**1. What is object detection?**
Finding *what* objects are in an image or frame and *where* they are — output as a bounding box, a class label, and a confidence score, for every object found.

**2. What is object tracking?**
Linking detections of the *same physical object* across consecutive frames so it keeps one consistent identity (an ID) over time, instead of being treated as a brand-new object every frame.

**3. What is YOLO?**
"You Only Look Once" — a convolutional neural network that looks at the whole image in a single pass and directly predicts all bounding boxes, classes, and confidences at once, which makes it fast enough for video.

**4. Why YOLO?**
It's fast (important for frame-by-frame video on CPU), accurate for common objects, and has a small, easy-to-use pretrained variant (`yolo26n.pt`) trained on the COCO dataset.

**5. What is Deep SORT?**
An extension of the SORT tracking algorithm that adds a deep-learning appearance embedding on top of a Kalman-filter motion model, so it can keep track of objects using both *where they're moving* and *what they look like*.

**6. Why Deep SORT?**
Plain detection has no memory between frames. Deep SORT is a well-established, easy-to-integrate method for turning a sequence of independent detections into consistent tracked identities, and it handles brief occlusions and visually similar objects better than motion-only tracking.

**7. Detection vs. tracking — what's the difference?**
Detection answers "what's in this single frame?" Tracking answers "is this the same object I saw a moment ago?" Detection has no concept of identity or time; tracking is built on top of detection to add both.

**8. What is a bounding box?**
The smallest rectangle that contains a detected object, usually given as `(x1, y1, x2, y2)` (top-left and bottom-right corners) or `(x, y, width, height)`.

**9. What is confidence?**
A score between 0 and 1 showing how sure the model is that a detected box actually contains the predicted class. Low-confidence detections are filtered out with a threshold (this project uses 0.40 by default).

**10. What is a class ID?**
A numeric label identifying *what kind* of object was detected (e.g., in COCO, `0` = person, `2` = car, `5` = bus). The class *name* is just a human-readable lookup of that ID.

**11. What is a tracking ID?**
A number assigned by the tracker (not the detector) to a specific tracked object, meant to stay the same across frames for as long as that object is being successfully tracked.

**12. How does Deep SORT maintain identity?**
Each frame, it predicts where existing tracks should be (Kalman filter), compares new detections to those predictions and to stored appearance features, and matches them using both motion distance and appearance similarity. Matched detections update their track; unmatched detections can start new tracks.

**13. What happens when an object disappears?**
Its track isn't deleted immediately — it's kept alive for up to `MAX_AGE` frames without a matching detection (e.g., brief occlusion behind another object). If it reappears within that window and matches well, it keeps its ID. If not, the track is dropped.

**14. Why can IDs switch?**
Long occlusion beyond `MAX_AGE`, two similar-looking objects crossing paths with ambiguous appearance features, or the detector simply missing the object for several consecutive frames (no detection means no update for the tracker).

**15. Why use a pretrained model instead of training one?**
Training a competitive detector from scratch needs a huge labeled dataset and heavy GPU compute — far beyond this task's scope. A strong pretrained model (COCO-trained YOLO) lets the project focus its effort on the detection→tracking pipeline itself, which is the actual task.

**16. Why is the model running on CPU?**
Because the development environment has no CUDA-capable GPU. The nano YOLO variant (`yolo26n.pt`) was specifically chosen because it's the fastest/lightest option, making CPU inference practical.

**17. What are the limitations?**
Not real-time on this CPU hardware (measured, not assumed); occasional ID switches under occlusion or crossing objects; the nano model misses more small/distant objects than a larger variant; no ground-truth labels exist for these videos, so tracking-accuracy metrics like MOTA/IDF1 aren't reported.

**18. How could the project be improved?**
GPU inference or ONNX/TensorRT export for speed; a larger YOLO variant for accuracy; tuning tracker parameters (`MAX_AGE`, `N_INIT`, `MAX_COSINE_DISTANCE`) against a labeled clip; adding a real MOT-format benchmark.

**19. How would GPU acceleration improve it?**
YOLO inference and the Deep SORT appearance embedder are both neural networks — GPUs run the underlying matrix operations far faster than CPUs, which would raise processing FPS, likely to or above the video's own FPS, enabling genuine real-time operation.

**20. How could this be deployed in a real application?**
Wrapped behind a small API/service that accepts a video or camera stream and returns/streams annotated results — e.g. for pedestrian counting, traffic monitoring, or retail analytics — typically running on GPU or edge hardware (like NVIDIA Jetson) for real-time throughput, with the same detect→track→draw→count pipeline used here.
