# 🤖 CodeAlpha AI Internship --- Machine Learning & Computer Vision Projects

```{=html}
<p align="center">
```
`<b>`{=html}Two practical AI projects developed for the CodeAlpha
Artificial Intelligence Internship`</b>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/Scikit--Learn-NLP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">`{=html}
`<img src="https://img.shields.io/badge/YOLO-Object%20Detection-111111?style=for-the-badge" alt="YOLO">`{=html}
`<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">`{=html}
`<img src="https://img.shields.io/badge/Deep%20SORT-Object%20Tracking-0F766E?style=for-the-badge" alt="Deep SORT">`{=html}
`<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 About This Repository

This repository contains my **CodeAlpha Artificial Intelligence
Internship** projects, focused on practical applications of **Natural
Language Processing (NLP)** and **Computer Vision**.

The projects were developed in **Jupyter Notebook** with an emphasis on:

-   reproducible machine-learning workflows
-   clear preprocessing and model pipelines
-   measurable evaluation
-   error analysis
-   practical computer-vision engineering
-   honest reporting of model limitations and performance

------------------------------------------------------------------------

# 📂 Projects

  -----------------------------------------------------------------------
  Task              Project           Main Technologies Status
  ----------------- ----------------- ----------------- -----------------
  **Task 2**        🎓 University FAQ Python, NLP,      ✅ Implemented &
                    Chatbot           TF-IDF, Logistic  evaluated
                                      Regression,       
                                      Cosine Similarity 

  **Task 4**        👁️ Object         YOLO, OpenCV,     🚧 Implementation
                    Detection &       Deep SORT,        in progress /
                    Tracking          PyTorch           ready for video
                                                        execution
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🎓 Task 2 --- University FAQ Chatbot

### Intelligent FAQ Retrieval Using NLP

The chatbot is designed to answer common university-related questions
covering:

-   Programs
-   Admission
-   Required Documents
-   Application Deadlines
-   Admission Requirements
-   Tuition Fees
-   Scholarships
-   Contact Information
-   University Location
-   Hostel / Accommodation
-   Online Learning
-   Program Duration

### 🧠 Approach

The project intentionally starts with a simple baseline and then
improves it.

``` text
                    USER QUESTION
                          │
                          ▼
                 Text Preprocessing
                          │
                          ▼
                    TF-IDF Vector
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Baseline Retrieval        Intent Classification
      Cosine Similarity          Logistic Regression
             │                         │
             │                    Predicted Intent
             │                         │
             │                         ▼
             │                  Category Filtering
             │                         │
             └────────────┬────────────┘
                          ▼
                  FAQ Cosine Similarity
                          │
                          ▼
                 Confidence Threshold
                          │
                  ┌───────┴───────┐
                  ▼               ▼
               Answer          Fallback
```

## 🔧 NLP Pipeline

The preprocessing pipeline includes:

1.  Lowercase conversion
2.  Punctuation removal
3.  Regex-based tokenization
4.  Stopword removal
5.  Basic plural normalization
6.  TF-IDF feature extraction

The project avoids dependence on downloadable NLTK tokenization
resources and uses a lightweight regex-based preprocessing approach.

## 📊 Baseline vs Hybrid Model

A baseline model was first evaluated using **TF-IDF + cosine similarity
across the complete FAQ bank**.

### Baseline

-   36 FAQ entries
-   12 categories
-   3 FAQ examples per category
-   24 clean unseen evaluation questions
-   **Accuracy: 41.67%**

The baseline demonstrated a clear weakness: word overlap does not
necessarily represent intent.

For example, admission, documents, deadlines, and fees questions can
contain similar vocabulary while asking fundamentally different things.

### Hybrid Model

The improved system combines:

-   120 intent-training examples
-   12 balanced categories
-   TF-IDF with unigram + bigram features
-   Logistic Regression intent classification
-   category-restricted FAQ retrieval
-   cosine similarity
-   confidence-based fallback handling

On the same 24-question clean unseen set:

  Model                                                       Accuracy
  ------------------------------------- ------------------------------
  Baseline TF-IDF + Cosine Similarity                       **41.67%**
  Hybrid Intent + Retrieval Model                           **62.50%**
  Improvement                             **+20.83 percentage points**

The notebook also reports a 4-fold cross-validation result of **66.67% ±
8.84%** for the selected Logistic Regression configuration on the
training portion.

> **Important:** these results are reported exactly as measured in the
> notebook. No accuracy values are fabricated or adjusted to look
> better.

## 🛡️ Confidence & Fallback

The chatbot does not blindly answer every question.

A similarity threshold is selected through a threshold sweep, with the
notebook choosing:

``` text
Confidence threshold = 0.15
```

If a question does not reach the threshold, the chatbot returns a
controlled fallback instead of inventing an answer.

Example:

``` text
User:
What is the recipe for chicken biryani?

Bot:
Sorry, I couldn't find a relevant answer to your question.
```

This is an important safety and reliability feature for an FAQ retrieval
system.

## 💬 Example Queries

``` text
User: How much is the tuition fee?

Bot: Tuition fees depend on the degree program and university fee structure.
```

``` text
User: Which documents do I need to apply?

Bot: You should provide your academic records, identification documents,
     and any additional documents required by the admission office.
```

``` text
User: Can I study online?

Bot: Whether you can study online depends on the program and university policies.
```

------------------------------------------------------------------------

# 👁️ Task 4 --- Object Detection & Tracking

### Real-World Computer Vision Pipeline

Task 4 combines **YOLO object detection** with **Deep SORT multi-object
tracking**.

The system is designed to process video/webcam frames and:

-   detect objects
-   identify object classes
-   calculate confidence scores
-   draw bounding boxes
-   assign persistent tracking IDs
-   count tracked objects
-   save annotated output video
-   provide a live webcam mode

## 🏗️ Architecture

``` text
                 VIDEO / WEBCAM
                       │
                       ▼
                    OpenCV
                 Frame Extraction
                       │
                       ▼
                 YOLO Detector
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Class        Confidence   Bounding Box
          └────────────┼────────────┘
                       ▼
                   Deep SORT
                       │
             Motion + Appearance
                       │
                       ▼
                Tracking IDs
                       │
                       ▼
              Annotated Frames
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Output Video          Live View
```

## 🧩 Technologies

### YOLO

A pretrained Ultralytics YOLO model is used for object detection.

The detector produces:

-   bounding boxes
-   class IDs
-   class names
-   confidence scores

The project uses the lightweight:

``` text
yolo26n.pt
```

model because the development environment is CPU-only.

### Deep SORT

Deep SORT links detections across consecutive frames to maintain object
identities.

It combines:

-   motion prediction
-   Kalman filtering
-   appearance embeddings
-   data association

Example output:

``` text
person | ID: 1
person | ID: 2
car    | ID: 3
```

The goal is for the same physical object to maintain its ID while it
remains trackable.

### OpenCV

OpenCV handles:

-   video input
-   frame extraction
-   video output
-   bounding-box drawing
-   labels
-   webcam input

------------------------------------------------------------------------

## 🖥️ Environment

Task 4 was developed in a dedicated Conda environment:

``` text
Environment: codealpha_cv
Python:      3.11.15
PyTorch:     2.13.0+cpu
OpenCV:      5.0.0
NumPy:       2.4.6
Ultralytics: 8.4.120
```

The system is **CPU-only**.

Because of this, processing speed depends heavily on video resolution
and CPU performance.

------------------------------------------------------------------------

## 📈 Task 4 Evaluation Philosophy

The project intentionally separates:

### Video FPS

The original video's frame rate.

### Processing FPS

How quickly the computer can actually process frames.

``` text
processing FPS =
frames processed / processing time
```

The project does **not** claim real-time performance unless it is
measured.

Likewise, tracking metrics such as MOTA or IDF1 are not fabricated when
ground-truth tracking labels are unavailable.

------------------------------------------------------------------------

# 📊 Object Counting

The Task 4 pipeline distinguishes between two types of counts:

### Current Object Count

Objects currently visible/tracked in the latest processed frame.

Example:

``` text
Currently tracked:
person: 3
car:    2
```

### Unique Objects Seen

Distinct tracking IDs observed during the processing run.

Example:

``` text
Unique persons observed: 7
Unique cars observed:    4
```

A person visible for 200 frames should not be counted as 200 different
people.

------------------------------------------------------------------------

# ⚠️ Limitations

## Task 2 --- FAQ Chatbot

-   The dataset is relatively small.
-   TF-IDF is vocabulary-based and does not provide deep semantic
    understanding.
-   Closely related intents can still be confused.
-   Answers are based on the supplied FAQ knowledge base.
-   Real deployment would require a larger, domain-specific FAQ dataset.

## Task 4 --- Object Detection & Tracking

-   CPU-only inference is slower than GPU inference.
-   Small or distant objects may be missed by the lightweight YOLO
    model.
-   Occlusion can cause tracking failures.
-   Similar-looking objects can sometimes cause ID switches.
-   Long disappearances can cause a new ID to be assigned.
-   No ground-truth tracking dataset is provided, so formal MOTA/IDF1
    accuracy is not claimed.

------------------------------------------------------------------------

# 🚀 Future Improvements

### FAQ Chatbot

-   Expand the FAQ dataset
-   Add sentence embeddings
-   Add semantic search
-   Add multilingual support
-   Connect to a real university knowledge base
-   Add a web interface
-   Add conversation history
-   Add retrieval-augmented generation

### Object Detection & Tracking

-   GPU/CUDA acceleration
-   Larger YOLO model when accuracy is more important than speed
-   ONNX/TensorRT optimization
-   Better tracking-parameter tuning
-   MOT-format benchmark data
-   MOTA / IDF1 evaluation with ground truth
-   Line-crossing analytics
-   Region-based counting
-   Multi-camera tracking
-   Web-based dashboard

------------------------------------------------------------------------

# 📁 Repository Structure

A recommended final GitHub structure is:

``` text
CodeAlpha-AI-Internship/
│
├── README.md
│
├── Task-2-FAQ-Chatbot/
│   ├── FAQ_Chatbot.ipynb
│   ├── data/
│   ├── models/
│   ├── requirements.txt
│   └── screenshots/
│
└── Task-4-Object-Detection-Tracking/
    ├── Object_Detection_Tracking.ipynb
    ├── videos/
    │   └── input.mp4
    ├── outputs/
    │   └── tracked_output.mp4
    ├── models/
    ├── src/
    │   ├── detector_tracker.py
    │   └── webcam_tracker.py
    ├── requirements.txt
    └── screenshots/
```

> The notebooks are the primary artifacts currently included in this
> submission. Add generated model files, output videos, screenshots, and
> the Task 4 `src/` scripts to the repository once they have been
> produced and verified.

------------------------------------------------------------------------

# 🛠️ Installation

## Clone the repository

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd CodeAlpha-AI-Internship
```

## Create a Python environment

For Task 4:

``` bash
conda create -n codealpha_cv python=3.11 -y
conda activate codealpha_cv
```

Install the required packages for the relevant task.

### Task 2

``` bash
pip install pandas numpy scikit-learn joblib matplotlib jupyter
```

### Task 4

``` bash
pip install ultralytics opencv-python deep-sort-realtime numpy torch torchvision
pip install "setuptools<81"
```

The `setuptools<81` compatibility pin is required by the Deep SORT
environment used in this project.

------------------------------------------------------------------------

# ▶️ Running the Projects

## Task 2

Open:

``` text
Task-2-FAQ-Chatbot/FAQ_Chatbot.ipynb
```

Select the appropriate Jupyter kernel and run the notebook from top to
bottom.

The notebook covers:

``` text
Dataset
   ↓
Preprocessing
   ↓
Baseline Retrieval
   ↓
Baseline Evaluation
   ↓
Expanded Intent Dataset
   ↓
TF-IDF
   ↓
Logistic Regression
   ↓
Intent Evaluation
   ↓
Hybrid Retrieval
   ↓
Confidence Threshold
   ↓
Final Evaluation
   ↓
Model Saving
```

## Task 4

Open:

``` text
Task-4-Object-Detection-Tracking/Object_Detection_Tracking.ipynb
```

Place a test video at:

``` text
videos/input.mp4
```

The notebook then performs:

``` text
Video
 ↓
OpenCV
 ↓
YOLO
 ↓
Deep SORT
 ↓
Tracking IDs
 ↓
Annotated Output
```

The intended output is:

``` text
outputs/tracked_output.mp4
```

For live webcam tracking, use the standalone webcam script described in
the notebook.

------------------------------------------------------------------------

# 🎯 Internship Learning Outcomes

Through these projects, the main practical concepts covered include:

### Machine Learning

-   Dataset construction
-   Train/test splitting
-   Feature extraction
-   Model training
-   Cross-validation
-   Classification metrics
-   Error analysis

### Natural Language Processing

-   Text normalization
-   Tokenization
-   Stopword removal
-   TF-IDF
-   N-grams
-   Intent classification
-   Cosine similarity
-   Confidence thresholds
-   Retrieval-based responses

### Computer Vision

-   Object detection
-   Bounding boxes
-   Confidence scores
-   Video frame processing
-   Multi-object tracking
-   Persistent tracking IDs
-   Object counting
-   Performance measurement

------------------------------------------------------------------------

# 📌 Key Results

  -----------------------------------------------------------------------
  Project                              Baseline          Improved / Final
  ------------------- ------------------------- -------------------------
  Task 2 FAQ Chatbot                 **41.67%**    **62.50%** on the same
                                                 24-question clean unseen
                                                                      set

  Task 4 Object                             --- Performance measured from
  Tracking                                        actual video processing
  -----------------------------------------------------------------------

Task 2 achieved a **+20.83 percentage-point improvement** over the
baseline on the clean unseen evaluation set.

Task 4 intentionally reports measured processing performance rather than
an invented "accuracy" percentage.

------------------------------------------------------------------------

# 👨‍💻 About

Developed as part of the **CodeAlpha Artificial Intelligence
Internship**.

The projects demonstrate practical application of:

**NLP • Machine Learning • Computer Vision • Object Detection • Object
Tracking**

------------------------------------------------------------------------

## ⭐ If you find this repository useful

Feel free to explore the notebooks, experiment with the datasets/models,
and improve the pipelines.

**Built with Python and a lot of debugging. 🚀**
