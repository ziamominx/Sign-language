# AI Sign Language Recognition System

A temporal, hand + face landmark based sign-language recognizer. It learns
both **hand shape/movement** and **where the hand is relative to the face**
(e.g. "thumb touching the forehead" vs. the same hand shape floating in the
air), because many signs are distinguished only by that contact point.

## Project structure

```
sign_language_ai/
├── collect_data.py        # 1. record labelled sequences from your webcam
├── train_model.py         # 2. train the model on everything collected so far
├── test_model.py          # 3. real-time recognition from your webcam
│
├── dataset/
│   ├── dataset.csv            # one row per (sequence, frame) — append-only
│   └── dataset_metadata.csv   # one row per sequence (summary)
│
├── models/
│   ├── sign_language_model.h5   # or sign_language_model.pkl (RandomForest fallback)
│   ├── label_encoder.pkl
│   ├── scaler.pkl
│   └── config.json
│
├── utils/
│   ├── config.py           # every shared constant (feature layout, paths, etc.)
│   ├── landmarks.py        # MediaPipe hands + face mesh wrapper, drawing
│   ├── features.py         # THE feature pipeline — same for collect/train/test
│   └── preprocessing.py    # resampling, scaling, leakage-safe sequence split
│
└── requirements.txt
```

`utils/features.py` is the single source of truth for "what a frame looks
like as numbers." All three programs import from it, so the feature
representation can never drift between data collection and inference.

## 1. Installation

```bash
cd sign_language_ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This project uses MediaPipe's **Tasks API** (`HandLandmarker` /
`FaceLandmarker`), not the older `mp.solutions.hands` / `mp.solutions.face_mesh`
API — recent `mediapipe` releases on PyPI removed that legacy API entirely,
so `pip install mediapipe` now gives you a version that only has the Tasks
API. On first run, `collect_data.py` / `test_model.py` will automatically
download two small model files (~10–15 MB total) from
`storage.googleapis.com` into `mp_models/` and reuse them on every run
after that — you need an internet connection the very first time you run
either program, but not after. If your network blocks that domain, the
error message tells you the exact URLs to download manually and where to
save them.

TensorFlow is used for the LSTM model but is optional: if it isn't
installed (or fails to install on your platform), `train_model.py` and
`test_model.py` automatically fall back to a scikit-learn `RandomForest`
trained on aggregated per-sequence statistics (mean/std/min/max over
time). You'll still get a working system, just without frame-by-frame
temporal modeling. If you want to skip TensorFlow, delete that line from
`requirements.txt` before installing.

## 2. Collecting your first dataset

```bash
python collect_data.py
```

```text
Enter the sign language you are collecting: ASL
Enter sign label: NO

Camera opened.
1 = Record NO
2 = Record NOT_NO
N = New label   S = Stop current recording early   Q = Quit
```

- **Press `1`** to record a positive example of the current label
  (~5 seconds; movement is captured, not a single frame).
- **Press `2`** to record a negative example — automatically labelled
  `NOT_<label>`.
- **Press `N`** to switch to a new sign label without restarting the camera
  or losing already-collected data (new labels start their own `1`/`2`
  pair automatically).
- **Press `S`** to stop a recording early if you finish the movement
  before 5 seconds are up.
- **Press `Q`** to quit.

Collect at least 15–20 positive and 15–20 negative sequences per sign,
from a few different distances/angles, for anything beyond a toy demo.
Every sequence is **appended** to `dataset/dataset.csv` — nothing is ever
overwritten, so you can stop and resume collection at any time, and even
re-run `collect_data.py` on a different day to add more signs.

## 3. Training

```bash
python train_model.py
```

No prompts — it automatically loads `dataset/dataset.csv`, discovers every
`(language, label)` combination present, reconstructs each temporal
sequence from its `sequence_id`, splits at the **sequence level** (so no
frame from one recording ever leaks across train/val/test), scales
features (fit only on training frames), trains an LSTM (or RandomForest
fallback), and writes everything into `models/`.

```text
Loading dataset...
Languages found: ASL
Classes found:
  ASL::NO
  ASL::NOT_NO
  ASL::FATHER
  ASL::NOT_FATHER

Sequences: 240
Frames: 18,430

Training sequences: 168
Validation sequences: 36
Test sequences: 36
Classes: 4

Training model...
Validation accuracy: 91.67%
Test accuracy: 88.89%

Saving model...
Model saved to:
  models/sign_language_model.h5
```

Re-run this script any time you've collected more data — it always
retrains from scratch on the full current dataset.

## 4. Real-time testing

```bash
python test_model.py
```

Opens the webcam immediately, loads the trained model, and shows a live
prediction panel plus a labelled bounding box around each detected hand:

```
┌─────────────────────────┐
│ SIGN: NO                │
│ CONFIDENCE: 94.2%        │
└─────────────────────────┘
```

Predictions are smoothed over a rolling buffer of recent frames (majority
vote + minimum-confidence threshold) so the display doesn't flicker —
if nothing meets the agreement/confidence bar, it shows `UNKNOWN` rather
than guessing.

## How the temporal + face-hand features work

**Temporal sequence.** Each `1`/`2` press records a *sequence* of frames
(not one snapshot) at roughly a fixed capture rate. In `dataset.csv`, every
frame is one row sharing a `sequence_id`, so `train_model.py` groups rows
by `sequence_id`, sorts by `frame`, and rebuilds the original time series.
Sequences are then resampled (uniformly sub-sampled if too long, padded by
repeating the last frame if too short) to one fixed length so they can be
batched, and fed into an LSTM, which learns patterns across time — not
just a static pose.

**Hand-to-face relationship.** For every frame, `utils/features.py` computes:
- Normalized hand landmarks (21 points × x/y/z, centered on the wrist,
  scaled by hand size) — for both hands independently.
- Normalized face landmarks (forehead, chin, nose, eyes, mouth, cheeks,
  centered on the nose tip, scaled by inter-eye distance).
- **Distances** between key hand points (wrist, thumb tip, index tip) and
  key face points (forehead, chin, nose, eyes, mouth), scaled by face
  size so they're camera-distance invariant.
- The hand's position relative to the face center.

Nothing hard-codes "thumb near forehead = FATHER." Those distance features
are simply part of the input vector; the model learns from your labelled
examples which distances matter for which sign. If you only ever record
`FATHER` with the thumb near the forehead, the model will learn that
association from the data — the same hand shape performed away from the
face will have large hand-to-forehead/chin/nose distances and won't match.

**Missing hands/face.** If a hand or the face isn't detected in a frame,
its landmark block is zero-filled and a `*_present` flag is set to `0`, so
the feature vector length never changes and the model can still recognize
one-handed signs or brief detection dropouts.

## Adding more signs

Just run `collect_data.py` again and either enter a new label at the
start prompt or press `N` mid-session. New labels/negatives are appended
to the same `dataset.csv`. Re-run `train_model.py` — it automatically
picks up every class present in the file; nothing needs to be changed in
the training code.

## Adding more sign languages

Run `collect_data.py` and enter a different value at the "Enter the sign
language you are collecting" prompt (e.g. `ISL`, `BSL`, or any name you
choose — the list isn't hard-coded). Internally, each training class is
`LANGUAGE::LABEL`, so the same label name (e.g. `NO`) in two different
languages is trained as two distinct classes automatically. `test_model.py`
displays both the recognized language and label.

## Notes on reliability

- The system never crashes when a hand or the face temporarily disappears
  — it simply zero-fills that part of the feature vector.
- Feature vector length is asserted in `utils/features.py`, so a mismatch
  (e.g. an old dataset collected with a different version of the pipeline)
  fails loudly in `train_model.py` instead of silently training on
  misaligned data.
- All CSV writes are appends (`open(..., "a")`), so re-running
  `collect_data.py` never deletes previous sessions.

## Desktop app

```bash
pip install -r requirements.txt
python app.py
```
Home -> ADD NEW SIGN (language + sign, keys 1/2/S/Q) -> TRAIN MODEL (runs `train_model.py` in the
background) -> TEST (large camera, subtitle box with ✕ to clear, ← BACK). UI code is in `ui/`;
data collection, features, training and prediction logic still live in the original scripts/`utils`.
