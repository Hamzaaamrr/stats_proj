# Driver Monitoring System

A webcam app that watches the driver and warns about **drowsiness**, **distraction**,
**phone use**, **eating** and **drinking**. Every event is saved to CSV files for the
statistical analysis in R.

> Educational prototype, not a certified safety system.

---

## 1. Setup (Windows, one time)

1. Install **Python 3.11** from https://www.python.org/downloads/windows/
   (tick **"Add python.exe to PATH"** in the installer). Python 3.12 also works;
   the setup script can use 3.14 if that is the Python version installed.
2. Copy the `driver_monitor` folder anywhere, e.g. `Documents\driver_monitor`.
3. Double-click **`setup.bat`**. It creates a virtual environment and installs the
   libraries (about 1 to 2 GB, mostly PyTorch for YOLO). This takes a few minutes.

The model files are already in `models/` (`yolo11n.pt`, `face_landmarker.task`), so the
app works offline after setup.

## 2. Running it

Double-click **`run.bat`** to launch the local web dashboard in your browser. It listens
only on this computer at `http://127.0.0.1:8765`; it is not exposed to your network.
Or open a terminal in the folder and run:

```bat
python main.py --driver D01
```

1. In the browser dashboard, choose a driver ID and camera, then click
   **Start monitoring**. The live preview and status cards show the current behavior,
   EAR, blink rate, PERCLOS and head pose.
2. For the first **5 seconds**, look at the screen/road with your eyes open. This is
   **calibration**. Risky behavior shows a red banner and can play an alert beep.
3. Click **Stop** or press **q** / **Esc** to end monitoring. Normal sessions are saved
   in `data\`; sessions run with **Test mode** enabled are not saved.

The previous desktop dashboard is still available with `python main.py --desktop`.

| Key | Action |
|---|---|
| `q` / `Esc` | stop and save |
| `c` | recalibrate (e.g. after moving the camera) |
| `m` | mute / unmute the beep |
| `0`-`5` | (test mode only) set the temporary true label |

The dashboard also provides buttons for stopping, recalibrating and muting, plus
session settings for lighting, camera position, glasses and test mode. You can upload
a recorded video instead of using the webcam. For automated video processing without
the dashboard, use
`python main.py --headless --video clip.mp4`.

**Test mode is temporary:** it lets you try monitoring and change the true label, but
does not write session, event, or sample data to the CSV files.

Use the **Session history** panel to search the current page, view a session's event
details, or select and delete saved sessions. Deletion also removes that session's
event and sample rows from the local CSV files, and is unavailable while monitoring
is active.

Useful options:

```
--driver D02                 who is driving (for comparing people)
--test-mode                  enable the true-label keys (for accuracy testing)
--lighting dim               saved in sessions.csv (bright / dim / night)
--camera-position left-45    saved in sessions.csv (front / left-45 / right-45)
--glasses yes                saved in sessions.csv
--camera 1                   use an external USB webcam
--video clip.mp4             analyse a recorded video instead of the webcam
--no-audio                   no beeps
--no-objects                 turn off YOLO (faster, face-only)
--desktop                    open the optional desktop dashboard
--port 8766                  use a different local web port
--no-browser                 start the web server without opening a browser tab
```

If the camera doesn't open, close other apps that use it (Teams, Zoom, the Camera app)
or try `--camera 1`. Check Windows Settings > Privacy > Camera if access is blocked.

---

## 3. How it works (for the presentation)

```
 webcam frame
     |
     +--> MediaPipe Face Landmarker (478 face points)
     |        +--> Eye Aspect Ratio (EAR) -------> Drowsiness tracker
     |        +--> 6 points -> solvePnP -> yaw/pitch/roll --> Distraction tracker
     |
     +--> YOLO11n (every 2nd frame): cell phone / cup, bottle / food --> Object tracker
     |
     +--> State machine: picks ONE state by priority, cuts time into events
              +--> on-screen banner + beep
              +--> CSV files (events, samples, sessions)
```

**Calibration.** Everyone's eyes are a different shape and the camera may be at an
angle, so the first 5 s measure *this* driver's open-eye EAR and "looking ahead" head
angle. Everything afterwards is compared to that.

**Drowsiness (EAR).** EAR = eye height / eye width, from 6 points per eye. Open is
about 0.25-0.35 and closed is about 0.1. Eyes count as closed below 75% of the
calibrated value. The driver is **Drowsy** if any of these happen:
- eyes closed for **1.5 s or more** in one go (microsleep)
- **PERCLOS** (percentage of eye closure) over 30% in the last 60 s
- more than **25 blinks per minute** (a blink is a closure under 0.5 s)

**Distraction (head pose).** OpenCV `solvePnP` fits a generic 3D head to 6 face points
(nose, chin, eye corners, mouth corners) to get yaw (left/right), pitch (up/down) and
roll. **Distracted** = head turned more than 30° sideways, more than 20° down or 25° up,
for **2 s or more**, or no face visible for 2 s.

**Phone / eating / drinking (YOLO).** YOLO11n is pretrained on the COCO dataset, which
already includes *cell phone*, *cup*, *bottle*, *wine glass* and foods such as banana,
apple, sandwich, pizza and donut. Rules:
- **Using Phone**: a phone anywhere in view (at the ear or texting low).
- **Drinking / Eating**: a drink or food **near the face**, so a cup in the holder
  doesn't count.
- An object must appear in **60% of the last 15 YOLO runs** to filter out one-frame mistakes.

**Choosing one state.** If several things fire at once, the most dangerous wins:
Drowsy > Using Phone > Distracted > Drinking > Eating > Normal. A state is held for
0.7 s after its signal stops so the label doesn't flicker.

**Confidence.** For YOLO behaviors it's YOLO's average score. For Drowsy/Distracted it's
a 0.5-1.0 score that grows with how far past the threshold the measurement is and how
long it lasted.

All thresholds live in **`config.yaml`** and can be tuned without touching the code.

**Known limitations.** YOLO only flags objects it can see, so a phone held on the far
side of the head (hidden from the camera) may be missed. Sunglasses break EAR. Dark
rooms reduce accuracy. Best camera position: in front of the driver, up to about 45°
to one side.

---

## 4. Output files (in `data\`)

All sessions are appended to the same three files and told apart by `Session_ID`.

**`events.csv`**: one row per event (a stretch of time in one state, including Normal)

| Column | Meaning |
|---|---|
| Session_ID, Event_ID, Driver_ID | identifiers |
| Timestamp, End_Time, Duration_s | when it started/ended and how long |
| Behavior | Normal, Drowsy, Distracted, Using Phone, Eating, Drinking |
| Trigger | why: eye_closure, perclos, blink_rate, head_turn, no_face, yolo:cell phone, ... |
| EAR, EAR_Min | mean and minimum EAR during the event |
| Head_Pose | most common direction: Forward, Left, Right, Down, Up, No Face |
| Yaw, Pitch, Roll | mean head angles (relative to calibration), degrees |
| Confidence, Confidence_Max | mean and max confidence |
| Alert | ON for risky behaviors, OFF for Normal |
| Blink_Rate | mean blinks per minute |

**`samples.csv`**: one row **per second**: Behavior, Trigger, EAR, Head_Pose, angles,
Confidence, Alert, Blink_Rate, PERCLOS, Face_Detected, Phone_Conf, Drink_Conf,
Food_Conf, **True_Label** (test mode), FPS. Use this for behavior percentages and for
accuracy, precision, recall and the confusion matrix.

**`sessions.csv`**: one row per session: driver, start/end, duration, lighting, camera
position, glasses, test mode, calibration values, number of events.

---

## 5. Collecting labeled test data (for accuracy / precision / recall)

Run with `--test-mode` and have a second person press the key for what the driver is
**really** doing as it happens (0 Normal, 1 Drowsy, 2 Distracted, 3 Using Phone,
4 Eating, 5 Drinking). Press `0` again when they go back to normal.

Suggested script for one ~5-minute session (always return to Normal for ~20 s between steps):

| # | Action | Length | Key |
|---|---|---|---|
| 1 | drive normally, look ahead, blink naturally | 40 s | 0 |
| 2 | close eyes slowly for 2-3 s, 3 times | 30 s | 1 |
| 3 | look at the side mirror / out the side window | 3 × 5 s | 2 |
| 4 | look down at the lap | 2 × 5 s | 2 |
| 5 | phone call held to the ear (near side) | 15 s | 3 |
| 6 | phone call held to the ear (far side) | 15 s | 3 |
| 7 | texting with the phone low | 15 s | 3 |
| 8 | drink from a cup or bottle | 3 sips | 5 |
| 9 | eat a snack (banana, sandwich, apple) | 20 s | 4 |

Repeat with 3-5 people and different conditions, and record them with `--lighting`,
`--camera-position` and `--glasses` so the R analysis can compare them. Example:

```bat
python main.py --driver D02 --test-mode --lighting dim --camera-position left-45 --glasses no
```

---

## 6. Project structure

```
driver_monitor/
  main.py            the app: reads frames, runs the detectors, draws, logs
  monitor/gui.py     desktop dashboard and session controls
  monitor/web.py     local-only web server and session API
  monitor/web_ui/    browser dashboard assets
  config.yaml        all thresholds and settings
  monitor/
    camera.py        webcam / video input
    face.py          MediaPipe face landmarks + EAR
    head_pose.py     solvePnP head pose (yaw, pitch, roll)
    calibration.py   per-driver baseline
    drowsiness.py    eye closure, PERCLOS, blink rate
    distraction.py   looking away / no face
    objects.py       YOLO phone, drink, food + smoothing
    state.py         priority rules and events
    alerts.py        banner + beep
    logger.py        CSV files
    overlay.py       drawing on the video
  models/            yolo11n.pt, face_landmarker.task
  data/              CSV output
  tests/             unit tests: python -m pytest -q
```
