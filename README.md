# Conveyor Part Tracking and Counting

An independent industrial-vision demonstrator for a fixed camera viewing coloured, separated parts on a conveyor.

![Demonstration output](outputs/preview.png)

## What it does

HSV saturation/value segmentation separates coloured parts from a neutral belt. Morphological opening removes small noise. Connected components provide bounding boxes and centroids. A distance-gated, one-to-one greedy association links detections to tracks. A line with a hysteresis band counts confirmed tracks moving left to right once per track. Short detection dropouts preserve identity; stale tracks expire.

## Quick start

Tested with Python 3.12 on Linux. These commands run without a camera or graphical display.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate  # Linux / WSL
python -m pip install -r requirements-dev.txt
python -m pytest -q
python app.py --demo
```

On Windows, replace the `source` line with the PowerShell activation command shown in the comment. On systems where the executable is named `python3`, use it for the first command. The demo writes `outputs/results.json` and visual outputs. No private images or industrial datasets are required.

## Verified result

6 tests passed. Five deterministic synthetic clips contained 60 expected crossings in total; the pipeline counted 60. Mean absolute count error was 0 per clip. These clips use separated lanes, colour contrast, lighting variation and sensor noise; they do not establish performance under touching parts or occlusion.

## Process a video

```bash
python app.py --video path/to/conveyor.mp4 --line-x 320 --output real-results
```

The real-video mode writes counts and crossing events to `real-results/results.json`. Set the line coordinate for the input resolution. Thresholds and track settings are configurable in `conveyor.py`; validate them on development footage before testing on a separate clip.

## Files

- `app.py`: CLI, input handling, synthetic demo and output generation.
- Core module: reusable algorithm and validation logic.
- `tests/`: regression checks including failure conditions.
- `outputs/`: generated evidence from the included experiment.
- `REPORT.md`: project rationale, method, evaluation and limitations.

## Limits and next step

The detector assumes coloured parts against a neutral background. Greedy centroid matching can exchange identities during close interactions, occlusion or large jumps. Merged components and tracks lost near the line can cause count errors. Camera vibration, reflections and unrelated saturated regions may create false detections. Processing times in results.json are environment-specific and exclude decoding/rendering.

Record a real conveyor or tabletop clip with manual crossing annotations. Include occlusions and similar-coloured parts. Compare this classical baseline with a learned detector and a motion-model tracker.

## References

- https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
- https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
