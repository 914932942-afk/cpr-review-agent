v4 frozen pipeline: 26 public RGB recordings, single view ch0

Purpose
Complete the raw-video acceptance of the current v4 classifier. No fitting,
threshold search, new data collection, new labels or Web/Agent integration.
Technical completion and recognition quality are separate decisions.

Locations
  Model: data/public_cpr/rgb-evaluation-v4/
  Media: data/public_cpr/rgb-acceptance-v1/
  Frozen run: data/public_cpr/rgb-video-acceptance-v4/
  Report: test-results/public-rgb-video-acceptance-v4/

Environment
Use C:\Users\gjj33\.conda\envs\ml\python.exe from CODE on Windows.
The exact Python, installed packages and GPU are recorded in environment.json.
Run serial GPU processes, one per recording; do not use the web .venv for ML.

Workflow (already versioned; never overwrite an existing run)
  python experiments/public_rgb_video_acceptance/freeze.py
  python experiments/public_rgb_video_acceptance/run_batch.py
  While the batch is running, before completion:
    python experiments/public_rgb_video_acceptance/freeze_scoring.py
  After all 26 attempts:
    python experiments/public_rgb_video_acceptance/summarize.py
    python experiments/public_rgb_video_acceptance/record_acceptance.py

The batch can resume only completed, checksum-verified attempt records and
untouched, unstarted jobs. An incomplete attempt folder or runner.lock requires
inspection; never delete it and silently retry. Completed failures are retained.
One attempt per video, timeout 900s. No automatic alternative recognition path.

Frozen execution contract
26 attempts, 26 valid classifications, 11215 decoded frames, matching video and
pipeline hashes, exact head/geometry replay, no labels supplied to inference.
If any video fails, full-set F1 is not reported using a convenient subset;
coverage and execution fail, with all failure records retained.

Frozen quality contract
Same numerical gates as the pre-existing v4 single-error protocol:
Macro-F1 >= .85, every supported error recall >= .60, classification coverage
100%, original Correct false alarm rate <= .10 (this batch has 2 Correct clips,
so that requires zero false alarms). The 26 videos have no composite examples.

Inference
The batch invokes the unchanged v4 infer_compact_video.py with an explicit
manifest, without --reference-record. Public labels are read for scoring only
after the batch completes. Source labels are cross-checked at freeze time
against the public annotation archive; all groups are test-only under v4.

Evidence
Each runs/<record>/ folder holds inference.json, inference.npz, process.txt and
attempt.json with output checksums. model-lock.json freezes weights, operating
thresholds, visual preprocessing and old application sources. Scoring sources
are separately frozen before the batch completes. Current candidate metadata
is updated only after final contracts pass; the v4 model/entrypoint stay fixed.

Interpretation
This is a repeatedly exposed engineering regression collection, not a blind
test. It contains 2 Correct and 11 single-error types, with few examples per
type. It cannot establish composite recognition, cross-person generalization,
physical compression depth, measured BPM, clinical effectiveness or exact
error onsets. Empty predictions do not prove correct CPR. All outputs remain
offline experimental results; no Web/Agent release is approved in this step.
