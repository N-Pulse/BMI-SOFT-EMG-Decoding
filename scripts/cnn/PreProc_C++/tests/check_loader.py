"""Compare the C++ JSON loader with the repo's own loader.py (oracle).
Usage: python check_loader.py <repo_root> <build_dir>"""
import importlib.util, json, os, subprocess, sys, tempfile
import numpy as np

repo, build = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("loader", os.path.join(repo, "src/bmiemg/data/epn612/loader.py"))
L = importlib.util.module_from_spec(spec); sys.modules["loader"] = L; spec.loader.exec_module(L)
exe = os.path.join(build, "loader_dump")
tmp = tempfile.mkdtemp()

def make_file(name, mutate=None):
    """Fake EPN-612 participant file: random int8 EMG, tricky trial order, optional fields missing."""
    rng = np.random.default_rng(0)
    d = {"generalInfo": {"samplingFrequencyInHertz": 200}, "userInfo": {"name": "user7"},
         "trainingSamples": {}, "testingSamples": {}}
    for k in [1, 2, 10, 3, 11, 20]:                       # must come back as 1, 2, 3, 10, 11, 20
        n = int(rng.integers(900, 1100))
        trial = {"emg": {f"ch{c}": rng.integers(-128, 128, n).tolist() for c in range(1, 9)},
                 "gestureName": ["fist", "open", "waveIn", "waveOut", "pinch", "noGesture"][k % 6],
                 "groundTruthIndex": [100, 400], "startPointforGestureExecution": 90,
                 "groundTruth": rng.integers(0, 2, n).tolist()}
        if k == 3:  del trial["gestureName"]               # optional field missing
        if k == 10: trial["groundTruthIndex"] = None       # explicit null
        if k == 11: del trial["startPointforGestureExecution"]
        if k == 20: del trial["groundTruth"]
        d["trainingSamples"][f"idx_{k}"] = trial
    d["testingSamples"]["idx_1"] = d["trainingSamples"]["idx_2"]
    if mutate: mutate(d)
    path = os.path.join(tmp, name)
    json.dump(d, open(path, "w"))
    return path

def run_cpp(path, section="trainingSamples"):
    out = os.path.join(tmp, "out.json")
    r = subprocess.run([exe, path, out, section], capture_output=True, text=True)
    return (json.load(open(out)) if r.returncode == 0 else None), r.stderr.strip()

# ---- 1. valid file: every field must match, for both sections ----
path = make_file("user7.json")
for section in ("trainingSamples", "testingSamples"):
    cpp, err = run_cpp(path, section)
    assert cpp is not None, f"C++ loader failed: {err}"
    py = list(L.load_user_trials(path, section))
    assert [t["trialId"] for t in cpp] == [t.trial_id for t in py], "trial order differs"
    for c, p in zip(cpp, py):
        assert c["userId"] == p.user_id and c["samplingRateHz"] == p.sampling_rate_hz
        assert (c["gesture"] or None) == p.gesture, f"{p.trial_id}: gesture"
        assert np.array_equal(np.array(c["emg"], np.float32), p.emg), f"{p.trial_id}: emg"
        assert (tuple(c["groundTruthIndex"]) if c["groundTruthIndex"] else None) == p.ground_truth_index, f"{p.trial_id}: groundTruthIndex"
        assert (None if c["gestureStart"] == -1 else c["gestureStart"]) == p.gesture_start, f"{p.trial_id}: gestureStart"
        assert (c["groundTruth"] or None) == (None if p.ground_truth is None else p.ground_truth.tolist()), f"{p.trial_id}: groundTruth"
    print(f"{section:16s} {len(cpp)} trials, order {[t['trialId'] for t in cpp]}: identical to loader.py OK")

# ---- 2. broken files: both loaders must refuse them ----
def drop_ch3(d): del d["trainingSamples"]["idx_1"]["emg"]["ch3"]
def unequal(d): d["trainingSamples"]["idx_1"]["emg"]["ch5"].pop()
def bad_rate(d): d["generalInfo"]["samplingFrequencyInHertz"] = 0
def bad_gti(d): d["trainingSamples"]["idx_2"]["groundTruthIndex"] = [400, 100]
def no_user(d): del d["userInfo"]
for mutate in (drop_ch3, unequal, bad_rate, bad_gti, no_user):
    p = make_file(f"{mutate.__name__}.json", mutate)
    cpp, err = run_cpp(p)
    try:
        list(L.load_user_trials(p)); py_failed = False
    except Exception:
        py_failed = True
    assert cpp is None and py_failed, f"{mutate.__name__}: C++ failed={cpp is None}, Python failed={py_failed}"
    print(f"{mutate.__name__:16s} rejected by both: {err}")

# ---- 3. missing file ----
cpp, err = run_cpp(os.path.join(tmp, "does_not_exist.json"))
assert cpp is None, "missing file should fail"
print(f"{'missing file':16s} rejected: {err}")
print("ALL LOADER CHECKS PASSED")
