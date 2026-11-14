"""Compare the C++ core with the repo's own adaptation.py (oracle). Usage: python check_vs_scipy.py <repo_root> <build_dir>"""
import importlib.util, itertools, subprocess, sys, tempfile, os
import numpy as np
repo, build = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("adaptation", os.path.join(repo, "scripts/epn612/adaptation.py"))
A = importlib.util.module_from_spec(spec); sys.modules["adaptation"] = A; spec.loader.exec_module(A)

def oracle(x, hp, lp, notch, rect, quant):
    # adaptation.py hardwires HP+LP; for toggled-off stages call scipy with the same code path.
    from scipy.signal import butter, sosfilt, iirnotch, lfilter
    y = x.astype(np.float32)
    if hp and lp and not notch:
        y = A.mimic_board_analog_filter(y, A.BoardAdaptationConfig(notch=False))
    elif hp and lp and notch:
        y = A.mimic_board_analog_filter(y, A.BoardAdaptationConfig(notch=True))
    else:
        if hp: y = sosfilt(butter(1, 20, "highpass", fs=2000, output="sos"), y, axis=-1)
        if lp: y = sosfilt(butter(1, 500, "lowpass", fs=2000, output="sos"), y, axis=-1)
        if notch:
            b, a = iirnotch(50, 30, fs=2000); y = lfilter(b, a, y, axis=-1)
        y = np.asarray(y, np.float32)
    if rect: y = np.abs(y)
    if quant: y = A.board_counts_to_normalized(A.quantize_signed_adc(y, 12), 12)
    return y

rng = np.random.default_rng(0)
n = 6000
t = np.arange(n) / 2000
x = (0.25 * rng.standard_normal((8, n)) + 0.3 * np.sin(2*np.pi*50*t) + 0.1).astype(np.float32)
x[0, 1000:1100] = 1.5          # drive into clipping
tmp = tempfile.mkdtemp(); fin = os.path.join(tmp, "in.f32")
np.ascontiguousarray(x.T).tofile(fin)

def run(exe, flags, phase=0, chunk=n):
    so, wo = os.path.join(tmp, "s.f32"), os.path.join(tmp, "w.f32")
    subprocess.run([os.path.join(build, "emg_cli"), fin, str(n), so, wo, *map(str, flags), str(phase), str(chunk)], check=True, capture_output=True)
    return np.fromfile(so, np.float32).reshape(n, 8).T, np.fromfile(wo, np.float32).reshape(-1, 8, 400)

worst = 0.0
print(f"{'hp lp no re qu':16s} {'max err':>10s} {'samples off by 1 LSB':>22s}")
for flags in itertools.product([0, 1], repeat=5):
    ref = oracle(x, *flags)
    sf, _ = run("emg_cli", flags)
    err = np.abs(sf - ref).max()
    if flags[4]:
        frac = np.mean(sf != ref)
        assert err <= 1 / 2048 + 1e-9, "more than 1 LSB off"
        print(f"{' '.join(map(str, flags)):16s} {err:10.2e} {frac:21.4%}")
    else:
        worst = max(worst, err)
        assert err < 1e-5, "pre-quantization error above 1e-5"
        print(f"{' '.join(map(str, flags)):16s} {err:10.2e} {'-':>22s}")
print("worst pre-quant float32 error:", f"{worst:.2e}", "(budget 1e-5)")

# windows + chunk invariance + phase
flags = (1, 1, 1, 1, 1); full = oracle(x, *flags)
base = run("emg_cli", flags, phase=137, chunk=n)[1]
for chunk in (1, 7, 200):
    w = run("emg_cli", flags, phase=137, chunk=chunk)[1]
    assert np.array_equal(w, base), f"chunk {chunk} differs"
starts = np.arange(137, n - 400 + 1, 200)
sig_f = run("emg_cli", flags)[0]
expect = np.stack([sig_f[:, s:s+400] for s in starts])
assert np.array_equal(base, expect), "window content/alignment mismatch"
print(f"chunk invariance 1/7/200/all: bit-identical OK ; {len(base)} windows at phase 137 OK")
