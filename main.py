#!/usr/bin/env python3
# main.py -- Herniation Risk Triage.
# Research prototype. Not a medical device.
#
# NOTE: there is NO `run` command.  HRI/BSI/DAR replay only happens
# inside the APK / desktop exe GUI, never in the terminal.
# The only command you ever need is `bootstrap`.

import argparse, csv, os, re, shlex, shutil, subprocess, sys, tempfile, threading, time
from collections import deque
from pathlib import Path

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    np = None
    _NP_OK = False

if _NP_OK:
    if hasattr(np, "trapezoid"): _trapz = np.trapezoid
    elif hasattr(np, "trapz"):   _trapz = np.trapz
    else:
        def _trapz(y, x=None, dx=1.0):
            y = np.asarray(y, dtype=float)
            x = (np.asarray(x, dtype=float) if x is not None
                 else np.arange(len(y), dtype=float) * dx)
            return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(x)))
else:
    _trapz = None

try:
    from scipy.signal import butter, filtfilt, iirnotch, welch
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

def _try_import(name):
    try: return __import__(name, fromlist=["*"])
    except ImportError: return None

PROFILES = {
    "desktop": {"fs": 250, "window_sec": 4.0, "step_samples": 50,
                "k_hold": 3, "hri_threshold": 0.65, "bsi_threshold": 0.25,
                "use_notch": True, "update_hz": 30},
    "mobile":  {"fs": 200, "window_sec": 3.0, "step_samples": 100,
                "k_hold": 3, "hri_threshold": 0.65, "bsi_threshold": 0.25,
                "use_notch": False, "update_hz": 15},
}
CFG = dict(PROFILES["desktop"])
ARTIFACT_LIMIT_V = 100e-6

# ==================== Signal processing ====================

def _bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band")

def preprocess(window, fs, use_notch=True):
    if np.any(np.abs(window) > ARTIFACT_LIMIT_V): return None
    if not SCIPY_OK: return window
    try:
        b, a = _bandpass(0.5, 45.0, fs)
        out = filtfilt(b, a, window, axis=0)
        if use_notch and fs > 100:
            bn, an = iirnotch(50.0, 30.0, fs)
            out = filtfilt(bn, an, out, axis=0)
    except Exception:
        return None
    if np.any(np.abs(out) > ARTIFACT_LIMIT_V): return None
    return out

def extract_features(window, fs):
    left  = window[:, 0]
    right = window[:, 1] if window.shape[1] > 1 else window[:, 0]
    bands = {"delta": (0.5, 4), "theta": (4, 8),
             "alpha": (8, 13), "beta": (13, 30)}
    if SCIPY_OK:
        nper = min(len(left), 256)
        f_l, pxx_l = welch(left,  fs=fs, nperseg=nper)
        f_r, pxx_r = welch(right, fs=fs, nperseg=nper)
        def bp(f, pxx, band):
            idx = (f >= band[0]) & (f <= band[1])
            return float(_trapz(pxx[idx], f[idx])) if np.any(idx) else 1e-12
        p = {k: bp(f_l, pxx_l, b) + bp(f_r, pxx_r, b) for k, b in bands.items()}
        la = bp(f_l, pxx_l, bands["alpha"]); ra = bp(f_r, pxx_r, bands["alpha"])
    else:
        v = float(np.var(left) + np.var(right) + 1e-12)
        p = {k: v for k in bands}; la = ra = v
    dar_log = float(np.log10(p["delta"] / (p["alpha"] + 1e-12) + 1e-12))
    bsi = float(np.abs(la - ra) / (la + ra + 1e-12))
    try:
        el, er = np.abs(left), np.abs(right)
        pers = (float(np.clip(np.corrcoef(el, er)[0, 1], 0, 1))
                if np.std(el) > 0 and np.std(er) > 0 else 0.0)
    except Exception:
        pers = 0.0
    return {"DAR_log": dar_log, "BSI": bsi, "persistence": pers}

def compute_hri(f):
    dar_s  = float(np.clip((f["DAR_log"] + 1.0) / 2.0, 0, 1))
    bsi_s  = float(np.clip(f["BSI"] / 0.5, 0, 1))
    pers_s = 1.0 - f["persistence"]
    return float(np.clip(0.45 * dar_s + 0.35 * bsi_s + 0.20 * pers_s, 0, 1))

# ==================== CSV source + pipeline ====================

def _load_csv(path, target_fs, file_fs=None):
    with path.open("r", newline="") as f:
        first_line = f.readline()
    has_header = any(ch.isalpha() for ch in first_line)
    try:
        arr = np.loadtxt(str(path), delimiter=",",
                         skiprows=1 if has_header else 0, ndmin=2, dtype=float)
    except ValueError as e:
        raise ValueError(f"{path}: not a numeric CSV ({e})")
    if arr.shape[0] < 8:
        raise ValueError(f"{path}: too few samples ({arr.shape[0]})")
    source_fs = float(file_fs) if file_fs else float(target_fs)
    if arr.shape[1] >= 3:
        col0 = arr[:, 0]; diffs = np.diff(col0)
        if (np.all(diffs > 0) and diffs.mean() < 0.1
                and diffs.std() < 0.05 * diffs.mean()):
            if file_fs is None:
                source_fs = 1.0 / float(np.median(diffs))
            arr = arr[:, 1:]
    if arr.shape[1] < 2:
        arr = np.column_stack([arr[:, 0], arr[:, 0]])
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0: arr = arr * 1e-6
    elif peak > 1e-3: arr = arr * 1e-3
    return arr.astype(np.float64), source_fs

class CSVSource:
    def __init__(self, path, target_fs, file_fs=None, loop=False):
        self.fs = target_fs; self.loop = loop
        data, source_fs = _load_csv(path, target_fs, file_fs=file_fs)
        if abs(source_fs - target_fs) > 1.0:
            data = self._resample(data, source_fs, target_fs)
        self.data = data; self.n = data.shape[0]; self._pos = 0
        self._lock = threading.Lock()
    @staticmethod
    def _resample(arr, src_fs, dst_fs):
        new_n = max(2, int(round(arr.shape[0] * dst_fs / src_fs)))
        src_idx = np.arange(arr.shape[0], dtype=float)
        new_idx = np.linspace(0, arr.shape[0] - 1, new_n)
        out = np.empty((new_n, arr.shape[1]), dtype=float)
        for c in range(arr.shape[1]):
            out[:, c] = np.interp(new_idx, src_idx, arr[:, c])
        return out
    def read(self, k):
        with self._lock:
            if self._pos + k > self.n:
                if self.loop and self.n > 0: self._pos = 0
                else: return None
            chunk = self.data[self._pos:self._pos + k].copy()
            self._pos += k; return chunk
    def stop(self): pass

class Pipeline:
    def __init__(self, source, cfg, callback):
        self.source = source; self.cfg = cfg; self.callback = callback
        self.running = False; self.alarm_counter = 0
        self.window_samples = int(cfg["fs"] * cfg["window_sec"])
        self.step_samples = int(cfg["step_samples"])
        self._buf = None
    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        fs = self.cfg["fs"]; prev_pos = 0
        try:
            while self.running:
                chunk = self.source.read(self.step_samples)
                if chunk is None: break
                cur_pos = getattr(self.source, "_pos", 0)
                if self.source.loop and cur_pos < prev_pos:
                    self._buf = None; self.alarm_counter = 0
                prev_pos = cur_pos
                self._buf = (chunk if self._buf is None
                             else np.vstack([self._buf, chunk]))
                if len(self._buf) > self.window_samples * 2:
                    self._buf = self._buf[-self.window_samples * 2:]
                if len(self._buf) >= self.window_samples:
                    win = self._buf[-self.window_samples:]
                    clean = preprocess(win, fs, self.cfg["use_notch"])
                    if clean is not None:
                        feats = extract_features(clean, fs)
                        hri = compute_hri(feats)
                        if hri > self.cfg["hri_threshold"]: self.alarm_counter += 1
                        else: self.alarm_counter = 0
                        alarm = self.alarm_counter >= self.cfg["k_hold"]
                        self.callback(clean, feats, hri, alarm)
                time.sleep(self.step_samples / fs)
        finally:
            self.running = False
    def stop(self): self.running = False

# ==================== GUI (only runs inside APK / exe) ====================

def _launch_gui(profile="desktop"):
    """Run GUI with auto-generated sample. Used by APK and desktop exe."""
    if not _NP_OK:
        msg = "ERROR: numpy missing. Install: pip install numpy scipy"
        sys.stderr.write(msg + "\n")
        try:
            from kivy.app import App
            from kivy.uix.label import Label
            class E(App):
                def build(self): return Label(text=msg)
            E().run()
        except Exception: pass
        return 1
    CFG.update(PROFILES[profile])
    if _try_import("kivy") is None:
        sys.stderr.write("ERROR: Kivy not installed.\n")
        return 1
    sample = Path("sample_eeg.csv")
    if not sample.exists():
        try: make_sample(path=sample, fs=CFG["fs"], seconds=30)
        except Exception as e: print(f"[gui] sample gen failed: {e}")
    if not sample.exists():
        try:
            with sample.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["t", "ch1", "ch2"])
                for i in range(200):
                    w.writerow([f"{i/200:.6f}", "0.000020", "0.000020"])
        except Exception as e: print(f"[gui] fallback CSV failed: {e}")

    from kivy.app import App
    from kivy.clock import Clock
    from kivy.graphics import Color, Line, Rectangle
    from kivy.uix.boxlayout import BoxLayout
    from kivy.uix.button import Button
    from kivy.uix.gridlayout import GridLayout
    from kivy.uix.label import Label
    from kivy.uix.widget import Widget
    ws = int(CFG["fs"] * CFG["window_sec"])

    class EEGPlot(Widget):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.data = deque(maxlen=ws * 3)
            with self.canvas:
                Color(0.08, 0.08, 0.12)
                self._bg = Rectangle(pos=self.pos, size=self.size)
                Color(0.2, 0.8, 0.2); self.line_l = Line(points=[], width=1.2)
                Color(0.8, 0.3, 0.3); self.line_r = Line(points=[], width=1.2)
            self.bind(pos=self._resize, size=self._resize)
            Clock.schedule_interval(self._draw, 1.0 / CFG["update_hz"])
        def _resize(self, *_):
            self._bg.pos, self._bg.size = self.pos, self.size
        def add(self, win):
            for s in win: self.data.append(s)
        def _draw(self, _dt):
            if len(self.data) < 2: return
            arr = np.array(self.data, dtype=float)
            w, h = self.size; x0, y0 = self.pos
            for ch, line in ((0, self.line_l), (1, self.line_r)):
                if arr.shape[1] <= ch: continue
                y = arr[:, ch]; rng = float(y.max() - y.min())
                yp = (np.full_like(y, y0 + h * 0.5) if rng < 1e-12
                      else (y - y.min()) / rng * (h * 0.8) + y0 + h * 0.1)
                xs = np.linspace(x0, x0 + w, len(yp))
                line.points = [(float(x), float(v)) for x, v in zip(xs, yp)]

    class Root(BoxLayout):
        def __init__(self, **kw):
            super().__init__(orientation="vertical", **kw)
            self.status = Label(text="Starting...", size_hint=(1, 0.07))
            self.add_widget(self.status)
            self.plot = EEGPlot(size_hint=(1, 0.55)); self.add_widget(self.plot)
            g = GridLayout(cols=4, size_hint=(1, 0.15))
            g.add_widget(Label(text="HRI")); self.hri_lbl = Label(text="-"); g.add_widget(self.hri_lbl)
            g.add_widget(Label(text="BSI")); self.bsi_lbl = Label(text="-"); g.add_widget(self.bsi_lbl)
            self.add_widget(g)
            self.alarm = Label(text="OK", color=(0, 1, 0, 1),
                               size_hint=(1, 0.13), font_size="24sp")
            self.add_widget(self.alarm)
            ctrl = BoxLayout(size_hint=(1, 0.10), spacing=8)
            ctrl.add_widget(Button(text="Start", on_press=self.start))
            ctrl.add_widget(Button(text="Stop", on_press=self.stop))
            self.add_widget(ctrl)
            self.source = self.pipe = None
        def start(self, *_):
            if self.pipe and self.pipe.running: return
            try:
                self.source = CSVSource(sample, CFG["fs"], loop=True)
            except Exception as e:
                self.status.text = f"ERROR: {e}"; return
            self.pipe = Pipeline(self.source, CFG, self._on_result)
            self.pipe.start(); self.status.text = f"Replaying {sample.name}"
        def stop(self, *_):
            if self.pipe: self.pipe.stop()
            if self.source: self.source.stop()
            self.status.text = "Stopped"
        def _on_result(self, win, feats, hri, alarm):
            Clock.schedule_once(lambda dt: self._update(win, feats, hri, alarm))
        def _update(self, win, feats, hri, alarm):
            self.plot.add(win)
            self.hri_lbl.text = f"{hri:.2f}"
            self.bsi_lbl.text = f"{feats['BSI']:.2f}"
            if alarm:
                self.alarm.text = "HERNIATION RISK"; self.alarm.color = (1, 0, 0, 1)
            else:
                self.alarm.text = "OK"; self.alarm.color = (0, 1, 0, 1)

    class TriApp(App):
        def build(self):
            self.title = "Herniation Triage (Research Prototype)"
            root = Root()
            Clock.schedule_once(lambda dt: root.start(), 0.5)
            return root

    TriApp().run()
    return 0

def make_sample(path=Path("sample_eeg.csv"), fs=250, seconds=60, seed=42):
    if not _NP_OK:
        sys.stderr.write("ERROR: numpy required for make-sample.\n"); return None
    path = Path(path)
    if path.exists(): print(f"  (overwriting {path})")
    n = int(fs * seconds); rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    left  = 20e-6 * np.sin(2*np.pi*10*t) + 0.5e-6 * rng.standard_normal(n)
    right = 20e-6 * np.sin(2*np.pi*10*t + 0.1) + 0.5e-6 * rng.standard_normal(n)
    event = (t > seconds / 2).astype(float)
    left  = left  * (1 - 0.75 * event) + 30e-6 * np.sin(2*np.pi*2*t) * event
    right = right + 5e-6 * np.sin(2*np.pi*2*t) * event
    with path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "ch1", "ch2"])
        for i in range(n):
            w.writerow([f"{t[i]:.6f}", f"{left[i]:.9f}", f"{right[i]:.9f}"])
    kb = path.stat().st_size // 1024
    print(f"Wrote {path} ({n} rows, {seconds}s @ {fs} Hz, ~{kb} KB)")
    return path

# ==================== Android detection ====================

def _is_apk_runtime():
    """True ONLY inside a p4a-built APK (not on plain Termux)."""
    return "P4A_BOOTSTRAP" in os.environ or "ANDROID_ARGUMENT" in os.environ

# ==================== Scaffold constants ====================

REQUIREMENTS_TXT = "kivy>=2.3.0,<2.4\nnumpy>=1.24,<3\nscipy>=1.10,<1.15\n"

BUILDOZER_SPEC = """\
[app]
title = Herniation Triage
package.name = herniationtriage
package.domain = org.research
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy==2.3.0,numpy==1.26.4
orientation = portrait
fullscreen = 0
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.numeric_version = 1
android.no_compile_py = 1
android.permissions = 

[buildozer]
log_level = 2
warn_on_root = 1
"""

GITIGNORE = """\
__pycache__/
*.py[cod]
*.so
build/
dist/
eggs/
*.egg-info/
*.egg
.venv/
venv/
env/
.buildozer/
.gradle/
gradle/
Android/
android/
kivy_env/
.kivy_env/
*.apk
*.aab
*.idsig
bin/
*.class
*.jar
*.war
*.ear
*.aar
target/
*.spec
!buildozer.spec
.pytest_cache/
.mypy_cache/
.tox/
.coverage
htmlcov/
.idea/
.vscode/
*.iml
*.swp
.DS_Store
Thumbs.db
node_modules/
sample_eeg.csv
*.tmp
*.log
"""

README_MD = """\
# Herniation Risk Triage (Research Prototype)

    python main.py bootstrap

Pushes to GitHub; Actions builds desktop executables + Android APK.
Not a medical device.
"""

DESKTOP_WORKFLOW = """\
name: Build Desktop Executables
"on":
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - name: Install dependencies
        run: pip install -r requirements.txt pyinstaller
      - name: Patch kivy.garden __path__
        shell: bash
        run: |
          python - <<'PY'
          import pathlib, sysconfig
          init = pathlib.Path(sysconfig.get_paths()["purelib"]) / "kivy" / "garden" / "__init__.py"
          if init.exists():
              txt = init.read_text()
              if "__path__ = 'kivy.garden'" in txt:
                  txt = txt.replace("__path__ = 'kivy.garden'",
                                    "__path__ = [__import__('os').path.dirname(__file__)]")
                  init.write_text(txt); print(f"patched {init}")
              else: print("already patched")
          else: print("kivy.garden not present")
          PY
      - name: Build executable
        run: >
          pyinstaller --onefile --windowed
          --name HerniationTriage
          --hidden-import kivy.core.window
          --collect-data kivy
          --collect-binaries kivy
          --exclude-module kivy.garden
          main.py
      - uses: actions/upload-artifact@v4
        with:
          name: HerniationTriage-${{ matrix.os }}
          path: dist/*
"""

ANDROID_WORKFLOW = """\
name: Build Android APK
"on":
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Restore Buildozer cache
        id: cache
        uses: actions/cache@v4
        with:
          path: |
            ~/.buildozer
            .buildozer
            ~/.cache/pip
          key: buildozer-${{ runner.os }}-${{ hashFiles('buildozer.spec', 'requirements.txt') }}
          restore-keys: |
            buildozer-${{ runner.os }}-
      - name: Report cache status
        run: |
          if [ "${{ steps.cache.outputs.cache-hit }}" = "true" ]; then
            echo "CACHE HIT -- ~6-10 min"
          else
            echo "CACHE MISS -- ~35-45 min"
          fi
          du -sh "$HOME/.buildozer" 2>/dev/null || true
      - name: Prepare cache dirs
        run: mkdir -p "$HOME/.buildozer" "$HOME/.cache/pip"
      - name: Build APK
        run: |
          docker run --rm \\
            --volume "$PWD":/home/user/hostcwd \\
            --volume "$HOME/.buildozer":/home/user/.buildozer \\
            --volume "$HOME/.cache/pip":/home/user/.cache/pip \\
            --workdir /home/user/hostcwd \\
            kivy/buildozer:latest \\
            buildozer -v android debug
      - uses: actions/upload-artifact@v4
        with:
          name: HerniationTriage-Android
          path: bin/*.apk
"""

WARM_CACHE_WORKFLOW = """\
name: Warm Build Cache
"on":
  schedule: [{cron: '0 3 * * 0'}]
  workflow_dispatch:

jobs:
  warm:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4
        with:
          path: |
            ~/.buildozer
            .buildozer
            ~/.cache/pip
          key: buildozer-${{ runner.os }}-${{ hashFiles('buildozer.spec', 'requirements.txt') }}
          restore-keys: |
            buildozer-${{ runner.os }}-
      - run: echo "Cache touched."
"""

def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path}")

def _ensure_main_py():
    if getattr(sys, "frozen", False): return
    src = Path(__file__).resolve(); target = Path("main.py").resolve()
    if src == target: return
    shutil.copy2(src, Path("main.py")); print("  synced main.py")

def cmd_setup_github():
    root = Path.cwd()
    print("[setup-github] writing project scaffolding...")
    _write(root / "requirements.txt", REQUIREMENTS_TXT)
    _write(root / "buildozer.spec", BUILDOZER_SPEC)
    _write(root / ".gitignore", GITIGNORE)
    _write(root / "README.md", README_MD)
    _write(root / ".github/workflows/build-desktop.yml", DESKTOP_WORKFLOW)
    _write(root / ".github/workflows/build-android.yml", ANDROID_WORKFLOW)
    _write(root / ".github/workflows/warm-cache.yml", WARM_CACHE_WORKFLOW)
    _ensure_main_py()
    print("[setup-github] done.")

# ==================== Git logging ====================

_LOG = {"git": True}
_PUSH_LAST_ERR = {"stderr": "", "code": 0}

def _git_log(msg, kind="info"):
    if not _LOG["git"]: return
    tag = {"info": "[git]", "cmd": "[git]", "ok": "[git-ok]",
           "warn": "[git-warn]", "err": "[git-err]"}.get(kind, "[git]")
    print(f"{tag} {time.strftime('%H:%M:%S')} {msg}", flush=True)

def _fmt_cmd(cmd):
    s = shlex.join(str(c) for c in cmd)
    return s if len(s) <= 240 else s[:237] + "..."

def _run(cmd, **kw):
    kw.setdefault("capture_output", True); kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)

def _git(*args, quiet=False, timeout=600, stream=False, **kw):
    cmd = ["git", *args]
    if stream and not quiet:
        stream_cmd = [*cmd, "--progress"]
        _git_log(f"$ {_fmt_cmd(stream_cmd)}  (streaming)")
        t0 = time.time()
        try: r = subprocess.run(stream_cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            _git_log(f"$ {_fmt_cmd(stream_cmd)}  TIMEOUT", "err"); raise
        _git_log(f"  exit={r.returncode} {time.time()-t0:.1f}s",
                 "ok" if r.returncode == 0 else "err")
        return r
    kw["capture_output"] = True; kw["text"] = True; kw["timeout"] = timeout
    t0 = time.time()
    try: r = subprocess.run(cmd, **kw)
    except subprocess.TimeoutExpired:
        _git_log(f"$ {_fmt_cmd(cmd)}  TIMEOUT", "err"); raise
    dur = time.time() - t0
    if quiet: return r
    ok = r.returncode == 0
    _git_log(f"$ {_fmt_cmd(cmd)}  [{'ok' if ok else 'FAIL=' + str(r.returncode)} {dur:.2f}s]",
             "cmd" if ok else "err")
    if args and args[0] in ("push","pull","fetch","clone","commit","tag",
                            "merge","rebase","ls-remote","remote","add",
                            "rm","reset","checkout","init","config"):
        for line in (r.stdout or "").rstrip().splitlines()[:12]:
            _git_log(f"  {line}", "cmd" if ok else "err")
        for line in (r.stderr or "").rstrip().splitlines()[:12]:
            _git_log(f"  {line}", "cmd" if ok else "err")
    elif not ok:
        for line in ((r.stderr or r.stdout or "")).rstrip().splitlines()[:12]:
            _git_log(f"  {line}", "err")
    if args and args[0] == "push":
        _PUSH_LAST_ERR["stderr"] = r.stderr or ""
        _PUSH_LAST_ERR["code"] = r.returncode
    return r

def _gh(*args, **kw):
    return _run(["gh", *args], **kw)

# ==================== Junk helpers ====================

JUNK_DIR_NAMES = {
    ".gradle","gradle","Android","android",".buildozer","bin","dist","build",
    "__pycache__",".venv","venv","env","node_modules",".pytest_cache",
    ".mypy_cache",".tox",".idea",".vscode","target","kivy_env",".kivy_env",
    "eggs","htmlcov",
}
JUNK_FILE_SUFFIXES = {".pyc", ".pyo", ".spec"}
JUNK_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
JUNK_FILE_PREFIXES = (".env.",)
SAFE_PATHS = ["main.py","requirements.txt","buildozer.spec",
              ".gitignore","README.md",".github"]

def _safe_paths():
    paths = list(SAFE_PATHS)
    inc = Path(".bootstrap-include")
    if inc.exists():
        try:
            for line in inc.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(line)
        except OSError: pass
    return paths

def _in_home_dir():
    try: return Path.cwd().resolve() == Path.home().resolve()
    except OSError: return False

def _is_junk(path_str):
    p = Path(path_str)
    if p.name in JUNK_FILE_NAMES: return True
    if p.name == "buildozer.spec": return False
    if p.suffix in JUNK_FILE_SUFFIXES: return True
    if any(p.name.startswith(pref) for pref in JUNK_FILE_PREFIXES): return True
    for part in p.parts[:-1]:
        if part in JUNK_DIR_NAMES: return True
    return False

def _untrack_junk():
    head = _git("rev-parse", "--verify", "HEAD", quiet=True)
    if head.returncode != 0: return
    r = _git("ls-files", quiet=True)
    if r.returncode != 0: return
    junk = [f for f in r.stdout.splitlines() if _is_junk(f)]
    if not junk:
        _git_log("no tracked junk found", "ok"); return
    _git_log(f"untracking {len(junk)} junk file(s):", "warn")
    for f in junk[:10]: _git_log(f"  - {f}", "warn")
    if len(junk) > 10: _git_log(f"  ... and {len(junk) - 10} more", "warn")
    for i in range(0, len(junk), 200):
        _git("rm", "--cached", "--ignore-unmatch", "--",
             *junk[i:i+200], quiet=True)
    _git_log(f"untracked {len(junk)} file(s)", "ok")

def _repo_health_check():
    r = _git("ls-files", "-z", quiet=True)
    if r.returncode != 0: return True
    files = [f for f in r.stdout.split("\0") if f and Path(f).is_file()]
    if not files: return True
    sizes = []
    for f in files:
        try: sizes.append((Path(f).stat().st_size, f))
        except OSError: pass
    if not sizes: return True
    sizes.sort(reverse=True)
    total = sum(s for s, _ in sizes)
    _git_log(f"tracked: {len(sizes)} file(s), {total/1024/1024:.1f} MB total")
    blocked = [(s, f) for s, f in sizes if s > 100 * 1024 * 1024]
    for s, f in sizes[:10]:
        if s > 40 * 1024 * 1024:
            tag = "BLOCKED" if s > 100 * 1024 * 1024 else "LARGE"
            _git_log(f"  {tag}: {f} = {s/1024/1024:.1f} MB",
                     "err" if s > 100 * 1024 * 1024 else "warn")
    if blocked:
        _git_log(f"{len(blocked)} file(s) exceed 100 MB", "err"); return False
    return True

# ==================== History inspection / surgery ====================

def _remote_has_commits(url):
    if not url: return None
    r = _git("ls-remote", "--heads", url, quiet=True, timeout=30)
    if r.returncode != 0: return None
    return bool(r.stdout.strip())

def _history_has_blobs_over(max_mb):
    r = _git("rev-list", "--objects", "--all", quiet=True)
    if r.returncode != 0 or not r.stdout.strip(): return []
    oids = [line.split(" ", 1)[0] for line in r.stdout.splitlines()
            if line and len(line.split(" ", 1)[0]) == 40]
    if not oids: return []
    try:
        r2 = subprocess.run(
            ["git", "cat-file",
             "--batch-check=%(objecttype) %(objectsize) %(objectname)"],
            input="\n".join(oids) + "\n",
            capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        _git_log("cat-file timeout", "warn"); return []
    if r2.returncode != 0: return []
    threshold = max_mb * 1024 * 1024
    big = []
    for line in r2.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "blob":
            try:
                size = int(parts[1])
                if size > threshold: big.append((size, parts[2]))
            except ValueError: pass
    return big

def _nuke_git_and_reinit(branch="main"):
    git_dir = Path(".git")
    if git_dir.is_dir():
        try:
            size_mb = sum(f.stat().st_size for f in git_dir.rglob("*")
                          if f.is_file()) / 1024 / 1024
        except OSError: size_mb = 0.0
        print(f"[nuke] removing .git ({size_mb:.1f} MB)")
        shutil.rmtree(git_dir)
    _git("init", "-b", branch)
    _git_log("fresh git repo initialized", "ok")

def _purge_large_blobs(url, max_mb=40):
    try: import git_filter_repo  # noqa
    except ImportError:
        print("[filter-repo] installing git-filter-repo...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--quiet", "git-filter-repo"])
        except subprocess.CalledProcessError:
            pass
    r = subprocess.run(
        ["git", "filter-repo", "--strip-blobs-bigger-than", f"{max_mb}M", "--force"],
        capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(
            [sys.executable, "-m", "git_filter_repo",
             "--strip-blobs-bigger-than", f"{max_mb}M", "--force"],
            capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"filter-repo failed: {(r.stderr or r.stdout or '')[:400]}")
    if url:
        _git("remote", "add", "origin", url, quiet=True)
        _git_log(f"  re-added origin: {url}")
    _git("reflog", "expire", "--expire=now", "--all", quiet=True)
    _git("gc", "--prune=now", quiet=True)
    _git_log("  expired reflog and gc'd", "ok")

# ==================== Running-process detector ====================

def _detect_running_main():
    """Find any python process running this file. Returns list of ps lines."""
    try:
        me = os.getpid()
        r = subprocess.run(["ps", "-eo", "pid,args"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode != 0: return []
        lines = []
        for line in r.stdout.splitlines():
            if "main.py" not in line: continue
            parts = line.strip().split(None, 1)
            if len(parts) < 2: continue
            try: pid = int(parts[0])
            except ValueError: continue
            if pid == me: continue
            if "grep" in parts[1]: continue
            lines.append(line.strip())
        return lines
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

# ==================== Bootstrap plumbing ====================

def _is_termux(): return "com.termux" in os.environ.get("PREFIX", "")
def _has(cmd):    return shutil.which(cmd) is not None

def _install_missing_tools():
    for name, tux, deb, brew in (("git","git","git","git"),
                                 ("gh","gh","gh","gh")):
        if _has(name): continue
        print(f"[preflight] installing {name}...")
        if _is_termux():
            installer = "pkg" if _has("pkg") else "apt"
            _run([installer, "install", "-y", tux], capture_output=False)
        elif _has("apt-get"):
            cmd = (["sudo","apt-get","install","-y",deb] if _has("sudo")
                   else ["apt-get","install","-y",deb])
            _run(cmd, capture_output=False)
        elif _has("brew"):
            _run(["brew","install",brew], capture_output=False)
        else:
            print(f"[preflight] cannot install {name}"); return False
    return _has("git") and _has("gh")

def _gh_authenticated():
    if not _has("gh"): return False
    if _run(["gh","auth","status"]).returncode == 0: return True
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        print("[preflight] trying gh auth via env token...")
        _run(["gh","auth","login","--with-token"], input=token)
        if _run(["gh","auth","status"]).returncode == 0: return True
    hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    return hosts.exists() and hosts.stat().st_size > 0

def _get_gh_token():
    if _has("gh"):
        r = _gh("auth", "token")
        if r.returncode == 0 and r.stdout.strip(): return r.stdout.strip()
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

def _bump_tag(tag):
    m = re.match(r"^(v?)(\d+)\.(\d+)\.(\d+)$", tag)
    if not m: return tag + ".1"
    p, a, b, c = m.groups()
    return f"{p}{a}.{b}.{int(c) + 1}"

def _parse_slug(url):
    s = url.rstrip("/")
    if s.endswith(".git"): s = s[:-4]
    if "://" in s: s = s.split("://",1)[1].split("/",1)[1]
    elif ":" in s and "@" in s: s = s.rsplit(":",1)[-1]
    return s

def _current_remote():
    r = _git("remote","get-url","origin", quiet=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None

def _default_repo_name():
    name = Path.cwd().name
    if not name or name in ("~","home",".") or name.startswith("com."):
        return "herniation-triage"
    return re.sub(r"[^A-Za-z0-9._-]","-",name)

def _resolve_remote(args):
    if args.repo:
        print(f"[remote] using provided URL: {args.repo}"); return args.repo
    existing = _current_remote()
    if existing:
        print(f"[remote] reusing origin: {existing}"); return existing
    if not _has("gh"):
        print("[remote] no origin and gh not installed"); return None
    name = args.repo_name or _default_repo_name()
    r = _gh("repo","view",name,"--json","url","--jq",".url")
    if r.returncode == 0 and r.stdout.strip():
        url = f"https://github.com/{_parse_slug(r.stdout.strip())}.git"
        print(f"[remote] found existing repo: {url}"); return url
    print(f"[remote] creating repo '{name}'...")
    flag = "--private" if args.private else "--public"
    r = _gh("repo","create",name,"--source=.","--remote=origin",flag)
    if r.returncode != 0:
        print(f"[remote] gh repo create failed: {r.stderr.strip()}"); return None
    r2 = _gh("repo","view",name,"--json","url","--jq",".url")
    if r2.returncode != 0: return None
    return f"https://github.com/{_parse_slug(r2.stdout.strip())}.git"

def _set_origin(url):
    if _current_remote(): _git("remote","set-url","origin",url, quiet=True)
    else:                 _git("remote","add","origin",url, quiet=True)

def _ensure_identity():
    if _git("config","--get","user.email", quiet=True).returncode != 0:
        email = "bootstrap@local"
        if _has("gh"):
            rr = _gh("api","user","--jq",".email // empty")
            if rr.returncode == 0 and rr.stdout.strip(): email = rr.stdout.strip()
        _git("config","user.email",email, quiet=True)
    if _git("config","--get","user.name", quiet=True).returncode != 0:
        name = "Herniation Bootstrap"
        if _has("gh"):
            rr = _gh("api","user","--jq",".login // empty")
            if rr.returncode == 0 and rr.stdout.strip(): name = rr.stdout.strip()
        _git("config","user.name",name, quiet=True)

def _commit_all(msg):
    if _in_home_dir():
        print("[git] selective staging (in $HOME)")
        for p in _safe_paths():
            path = Path(p)
            if path.exists():
                _git("add","--",str(path), quiet=True)
                _git_log(f"  staged: {p}")
    else:
        _git("add","-A")
    if _git("diff","--cached","--quiet", quiet=True).returncode != 0:
        _git("commit","-m",msg)
    else:
        _git_log("nothing to commit (clean)")

def _ensure_branch(name):
    cur = _git("rev-parse","--abbrev-ref","HEAD", quiet=True)
    cur_name = cur.stdout.strip() if cur.returncode == 0 else ""
    if cur_name == name: return
    if _git("rev-parse","--verify","HEAD", quiet=True).returncode == 0:
        _git("branch","-M",name)
    else:
        _git("checkout","-b",name)

def _push_branch(branch, stream=False):
    url = _current_remote()
    if not url:
        print("[push] ERROR: no origin"); return False
    print(f"[push] remote: {url}"); print(f"[push] branch: {branch}")

    print("[push] strategy 1/4: direct push...")
    r = _git("push","-u","origin",branch, stream=stream, timeout=1800)
    if r.returncode == 0: print("[push] OK (direct)"); return True
    print("[push] strategy 1/4 failed")
    if "GH001" in _PUSH_LAST_ERR.get("stderr","") or \
       "exceeds GitHub" in _PUSH_LAST_ERR.get("stderr",""):
        print("[push] GH001 -- no fallback helps"); return False

    heads = _git("ls-remote","--heads","origin",branch, quiet=True)
    if heads.returncode == 0 and heads.stdout.strip():
        print("[push] strategy 2/4: fetch + push...")
        _git("fetch","origin",branch)
        r = _git("push","-u","origin",branch, stream=stream, timeout=1800)
        if r.returncode == 0: print("[push] OK (fetch+push)"); return True
        print("[push] strategy 2/4 failed")
    else:
        print("[push] strategy 2/4 skipped (branch absent)")

    token = _get_gh_token()
    if token and "github.com" in url:
        print("[push] strategy 3/4: token URL...")
        slug = _parse_slug(url)
        token_url = f"https://x-access-token:{token}@github.com/{slug}.git"
        r = _git("push",token_url,f"HEAD:refs/heads/{branch}",
                 stream=stream, timeout=1800)
        if r.returncode == 0:
            print("[push] OK (via token)")
            _git("remote","set-url","origin",url, quiet=True); return True
        print("[push] strategy 3/4 failed")
    else:
        print("[push] strategy 3/4 skipped (no token)")

    if "github.com" in url:
        print("[push] strategy 4/4: SSH...")
        slug = _parse_slug(url)
        ssh_url = f"git@github.com:{slug}.git"
        r = _git("push",ssh_url,f"HEAD:refs/heads/{branch}",
                 stream=stream, timeout=1800)
        if r.returncode == 0:
            print("[push] OK (via SSH)")
            _git("remote","set-url","origin",ssh_url, quiet=True); return True
        print("[push] strategy 4/4 failed")

    print("[push] ALL STRATEGIES FAILED"); return False

def _tag_and_push(tag):
    if not _git("tag","-l",tag, quiet=True).stdout.strip():
        _git("tag","-a",tag,"-m",f"Release {tag}")
    local = _git("rev-parse",tag, quiet=True)
    sha = local.stdout.strip() if local.returncode == 0 else ""
    if sha:
        r = _git("ls-remote","--tags","origin",f"refs/tags/{tag}", quiet=True)
        if r.returncode == 0 and sha in r.stdout:
            print(f"[git] tag {tag} already on remote"); return True
    print(f"[git] pushing tag {tag}...")
    if _git("push","origin",tag).returncode == 0:
        print(f"[git] tag {tag} pushed"); return True
    token = _get_gh_token(); url = _current_remote()
    if token and url and "github.com" in url:
        slug = _parse_slug(url)
        token_url = f"https://x-access-token:{token}@github.com/{slug}.git"
        if _git("push",token_url,tag).returncode == 0:
            print(f"[git] tag {tag} pushed (via token)"); return True
    print(f"[git] tag {tag} push failed"); return False

def _analyse():
    script = str(Path(__file__).resolve())
    print("[analyse] py_compile...")
    if _run([sys.executable,"-m","py_compile",script]).returncode != 0:
        return 1
    if not _NP_OK:
        print("[analyse] skipping selftest (numpy not installed locally)")
        print("[analyse] CI will install numpy/scipy and run the build")
        return 0
    print("[analyse] selftest...")
    return _run([sys.executable, script, "selftest"],
                capture_output=False).returncode

def run_selftest():
    if not _NP_OK:
        print("[selftest] SKIP: numpy not installed"); return 0
    print("[selftest] verifying core functions...")
    big = np.full((500, 2), 1e-3)
    if preprocess(big, 250) is not None:
        print("[selftest] FAIL"); return 1
    rng = np.random.default_rng(0)
    normal = 20e-6 * rng.standard_normal((500, 2))
    if preprocess(normal, 250) is None:
        print("[selftest] FAIL"); return 1
    hri = compute_hri({"DAR_log":0.0,"BSI":0.0,"persistence":1.0})
    if not (0.0 <= hri <= 1.0):
        print(f"[selftest] FAIL: {hri}"); return 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                     delete=False, newline="") as f:
        w = csv.writer(f); w.writerow(["t","ch1","ch2"])
        for i in range(500):
            w.writerow([f"{i/250:.6f}","0.000020","0.000020"])
        tmp = Path(f.name)
    try:
        arr, fs = _load_csv(tmp, 250)
        if arr.shape != (500,2): print("[selftest] FAIL: shape"); return 1
    finally: tmp.unlink(missing_ok=True)
    print("[selftest] OK"); return 0

def _print_success(url, tag):
    slug = _parse_slug(url)
    print()
    print("=" * 64)
    print(" BOOTSTRAP SUCCESS -- GitHub is now building everything")
    print("=" * 64)
    print(f" Repo:    https://github.com/{slug}")
    print(f" Tag:     {tag}")
    print(f" Actions: https://github.com/{slug}/actions")
    print()
    print(" Desktop:  cold ~3-5 min,  warm ~1-2 min")
    print(" Android:  cold ~35-45 min, warm ~6-10 min  (cache HIT)")
    print()
    print(f" Download artifacts: https://github.com/{slug}/actions")
    print("=" * 64)

def _warn_if_home(args):
    cwd = Path.cwd().resolve(); home = Path.home().resolve()
    if cwd != home: return True
    print()
    print("!" * 68)
    print("  WARNING: Bootstrap is running from your HOME directory")
    print("  Safe mode active -- only these paths will be staged:")
    for p in _safe_paths(): print(f"        {p}")
    print("!" * 68)
    if args.force:
        print("  --force given; continuing"); return True
    if not sys.stdin.isatty():
        print("  (no TTY; continuing)"); return True
    try: ans = input("  Continue anyway? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt): return False
    return ans in ("y", "yes")

# ==================== Bootstrap ====================

def _bootstrap_once(args):
    cmd_setup_github()
    if not args.skip_analyse and _analyse() != 0:
        return False, "analyse"

    root = Path.cwd()
    if not (root / ".git").is_dir():
        _git("init","-b",args.branch)
    _ensure_branch(args.branch)
    _ensure_identity()

    url = _resolve_remote(args)
    if not url: return False, "remote"

    if not args.keep_history:
        big_blobs = _history_has_blobs_over(40)
        if big_blobs:
            print(f"[bootstrap] history has {len(big_blobs)} blob(s) > 40 MB:")
            for size, oid in big_blobs[:5]:
                print(f"[bootstrap]   {size/1024/1024:8.1f} MB  {oid[:12]}")
            remote_state = _remote_has_commits(url)
            if remote_state is False:
                print("[bootstrap] remote empty -> nuking .git")
                _nuke_git_and_reinit(args.branch)
                _ensure_identity()
                url = _resolve_remote(args) or url
            elif remote_state is True:
                print("[bootstrap] remote populated -> filter-repo")
                try: _purge_large_blobs(url, max_mb=40)
                except Exception as e:
                    print(f"[bootstrap] filter-repo failed: {e}"); return False, "size"
                _ensure_identity()
                url = _resolve_remote(args) or url
            else:
                print("[bootstrap] remote state unknown; pushing anyway")

    _untrack_junk()
    _commit_all(f"bootstrap: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    head = _git("rev-parse","HEAD", quiet=True)
    if head.returncode != 0:
        print("[bootstrap] no commits"); return False, "analyse"
    head_sha = head.stdout.strip()

    if not _repo_health_check(): return False, "size"

    if args.tag:
        existing = _git("rev-parse",f"refs/tags/{args.tag}", quiet=True)
        if existing.returncode == 0:
            tag_sha = existing.stdout.strip()
            if tag_sha and tag_sha != head_sha:
                old = args.tag; args.tag = _bump_tag(args.tag)
                print(f"[bootstrap] tag {old} -> {tag_sha[:8]}, "
                      f"HEAD -> {head_sha[:8]}; bumped to {args.tag}")

    _set_origin(url)
    if not _push_branch(args.branch, stream=args.stream_progress):
        if "GH001" in _PUSH_LAST_ERR.get("stderr",""):
            print("[recovery] GH001; retrying nuke")
            if _remote_has_commits(url) is False:
                _nuke_git_and_reinit(args.branch)
                _ensure_identity()
                _commit_all(f"bootstrap-retry: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                _set_origin(url)
                if not _push_branch(args.branch, stream=args.stream_progress):
                    return False, "push"
            else:
                return False, "size"
        else:
            return False, "push"

    if not _tag_and_push(args.tag): return False, "tag"
    _print_success(url, args.tag)
    return True, args.tag

def cmd_bootstrap(args):
    # -------- UNMISTAKABLE BANNER --------
    print()
    print("=" * 68)
    print(f" BOOTSTRAP STARTED  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"(PID {os.getpid()})")
    print("=" * 68)
    print(" This command pushes code to GitHub.  It prints [git] / [push]")
    print(" lines and exits in ~15 seconds.  It NEVER prints HRI/BSI/DAR.")
    print(" If you see HRI/BSI/DAR after this banner, another `python`")
    print(" process is printing to your terminal -- kill it as shown below.")
    print("=" * 68)
    print()

    # -------- REFUSE IF ANY main.py PROCESS IS ALREADY RUNNING --------
    procs = _detect_running_main()
    if procs:
        print("[bootstrap] FATAL: another 'main.py' process is already running:")
        for p in procs: print(f"  {p}")
        print()
        print("[bootstrap] Its output will interleave with bootstrap and")
        print("[bootstrap] confuse you.  Kill it first:")
        print()
        for p in procs:
            pid = p.split()[0]
            print(f"    kill -9 {pid}")
        print()
        if args.kill_stuck:
            print("[bootstrap] --kill-stuck given; killing them now...")
            for p in procs:
                pid = p.split()[0]
                try:
                    subprocess.run(["kill","-9",pid], capture_output=True)
                except Exception: pass
            time.sleep(0.7)
            remaining = _detect_running_main()
            if remaining:
                print("[bootstrap] still running after kill -9:")
                for p in remaining: print(f"  {p}")
                print("[bootstrap] aborting"); return 1
            print("[bootstrap] all clear; continuing")
        else:
            print("[bootstrap] re-run with --kill-stuck to auto-kill.")
            return 1
        print()

    if not _warn_if_home(args):
        print("[bootstrap] aborted by user"); return 1
    if not _install_missing_tools():
        print("[bootstrap] FATAL: cannot install git/gh"); return 1
    if not _gh_authenticated():
        print("[bootstrap] FATAL: gh not authenticated.")
        print("  One-time:  gh auth login")
        return 2
    _run(["gh","auth","setup-git"])

    for attempt in range(1, args.max_attempts + 1):
        print(); print(f"----- attempt {attempt}/{args.max_attempts} -----")
        try:
            ok, reason = _bootstrap_once(args)
        except KeyboardInterrupt:
            print("[bootstrap] interrupted"); return 130
        except Exception as e:
            ok, reason = False, f"exception: {e}"
        if ok: return 0
        print(f"[bootstrap] attempt {attempt} failed: {reason}")
        if reason in ("analyse","size"):
            print(f"[bootstrap] fatal ({reason})"); return 3
        if attempt < args.max_attempts:
            time.sleep(2)
    print("[bootstrap] gave up"); return 4

def cmd_clean_history(args):
    print("=" * 64); print(" HISTORY CLEANUP"); print("=" * 64)
    if not Path(".git").is_dir(): print("[clean-history] no .git"); return 1
    url = _current_remote() or args.repo
    if not url: print("[clean-history] no origin; pass --repo <url>"); return 1
    big = _history_has_blobs_over(40)
    if not big: print("[clean-history] nothing to do"); return 0
    print(f"[clean-history] found {len(big)} blob(s) > 40 MB")
    remote_state = _remote_has_commits(url)
    if remote_state is False or args.nuke_history:
        print("[clean-history] nuking .git")
        _nuke_git_and_reinit(args.branch)
        _ensure_identity()
        _commit_all(f"clean: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        _set_origin(url)
        if not _repo_health_check(): return 1
        if not _push_branch(args.branch, stream=args.stream_progress): return 1
    elif remote_state is True:
        print("[clean-history] running filter-repo")
        try: _purge_large_blobs(url, max_mb=40)
        except Exception as e: print(f"[clean-history] failed: {e}"); return 1
        _set_origin(url)
        r = _git("push","--force-with-lease","origin",args.branch)
        if r.returncode != 0: print("[clean-history] force push failed"); return 1
    else:
        print("[clean-history] remote unknown; refusing"); return 1
    print("[clean-history] done.")
    return 0

def cmd_clone(args):
    url = args.repo
    if not url: print("[clone] ERROR: --repo <url> required"); return 1
    dest = Path(args.dest) if args.dest else Path(_default_repo_name())
    if dest.exists() and any(dest.iterdir()):
        print(f"[clone] {dest} exists and not empty; refusing"); return 1
    extra = []
    if args.depth:  extra.extend(["--depth", str(args.depth)])
    if args.branch: extra.extend(["--branch", args.branch])
    print(f"[clone] cloning {url} -> {dest}")
    r = _git("clone",*extra,url,str(dest), stream=args.stream_progress)
    if r.returncode == 0: print(f"[clone] OK -> {dest}"); return 0
    print("[clone] FAILED"); return 1

# ==================== Entry point ====================

def main():
    is_frozen = getattr(sys, "frozen", False)
    argv = sys.argv[1:]

    # Only enter GUI when there are ZERO args AND we are truly in an APK
    # or a frozen (double-clicked) executable.  Never intercept CLI args.
    if not argv:
        if _is_apk_runtime():
            return _launch_gui("mobile")
        if is_frozen:
            return _launch_gui("desktop")

    ap = argparse.ArgumentParser(
        prog="main.py",
        description="Herniation Triage. Run 'bootstrap' to push to GitHub.",
    )
    ap.add_argument("command",
                    choices=["bootstrap","make-sample","selftest",
                             "setup-github","clean-history","clone","gui"])
    ap.add_argument("--file", type=Path)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--repo-name", default=None)
    ap.add_argument("--tag", default="v1.0.0")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--skip-analyse", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--git-quiet", action="store_true")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--nuke-history", action="store_true")
    ap.add_argument("--keep-history", action="store_true")
    ap.add_argument("--stream-progress", action="store_true")
    ap.add_argument("--kill-stuck", action="store_true")
    args = ap.parse_args()

    if args.git_quiet: _LOG["git"] = False

    if args.command == "bootstrap":     sys.exit(cmd_bootstrap(args))
    if args.command == "make-sample":
        p = make_sample(path=args.file or Path("sample_eeg.csv"))
        sys.exit(0 if p else 1)
    if args.command == "selftest":      sys.exit(run_selftest())
    if args.command == "setup-github":  cmd_setup_github(); return
    if args.command == "clean-history": sys.exit(cmd_clean_history(args))
    if args.command == "clone":         sys.exit(cmd_clone(args))
    if args.command == "gui":           return _launch_gui("desktop")


if __name__ == "__main__":
    main()
