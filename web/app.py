# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Flask web app to upload videos, process them into clips + descriptions, and
create a compilation video for a given query.

Pipeline wrappers:
    Upload   -> POST /upload       (multipart files)
    Process  -> POST /process      generate_clip.py + extract_description.py
    Create   -> POST /create       find_clip.py with --plm_verify --compile

Run:
    cd perception_models
    python -m web.app                # http://localhost:5000
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "web" / "workspace"
VIDEOS_DIR = WORKSPACE / "videos"
CLIPS_DIR = VIDEOS_DIR / "clips"
CLIPS_JSON = CLIPS_DIR / "clips.json"
DESCRIPTIONS_JSON = WORKSPACE / "descriptions.json"
COMPILATION_MP4 = WORKSPACE / "compilation.mp4"

ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB per request

# ──────────────────────────────────────────────────────────────────────────────
# App state (single-user dev server)
# ──────────────────────────────────────────────────────────────────────────────

MAX_LOG_LINES = 500

STATE = {
    "uploaded": [],      # filenames present in videos/
    "processing": False, # a processing job is running
    "processed": False,  # descriptions.json is ready
    "creating": False,   # a create job is running
    "compilation": None, # token to bust the browser video cache
    "last_error": None,
    "log": [],           # recent stdout lines from subprocess + app events
}
STATE_LOCK = threading.Lock()


def _log(message: str) -> None:
    """Append a line to the shared log buffer (bounded)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    logger.info(message)
    with STATE_LOCK:
        STATE["log"].append(line)
        if len(STATE["log"]) > MAX_LOG_LINES:
            del STATE["log"][: len(STATE["log"]) - MAX_LOG_LINES]


HAS_FFPROBE = shutil.which("ffprobe") is not None
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _list_uploaded() -> list[str]:
    if not VIDEOS_DIR.exists():
        return []
    return sorted(
        p.name for p in VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _probe_creation_time(path: Path) -> Optional[datetime]:
    """Read the MOV/MP4 container creation_time tag via ffprobe."""
    if not HAS_FFPROBE:
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format_tags=creation_time",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            text=True, timeout=20,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    try:
        return datetime.fromisoformat(out.replace("Z", "+00:00"))
    except ValueError:
        return None


def _transcode_for_browser(path: Path) -> None:
    """Re-encode `path` in place to H.264 so <video> tags can play it.

    find_clip.py writes compilation.mp4 with the OpenCV `mp4v` fourcc
    (MPEG-4 Part 2), which Chrome/Safari/Firefox don't support. We
    convert it to H.264 + yuv420p + faststart using ffmpeg.
    """
    if not HAS_FFMPEG:
        _log("ffmpeg not found; browser may be unable to play compilation.mp4")
        return
    tmp = path.with_suffix(".browser.mp4")
    _log("Re-encoding compilation to H.264 for browser playback...")
    _run([
        "ffmpeg", "-y", "-i", str(path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(tmp),
    ])
    tmp.replace(path)


def _next_index_for_day(date_str: str, ext: str) -> int:
    """Return the next available 1-based index for `<date_str>-<n><ext>`."""
    pattern = re.compile(rf"^{re.escape(date_str)}-(\d+)$", re.I)
    used = 0
    if VIDEOS_DIR.exists():
        for p in VIDEOS_DIR.iterdir():
            if p.suffix.lower() != ext:
                continue
            m = pattern.match(p.stem)
            if m:
                used = max(used, int(m.group(1)))
    return used + 1


def _run(cmd: list[str]) -> None:
    """Run a subprocess and stream its stdout/stderr into the shared log."""
    _log("$ " + " ".join(cmd))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            _log(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}")


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    with STATE_LOCK:
        return jsonify({
            "uploaded": _list_uploaded(),
            "processing": STATE["processing"],
            "processed": STATE["processed"],
            "creating": STATE["creating"],
            "compilation": STATE["compilation"],
            "last_error": STATE["last_error"],
        })


@app.route("/log")
def log_route():
    with STATE_LOCK:
        return jsonify({"lines": list(STATE["log"])})


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No files received."}), 400

    # Client sends one mtime (ms since epoch) per file, same order.
    mtimes = request.form.getlist("mtimes")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    # Pre-sort by client mtime so same-day indices are chronological.
    order = sorted(
        range(len(files)),
        key=lambda i: int(mtimes[i]) if i < len(mtimes) and mtimes[i].isdigit() else 0,
    )

    saved = []
    rejected = []
    for i in order:
        f = files[i]
        orig = secure_filename(f.filename or "")
        if not orig:
            continue
        ext = Path(orig).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(orig)
            continue

        # Save to a temp name first so ffprobe can read the container.
        tmp = VIDEOS_DIR / f".upload-{uuid.uuid4().hex}{ext}"
        f.save(str(tmp))

        # 1) container creation_time  2) browser lastModified  3) now
        dt = _probe_creation_time(tmp)
        if dt is None and i < len(mtimes) and mtimes[i].isdigit():
            dt = datetime.fromtimestamp(int(mtimes[i]) / 1000.0, tz=timezone.utc)
        if dt is None:
            dt = datetime.now(tz=timezone.utc)

        date_str = dt.astimezone().strftime("%Y-%m-%d")
        idx = _next_index_for_day(date_str, ext)
        dest = VIDEOS_DIR / f"{date_str}-{idx}{ext}"
        tmp.rename(dest)
        saved.append(dest.name)
        _log(f"Uploaded {orig} -> {dest.name}")

    # Any new upload invalidates a prior processed state.
    with STATE_LOCK:
        STATE["processed"] = False
        STATE["compilation"] = None
        STATE["last_error"] = None

    return jsonify({
        "saved": saved,
        "rejected": rejected,
        "uploaded": _list_uploaded(),
    })


@app.route("/clear", methods=["POST"])
def clear():
    """Wipe the workspace so a fresh upload starts from scratch."""
    with STATE_LOCK:
        if STATE["processing"] or STATE["creating"]:
            return jsonify({"error": "A job is running."}), 409
        if WORKSPACE.exists():
            shutil.rmtree(WORKSPACE)
        STATE["processed"] = False
        STATE["compilation"] = None
        STATE["last_error"] = None
    return jsonify({"uploaded": []})


def _do_process() -> None:
    try:
        _log("=== Processing started ===")
        # 1) split into 5-second clips
        _run([
            sys.executable, "generate_clip.py",
            "--input_dir", str(VIDEOS_DIR),
            "--clip_duration", "5",
        ])
        if not CLIPS_JSON.exists():
            raise RuntimeError(f"clips.json not found at {CLIPS_JSON}")

        # 2) extract descriptions for every clip (--all)
        _run([
            sys.executable, "extract_description.py",
            "--clips_json", str(CLIPS_JSON),
            "--all",
            "--output", str(DESCRIPTIONS_JSON),
        ])
        if not DESCRIPTIONS_JSON.exists():
            raise RuntimeError(f"descriptions.json not found at {DESCRIPTIONS_JSON}")

        _log("=== Processing complete ===")
        with STATE_LOCK:
            STATE["processed"] = True
            STATE["last_error"] = None
    except Exception as exc:
        logger.exception("Processing failed")
        _log(f"ERROR: {exc}")
        with STATE_LOCK:
            STATE["processed"] = False
            STATE["last_error"] = str(exc)
    finally:
        with STATE_LOCK:
            STATE["processing"] = False


@app.route("/process", methods=["POST"])
def process():
    with STATE_LOCK:
        if STATE["processing"]:
            return jsonify({"error": "Already processing."}), 409
        if not _list_uploaded():
            return jsonify({"error": "No videos uploaded."}), 400
        STATE["processing"] = True
        STATE["processed"] = False
        STATE["last_error"] = None

    threading.Thread(target=_do_process, daemon=True).start()
    return jsonify({"started": True})


def _do_create(query: str) -> None:
    try:
        _log(f'=== Creating compilation for query: "{query}" ===')
        if COMPILATION_MP4.exists():
            COMPILATION_MP4.unlink()
        _run([
            sys.executable, "find_clip.py",
            "--descriptions", str(DESCRIPTIONS_JSON),
            "--query", query,
            "--plm_verify",
            "--compile", str(COMPILATION_MP4),
        ])
        if not COMPILATION_MP4.exists():
            raise RuntimeError("find_clip.py produced no compilation.mp4")
        _transcode_for_browser(COMPILATION_MP4)
        _log("=== Compilation ready ===")
        with STATE_LOCK:
            STATE["compilation"] = uuid.uuid4().hex
            STATE["last_error"] = None
    except Exception as exc:
        logger.exception("Create failed")
        _log(f"ERROR: {exc}")
        with STATE_LOCK:
            STATE["compilation"] = None
            STATE["last_error"] = str(exc)
    finally:
        with STATE_LOCK:
            STATE["creating"] = False


@app.route("/create", methods=["POST"])
def create():
    query = (request.json or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is empty."}), 400

    with STATE_LOCK:
        if STATE["creating"]:
            return jsonify({"error": "Already creating."}), 409
        if not STATE["processed"]:
            return jsonify({"error": "You must run Processing first."}), 409
        STATE["creating"] = True
        STATE["compilation"] = None
        STATE["last_error"] = None

    threading.Thread(target=_do_create, args=(query,), daemon=True).start()
    return jsonify({"started": True})


@app.route("/compilation.mp4")
def compilation():
    if not COMPILATION_MP4.exists():
        return "Not found", 404
    return send_from_directory(
        COMPILATION_MP4.parent, COMPILATION_MP4.name, mimetype="video/mp4"
    )


if __name__ == "__main__":
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
