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
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

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

STATE = {
    "uploaded": [],      # filenames present in videos/
    "processing": False, # a processing job is running
    "processed": False,  # descriptions.json is ready
    "creating": False,   # a create job is running
    "compilation": None, # token to bust the browser video cache
    "last_error": None,
}
STATE_LOCK = threading.Lock()


def _list_uploaded() -> list[str]:
    if not VIDEOS_DIR.exists():
        return []
    return sorted(
        p.name for p in VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _run(cmd: list[str]) -> None:
    """Run a subprocess with the repo root as CWD; raise on non-zero exit."""
    logger.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


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


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No files received."}), 400

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    rejected = []
    for f in files:
        name = secure_filename(f.filename or "")
        if not name:
            continue
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            rejected.append(name)
            continue
        dest = VIDEOS_DIR / name
        f.save(str(dest))
        saved.append(name)

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

        with STATE_LOCK:
            STATE["processed"] = True
            STATE["last_error"] = None
    except Exception as exc:
        logger.exception("Processing failed")
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
        with STATE_LOCK:
            STATE["compilation"] = uuid.uuid4().hex
            STATE["last_error"] = None
    except Exception as exc:
        logger.exception("Create failed")
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
