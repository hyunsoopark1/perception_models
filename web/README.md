# Video Compilation Web UI

Minimal Flask front-end for the clip pipeline.

## Install

```
pip install flask
```

## Run

From the repo root:

```
python -m web.app
```

Open http://localhost:5000.

## Flow

1. **Upload** one or more local video files (`.mov`, `.mp4`, `.avi`, `.mkv`,
   `.webm`, `.flv`, `.m4v`). They are saved under `web/workspace/videos/`.
2. **Processing** splits each video into 5-second clips
   (`generate_clip.py --input_dir ... --clip_duration 5`) and then extracts
   descriptions (`extract_description.py --clips_json ... --all`). The
   Creating button is locked until this step finishes.
3. **Creating** runs `find_clip.py --plm_verify --compile compilation.mp4`
   with the user-entered query and streams the result back to the page.

`Clear workspace` wipes uploaded videos, clips, and descriptions so you can
start over.
