#!/usr/bin/env python3
"""
Video crawler that searches and downloads videos from YouTube (and other
yt-dlp-compatible platforms) based on configurable search keywords.

Output is saved as JSONL metadata files compatible with the VideoTextDataset
used in this repo (see core/data/data.py).

Usage
-----
# Search with inline keywords
python crawl_videos.py --queries "cat playing piano" "dog tricks" \
                       --output_dir data/crawled

# Search with a keyword file (one query per line)
python crawl_videos.py --query_file queries.txt --output_dir data/crawled

# Control download behaviour
python crawl_videos.py --queries "nature documentary" \
                       --output_dir data/crawled \
                       --max_results 50 \
                       --workers 4 \
                       --max_duration 600 \
                       --download_video      # omit to save metadata only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    """Metadata record written to the JSONL output file."""
    video_id: str
    title: str
    description: str
    url: str
    webpage_url: str
    channel: str
    channel_id: str
    duration: float          # seconds
    view_count: int
    like_count: int
    upload_date: str         # YYYYMMDD
    search_query: str
    local_path: Optional[str] = None   # set after download
    thumbnail_url: Optional[str] = None
    tags: list = field(default_factory=list)
    categories: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

def _ensure_yt_dlp() -> None:
    """Install yt-dlp if it is not available in the current environment."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        log.info("yt-dlp not found – installing via pip …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "yt-dlp", "-q"]
        )
        log.info("yt-dlp installed successfully.")


def _import_yt_dlp():
    import yt_dlp
    return yt_dlp


def _base_ydl_opts(
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> dict:
    """Return common yt-dlp options shared across search, enrich, and download.

    Authentication priority (only one cookie method is applied):
      1. cookies_from_browser – pull live cookies from an installed browser
      2. cookies_file         – load a pre-exported Netscape cookies.txt
      3. username + password  – site credential login (NOT supported by YouTube;
                                works on Vimeo, Dailymotion, and other sites)

    Parameters
    ----------
    cookies_from_browser:
        Browser name to pull cookies from, e.g. ``"chrome"``, ``"firefox"``,
        ``"safari"``, ``"edge"``.
    cookies_file:
        Path to a Netscape-format cookies.txt file.
    username:
        Account username / e-mail for sites that support credential login.
        YouTube does NOT support this – use a cookies option instead.
    password:
        Account password paired with *username*.
    """
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        # Mimic a real browser to reduce bot-detection signals
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
        log.debug("Using cookies from browser: %s", cookies_from_browser)
    elif cookies_file:
        opts["cookiefile"] = cookies_file
        log.debug("Using cookies file: %s", cookies_file)
    elif username:
        opts["username"] = username
        opts["password"] = password or ""
        log.debug("Using username/password auth for user: %s", username)
    return opts


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_videos(
    query: str,
    max_results: int = 20,
    max_duration: Optional[int] = None,
    min_duration: Optional[int] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> list[VideoMeta]:
    """Return a list of VideoMeta objects matching *query*.

    Parameters
    ----------
    query:                YouTube search string.
    max_results:          Maximum number of videos to return.
    max_duration:         Skip videos longer than this many seconds.
    min_duration:         Skip videos shorter than this many seconds.
    cookies_from_browser: Browser name to pull cookies from (e.g. "chrome").
    cookies_file:         Path to a Netscape-format cookies.txt file.
    username:             Site credential login (not supported by YouTube).
    password:             Password paired with username.
    """
    yt_dlp = _import_yt_dlp()

    search_url = f"ytsearch{max_results}:{query}"

    ydl_opts = _base_ydl_opts(cookies_from_browser, cookies_file, username, password)
    ydl_opts.update({
        "extract_flat": "in_playlist",  # fast: fetch only metadata
        "skip_download": True,
    })

    log.info("Searching: %r  (max %d results)", query, max_results)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_url, download=False)

    entries = info.get("entries", []) if info else []
    results: list[VideoMeta] = []

    for entry in entries:
        if entry is None:
            continue

        duration = entry.get("duration") or 0

        # Duration filters
        if max_duration is not None and duration > max_duration:
            log.debug("Skipping %s – duration %ds > max %ds",
                      entry.get("id"), duration, max_duration)
            continue
        if min_duration is not None and duration < min_duration:
            log.debug("Skipping %s – duration %ds < min %ds",
                      entry.get("id"), duration, min_duration)
            continue

        meta = VideoMeta(
            video_id=entry.get("id", ""),
            title=entry.get("title", ""),
            description=entry.get("description") or "",
            url=entry.get("url", ""),
            webpage_url=entry.get("webpage_url")
                        or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
            channel=entry.get("uploader") or entry.get("channel") or "",
            channel_id=entry.get("channel_id") or entry.get("uploader_id") or "",
            duration=duration,
            view_count=entry.get("view_count") or 0,
            like_count=entry.get("like_count") or 0,
            upload_date=entry.get("upload_date") or "",
            search_query=query,
            thumbnail_url=entry.get("thumbnail") or "",
            tags=entry.get("tags") or [],
            categories=entry.get("categories") or [],
        )
        results.append(meta)

    log.info("  Found %d videos for query %r", len(results), query)
    return results


def enrich_metadata(
    meta: VideoMeta,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> VideoMeta:
    """Fetch full metadata for a single video (fills description, tags, etc.)."""
    yt_dlp = _import_yt_dlp()
    ydl_opts = _base_ydl_opts(cookies_from_browser, cookies_file, username, password)
    ydl_opts["skip_download"] = True
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(meta.webpage_url, download=False)
        if info:
            meta.description = info.get("description") or meta.description
            meta.tags = info.get("tags") or meta.tags
            meta.categories = info.get("categories") or meta.categories
            meta.view_count = info.get("view_count") or meta.view_count
            meta.like_count = info.get("like_count") or meta.like_count
            meta.upload_date = info.get("upload_date") or meta.upload_date
            meta.channel = info.get("uploader") or meta.channel
            meta.channel_id = info.get("channel_id") or meta.channel_id
            meta.thumbnail_url = info.get("thumbnail") or meta.thumbnail_url
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not enrich metadata for %s: %s", meta.video_id, exc)
    return meta


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_video(
    meta: VideoMeta,
    output_dir: Path,
    format_spec: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    retries: int = 3,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Optional[Path]:
    """Download a single video and return its local path.

    Returns None if the download fails after all retries.
    """
    yt_dlp = _import_yt_dlp()

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(video_dir / "%(id)s.%(ext)s")

    ydl_opts = _base_ydl_opts(cookies_from_browser, cookies_file, username, password)
    ydl_opts.update({
        "format": format_spec,
        "outtmpl": outtmpl,
        "retries": retries,
        "merge_output_format": "mp4",
        "noplaylist": True,
    })

    for attempt in range(1, retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([meta.webpage_url])

            # Locate the downloaded file
            for ext in ("mp4", "mkv", "webm", "avi"):
                candidate = video_dir / f"{meta.video_id}.{ext}"
                if candidate.exists():
                    log.info("Downloaded: %s → %s", meta.video_id, candidate)
                    return candidate

            log.warning("Download succeeded but file not found for %s", meta.video_id)
            return None

        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            log.warning(
                "Download attempt %d/%d failed for %s: %s (retry in %ds)",
                attempt, retries, meta.video_id, exc, wait,
            )
            if attempt < retries:
                time.sleep(wait)

    log.error("All %d download attempts failed for %s", retries, meta.video_id)
    return None


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------

def _load_done_ids(jsonl_path: Path) -> set[str]:
    """Return the set of video IDs already present in the JSONL output file."""
    done: set[str] = set()
    if jsonl_path.exists():
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if vid := record.get("video_id"):
                            done.add(vid)
                    except json.JSONDecodeError:
                        pass
    return done


def _append_record(jsonl_path: Path, meta: VideoMeta) -> None:
    with jsonl_path.open("a") as fh:
        fh.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main crawl logic
# ---------------------------------------------------------------------------

def crawl(
    queries: list[str],
    output_dir: Path,
    max_results: int = 20,
    max_duration: Optional[int] = None,
    min_duration: Optional[int] = None,
    download_video: bool = False,
    format_spec: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    workers: int = 4,
    rate_limit_secs: float = 1.0,
    enrich: bool = False,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> None:
    """Crawl videos for all *queries* and write metadata (+ optional videos).

    Parameters
    ----------
    queries:              List of search strings.
    output_dir:           Root directory for all outputs.
    max_results:          Videos to fetch per query.
    max_duration:         Maximum video duration in seconds (None = no filter).
    min_duration:         Minimum video duration in seconds (None = no filter).
    download_video:       If True, download the video files.
    format_spec:          yt-dlp format selector used when downloading.
    workers:              Number of parallel download threads.
    rate_limit_secs:      Minimum delay (seconds) between searches.
    enrich:               Fetch full metadata for each video.
    cookies_from_browser: Browser name to pull cookies from (e.g. "chrome").
    cookies_file:         Path to a Netscape-format cookies.txt file.
    username:             Site credential login (not supported by YouTube).
    password:             Password paired with username.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "metadata.jsonl"

    done_ids = _load_done_ids(jsonl_path)
    if done_ids:
        log.info("Resuming – %d videos already in %s", len(done_ids), jsonl_path)

    # ---- Collect all candidates ----
    all_metas: list[VideoMeta] = []
    for i, query in enumerate(queries):
        results = search_videos(
            query,
            max_results=max_results,
            max_duration=max_duration,
            min_duration=min_duration,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
            username=username,
            password=password,
        )
        new = [m for m in results if m.video_id not in done_ids]
        log.info("  %d new videos (skipping %d already done)",
                 len(new), len(results) - len(new))
        all_metas.extend(new)
        # Polite delay between searches (skip after last query)
        if i < len(queries) - 1:
            time.sleep(rate_limit_secs)

    if not all_metas:
        log.info("Nothing new to process. Exiting.")
        return

    log.info("Total new candidates: %d", len(all_metas))

    # ---- Optionally enrich metadata ----
    if enrich:
        log.info("Enriching metadata for %d videos …", len(all_metas))
        def _enrich_one(m):
            time.sleep(0.5)   # small delay per request
            return enrich_metadata(
                m,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
                username=username,
                password=password,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_enrich_one, m): m for m in all_metas}
            enriched = []
            for fut in as_completed(futures):
                enriched.append(fut.result())
        all_metas = enriched

    # ---- Download or just save metadata ----
    if download_video:
        log.info("Downloading %d videos with %d workers …", len(all_metas), workers)

        def _do_download(meta: VideoMeta) -> VideoMeta:
            path = _download_video(
                meta, output_dir, format_spec,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
                username=username,
                password=password,
            )
            if path:
                meta.local_path = str(path)
            time.sleep(rate_limit_secs)
            return meta

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do_download, m): m for m in all_metas}
            for fut in as_completed(futures):
                finished_meta = fut.result()
                _append_record(jsonl_path, finished_meta)
                done_ids.add(finished_meta.video_id)
    else:
        # Metadata-only mode
        for meta in all_metas:
            _append_record(jsonl_path, meta)
            done_ids.add(meta.video_id)

    log.info("Done. Metadata written to %s", jsonl_path)
    log.info("Total records in file: %d", len(done_ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl videos from YouTube (and other platforms) by search keyword.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Queries
    q_group = p.add_mutually_exclusive_group(required=True)
    q_group.add_argument(
        "--queries", nargs="+", metavar="QUERY",
        help="One or more search queries (wrap multi-word queries in quotes).",
    )
    q_group.add_argument(
        "--query_file", type=Path, metavar="FILE",
        help="Plain-text file with one search query per line.",
    )

    # Output
    p.add_argument(
        "--output_dir", type=Path, default=Path("data/crawled"),
        help="Root directory for metadata JSONL and (optionally) video files.",
    )

    # Search options
    p.add_argument(
        "--max_results", type=int, default=20,
        help="Maximum number of videos to retrieve per query.",
    )
    p.add_argument(
        "--max_duration", type=int, default=None, metavar="SECONDS",
        help="Discard videos longer than this many seconds.",
    )
    p.add_argument(
        "--min_duration", type=int, default=None, metavar="SECONDS",
        help="Discard videos shorter than this many seconds.",
    )

    # Download options
    p.add_argument(
        "--download_video", action="store_true",
        help="Download actual video files (default: metadata only).",
    )
    p.add_argument(
        "--format", dest="format_spec",
        default="bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        help="yt-dlp format selector (only used with --download_video).",
    )

    # Performance / politeness
    p.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads.",
    )
    p.add_argument(
        "--rate_limit", type=float, default=1.0, dest="rate_limit_secs",
        metavar="SECONDS",
        help="Minimum delay between requests (seconds).",
    )
    p.add_argument(
        "--enrich", action="store_true",
        help="Fetch full per-video metadata (slower, more complete).",
    )

    # Authentication (bypass YouTube bot-detection)
    # --cookies-from-browser / --cookies are mutually exclusive cookie methods.
    # --username / --password work on many sites but NOT on YouTube.
    auth_group = p.add_mutually_exclusive_group()
    auth_group.add_argument(
        "--cookies-from-browser",
        dest="cookies_from_browser",
        metavar="BROWSER",
        help=(
            "Pull live cookies from an installed browser session to authenticate. "
            "BROWSER: chrome, chromium, firefox, safari, edge, opera, brave, vivaldi. "
            "You must be logged in to YouTube in that browser. "
            "Recommended for YouTube."
        ),
    )
    auth_group.add_argument(
        "--cookies",
        dest="cookies_file",
        metavar="FILE",
        help=(
            "Path to a Netscape-format cookies.txt file (e.g. exported with "
            "the 'Get cookies.txt LOCALLY' browser extension). "
            "Recommended for YouTube."
        ),
    )
    auth_group.add_argument(
        "--username",
        dest="username",
        metavar="USER",
        help=(
            "Account username/e-mail for sites that support credential login "
            "(Vimeo, Dailymotion, etc.). "
            "WARNING: YouTube dropped username/password support – use "
            "--cookies-from-browser or --cookies instead."
        ),
    )
    p.add_argument(
        "--password",
        dest="password",
        metavar="PASS",
        help="Password paired with --username (only used when --username is set).",
    )

    # Misc
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve queries
    if args.query_file:
        queries = [
            line.strip()
            for line in args.query_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not queries:
            log.error("No queries found in %s", args.query_file)
            sys.exit(1)
    else:
        queries = args.queries

    log.info("Queries (%d): %s", len(queries), queries)

    # Ensure yt-dlp is available
    _ensure_yt_dlp()

    # Validate --password requires --username
    if args.password and not args.username:
        log.error("--password requires --username.")
        sys.exit(1)

    # Warn appropriately based on auth method chosen
    if args.username:
        log.warning(
            "Username/password auth is NOT supported by YouTube. "
            "If you are crawling YouTube, use --cookies-from-browser or --cookies."
        )
    elif not args.cookies_from_browser and not args.cookies_file:
        log.warning(
            "No authentication provided. YouTube may block requests with "
            "'Sign in to confirm you're not a bot'. "
            "Use --cookies-from-browser BROWSER or --cookies FILE to fix this."
        )

    # Run the crawl
    crawl(
        queries=queries,
        output_dir=args.output_dir,
        max_results=args.max_results,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        download_video=args.download_video,
        format_spec=args.format_spec,
        workers=args.workers,
        rate_limit_secs=args.rate_limit_secs,
        enrich=args.enrich,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies_file,
        username=getattr(args, "username", None),
        password=getattr(args, "password", None),
    )


if __name__ == "__main__":
    main()
