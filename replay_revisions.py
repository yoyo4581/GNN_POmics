"""
Phase 2: Replay Drive revisions as git commits with their original timestamps.

Reads revision_manifest.json (from inventory_revisions.py), interleaves every
revision across all files into one chronological sequence, and for each
timestamp writes out the state of the tree at that moment and commits it with
GIT_AUTHOR_DATE / GIT_COMMITTER_DATE set to the real modification time.

Run this against a FRESH repo directory, not your existing one. Reconstructed
history is then merged or grafted onto current work.

Usage in Colab:
    %run replay_revisions.py
"""

import io
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta

from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ─── Config ───────────────────────────────────────────────────────
MANIFEST = "revision_manifest.json"
REPO_DIR = "/content/pognn_reconstructed"

# Revisions closer together than this are collapsed into one commit,
# so a burst of autosaves doesn't become 40 commits.
BUCKET = timedelta(hours=2)

AUTHOR_NAME = "Your Name"
AUTHOR_EMAIL = "you@users.noreply.github.com"
# ──────────────────────────────────────────────────────────────────

auth.authenticate_user()
drive = build("drive", "v3")


def download_revision(file_id, revision_id):
    """Fetch the bytes of one revision."""
    request = drive.revisions().get_media(fileId=file_id, revisionId=revision_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_events(manifest):
    """Flatten all revisions into one chronological list of events."""
    events = []
    for path, info in manifest.items():
        for rev in info["revisions"]:
            events.append({
                "time": parse_time(rev["modifiedTime"]),
                "path": path,
                "file_id": info["file_id"],
                "revision_id": rev["id"],
            })
    return sorted(events, key=lambda e: e["time"])


def bucket_events(events):
    """Group events into commits by time proximity."""
    buckets, current, anchor = [], [], None

    for ev in events:
        if anchor is None or ev["time"] - anchor <= BUCKET:
            anchor = anchor or ev["time"]
            current.append(ev)
        else:
            buckets.append(current)
            current, anchor = [ev], ev["time"]

    if current:
        buckets.append(current)
    return buckets


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=REPO_DIR, check=True, **kwargs)


def main():
    with open(MANIFEST) as fh:
        manifest = json.load(fh)

    events = build_events(manifest)
    buckets = bucket_events(events)
    print(f"{len(events)} revisions → {len(buckets)} commits\n")

    os.makedirs(REPO_DIR, exist_ok=True)
    run(["git", "init", "-q", "-b", "main"])

    # Latest content seen per path, so each commit reflects full tree state.
    state = {}
    cache = defaultdict(dict)

    for i, bucket in enumerate(buckets, 1):
        stamp = max(e["time"] for e in bucket)
        touched = []

        for ev in bucket:
            key = (ev["file_id"], ev["revision_id"])
            if key not in cache:
                try:
                    cache[key] = download_revision(*key)
                except Exception as exc:
                    print(f"  skip {ev['path']} @ {ev['revision_id']}: {exc}")
                    continue
            state[ev["path"]] = cache[key]
            touched.append(ev["path"])

        if not touched:
            continue

        for path, content in state.items():
            dest = os.path.join(REPO_DIR, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(content)

        names = sorted(set(os.path.basename(p) for p in touched))
        subject = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        message = f"Update {subject}"

        iso = stamp.isoformat()
        env = {
            **os.environ,
            "GIT_AUTHOR_DATE": iso,
            "GIT_COMMITTER_DATE": iso,
            "GIT_AUTHOR_NAME": AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": AUTHOR_EMAIL,
        }

        run(["git", "add", "-A"])
        result = subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=REPO_DIR, env=env, capture_output=True, text=True,
        )
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            print(f"  commit failed: {result.stdout}{result.stderr}")
            continue

        print(f"[{i}/{len(buckets)}] {stamp:%Y-%m-%d %H:%M}  {message}")

    print(f"\nDone. Inspect with:\n  cd {REPO_DIR} && git log --format=fuller")


if __name__ == "__main__":
    main()