"""
Phase 1: Inventory what revision history actually survives in Drive.

Run this in a Colab notebook cell BEFORE attempting any reconstruction.
Drive prunes revisions of non-Google-native files (.ipynb, .py) after
~30 days or ~100 versions, so the answer may be "not much".

Usage in Colab:
    !pip install -q google-api-python-client
    %run inventory_revisions.py
"""

import json
from collections import defaultdict

from google.colab import auth
from googleapiclient.discovery import build

# ─── Config ───────────────────────────────────────────────────────
FOLDER_NAME = "POGNN_Module"     # top-level project folder in MyDrive
EXTENSIONS = (".py", ".ipynb")   # file types to inventory
# ──────────────────────────────────────────────────────────────────

auth.authenticate_user()
drive = build("drive", "v3")


def find_folder(name):
    """Locate the project folder by name. Returns its Drive fileId."""
    resp = drive.files().list(
        q=(
            f"name = '{name}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false"
        ),
        fields="files(id, name)",
        spaces="drive",
    ).execute()

    folders = resp.get("files", [])
    if not folders:
        raise SystemExit(f"No folder named {name!r} found in Drive.")
    if len(folders) > 1:
        print(f"Warning: {len(folders)} folders named {name!r}; using the first.")
    return folders[0]["id"]


def walk(folder_id, prefix=""):
    """Recursively yield (path, fileId) for files under folder_id."""
    page_token = None
    while True:
        resp = drive.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()

        for f in resp.get("files", []):
            path = f"{prefix}{f['name']}"
            if f["mimeType"] == "application/vnd.google-apps.folder":
                yield from walk(f["id"], prefix=f"{path}/")
            elif path.endswith(EXTENSIONS):
                yield path, f["id"]

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def list_revisions(file_id):
    """Return all revisions for a file, oldest first."""
    revisions, page_token = [], None
    while True:
        resp = drive.revisions().list(
            fileId=file_id,
            fields=(
                "nextPageToken, "
                "revisions(id, modifiedTime, keepForever, size, "
                "lastModifyingUser(displayName))"
            ),
            pageSize=1000,
            pageToken=page_token,
        ).execute()

        revisions.extend(resp.get("revisions", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return sorted(revisions, key=lambda r: r["modifiedTime"])


def main():
    folder_id = find_folder(FOLDER_NAME)
    print(f"Found {FOLDER_NAME} (id={folder_id})\n")

    manifest = {}
    timeline = defaultdict(list)
    total_revs = 0

    for path, file_id in walk(folder_id):
        try:
            revs = list_revisions(file_id)
        except Exception as exc:
            print(f"  {path}: could not list revisions ({exc})")
            continue

        if not revs:
            continue

        manifest[path] = {"file_id": file_id, "revisions": revs}
        total_revs += len(revs)

        first, last = revs[0]["modifiedTime"], revs[-1]["modifiedTime"]
        kept = sum(1 for r in revs if r.get("keepForever"))
        print(
            f"  {path:<50} {len(revs):>4} revs  "
            f"{first[:10]} → {last[:10]}"
            + (f"  ({kept} keepForever)" if kept else "")
        )

        for r in revs:
            timeline[r["modifiedTime"][:10]].append(path)

    if not manifest:
        raise SystemExit("\nNo revision history found. Nothing to reconstruct.")

    days = sorted(timeline)
    print(f"\n{'─' * 60}")
    print(f"Files with history : {len(manifest)}")
    print(f"Total revisions    : {total_revs}")
    print(f"Date range         : {days[0]} → {days[-1]}")
    print(f"Distinct days      : {len(days)}")

    with open("revision_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("\nWrote revision_manifest.json")


if __name__ == "__main__":
    main()