"""Fetch a pinned two-domain data release; never fetch final-test files here."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests


REPO = "AkaliKong/MiniOneRec"
COMMIT = "0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed"
ROOT = Path("data/raw/minionerec_e005")
DOMAINS = ["Office_Products", "Industrial_and_Scientific"]


def git_hash(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1", timeout=30)
    response.raise_for_status()
    tree = {entry["path"]: entry for entry in response.json()["tree"] if entry["type"] == "blob"}
    paths = []
    for domain in DOMAINS:
        paths += [f"data/Amazon/index/{domain}.{suffix}.json" for suffix in ["index", "item"]]
        paths += [f"data/Amazon/{split}/{domain}_5_2016-10-2018-11.csv" for split in ["train", "valid"]]
        paths.append(f"data/Amazon/info/{domain}_5_2016-10-2018-11.txt")

    def fetch(path):
        entry = tree[path]
        target = ROOT / path.removeprefix("data/Amazon/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and git_hash(target.read_bytes()) == entry["sha"]:
            content = target.read_bytes()
        else:
            content = None
            for attempt in range(3):
                try:
                    if attempt == 0:
                        result = requests.get(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}", timeout=35)
                        result.raise_for_status()
                        payload = result.content
                    else:
                        result = requests.get(f"https://api.github.com/repos/{REPO}/git/blobs/{entry['sha']}", timeout=60)
                        result.raise_for_status()
                        payload = base64.b64decode(result.json()["content"])
                    if git_hash(payload) != entry["sha"]:
                        raise ValueError(f"Git blob mismatch: {path}")
                    content = payload
                    break
                except Exception as error:
                    print("FETCH_RETRY", path, attempt, type(error).__name__, flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(1)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(content)
            temporary.replace(target)
        print("VERIFIED", path, len(content), flush=True)
        return {"upstream_path": path, "local_path": str(target), "git_blob_sha": entry["sha"],
                "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

    with ThreadPoolExecutor(max_workers=3) as executor:
        files = list(executor.map(fetch, paths))
    (ROOT / "provenance.json").write_text(json.dumps({"repo": REPO, "commit": COMMIT, "files": files,
        "domains": DOMAINS, "final_test_downloaded": False}, indent=2), encoding="utf-8")
    print("DOWNLOAD_COMPLETE", len(files), flush=True)


if __name__ == "__main__":
    main()
