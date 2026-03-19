#!/usr/bin/env python3

import os
import sys
from urllib import error, request

OUT_DIR = "data/spacelog"
MISSION_TRANSCRIPTS = {
    "a11": ["CM", "TEC"],
    "a13": ["TEC"],
    "a8": ["TEC"],
    "g3": ["TEC"],
    "g4": ["TEC"],
    "g6": ["TEC"],
    "g8": ["TEC"],
    "ma6": ["TEC"],
    "ma7": ["TEC"],
    "ma8": ["TEC"],
    "mr3": ["ATG", "PAO"],
    "mr3": ["ATG"],
    "vostok1": ["en"]
}

def _download_bytes(url: str) -> bytes:
    req = request.Request(url=url)
    with request.urlopen(req) as resp:
        return resp.read()


def _write_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def main():
    for mission, transcripts in MISSION_TRANSCRIPTS.items():
        for transcript in transcripts:
            out_path = os.path.join(OUT_DIR, f"{transcript}_{mission}.txt")
            if os.path.exists(out_path):
                print(f"{out_path} exists")
                continue

            try:
                content = _download_bytes(url=f"https://raw.githubusercontent.com/Spacelog/Spacelog/main/missions/{mission}/transcripts/{transcript}")
                _write_bytes(out_path, content)
                print(f"{out_path} successful")
            except error.HTTPError as exc:
                msg = f"HTTP {exc.code}"
                print(f"{out_path} failed ({msg})", file=sys.stderr)
            except error.URLError as exc:
                msg = f"URL error: {exc}"
                print(f"{out_path} failed ({msg})", file=sys.stderr)

if __name__ == "__main__":
    main()