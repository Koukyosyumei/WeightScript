"""Download only the trained pilot checkpoint from a pinned upstream revision."""
import argparse
import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

REVISION = "787398b0f4393a90ad02cd8e71be48d1e85d2b21"
MODEL = "bin_majority-@1l1h16d3lr01drop"
EXPECTED = {
    "config.json": "bb64f51d1ccc2d501fd2ed43621b6bd320137b6303b4d85d4c30381dad6be262",
    "model.safetensors": "a7652c98012b4897efb33fe7bdf45f5a07a0103427d3bdc8d8cd02b979959b76",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", nargs="?", default="artifacts/upstream-majority")
    args = p.parse_args()
    target = Path(args.output); target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors"):
        destination = target / name
        if destination.exists():
            raise ValueError(f"refusing to overwrite {destination}")
        suffix = urllib.parse.quote(f"share/saved_models/{MODEL}/{name}")
        url = f"https://raw.githubusercontent.com/lacoco-lab/decompiling_transformers/{REVISION}/{suffix}"
        content = urllib.request.urlopen(url, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != EXPECTED[name]:
            raise ValueError(f"upstream content hash mismatch for {name}")
        destination.write_bytes(content)
        print(name, "sha256:", hashlib.sha256(content).hexdigest())
    print("Source: https://github.com/lacoco-lab/decompiling_transformers/tree/" + REVISION)


if __name__ == "__main__":
    main()
