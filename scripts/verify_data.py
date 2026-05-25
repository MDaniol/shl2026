#!/usr/bin/env python3
"""Recompute SHA-256 over every raw SHL file and compare to MANIFEST.sha256.

Two modes:
    --generate   walk a raw-data root and emit MANIFEST.sha256 (one-time use)
    --manifest   read MANIFEST.sha256 and verify every listed file (default)

Exit code is 0 on success, 1 on any mismatch. Run as the first DVC stage so
no downstream computation ever operates on silently-altered data.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def cmd_generate(root: Path, out: Path) -> int:
    files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    if not files:
        print(f"no *.txt files under {root}", file=sys.stderr)
        return 1
    with out.open("w") as f:
        for p in files:
            digest = sha256_of(p)
            rel = p.relative_to(root)
            f.write(f"{digest}  {rel}\n")
    print(f"wrote {out} ({len(files)} files)")
    return 0


def cmd_verify(manifest: Path) -> int:
    root = manifest.parent
    bad: list[str] = []
    n = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        path = root / rel
        if not path.exists():
            bad.append(f"missing: {rel}")
            continue
        actual = sha256_of(path)
        if actual != digest:
            bad.append(f"hash mismatch: {rel} (expected {digest}, got {actual})")
        n += 1
    if bad:
        for msg in bad:
            print(msg, file=sys.stderr)
        print(f"FAIL: {len(bad)}/{n} files diverged from {manifest}", file=sys.stderr)
        return 1
    print(f"OK: {n} files match {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    g = sub.add_parser("generate", help="emit MANIFEST.sha256 from a raw-data root")
    g.add_argument("--root", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)

    v = sub.add_parser("verify", help="verify files against MANIFEST.sha256")
    v.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.action == "generate":
        return cmd_generate(args.root, args.out)
    if args.action == "verify":
        return cmd_verify(args.manifest)
    parser.print_help()
    return 64


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
