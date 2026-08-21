"""Build the release candidate from the quantized checkpoint.

WHY THIS SCRIPT EXISTS
    The base model declares a vision tower and ships ZERO trained weights for it.
    Transformers materialises those missing parameters as random values at load
    time, reap then saves them, and the quantizer's ignore list (`re:visual.*`)
    protects them from compression - so 333 randomly-initialised BF16 tensors,
    0.83 GiB, survive all the way into the release artifact.

    Stripping `vision_config` out of config.json (the old runbook step) removes
    the DECLARATION but not the WEIGHTS. This script removes both, which is what
    "vision-stripped" was always documented to mean:

        13.28 GiB (as quantized)  -  0.83 GiB (phantom tower)  =  12.45 GiB

    See docs/vision_weight_regression_2026-08-20.md for the full investigation.

WHAT IT DOES
    1. copies every sidecar file from the quantized checkpoint
    2. drops the vision keys from config.json
    3. drops the now-dead `model.visual.*` entries from quantization_config.ignore
    4. rewrites model.safetensors without any `visual` tensor, by byte-range copy
       (no torch, no dtype round-trip, no full-file RAM load)
    5. verifies the result by artifact, never by exit code
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import struct
import sys

GiB = 2 ** 30
CHUNK = 32 * 1024 * 1024

SRC = pathlib.Path(os.environ.get("RC_SRC", str(pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16")))
DST = pathlib.Path(
    os.environ.get("RC_DST", str(pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16-renorm-stripped"))
)

# Anything whose tensor name matches this is the phantom tower.
VISUAL = "visual"


def read_header(path: pathlib.Path):
    """Return (header_dict, metadata, data_start_offset)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    return header, meta, 8 + n


def strip_visual_tensors(src_st: pathlib.Path, dst_st: pathlib.Path):
    header, meta, data_start = read_header(src_st)

    keep = {k: v for k, v in header.items() if VISUAL not in k}
    drop = {k: v for k, v in header.items() if VISUAL in k}
    dropped_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in drop.values())

    # Preserve on-disk order so the copy stays sequential.
    ordered = sorted(keep.items(), key=lambda kv: kv[1]["data_offsets"][0])

    new_header, plan, off = {}, [], 0
    for name, info in ordered:
        s, e = info["data_offsets"]
        size = e - s
        new_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [off, off + size]}
        plan.append((s, e))
        off += size
    if meta is not None:
        new_header["__metadata__"] = meta

    hb = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - (len(hb) % 8)) % 8)  # safetensors pads the header to 8 bytes

    with open(src_st, "rb") as fin, open(dst_st, "wb") as fout:
        fout.write(struct.pack("<Q", len(hb)))
        fout.write(hb)
        for s, e in plan:
            fin.seek(data_start + s)
            remaining = e - s
            while remaining:
                buf = fin.read(min(remaining, CHUNK))
                if not buf:
                    raise IOError(f"short read at offset {s}")
                fout.write(buf)
                remaining -= len(buf)

    return len(keep), len(drop), dropped_bytes


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"source checkpoint missing: {SRC}")
    print(f"source : {SRC}", flush=True)
    print(f"dest   : {DST}", flush=True)

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # 1. sidecars
    for p in sorted(SRC.iterdir()):
        if p.is_file() and p.name != "model.safetensors":
            shutil.copy2(p, DST / p.name)
            print(f"  copied {p.name}")

    # 2 + 3. config surgery
    cfg = json.loads((SRC / "config.json").read_text())
    removed_keys = [k for k in list(cfg) if "vision" in k.lower() or "visual" in k.lower()]
    for k in removed_keys:
        cfg.pop(k)

    q = cfg.get("quantization_config", {})
    before = len(q.get("ignore", []))
    if "ignore" in q:
        q["ignore"] = [e for e in q["ignore"] if VISUAL not in e]
    removed_ignores = before - len(q.get("ignore", []))

    (DST / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  config: dropped keys {removed_keys or '(none)'}")
    print(f"  config: dropped {removed_ignores} dead visual entries from quantization_config.ignore")

    # 4. the weights
    print("  rewriting model.safetensors without the phantom tower...", flush=True)
    kept, dropped, dropped_bytes = strip_visual_tensors(SRC / "model.safetensors", DST / "model.safetensors")

    # 5. verify by artifact
    src_sz = (SRC / "model.safetensors").stat().st_size
    dst_sz = (DST / "model.safetensors").stat().st_size
    print("\n=== verification ===")
    print(f"  tensors kept    : {kept:,}")
    print(f"  tensors dropped : {dropped:,}  ({dropped_bytes/GiB:.4f} GiB)")
    print(f"  before          : {src_sz:,} bytes = {src_sz/GiB:.4f} GiB")
    print(f"  after           : {dst_sz:,} bytes = {dst_sz/GiB:.4f} GiB")
    print(f"  reduction       : {(src_sz-dst_sz)/GiB:.4f} GiB")

    # re-read the written header: nothing visual may remain, offsets must be sane
    h2, _, _ = read_header(DST / "model.safetensors")
    leftover = [k for k in h2 if VISUAL in k]
    covered = max((v["data_offsets"][1] for v in h2.values()), default=0)
    print(f"  visual tensors remaining: {len(leftover)}")
    print(f"  header re-parsed OK     : {len(h2):,} tensors, data buffer {covered:,} bytes")

    ok = True
    if leftover:
        print("  !! VISUAL TENSORS STILL PRESENT")
        ok = False
    if dst_sz / GiB > 13.0:
        print(f"  !! larger than expected ({dst_sz/GiB:.2f} GiB)")
        ok = False
    if abs(dst_sz / GiB - 12.45) > 0.05:
        print(f"  ?? {dst_sz/GiB:.4f} GiB does not match the documented 12.45 GiB")
    else:
        print(f"  OK: {dst_sz/GiB:.4f} GiB matches the documented 12.45 GiB")

    print("RELEASE_CANDIDATE_OK" if ok else "RELEASE_CANDIDATE_FAIL", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
