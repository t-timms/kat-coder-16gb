"""Split a single-file safetensors checkpoint into sharded form.

WHY
    A 12.45 GiB single file is one interrupted download away from starting over.
    The convention every large checkpoint follows - including this model's own base
    - is ~5 GiB shards plus a model.safetensors.index.json weight map, so a failed
    transfer resumes at shard granularity and loaders can stream.

HOW
    Byte-range copy, the same approach build_release_candidate.py uses. The
    checkpoint mixes U8, F8_E4M3, BF16 and F32; a byte copy cannot silently coerce a
    dtype the way a load-and-resave through a tensor library can, and it never holds
    more than one chunk in memory.

PROOF
    Sharding a released artifact is only acceptable if it is provably lossless, so
    this does not merely re-read the headers. It hashes every tensor's bytes in the
    source and again in the shards, and requires all N digests to match. Any
    mismatch, missing tensor, or extra tensor fails the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import struct
import sys

GiB = 2 ** 30
CHUNK = 32 * 1024 * 1024
MAX_SHARD = int(os.environ.get("MAX_SHARD_BYTES", 5 * GiB))

SRC = pathlib.Path(
    os.environ.get("SHARD_SRC", str(pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16-renorm-stripped"))
)
DST = pathlib.Path(
    os.environ.get("SHARD_DST", str(pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16-sharded"))
)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None)
    return header, meta, 8 + n


def copy_range(fin, fout, start, end):
    fin.seek(start)
    remaining = end - start
    while remaining:
        buf = fin.read(min(remaining, CHUNK))
        if not buf:
            raise IOError(f"short read at {start}")
        fout.write(buf)
        remaining -= len(buf)


def digest_range(path, start, end):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start
        while remaining:
            buf = f.read(min(remaining, CHUNK))
            if not buf:
                raise IOError("short read while hashing")
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()


def main() -> None:
    src_st = SRC / "model.safetensors"
    if not src_st.is_file():
        raise SystemExit(f"source checkpoint missing: {src_st}")

    header, meta, data_start = read_header(src_st)
    ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    total_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in header.values())

    print(f"source : {src_st}")
    print(f"dest   : {DST}")
    print(f"tensors: {len(ordered):,}   data: {total_bytes/GiB:.4f} GiB")
    print(f"target shard size: {MAX_SHARD/GiB:.2f} GiB")

    # --- plan: greedy pack, never splitting a tensor -----------------------
    shards, current, current_size = [], [], 0
    for name, info in ordered:
        size = info["data_offsets"][1] - info["data_offsets"][0]
        if current and current_size + size > MAX_SHARD:
            shards.append(current)
            current, current_size = [], 0
        current.append((name, info))
        current_size += size
    if current:
        shards.append(current)
    n_shards = len(shards)
    print(f"plan   : {n_shards} shards\n")

    if DST.exists():
        import shutil

        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # --- sidecars ----------------------------------------------------------
    for p in sorted(SRC.iterdir()):
        if p.is_file() and p.name != "model.safetensors":
            (DST / p.name).write_bytes(p.read_bytes())

    # --- write shards ------------------------------------------------------
    weight_map, expected = {}, {}
    with open(src_st, "rb") as fin:
        for i, group in enumerate(shards, start=1):
            fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
            weight_map.update({name: fname for name, _ in group})

            new_header, plan, off = {}, [], 0
            for name, info in group:
                s, e = info["data_offsets"]
                size = e - s
                new_header[name] = {
                    "dtype": info["dtype"],
                    "shape": info["shape"],
                    "data_offsets": [off, off + size],
                }
                plan.append((name, s, e))
                off += size
            if meta is not None:
                new_header["__metadata__"] = meta

            hb = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
            hb += b" " * ((8 - (len(hb) % 8)) % 8)

            with open(DST / fname, "wb") as fout:
                fout.write(struct.pack("<Q", len(hb)))
                fout.write(hb)
                for name, s, e in plan:
                    expected[name] = digest_range(src_st, data_start + s, data_start + e)
                    copy_range(fin, fout, data_start + s, data_start + e)

            print(f"  {fname}  {len(group):>6,} tensors  {off/GiB:7.4f} GiB")

    # --- index -------------------------------------------------------------
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"\n  model.safetensors.index.json  ({len(weight_map):,} entries)")

    # --- proof: every tensor's bytes must hash identically -----------------
    print("\n=== verifying every tensor byte-for-byte ===")
    seen, mismatch = set(), []
    for i in range(1, n_shards + 1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        h2, _, ds2 = read_header(DST / fname)
        for name, info in h2.items():
            s, e = info["data_offsets"]
            got = digest_range(DST / fname, ds2 + s, ds2 + e)
            if name in seen:
                mismatch.append(f"{name}: appears in more than one shard")
            seen.add(name)
            if got != expected.get(name):
                mismatch.append(f"{name}: digest differs")
            if info["dtype"] != header[name]["dtype"]:
                mismatch.append(f"{name}: dtype changed")
            if info["shape"] != header[name]["shape"]:
                mismatch.append(f"{name}: shape changed")

    missing = set(header) - seen
    extra = seen - set(header)
    if missing:
        mismatch.append(f"{len(missing)} tensor(s) missing, e.g. {sorted(missing)[:3]}")
    if extra:
        mismatch.append(f"{len(extra)} unexpected tensor(s), e.g. {sorted(extra)[:3]}")

    shard_total = sum((DST / f"model-{i:05d}-of-{n_shards:05d}.safetensors").stat().st_size
                      for i in range(1, n_shards + 1))
    print(f"  tensors verified : {len(seen):,} / {len(header):,}")
    print(f"  digests matched  : {len(seen) - len([m for m in mismatch if 'digest' in m]):,}")
    print(f"  source file      : {src_st.stat().st_size:,} bytes")
    print(f"  shards total     : {shard_total:,} bytes")

    if mismatch:
        print("\nPROBLEMS:")
        for m in mismatch[:20]:
            print("  -", m)
        print("SHARD_FAIL")
        sys.exit(1)

    print("\n  every tensor byte-identical to the source")
    print("SHARD_OK")


if __name__ == "__main__":
    main()
