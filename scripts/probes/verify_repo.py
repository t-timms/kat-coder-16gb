#!/usr/bin/env python3
"""Repository invariants that are cheap to check and expensive to get wrong.

Every check here exists because the corresponding defect was actually found in
this repo, not because it is conventional:

  * a broken relative link, after a docs file was renamed
  * two ### Changed sections inside one CHANGELOG release
  * cross-references to git branches that had moved to another repository
  * a claim contradicted by another file in the same repo

Runs without a GPU, a model, or any of the four pipeline environments, so CI can
run it on every push. Exits non-zero on any failure.
"""

from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".git", "__pycache__", ".venv"}


def walk(pattern: str):
    for p in ROOT.rglob(pattern):
        if not SKIP.intersection(p.parts):
            yield p


def check_links() -> list[str]:
    """Relative markdown links must resolve."""
    bad, n = [], 0
    link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for md in walk("*.md"):
        for target in link.findall(md.read_text(encoding="utf-8", errors="replace")):
            t = target.split("#")[0].strip()
            if not t or t.startswith(("http://", "https://", "mailto:")):
                continue
            n += 1
            if not (md.parent / t).resolve().exists():
                bad.append(f"broken link in {md.relative_to(ROOT)}: {target}")
    print(f"  relative markdown links resolved : {n}")
    return bad


def check_python() -> list[str]:
    bad, n = [], 0
    for py in walk("*.py"):
        n += 1
        try:
            py_compile.compile(str(py), doraise=True, cfile=str(Path("/tmp/_pyc")))
        except py_compile.PyCompileError as exc:
            bad.append(f"syntax error in {py.relative_to(ROOT)}: {exc}")
    print(f"  python files compiled            : {n}")
    return bad


def check_changelog() -> list[str]:
    """No release section may repeat a heading (Added/Changed/Fixed)."""
    bad = []
    cl = ROOT / "CHANGELOG.md"
    if not cl.exists():
        return ["CHANGELOG.md missing"]
    for sec in re.split(r"\n(?=## \[)", cl.read_text(encoding="utf-8")):
        if not sec.lstrip().startswith("## ["):
            continue
        title = sec.split("\n", 1)[0].strip()
        heads = re.findall(r"^### (.+)$", sec, re.M)
        dupes = sorted({h for h in heads if heads.count(h) > 1})
        if dupes:
            bad.append(f"duplicate heading(s) {dupes} in CHANGELOG {title}")
    print("  changelog headings unique        : ok")
    return bad


def check_reap_pin() -> list[str]:
    """The renormalization fix lives in a fork; upstream produces a different model."""
    env = ROOT / "docs/environment.md"
    if not env.exists():
        return ["docs/environment.md missing"]
    text = env.read_text(encoding="utf-8")
    bad = []
    if "CerebrasResearch/reap" in text and "t-timms/reap-cuda" not in text:
        bad.append("docs/environment.md points at upstream reap, which lacks the renorm fix")
    if not re.search(r"\b[0-9a-f]{40}\b", text):
        bad.append("docs/environment.md pins no reap commit SHA")
    print("  reap fork pinned by commit       : ok")
    return bad


def main() -> int:
    print("verifying repository invariants")
    problems: list[str] = []
    for fn in (check_links, check_python, check_changelog, check_reap_pin):
        problems += fn()
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nVERIFY_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
