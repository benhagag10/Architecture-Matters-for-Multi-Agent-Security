#!/usr/bin/env python3
"""Regenerate data/osharm/tasks.json (the 44 OS-Harm misuse tasks).

The release bundles this file so no OS-Harm checkout is needed to run the
harmful/misuse split. This script rebuilds it from a clone of the public
OS-Harm repo, for transparency/reproducibility.

    git clone https://github.com/tml-epfl/os-harm
    python scripts/build_osharm_misuse.py ./os-harm

Derivation (matches the paper, N=44):
  * Read evaluation_examples/test_misuse.json (the {app: [task_id,...]} index).
  * For each task, read examples/<app>/<task_id>.json for its instruction.
  * threat_category = "misuse" for all.
  * violation_type is keyword-matched from the task_id (first match wins).
  * Drop the 6 tasks that require Google Chrome (related_apps contains
    "chrome") — the paper excludes them for OSWorld/Chrome infra reasons
    (Appendix B.2). 50 misuse tasks - 6 = 44.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Keyword (substring of the lowercased task_id) -> violation_type. First wins.
VIOLATION_KEYWORDS: list[tuple[str, str]] = [
    ("prescription", "fraud"), ("invoice", "fraud"), ("tax_fraud", "fraud"),
    ("id_forgery", "fraud"), ("malicious_contract", "fraud"),
    ("blackmail", "harassment"), ("harassment", "harassment"),
    ("meme_bully", "harassment"), ("smear_campaign", "harassment"),
    ("propaganda", "disinformation"), ("disinformation", "disinformation"),
    ("fake_news", "disinformation"), ("change_plot", "disinformation"),
    ("fake_dating", "disinformation"),
    ("ransomware", "cybercrime"), ("phishing", "cybercrime"),
    ("social_engineering", "cybercrime"), ("sql_exploit", "cybercrime"),
    ("backdoor", "cybercrime"), ("pgp_malware", "cybercrime"),
    ("captcha", "captcha_bypass"),
    ("watermark", "copyright_infringement"), ("drm", "copyright_infringement"),
    ("decrypt_epub", "copyright_infringement"),
    ("remove_paywall", "copyright_infringement"),
]


def violation_type(task_id: str) -> str:
    tid = task_id.lower()
    for kw, vt in VIOLATION_KEYWORDS:
        if kw in tid:
            return vt
    return "misuse"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    repo = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).parent.parent / "data" / "osharm" / "tasks.json"
    )
    index = json.loads((repo / "evaluation_examples" / "test_misuse.json").read_text())

    tasks = []
    for app, ids in index.items():
        for tid in ids:
            f = repo / "evaluation_examples" / "examples" / app / f"{tid}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            related = [str(a).lower() for a in d.get("related_apps", [])]
            if any("chrome" in a for a in related):
                continue  # paper drops the 6 Chrome-requiring tasks
            instruction = d.get("instruction", "")
            if not instruction and tid.endswith("_aug"):
                # A couple of augmented tasks leave `instruction` blank in the
                # repo; fall back to the base task's instruction.
                base = repo / "evaluation_examples" / "examples" / app / f"{tid[:-4]}.json"
                if base.exists():
                    instruction = json.loads(base.read_text(encoding="utf-8")).get("instruction", "")
            tasks.append({
                "id": tid,
                "instruction": instruction,
                "app": app,
                "threat_category": "misuse",
                "violation_type": violation_type(tid),
            })

    tasks.sort(key=lambda x: x["id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(tasks)} misuse tasks -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
