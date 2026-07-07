#!/usr/bin/env python3
"""
Pre-flight check for the ICML release. Run this after ``pip install`` and
before any sweep — it surfaces every prerequisite gotcha at once so you don't
discover them mid-sweep.

    python check_setup.py
    python check_setup.py --scenario browserart   # only check one
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def ok(msg: str)   -> None: print(f"{GREEN}OK  {RESET}  {msg}")
def warn(msg: str) -> None: print(f"{YELLOW}WARN{RESET}  {msg}")
def fail(msg: str) -> None: print(f"{RED}FAIL{RESET}  {msg}")
def hint(msg: str) -> None: print(f"      {DIM}→ {msg}{RESET}")


def check_python() -> bool:
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor} — need ≥ 3.10")
    hint("install Python 3.10+ and recreate your venv")
    return False


def check_package(name: str, min_version: str | None = None) -> bool:
    try:
        mod = importlib.import_module(name.replace("-", "_"))
    except ImportError:
        fail(f"{name} is not installed")
        hint(f"pip install '{name}{f'>={min_version}' if min_version else ''}'")
        return False
    have = getattr(mod, "__version__", None)
    if min_version and have:
        from packaging.version import Version
        if Version(have) < Version(min_version):
            fail(f"{name} {have} — need ≥ {min_version}")
            hint(f"pip install --upgrade '{name}>={min_version}'")
            return False
    ok(f"{name} {have or '(version unknown)'}")
    return True


def check_env_var(name: str, *, required: bool, used_for: str) -> bool:
    if os.environ.get(name):
        ok(f"{name} set ({used_for})")
        return True
    if required:
        fail(f"{name} not set ({used_for})")
        hint(f"export {name}=...")
        return False
    warn(f"{name} not set ({used_for})")
    hint(f"only needed if you use {used_for}")
    return True


def check_data_files() -> bool:
    here = Path(__file__).parent
    ok_all = True
    bart_h = here / "data" / "browserart" / "hbb.json"
    bart_b = here / "data" / "browserart" / "hbb_benign.json"
    rcg    = here / "data" / "redcode_gen"
    for p, expected_count, label in [
        (bart_h, None, "BrowserART harmful tasks"),
        (bart_b, None, "BrowserART benign tasks"),
    ]:
        if p.exists():
            ok(f"{label}: {p.relative_to(here)}")
        else:
            fail(f"{label} missing: {p}")
            ok_all = False
    if rcg.is_dir() and any(rcg.glob("*/*.py")):
        n = sum(1 for _ in rcg.glob("*/*.py"))
        ok(f"RedCode-Gen tasks: {n} files under data/redcode_gen/")
        if n != 160:
            warn(f"  expected 160 (8 categories × 20), have {n}")
    else:
        fail(f"RedCode-Gen task files missing under {rcg}")
        ok_all = False
    return ok_all


def check_conditions_loadable() -> bool:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from conditions import list_conditions
        cs = list_conditions()
        if len(cs) != 13:
            fail(f"conditions.py registers {len(cs)} conditions (expected 13)")
            return False
        ok(f"conditions.py loads — 13 conditions registered")
        return True
    except Exception as e:
        fail(f"conditions.py import error: {type(e).__name__}: {e}")
        return False


def check_scenario(name: str, condition: str = "single_agent") -> bool:
    """Try to construct a Task for the given scenario without invoking models."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        if name == "browserart":
            import browserart as M
            t = M.browserart(condition=condition, kind="harmful", limit=1)
            ok(f"browserart.browserart(...) builds — n_samples={len(t.dataset)}")
        elif name == "osharm":
            try:
                import osharm as M
                t = M.osharm(condition=condition, kind="harmful", limit=1)
                ok(f"osharm.osharm(...) builds — n_samples={len(t.dataset)}")
            except FileNotFoundError as e:
                warn(f"osharm requires checkout: {e}")
                hint("clone https://github.com/tml-epfl/os-harm and set "
                     "OSHARM_PATH=... (must provide data/misuse/tasks.json)")
        elif name == "redcode_gen":
            import redcode_gen as M
            t = M.redcode_gen(condition=condition, limit=1)
            ok(f"redcode_gen.redcode_gen(...) builds — n_samples={len(t.dataset)}")
        elif name == "bigcodebench":
            try:
                import bigcodebench as M
                t = M.bigcodebench(condition=condition, limit=1)
                ok(f"bigcodebench.bigcodebench(...) builds — n_samples={len(t.dataset)}")
            except Exception as e:
                warn(f"bigcodebench needs HuggingFace fetch: {type(e).__name__}: {str(e)[:80]}")
                hint("first run will download bigcode/bigcodebench from HF")
        return True
    except Exception as e:
        fail(f"{name}: {type(e).__name__}: {str(e)[:120]}")
        return False


def check_docker() -> bool:
    if shutil.which("docker") is None:
        warn("docker not on PATH")
        hint("required for redcode_gen, bigcodebench, osharm sandboxes")
        return False
    ok("docker on PATH")
    return True


def check_playwright() -> bool:
    if shutil.which("playwright") is None:
        warn("playwright CLI not on PATH (browserart needs it)")
        hint("pip install playwright && playwright install chromium")
        return False
    ok("playwright CLI on PATH")
    hint(f"{DIM}make sure you also ran: playwright install chromium{RESET}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ICML release setup.")
    parser.add_argument(
        "--scenario", choices=["all", "browserart", "osharm", "redcode_gen", "bigcodebench"],
        default="all",
    )
    args = parser.parse_args()

    print("\n=== Python & core deps ===")
    check_python()
    check_package("inspect_ai", "0.3.179")
    check_package("openai", "2.17.0")
    check_package("anthropic")
    check_package("together")

    print("\n=== Data ===")
    check_data_files()

    print("\n=== Module imports ===")
    check_conditions_loadable()

    print("\n=== API keys ===")
    check_env_var("OPENAI_API_KEY",   required=True,
                  used_for="agent + OS-Harm/RedCode/BCB judges")
    check_env_var("TOGETHER_API_KEY", required=False,
                  used_for="BrowserART HarmBench judge")
    check_env_var("ANTHROPIC_API_KEY", required=False,
                  used_for="if you swap --model to anthropic/...")

    print("\n=== System tools ===")
    check_docker()
    check_playwright()

    print(f"\n=== Per-scenario Task construction ({args.scenario}) ===")
    if args.scenario in ("all", "browserart"):
        check_scenario("browserart")
    if args.scenario in ("all", "osharm"):
        check_scenario("osharm")
    if args.scenario in ("all", "redcode_gen"):
        check_scenario("redcode_gen")
    if args.scenario in ("all", "bigcodebench"):
        check_scenario("bigcodebench")

    print(f"\n{DIM}If everything above is OK or WARN, you're good to run a smoke test:{RESET}")
    print(f"{DIM}  python redcode_gen.py --condition single_agent --limit 1 --sandbox local{RESET}")
    print(f"{DIM}or a full sweep:{RESET}")
    print(f"{DIM}  ./run_all.sh openai/gpt-4o 42{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
