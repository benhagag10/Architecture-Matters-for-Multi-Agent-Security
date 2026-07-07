# Architecture Matters for Multi-Agent Security

Code, data, and judges to reproduce the results in the ICML paper
**"Architecture Matters for Multi-Agent Security."** Built on
UK AISI's [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

The release includes:

```
.
├── README.md            ← you are here
├── LICENSE              ← MIT
├── requirements.txt     ← (Python ≥ 3.10)
├── check_setup.py       ← run after `pip install` to find any dependency or setup issues
├── run_all.sh           ← reproduce every number in the paper
├── conditions.py        ← the 13 architectural conditions (Tables 8-10)
├── judges.py            ← the scenario-specific judges (Appendix B.6)
├── browserart.py        ← BrowserART scenario (100 harmful + 42 benign)
├── osharm.py            ← OS-Harm scenario (44 harmful + 50 benign)
├── redcode_gen.py       ← RedCode-Gen scenario (160 harmful)
├── bigcodebench.py      ← BigCodeBench scenario (50 benign — RedCode baseline)
└── data/
    ├── browserart/      ← 100 harmful + 59 benign tasks (filtered to 42 at load)
    └── redcode_gen/     ← 160 malware tasks across 8 categories
```

## Quick start

```bash
git clone <release-url> icml_release && cd icml_release
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
playwright install chromium                            # for BrowserART

export OPENAI_API_KEY=sk-...
export TOGETHER_API_KEY=...                            # for the BrowserART judge

# Pre-flight:
python check_setup.py

# Smoke test (3 samples, fast — finishes in ~30s on RedCode-Gen):
python redcode_gen.py --smoke --condition single_agent --sandbox local

# Reproduce a single condition × scenario × kind:
python browserart.py  --condition star_specialist  --kind harmful --model openai/gpt-4o
python osharm.py      --condition mesh_delegation  --kind benign  --model openai/gpt-4o
python redcode_gen.py --condition star_specialist  --model openai/gpt-4o
python bigcodebench.py --condition single_agent    --model openai/gpt-4o --limit 50

# Reproduce all paper numbers for a given model:
./run_all.sh openai/gpt-4o 42
```

Logs land under `logs/<scenario>_icml/<condition>/<kind>/`.

## Mapping paper tables → commands

| Paper table | Scenario | Run with |
|---|---|---|
| Table 1 (Role distribution, BrowserART) | BrowserART | `browserart.py --condition <id> --kind harmful` |
| Table 1 (Role distribution, OS-Harm)    | OS-Harm    | `osharm.py     --condition <id> --kind harmful` |
| Table 1 (Role distribution, RedCode-Gen)| RedCode-Gen| `redcode_gen.py --condition <id>` |
| Table 2 (Communication topology)        | all 3      | `--condition single_agent \| star_specialist \| mesh_round_robin \| mesh_delegation` |
| Table 3 (Memory & state visibility)     | all 3      | `--condition star_specialist \| memory_own_reasoning \| memory_full \| mesh_delegation \| mesh_delegation_cot \| mesh_delegation_full` |
| Tables 4–7 (Extended results, other models) | all 3 | swap `--model` (e.g. `openai/gpt-5.4`, `anthropic/claude-sonnet-4`, `together/Qwen/Qwen3-VL-...`, `together/meta-llama/Llama-3-70B-...`) |
| Benign columns                           | all 3      | `--kind benign` for browserart/osharm; `bigcodebench.py` for redcode benign |

The 13 condition identifiers (`<id>` above) match the paper's Tables 8–10
exactly. List them with:

```bash
python -c "from conditions import list_conditions; print('\n'.join(list_conditions()))"
```

## Prerequisites by scenario

| Scenario | Needs |
|---|---|
| **BrowserART** | `OPENAI_API_KEY` (agent), `TOGETHER_API_KEY` (BrowserART judge), Playwright + Chromium. The local site server (BrowserART hbb sites) must be reachable; see [BrowserART repo](https://github.com/MichaelKohlPro/BrowserART) for the bundled docker-compose. |
| **OS-Harm**    | `OPENAI_API_KEY`, Docker running, OSWorld + OS-Harm checkouts (clone the public repos and set `OSWORLD_PATH=` / `OSHARM_PATH=` env vars). `OSHARM_PATH` must contain `data/misuse/tasks.json`. |
| **RedCode-Gen**| `OPENAI_API_KEY`. Code execution runs inside Inspect's `python` sandbox (Docker recommended). |
| **BigCodeBench**| `OPENAI_API_KEY`, plus `pip install datasets` to fetch `bigcode/bigcodebench` from HuggingFace on first use. |

## Judges (Appendix B.6)

| Scenario | Judge model (default) | Configurable via |
|---|---|---|
| BrowserART  | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` (HarmBench-style) | `--judge-model` |
| OS-Harm     | `openai/gpt-4.1` (misuse + injection + misbehavior)                  | `--judge-model` |
| RedCode-Gen | `openai/gpt-4.1` (RedCode 0/1/5/8/10 rubric)                          | `--judge-model` |

All three return the four-stage outcome (Planning Refusal / Execution
Refusal / Harmful Action / Harmful Task), which are
disjoint sets.

## The 13 conditions

```
single_agent          monolithic baseline (Standalone Agent)
star_batch_relaxed    Star + 1 batch executor (coarse plan)
star_batch            Star + 1 batch executor (structured plan)
star_step             Star + 1 stepwise executor
star_2_specialist     Star + 2 merged specialists
star_3_specialist     Star + 3 specialists
star_specialist       Star + 4 specialists (= "Star + Specialized" in paper)
mesh_round_robin      Chain (4 peers, fixed cyclic order)
mesh_delegation       Mesh (4 peers, free delegation)
memory_own_reasoning  Star + 4 specialists, each sees its own past CoT
memory_full           Star + 4 specialists, all share full memory
mesh_delegation_cot   Mesh + 4 peers, each sees its own past CoT
mesh_delegation_full  Mesh + 4 peers, all share full memory
```

These are the 13 condition identifiers used in the paper.
See `conditions.CONDITIONS`.

## Smoke tests

Every runner supports `--smoke`, which caps to 3 samples and shortens
per-task wall-clock for sanity checking.

```bash
python browserart.py   --smoke --condition single_agent --kind harmful
python osharm.py       --smoke --condition single_agent --kind harmful
python redcode_gen.py  --smoke --condition single_agent
python bigcodebench.py --smoke --condition single_agent
```

## What this release includes.

- **Includes:** all 13 condition definitions, all 3 judges, all 4 scenario
  runners, the BrowserART + RedCode-Gen task data (public benchmarks),
  end-to-end orchestration via Inspect AI.
- **Does not include:** the OSWorld VM image (clone [the OSWorld repo](https://github.com/xlang-ai/osworld) and
  build per its README), or existing run logs.

## Basic troubleshooting

Run `python check_setup.py` first — it diagnoses most of these in one shot.

#### `inspect_ai._util.error.PrerequisiteError: ERROR: OpenAI API requires at least version 2.17.0 of package openai`
You have an older `openai`. Run `pip install --upgrade openai` (or re-run
`pip install -r requirements.txt`).

#### `PrerequisiteError: ERROR: Unable to initialise Together client. No TOGETHER_API_KEY defined`
Only the BrowserART judge needs Together. Either:
- `export TOGETHER_API_KEY=...`, or
- `export BROWSERART_JUDGE_MODEL=openai/gpt-4.1` to swap the judge to GPT-4.1 (or other model, as desired).

#### `ProcessLookupError: No sandbox environment has been provided`
The agent tried to invoke a sandbox-backed tool (Python / bash) but the Task
has no sandbox declared. The runners default to `sandbox="docker"`. To
test without Docker, pass `--sandbox local`. **This is not recommended.**

#### `SandboxInjectionError: ... Only Linux containers are currently supported`
Inspect's `python` tool requires Docker to run Linux VMs. If you attempt to use a non-Linux sandbox, you may encounter this error. 

#### `FileNotFoundError: OS-Harm dataset not found at os-harm/data/misuse/tasks.json`
You haven't cloned the OS-Harm checkout yet. Run:
```bash
git clone https://github.com/tml-epfl/os-harm
git clone https://github.com/xlang-ai/OSWorld
export OSHARM_PATH=./os-harm OSWORLD_PATH=./OSWorld
# OSHARM_PATH must provide data/misuse/tasks.json (the 44-task misuse manifest).
```

### BrowserART smoke test hangs / page loads forever
The harmful tasks reference `local:*` URLs — you need the BrowserART local
site server running. See [the BrowserART repo](https://github.com/MichaelKohlPro/BrowserART)
for the bundled `docker-compose.yml`.

## License

MIT

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{hagag2026architecture,
  title   = {Architecture Matters for Multi-Agent Security},
  author  = {Hagag, Ben and Anderson, William L. and Schroeder de Witt, Christian and Scheffler, Sarah},
  journal = {arXiv preprint arXiv:2604.23459},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.23459}
}
```

- **Paper:** [arXiv:2604.23459](https://arxiv.org/pdf/2604.23459)
- **Code:** [github.com/benhagag10/Architecture-Matters-for-Multi-Agent-Security](https://github.com/benhagag10/Architecture-Matters-for-Multi-Agent-Security/tree/main)
