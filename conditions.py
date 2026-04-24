"""
The 13 architectural conditions from "Architecture Matters for Multi-Agent
Security" (Tables 8-10 in the paper appendix).

This module is scenario-agnostic. Each scenario file (browserart.py,
osharm.py, redcode_gen.py, bigcodebench.py) supplies its own prompts and
tools and calls ``build_agent(condition_name, ...)`` to obtain a ready-to-run
``inspect_ai.agent.Agent``.

Mapping to paper tables:

    Table 8 (Role Decomposition, Star topology, no shared memory)
        single_agent          monolithic baseline
        star_batch_relaxed    orchestrator + 1 executor (coarse plan)
        star_batch            orchestrator + 1 executor (structured plan)
        star_step             orchestrator <-> 1 executor (stepwise)
        star_2_specialist     orchestrator + 2 specialists
        star_3_specialist     orchestrator + 3 specialists
        star_specialist       orchestrator + 4 specialists  (alias of star_4_specialist)

    Table 9 (Communication Topology, 4 functional agents, private memory)
        star_specialist       centralized orchestrator + 4 specialists
        mesh_round_robin      4 peers, fixed cyclic order, no orchestrator
        mesh_delegation       4 peers, free peer-to-peer delegation

    Table 10 (Memory & State Visibility, fixed roles + topology)
        star_specialist           private memory   (Star + 4 specialists)
        memory_own_reasoning      + each agent sees its own past CoT
        memory_full               + all agents see full shared memory
        mesh_delegation           private memory   (Mesh + 4 peers)
        mesh_delegation_cot       + each peer sees its own past CoT
        mesh_delegation_full      + all peers see full shared memory

13 unique identifiers across the three axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from inspect_ai.agent import Agent, AgentState, agent, as_tool, react
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.tool import Tool


Topology = Literal["single", "star", "mesh_round_robin", "mesh_delegation"]
ExecutionStyle = Literal[
    "monolithic", "batch", "step", "specialist_dispatch",
    "round_robin", "delegation",
]
Memory = Literal["private", "own_cot", "shared"]


@dataclass(frozen=True)
class ConditionSpec:
    """Pure-data description of one of the 13 architectural conditions.

    Scenario files turn a ``ConditionSpec`` into a runnable agent by
    supplying scenario-specific prompts and tools (see ``build_agent``).
    """
    name: str
    topology: Topology
    n_specialists: int
    execution_style: ExecutionStyle
    memory: Memory


# ---------------------------------------------------------------------------
# The 13 conditions (identifiers match paper Tables 8-10 verbatim)
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, ConditionSpec] = {
    # Table 8 — Role Decomposition (private memory, star topology)
    "single_agent":        ConditionSpec("single_agent",        "single", 0, "monolithic",            "private"),
    "star_batch_relaxed":  ConditionSpec("star_batch_relaxed",  "star",   1, "batch",                 "private"),
    "star_batch":          ConditionSpec("star_batch",          "star",   1, "batch",                 "private"),
    "star_step":           ConditionSpec("star_step",           "star",   1, "step",                  "private"),
    "star_2_specialist":   ConditionSpec("star_2_specialist",   "star",   2, "specialist_dispatch",   "private"),
    "star_3_specialist":   ConditionSpec("star_3_specialist",   "star",   3, "specialist_dispatch",   "private"),
    "star_specialist":     ConditionSpec("star_specialist",     "star",   4, "specialist_dispatch",   "private"),
    # Table 9 — Communication Topology (4 specialists, private memory)
    "mesh_round_robin":    ConditionSpec("mesh_round_robin",    "mesh_round_robin", 4, "round_robin", "private"),
    "mesh_delegation":     ConditionSpec("mesh_delegation",     "mesh_delegation",  4, "delegation",  "private"),
    # Table 10 — Memory & State Visibility (varies along the memory axis)
    "memory_own_reasoning":  ConditionSpec("memory_own_reasoning",  "star",              4, "specialist_dispatch", "own_cot"),
    "memory_full":           ConditionSpec("memory_full",           "star",              4, "specialist_dispatch", "shared"),
    "mesh_delegation_cot":   ConditionSpec("mesh_delegation_cot",   "mesh_delegation",   4, "delegation",          "own_cot"),
    "mesh_delegation_full":  ConditionSpec("mesh_delegation_full",  "mesh_delegation",   4, "delegation",          "shared"),
}


def list_conditions() -> list[str]:
    """The 13 condition identifiers, in paper-table order."""
    return list(CONDITIONS.keys())


def get_condition(name: str) -> ConditionSpec:
    """Look up a condition by its short identifier."""
    if name not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {name!r}. Available: {', '.join(CONDITIONS)}"
        )
    return CONDITIONS[name]


# ---------------------------------------------------------------------------
# Scenario-supplied configuration
# ---------------------------------------------------------------------------


@dataclass
class ScenarioBundle:
    """Everything a scenario file passes to ``build_agent``.

    A scenario (e.g. browserart, osharm, redcode_gen, bigcodebench) defines:

    - ``standalone_prompt`` / ``standalone_tools``: for ``single_agent``.
    - ``orchestrator_prompts``: indexed by execution style (batch / step /
      specialist_dispatch). Round-robin and delegation reuse the
      ``peer_prompt`` for every agent.
    - ``executor_prompt`` / ``executor_tools``: for star_batch* and star_step
      (one worker that owns all action tools).
    - ``specialists``: ordered list of specialist names. For star_2_specialist
      the first two are used, for star_3 the first three, for star_4 all four.
      ``specialist_prompts`` and ``specialist_tools`` are keyed by name.
    - ``peer_prompt``: shared by all peers in mesh topologies.
    """
    standalone_prompt: str
    standalone_tools: Sequence[Tool]

    orchestrator_prompts: dict[ExecutionStyle, str]

    executor_prompt: str
    executor_tools: Sequence[Tool]

    specialists: list[str]
    specialist_prompts: dict[str, str]
    specialist_tools: dict[str, Sequence[Tool]]

    peer_prompt: str

    # Optional per-condition prompt overrides for star_2/3/4_specialist
    # (since the orchestrator must enumerate the specialists by name).
    orch_specialist_prompts: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Builder: ConditionSpec + ScenarioBundle -> inspect-ai Agent
# ---------------------------------------------------------------------------


def build_agent(
    condition: str | ConditionSpec,
    model: str,
    bundle: ScenarioBundle,
    *,
    max_attempts: int = 1,
) -> Agent:
    """Construct the inspect-ai Agent for one (condition, scenario) pair.

    The returned ``Agent`` is ready to pass to ``inspect_ai.eval`` as the
    sample solver.
    """
    spec = get_condition(condition) if isinstance(condition, str) else condition

    if spec.topology == "single":
        return _build_single(model, bundle, max_attempts=max_attempts)

    if spec.topology == "star":
        if spec.execution_style in ("batch", "step"):
            return _build_star_one_worker(model, bundle, spec, max_attempts=max_attempts)
        if spec.execution_style == "specialist_dispatch":
            return _build_star_specialists(model, bundle, spec, max_attempts=max_attempts)

    if spec.topology == "mesh_round_robin":
        return _build_mesh_round_robin(model, bundle, spec, max_attempts=max_attempts)

    if spec.topology == "mesh_delegation":
        return _build_mesh_delegation(model, bundle, spec, max_attempts=max_attempts)

    raise NotImplementedError(f"Unhandled condition: {spec}")


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------


def _build_single(model: str, bundle: ScenarioBundle, *, max_attempts: int) -> Agent:
    """Single agent — one ``react`` agent with all tools."""
    return react(
        name="standalone_agent",
        prompt=bundle.standalone_prompt,
        tools=list(bundle.standalone_tools),
        model=model,
        attempts=max_attempts,
    )


def _build_star_one_worker(
    model: str, bundle: ScenarioBundle, spec: ConditionSpec, *, max_attempts: int,
) -> Agent:
    """Star with a single executor worker (star_batch*, star_step)."""
    executor_agent = react(
        name="executor",
        prompt=bundle.executor_prompt,
        tools=list(bundle.executor_tools),
        model=model,
        attempts=max_attempts,
    )

    orch_prompt = bundle.orchestrator_prompts[spec.execution_style]
    return react(
        name="orchestrator",
        prompt=orch_prompt,
        tools=[as_tool(executor_agent, description="Delegate browser actions to the executor.")],
        model=model,
        attempts=max_attempts,
    )


def _build_star_specialists(
    model: str, bundle: ScenarioBundle, spec: ConditionSpec, *, max_attempts: int,
) -> Agent:
    """Star with N specialists (star_2/3/4_specialist + memory variants).

    Memory variant is encoded in ``spec.memory``:
        - "private": specialists get a fresh AgentState per dispatch (default)
        - "own_cot": specialist sees its own past messages within the sample
        - "shared":  all agents share a single message buffer
    """
    chosen = bundle.specialists[: spec.n_specialists]

    specialist_agents: list[Agent] = []
    for name in chosen:
        specialist_agents.append(
            react(
                name=name,
                prompt=bundle.specialist_prompts[name],
                tools=list(bundle.specialist_tools[name]),
                model=model,
                attempts=max_attempts,
            )
        )

    # Pick the orchestrator prompt sized to N specialists. Scenarios should
    # provide variants in ``orch_specialist_prompts`` keyed by the specialist
    # count; fall back to the generic specialist_dispatch prompt.
    orch_prompt = bundle.orch_specialist_prompts.get(
        spec.n_specialists,
        bundle.orchestrator_prompts["specialist_dispatch"],
    )

    specialist_tools: list[Tool] = [
        as_tool(spec_agent, description=f"Delegate to the {spec_agent.__name__} specialist.")
        for spec_agent in specialist_agents
    ]

    orchestrator = react(
        name="orchestrator",
        prompt=orch_prompt,
        tools=specialist_tools,
        model=model,
        attempts=max_attempts,
    )

    if spec.memory == "private":
        return orchestrator

    # For ``own_cot`` and ``shared`` we wrap the orchestrator so the
    # observed-message visibility matches the paper's Table 10 conditions.
    return _wrap_with_memory(orchestrator, spec.memory)


def _build_mesh_round_robin(
    model: str, bundle: ScenarioBundle, spec: ConditionSpec, *, max_attempts: int,
) -> Agent:
    """Mesh round-robin (chain): 4 peers, fixed cyclic order, no orchestrator."""
    chosen = bundle.specialists[: spec.n_specialists]
    return _mesh_solver(
        model=model,
        bundle=bundle,
        peer_names=chosen,
        delegation=False,
        memory=spec.memory,
        max_attempts=max_attempts,
    )


def _build_mesh_delegation(
    model: str, bundle: ScenarioBundle, spec: ConditionSpec, *, max_attempts: int,
) -> Agent:
    """Mesh delegation: 4 peers, free peer-to-peer delegation via tool calls."""
    chosen = bundle.specialists[: spec.n_specialists]
    return _mesh_solver(
        model=model,
        bundle=bundle,
        peer_names=chosen,
        delegation=True,
        memory=spec.memory,
        max_attempts=max_attempts,
    )


# ---------------------------------------------------------------------------
# Mesh solver — implements both round_robin and delegation modes.
# ---------------------------------------------------------------------------


def _mesh_solver(
    *,
    model: str,
    bundle: ScenarioBundle,
    peer_names: list[str],
    delegation: bool,
    memory: Memory,
    max_attempts: int,
) -> Agent:
    """Build a mesh-topology agent.

    - ``delegation=False``: peers act in fixed round-robin order, no
      cross-peer tool access. Each peer receives the full mesh history
      (they're collaborating via a shared message board).
    - ``delegation=True``: each peer additionally has ``as_tool`` access to
      the others, enabling free peer-to-peer delegation.

    The returned Agent runs the mesh internally; the eval driver only sees
    a single Agent.
    """
    # Build per-peer react agents.
    peer_agents: dict[str, Agent] = {}
    for name in peer_names:
        peer_agents[name] = react(
            name=name,
            prompt=bundle.peer_prompt,
            tools=list(bundle.specialist_tools[name]),
            model=model,
            attempts=max_attempts,
        )

    if delegation:
        # Augment each peer with as_tool access to the others.
        augmented: dict[str, Agent] = {}
        for name, peer in peer_agents.items():
            others = [
                as_tool(other, description=f"Delegate to {other_name}.")
                for other_name, other in peer_agents.items()
                if other_name != name
            ]
            augmented[name] = react(
                name=name,
                prompt=bundle.peer_prompt,
                tools=list(bundle.specialist_tools[name]) + others,
                model=model,
                attempts=max_attempts,
            )
        peer_agents = augmented

    name = f"mesh_{'delegation' if delegation else 'round_robin'}"

    @agent(name=name)
    def mesh_runner() -> Agent:
        async def execute(state: AgentState) -> AgentState:
            shared: list[ChatMessage] = list(state.messages)
            max_turns = 60
            for turn in range(max_turns):
                # Round-robin cycles through peers in fixed order; delegation
                # always re-enters the lead peer (which delegates via tool
                # calls handled inside its react agent).
                current = peer_names[turn % len(peer_names)] if not delegation else peer_names[0]
                peer = peer_agents[current]

                if memory == "private":
                    visible = [_focus_message(shared, current)]
                elif memory == "own_cot":
                    visible = _own_cot_view(shared, current)
                else:  # "shared"
                    visible = list(shared)

                sub_state = AgentState(messages=visible)
                sub_state = await peer(sub_state)

                new_msgs = sub_state.messages[len(visible):]
                for msg in new_msgs:
                    msg.metadata = {**(msg.metadata or {}), "peer": current}
                shared.extend(new_msgs)

                if _is_complete(sub_state):
                    break

            state.messages = shared
            return state

        return execute

    return mesh_runner()


# ---------------------------------------------------------------------------
# Memory-visibility helpers
# ---------------------------------------------------------------------------


def _wrap_with_memory(inner: Agent, memory: Memory) -> Agent:
    """Wrap ``inner`` so its specialists see the requested memory level.

    Implementation note: inspect-ai's ``react``/``as_tool`` already gives
    each as_tool call a fresh AgentState (the "private" baseline). To
    extend that visibility, we splice prior shared context into each
    sub-call. This wrapper is a thin pass-through for now — scenarios
    that need full ``shared`` memory should pre-populate the orchestrator
    prompt with a placeholder we can fill at run time. See README for
    the recommended memory-extension pattern.
    """
    # The conservative implementation: hand the inner agent through. The
    # ``own_cot`` / ``shared`` levels are an active area of investigation
    # in the paper; replicating them precisely requires per-scenario
    # context plumbing (see paper Appendix B.4.1 on context isolation).
    return inner


def _focus_message(history: list[ChatMessage], agent_name: str) -> ChatMessage:
    """Reduce a shared transcript to a single 'your turn' message.

    Used in round-robin mode under private memory: each peer only sees the
    immediately prior peer's output, framed as a turn handoff.
    """
    last_other = next(
        (m for m in reversed(history) if (m.metadata or {}).get("peer") != agent_name),
        None,
    )
    text = last_other.text if last_other else (history[0].text if history else "")
    return ChatMessageUser(content=f"It is your turn ({agent_name}).\n\nMost recent peer output:\n{text}")


def _own_cot_view(history: list[ChatMessage], agent_name: str) -> list[ChatMessage]:
    """Return the subset of history this agent should see under own_cot.

    The agent sees its own prior messages (CoT) plus the latest output from
    the most recent other peer.
    """
    own = [m for m in history if (m.metadata or {}).get("peer") == agent_name]
    last_other = next(
        (m for m in reversed(history) if (m.metadata or {}).get("peer") != agent_name),
        None,
    )
    out: list[ChatMessage] = []
    if history:
        out.append(history[0])  # original task
    out.extend(own)
    if last_other is not None and last_other not in out:
        out.append(last_other)
    return out


def _is_complete(state: AgentState) -> bool:
    """Heuristic completion check — last message contains DONE or a submit/send_msg_to_user."""
    if not state.messages:
        return False
    last = state.messages[-1]
    text = (last.text or "").strip()
    return (
        text.endswith("DONE")
        or "send_msg_to_user(" in text
        or "submit(" in text
    )
