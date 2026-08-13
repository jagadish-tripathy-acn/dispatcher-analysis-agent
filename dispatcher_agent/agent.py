"""
Dispatcher AI Analysis Agent
=============================

A LangGraph-based agent that turns the `/api/stats` payload into a
structured AI analysis using AWS Bedrock (Claude, via the Converse API).

Usage:
    agent = DispatcherGraph()                       # uses DEFAULT_MODEL
    agent = DispatcherGraph("us.anthropic.claude-sonnet-4-6")  # or override
    response = agent.invoke(stats_payload)           # -> AnalysisResponse

Workflow:  validate -> analyze -> assemble
Invalid/empty payloads short-circuit straight to `assemble` with status="error".
`analyze` makes exactly ONE Bedrock call that produces the full analysis,
anomalies, recommendations, and executive summary together (one combined
structured schema) -- this keeps latency to a single round-trip instead of
several sequential ones. If it fails, the response degrades to
status="partial" instead of raising -- the agent never crashes.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# Durations in this payload are minutes, which is right for the charts and wrong
# for prose — "706622 min" is unreadable. Shared with the parser so the RCA text
# and the anomaly text phrase the same number the same way.
from parser import fmt_duration as _dur


def _timed(minutes):
    """A duration for prose, or None when there is no measurement behind it.

    parser's averages and percentiles are 0.0 when nothing was sampled — a job
    type whose tasks all die before TRANSLATING has no timing at all — so
    printing "0s" would claim a measurement that was never taken. Call sites
    check for None and drop or reword the clause instead.
    """
    return _dur(minutes) if minutes else None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MIN_TASKS_FOR_HEALTH_SCORE = 5

# A single provider/user/group behind this share of a job type's failures is
# treated as a concentration worth naming as a suspect.
CONCENTRATION = 0.7

SYSTEM_PROMPT = (
    "You are a senior site reliability engineer analyzing telemetry from a Teamcenter "
    "Dispatcher job-processing system. Use ONLY the data given below -- never invent metrics, "
    "jobs, users, or timestamps. If something needed is missing, say so instead of guessing. "
    "Respond only via the requested structured schema, no extra text.\n\n"
    "Produce, in this single response:\n"
    "1. health_assessment: a single concise sentence on overall system health.\n"
    "2. anomalies: notable deviations or warning signs worth investigating, each with a severity.\n"
    "3. recommendations: 3-5 concrete, prioritized actions, each traceable to an anomaly.\n"
    "4. executive_summary: STRICT LIMIT of 1-2 short sentences (max ~40 words total). A "
    "plain-language headline naming the overall health and, if unhealthy, only the single "
    "biggest issue. Do not enumerate metrics, anomalies, or recommendations here.\n"
    "5. reasoning: 1-2 sentences on how you reached these conclusions.\n"
    "6. confidence: 0.0-1.0 based on how complete the input data was.\n"
    "7. health_score: 0.0-100.0 overall score, or null if the sample size is too small to be "
    "told sufficient below."
)

RCA_SYSTEM_PROMPT = (
    "You are a senior Teamcenter Dispatcher administrator performing root cause analysis on ONE "
    "job type whose telemetry has been flagged as anomalous.\n\n"
    "Domain facts you may rely on:\n"
    "- A task's lifecycle is INITIAL -> PREPARING -> SCHEDULED -> TRANSLATING -> LOADING -> "
    "COMPLETE, and it can end in TERMINAL (failed), DELETE or DUPLICATE.\n"
    "- Queue time (INITIAL->TRANSLATING) is time waiting for a free translator/module slot; it "
    "points at scheduling, module instance count, or provider availability.\n"
    "- Processing time (TRANSLATING->end) is the translator doing work; it points at input size, "
    "translator/host resources, or the target system.\n"
    "- Dispatcher components involved: DispatcherClient, Scheduler, Module (translators), and the "
    "per-job configuration in the Dispatcher module's XML/properties files.\n\n"
    "Rules: use ONLY the evidence given. Never invent task IDs, users, providers, counts or "
    "timestamps. Quote the specific numbers from the evidence in your reasoning. If the evidence "
    "cannot distinguish between causes, say so and rank them by likelihood. Prefer a queue-side "
    "cause when most of the elapsed time is queue time, and a processing-side cause when most of "
    "it is processing time. Durations in the evidence are in minutes; when you quote one, write it "
    "in the units a reader thinks in — '35m', '2h 10m', '3d 4h' — never a bare minute count like "
    "'4332 min'. Respond only via the requested structured schema.\n\n"
    "Produce:\n"
    "1. summary: 2-3 sentences stating what is wrong with this job type and the single most "
    "likely reason.\n"
    "2. primary_cause: one short sentence — the most probable root cause.\n"
    "3. impact: what continues to happen if this is not addressed.\n"
    "4. causes: 2-4 candidate root causes, most likely first, each with the evidence that "
    "supports it and a category.\n"
    "5. actions: 3-5 concrete actions an administrator can actually take, ordered by priority, "
    "each tagged immediate / short_term / preventive, with the expected effect.\n"
    "6. checks: 2-4 specific things to inspect to confirm or rule out the primary cause "
    "(name the log, config file, or metric).\n"
    "7. confidence: 0.0-1.0, based on how conclusive the evidence is."
)


# --------------------------------------------------------------------------- #
# Structured output schemas
# --------------------------------------------------------------------------- #
class Anomaly(BaseModel):
    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class Recommendation(BaseModel):
    title: str
    description: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class AnalysisResult(BaseModel):
    """Everything the single Bedrock call produces, in one combined schema."""

    health_assessment: str
    anomalies: List[Anomaly] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    executive_summary: str = Field(description="1-2 sentences max, ~40 words.")
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    health_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class AnalysisResponse(BaseModel):
    """The single object `DispatcherGraph.invoke()` returns, success or failure."""

    status: Literal["success", "partial", "error"] = "success"
    executive_summary: str = ""
    health_assessment: str = ""
    anomalies: List[Anomaly] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    confidence: float = 0.0
    health_score: Optional[float] = None
    reasoning: str = ""
    error: Optional[str] = None


class RootCause(BaseModel):
    cause: str
    category: Literal[
        "capacity", "configuration", "input_data", "dependency",
        "scheduling", "infrastructure", "user_behaviour", "unknown",
    ] = "unknown"
    likelihood: Literal["low", "medium", "high"] = "medium"
    evidence: str = ""


class RCAAction(BaseModel):
    action: str
    horizon: Literal["immediate", "short_term", "preventive"] = "immediate"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    rationale: str = ""
    expected_impact: str = ""


class DiagnosticCheck(BaseModel):
    check: str
    where: str = ""


class RCAResult(BaseModel):
    """What the RCA Bedrock call produces for one job type."""

    summary: str
    primary_cause: str
    impact: str = ""
    causes: List[RootCause] = Field(default_factory=list)
    actions: List[RCAAction] = Field(default_factory=list)
    checks: List[DiagnosticCheck] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RCAResponse(BaseModel):
    """The object `DispatcherGraph.rca()` returns, success or degraded."""

    status: Literal["success", "heuristic", "error"] = "success"
    source: Literal["bedrock", "rules", "none"] = "bedrock"
    job: str = ""
    severity: str = "ok"
    anomaly_types: List[str] = Field(default_factory=list)
    summary: str = ""
    primary_cause: str = ""
    impact: str = ""
    causes: List[RootCause] = Field(default_factory=list)
    actions: List[RCAAction] = Field(default_factory=list)
    checks: List[DiagnosticCheck] = Field(default_factory=list)
    confidence: float = 0.0
    error: Optional[str] = None


class GraphState(TypedDict, total=False):
    stats: dict
    is_valid: bool
    error: Optional[str]
    result: Optional[AnalysisResult]
    response: AnalysisResponse


class RCAState(TypedDict, total=False):
    detail: dict
    is_valid: bool
    error: Optional[str]
    result: Optional[RCAResult]
    response: RCAResponse


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
class DispatcherGraph:
    """LangGraph-based AI analysis engine for dispatcher system statistics."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = ChatBedrockConverse(model=model_name)
        self._graph = self._build_graph()
        self._rca_graph = self._build_rca_graph()

    def invoke(self, stats: dict) -> AnalysisResponse:
        """Run the full workflow over a `/api/stats`-shaped payload."""
        state: GraphState = {"stats": stats or {}}
        result = self._graph.invoke(state)
        return result["response"]

    def rca(self, detail: dict) -> RCAResponse:
        """Root cause analysis for one job type, from a `job_detail()` payload."""
        state: RCAState = {"detail": detail or {}}
        return self._rca_graph.invoke(state)["response"]

    # -- graph wiring ------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("validate", self._validate)
        g.add_node("analyze", self._analyze)
        g.add_node("assemble", self._assemble)

        g.add_edge(START, "validate")
        g.add_conditional_edges(
            "validate",
            lambda s: "analyze" if s.get("is_valid") else "assemble",
            {"analyze": "analyze", "assemble": "assemble"},
        )
        g.add_edge("analyze", "assemble")
        g.add_edge("assemble", END)
        return g.compile()

    def _build_rca_graph(self):
        """validate -> diagnose -> assemble, with a rules-based safety net.

        `assemble` falls back to the deterministic playbook in `heuristic_rca`
        when the Bedrock call fails, so the "why is this job flagged?" button
        always answers with something actionable.
        """
        g = StateGraph(RCAState)
        g.add_node("validate", self._rca_validate)
        g.add_node("diagnose", self._rca_diagnose)
        g.add_node("assemble", self._rca_assemble)

        g.add_edge(START, "validate")
        g.add_conditional_edges(
            "validate",
            lambda s: "diagnose" if s.get("is_valid") else "assemble",
            {"diagnose": "diagnose", "assemble": "assemble"},
        )
        g.add_edge("diagnose", "assemble")
        g.add_edge("assemble", END)
        return g.compile()

    # -- nodes ------------------------------------------------------------
    def _validate(self, state: GraphState) -> dict:
        stats = state.get("stats")
        if not isinstance(stats, dict) or not stats.get("has_data"):
            return {"is_valid": False, "error": "Payload is missing or has no data to analyze."}
        return {"is_valid": True}

    def _analyze(self, state: GraphState) -> dict:
        """Single Bedrock call producing the full analysis, anomalies,
        recommendations, and executive summary together."""
        try:
            result = self._ask(SYSTEM_PROMPT, self._user_prompt(state["stats"]), AnalysisResult)
        except Exception as exc:
            logger.error("analyze failed: %s", exc)
            return {"result": None, "error": f"Analysis failed: {exc}"}
        return {"result": result}

    def _assemble(self, state: GraphState) -> dict:
        if not state.get("is_valid"):
            return {"response": AnalysisResponse(status="error", error=state.get("error") or "Invalid input.")}

        result = state.get("result")
        if not result:
            return {
                "response": AnalysisResponse(
                    status="partial",
                    executive_summary="Analysis could not be generated.",
                    error=state.get("error") or "The analysis step failed.",
                )
            }

        return {
            "response": AnalysisResponse(
                status="success",
                executive_summary=result.executive_summary,
                health_assessment=result.health_assessment,
                anomalies=result.anomalies,
                recommendations=result.recommendations,
                confidence=result.confidence,
                health_score=result.health_score,
                reasoning=result.reasoning,
            )
        }

    # -- RCA nodes ---------------------------------------------------------
    def _rca_validate(self, state: RCAState) -> dict:
        detail = state.get("detail")
        if not isinstance(detail, dict) or not detail.get("job"):
            return {"is_valid": False, "error": "No job detail supplied to analyse."}
        if not (detail.get("metrics") or {}).get("count"):
            return {"is_valid": False, "error": "That job type has no tasks in the selected window."}
        return {"is_valid": True}

    def _rca_diagnose(self, state: RCAState) -> dict:
        try:
            result = self._ask(RCA_SYSTEM_PROMPT, rca_evidence(state["detail"]), RCAResult)
        except Exception as exc:
            logger.error("rca failed: %s", exc)
            return {"result": None, "error": f"AI root cause analysis unavailable: {exc}"}
        return {"result": result}

    def _rca_assemble(self, state: RCAState) -> dict:
        detail = state.get("detail") or {}
        if not state.get("is_valid"):
            return {"response": RCAResponse(status="error", source="none",
                                            job=detail.get("job", ""),
                                            error=state.get("error") or "Invalid input.")}

        anomalies = detail.get("anomalies") or []
        common = {
            "job": detail.get("job", ""),
            "severity": detail.get("severity", "ok"),
            "anomaly_types": [a.get("type", "") for a in anomalies],
        }

        result = state.get("result")
        if not result:
            fallback = heuristic_rca(detail)
            fallback.error = state.get("error")
            return {"response": fallback}

        return {"response": RCAResponse(
            status="success", source="bedrock", **common,
            summary=result.summary, primary_cause=result.primary_cause, impact=result.impact,
            causes=result.causes, actions=result.actions, checks=result.checks,
            confidence=result.confidence,
        )}

    # -- small helpers -------------------------------------------------------
    def _ask(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> BaseModel:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        return self.model.with_structured_output(schema).invoke(messages)

    @staticmethod
    def _user_prompt(stats: dict) -> str:
        sample_size = (stats.get("kpis") or {}).get("total_tasks", 0)
        note = (
            "sufficient for a numeric health score."
            if sample_size >= MIN_TASKS_FOR_HEALTH_SCORE
            else "too small -- return null for health_score."
        )
        return f"{DispatcherGraph._context(stats)}\n\nSample size is {note}"

    @staticmethod
    def _context(stats: dict) -> str:
        k = stats.get("kpis") or {}
        lines = [
            f"Total tasks: {k.get('total_tasks', 0)}",
            f"Completed: {k.get('completed', 0)}",
            f"In Queue (awaiting translation): {k.get('in_queue', 0)} ({k.get('in_queue_pct', 0)}%)",
            f"Terminal/failed: {k.get('terminal_jobs', 0)} ({k.get('terminal_pct', 0)}%)",
            f"Avg queue time: {k.get('avg_queue_min', 0)} min (max {k.get('max_queue_min', 0)} min)",
            f"Avg processing time: {k.get('avg_proc_min', 0)} min",
            f"Avg total lifecycle time: {k.get('avg_total_min', 0)} min",
        ]
        jobs = stats.get("job_breakdown") or []
        if jobs:
            lines.append(
                "Job breakdown: "
                + "; ".join(f"{j.get('job')} (count={j.get('count')}, avg_total={j.get('avg_total')}m)" for j in jobs[:10])
            )
        stuck = stats.get("in_queue_tasks") or []
        if stuck:
            lines.append(
                f"Stuck tasks ({len(stuck)} total, sample): "
                + ", ".join(f"{t.get('task_id')}/{t.get('status')}/{t.get('job')}" for t in stuck[:10])
            )
        jh = stats.get("job_health") or []
        flagged = [j for j in jh if j.get("severity") not in (None, "ok")]
        if flagged:
            lines.append("Job types flagged by the anomaly detector: " + "; ".join(
                f"{j['job']} [{j['severity']}] " + ", ".join(a["type"] for a in j.get("anomalies", []))
                for j in flagged[:8]
            ))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Root cause analysis: evidence formatting + rules-based fallback
# --------------------------------------------------------------------------- #
def _lead(entries):
    """Biggest contributor in a [{name,count}] list and its share of the total."""
    total = sum(e.get("count", 0) for e in entries or [])
    if not total:
        return None, 0.0
    top = max(entries, key=lambda e: e.get("count", 0))
    return top, top.get("count", 0) / total


def _suspect(fail_entries, all_entries, total_tasks, min_lift=1.15):
    """Entity that owns most of the failures *and* more of them than its share
    of the traffic explains.

    Without the lift test the top submitter of a job type always looks guilty:
    if one user files every task, they also own every failure.
    Returns (entry, failure_share, submission_share) or (None, ...).
    """
    top, share = _lead(fail_entries)
    if not top or share < CONCENTRATION:
        return None, share, 0.0
    submitted = next((e["count"] for e in (all_entries or []) if e["name"] == top["name"]), 0)
    base = (submitted / total_tasks) if total_tasks else 0.0
    if base and share < base * min_lift:
        return None, share, base
    return top, share, base


def _peak_fail_hour(by_hour):
    """Hour of day holding the largest share of this job's failures."""
    total = sum(h.get("fail", 0) for h in by_hour or [])
    if not total:
        return None, 0.0
    top = max(by_hour, key=lambda h: h.get("fail", 0))
    return top, top.get("fail", 0) / total


def rca_evidence(detail: dict) -> str:
    """Flatten a `job_detail()` payload into the evidence block for the model."""
    m = detail.get("metrics") or {}
    b = detail.get("baselines") or {}
    win = detail.get("window") or {}
    lines = [
        f"JOB TYPE: {detail.get('job')}",
        f"Window: {win.get('from') or 'all data'} -> {win.get('to') or 'all data'}",
        "",
        "-- Volume and outcomes --",
        f"Tasks: {m.get('count')} | completed {m.get('completed')} ({m.get('completed_pct')}%) | "
        f"TERMINAL {m.get('terminal')} ({m.get('terminal_pct')}%) | DELETE {m.get('deleted')} | "
        f"DUPLICATE {m.get('duplicate')} | still SCHEDULED {m.get('in_queue')} | "
        f"mid-flight {m.get('active')}",
        f"First task {m.get('first_seen')}, last task {m.get('last_seen')}",
        "",
        "-- Timing (minutes) --",
        f"Completion: avg {m.get('avg_total')}, p50 {m.get('p50_total')}, p95 {m.get('p95_total')}, "
        f"max {m.get('max_total')}",
        f"Queue wait: avg {m.get('avg_queue')}, p95 {m.get('p95_queue')}",
        f"Processing: avg {m.get('avg_proc')}, p95 {m.get('p95_proc')}",
        f"Share of elapsed time spent queueing: {m.get('queue_share')}%",
        f"Own historical baseline p50 {m.get('baseline_p50')} -> recent p50 {m.get('recent_p50')} "
        f"({m.get('drift_pct')}% change)",
        "",
        "-- System baseline for comparison --",
        f"All job types: TERMINAL {b.get('terminal_pct')}%, still SCHEDULED {b.get('in_queue_pct')}%, "
        f"completion p50 {b.get('p50_total')} min, p95 {b.get('p95_total')} min, "
        f"avg queue {b.get('avg_queue')} min, avg processing {b.get('avg_proc')} min "
        f"across {b.get('job_types')} job types",
        "",
        "-- Anomalies detected for this job type --",
    ]
    for a in detail.get("anomalies") or []:
        lines.append(f"[{a.get('severity','').upper()}] {a.get('type')}: {a.get('detail')}"
                     + (f" Sample task IDs: {', '.join(a.get('sample') or [])}" if a.get("sample") else ""))
    if not (detail.get("anomalies") or []):
        lines.append("(none — explain why the job looks healthy instead)")

    def block(title, entries):
        if entries:
            lines.append("")
            lines.append(f"-- {title} --")
            lines.append(", ".join(f"{e['name']}={e['count']}" for e in entries))

    block("Submitting users (all tasks)", detail.get("top_users"))
    block("Submitting groups (all tasks)", detail.get("top_groups"))
    block("Providers (all tasks)", detail.get("top_providers"))
    block("Users owning the TERMINAL tasks", detail.get("failing_users"))
    block("Providers owning the TERMINAL tasks", detail.get("failing_providers"))

    hours = [h for h in (detail.get("failure_by_hour") or []) if h.get("fail")]
    if hours:
        lines += ["", "-- TERMINAL tasks by hour of submission (hour: failed/total) --",
                  ", ".join(f"{h['hour']:02d}h: {h['fail']}/{h['total']}" for h in hours)]

    def tasks(title, rows, fields):
        if rows:
            lines.append("")
            lines.append(f"-- {title} --")
            for t in rows[:6]:
                lines.append(" | ".join(f"{f}={t.get(f)}" for f in fields))

    tasks("Slowest completed tasks", detail.get("slowest_tasks"),
          ["task_id", "user", "provider", "queue_min", "proc_min", "total_min", "initial"])
    tasks("Most recent TERMINAL tasks", detail.get("failed_tasks"),
          ["task_id", "user", "provider", "status", "queue_min", "proc_min", "initial", "end_time"])
    tasks("Tasks still not finished", detail.get("stuck_tasks"),
          ["task_id", "user", "provider", "status", "initial", "translating"])
    return "\n".join(lines)


def heuristic_rca(detail: dict) -> RCAResponse:
    """Deterministic root cause analysis derived from the anomaly types.

    Used when Bedrock is unreachable. It is a dispatcher troubleshooting
    playbook keyed on anomaly type, narrowed with the same evidence the model
    would get — a single provider or user behind most failures, a failure peak
    at one hour of the day, and whether the time is going to queueing or to
    translation.
    """
    m = detail.get("metrics") or {}
    b = detail.get("baselines") or {}
    anomalies = detail.get("anomalies") or []
    types = [a.get("type") for a in anomalies]
    job = detail.get("job", "")
    queue_bound = (m.get("queue_share") or 0) >= 60

    causes: List[RootCause] = []
    actions: List[RCAAction] = []
    checks: List[DiagnosticCheck] = []

    def cause(text, category, likelihood, evidence):
        causes.append(RootCause(cause=text, category=category, likelihood=likelihood, evidence=evidence))

    def action(text, horizon, priority, rationale="", impact=""):
        actions.append(RCAAction(action=text, horizon=horizon, priority=priority,
                                 rationale=rationale, expected_impact=impact))

    def check(text, where=""):
        checks.append(DiagnosticCheck(check=text, where=where))

    # One playbook per anomaly type. They are applied in the order the detector
    # ranked the anomalies (most severe first), so the primary cause always
    # comes from the worst signal.
    def _terminal_spike(a):
        total = m.get("count") or 0
        prov, prov_share, prov_base = _suspect(
            detail.get("failing_providers"), detail.get("top_providers"), total)
        user, user_share, user_base = _suspect(
            detail.get("failing_users"), detail.get("top_users"), total)
        hour, hour_share = _peak_fail_hour(detail.get("failure_by_hour"))
        # How long a failing task ran before dying, taken from the failed tasks
        # themselves rather than the job's overall average.
        fail_procs = [t.get("proc_min") for t in (detail.get("failed_tasks") or [])
                      if t.get("proc_min") is not None]
        fail_proc = sum(fail_procs) / len(fail_procs) if fail_procs else None
        # No failing task even reached TRANSLATING: the strongest form of the
        # same signal, and not the same thing as "translated for 0 minutes".
        never_translated = not fail_procs and not m.get("avg_proc")
        fast_fail = never_translated or (fail_proc is not None and fail_proc <= 1)

        if prov:
            cause(f"Translator provider '{prov['name']}' is failing this job type",
                  "dependency", "high",
                  f"{prov['count']} of the failures ({round(prov_share * 100)}%) ran on "
                  f"'{prov['name']}', which handles {round(prov_base * 100)}% of this job type's "
                  f"tasks — failures are over-represented there.")
        if user:
            cause(f"Input data submitted by '{user['name']}' is being rejected by the translator",
                  "input_data", "high",
                  f"{user['count']} of the failures ({round(user_share * 100)}%) came from "
                  f"'{user['name']}', who submits {round(user_base * 100)}% of this job type's "
                  f"tasks, so the failures track that submitter's datasets or arguments rather "
                  f"than the service itself.")
        if fast_fail:
            cause("Tasks are rejected at translator start-up, not part-way through translation",
                  "configuration", "high",
                  ("No failing task ever recorded a TRANSLATING phase, "
                   if never_translated else
                   f"Failing tasks ran for only {_dur(fail_proc)} before dying, ")
                  + "the signature of a missing licence, unreachable target, bad argument or "
                    "missing input file rather than a translation that ran and broke.")
        cause("Job-type configuration or its translator is broken for this workload",
              "configuration", "medium" if fast_fail else "high",
              f"{m.get('terminal')} of {m.get('count')} tasks ({m.get('terminal_pct')}%) end "
              f"TERMINAL against a {b.get('terminal_pct')}% system baseline.")
        if hour and hour_share >= 0.4:
            cause(f"Failures cluster around {hour['hour']:02d}:00, so a scheduled batch window is "
                  f"contending for the same resource",
                  "scheduling", "medium",
                  f"{hour['fail']} of the failures ({round(hour_share * 100)}%) were submitted in "
                  f"the {hour['hour']:02d}:00 hour.")

        action(f"Pull the Module (translator) log for the sample failing task IDs of {job} and read "
               f"the first error, not the last", "immediate", "critical",
               "The dispatcher history log records that a task went TERMINAL but not why; only the "
               "translator log carries the underlying error.",
               "Identifies the exact failure reason for this job type.")
        action("Re-run one failing task manually with the translator's verbose/debug flag",
               "immediate", "high",
               "Reproducing outside the dispatcher separates a dispatcher/scheduling problem from a "
               "translator or data problem.",
               "Confirms or eliminates the translator as the cause.")
        if prov:
            action(f"Check licence availability, disk space and service health on provider "
                   f"'{prov['name']}'", "immediate", "high",
                   "Failures are over-represented on that one provider.",
                   "Restores capacity if the provider is the fault.")
        if user:
            action(f"Review a sample of '{user['name']}'s input datasets and submission arguments "
                   f"against the job's expected input", "short_term", "medium",
                   "The failures track one submitter beyond their share of traffic.",
                   "Stops the same bad input being resubmitted.")
        action(f"Add a bounded retry with backoff for {job}, and alert on its terminal rate "
               f"crossing {b.get('terminal_pct')}%", "preventive", "medium",
               "Transient provider errors currently consume the task instead of retrying.",
               "Recovers transient failures and surfaces regressions in hours, not weeks.")
        check("Module/translator log around the timestamps of the sample failing tasks",
              "Dispatcher Module logs")
        check("Job-type definition: translator command, argument list and expected input types",
              "Dispatcher module configuration XML/properties")
        check("Licence server availability and target site reachability during the failure window",
              "Licence server + target site logs")

    def _queue_backlog(a):
        p95q = _timed(m.get("p95_queue"))
        cause("Not enough translator/module slots for this job type's submission rate",
              "capacity", "high",
              f"{m.get('in_queue')} tasks ({m.get('in_queue_pct')}%) are sitting in SCHEDULED "
              f"against a {b.get('in_queue_pct')}% system baseline"
              + (f", with p95 queue wait {p95q}." if p95q else "."))
        proc = _timed(m.get("avg_proc"))
        cause("The scheduler is starving this job type behind higher-priority or longer-running work",
              "scheduling", "medium",
              f"Queueing accounts for {m.get('queue_share')}% of elapsed time"
              + (f" while processing averages only {proc}." if proc else
                 ", and no task recorded meaningful translation time."))
        cause("A provider that serves this job type is down or unregistered, so tasks are never picked up",
              "dependency", "medium",
              f"Tasks reach SCHEDULED and stop there; {m.get('active')} more are mid-flight.")
        action("Compare the module instance count for this job type against its arrival rate and "
               "raise it", "immediate", "high",
               "Backlog with short processing time is an intake-capacity problem.",
               "Drains the queue and cuts p95 queue wait.")
        action("Confirm every provider registered for this job type is running and accepting tasks",
               "immediate", "high", "A silent provider looks exactly like a capacity shortfall.",
               "Rules out a dead provider in minutes.")
        action("Stagger or throttle bulk submissions of this job type away from peak hours",
               "short_term", "medium",
               "Arrival spikes queue behind a fixed number of slots.",
               "Flattens the queue without new hardware.")
        action("Alert when SCHEDULED depth for one job type exceeds its normal band",
               "preventive", "medium", "The backlog was only visible after the fact.",
               "Turns a backlog into a page instead of a complaint.")
        check("Module instance/slot configuration and scheduler priority for this job type",
              "Dispatcher Scheduler + Module configuration")
        check("Provider registration and heartbeat status", "Dispatcher Scheduler log")

    def _latency_drift(a):
        drift = a
        cause("Load grew while capacity stayed flat, so waiting time increased"
              if queue_bound else "Translation work per task got heavier",
              "capacity" if queue_bound else "input_data", "high",
              f"Recent median completion {_dur(drift.get('value'))} vs {_dur(drift.get('baseline'))} "
              f"historically, with {m.get('queue_share')}% of elapsed time spent queueing.")
        cause("Resource contention on the translator host (CPU, memory, disk or network)",
              "infrastructure", "medium",
              f"p95 completion is {_dur(m.get('p95_total'))} against a p50 of {_dur(m.get('p50_total'))}.")
        cause("A dependency this job type calls out to has slowed down (database, target site, file store)",
              "dependency", "medium",
              f"Processing time averages {_dur(m.get('avg_proc'))} (p95 {_dur(m.get('p95_proc'))}).")
        action("Compare input dataset sizes for recent tasks against older ones",
               "immediate", "high", "Rules the workload itself in or out first.",
               "Separates 'more work' from 'less capacity'.")
        action("Check host CPU/memory/disk on the translator machines over the drift window",
               "immediate", "high", "Contention is the most common cause of gradual drift.",
               "Finds saturation directly.")
        action("Chart what else runs concurrently in that window and move overlapping batches",
               "short_term", "medium", "Drift often tracks another job type's schedule.",
               "Removes the overlap that caused the slowdown.")
        action("Track p50/p95 completion time per job type over time with a threshold alert",
               "preventive", "medium", "The drift was only found by comparing two halves after the fact.",
               "Catches the next regression as it starts.")
        check("Host resource metrics for the translator machines across the drift window", "OS/host monitoring")
        check("Input dataset sizes of recent vs older tasks for this job type", "Dispatcher staging area")

    def _latency_outlier(a):
        if queue_bound:
            cause("This job type waits far longer than it runs — it is under-served, not slow",
                  "scheduling", "high",
                  f"{m.get('queue_share')}% of a {_dur(m.get('p50_total'))} median lifecycle is queue "
                  f"wait; the system median is {_dur(b.get('p50_total'))}.")
            action("Give this job type its own module instances or a higher scheduler priority",
                   "short_term", "high", "The translation itself is not the bottleneck.",
                   f"Cuts median completion toward the {_dur(m.get('avg_proc'))} of actual work.")
        else:
            cause("The translation itself is expensive for this job type",
                  "input_data", "high",
                  f"Processing averages {_dur(m.get('avg_proc'))} (p95 {_dur(m.get('p95_proc'))}) with "
                  f"only {m.get('queue_share')}% of time spent queueing.")
            action("Profile one representative task end to end and size the translator host for it",
                   "short_term", "high", "Cost is in translation, so tuning the queue will not help.",
                   "Targets the real bottleneck.")
        check("Per-job-type slot allocation vs its arrival rate", "Dispatcher Scheduler configuration")

    def _stalled_tasks(a):
        stalled = a
        cause("Tasks were orphaned by a dispatcher/module restart and never reconciled",
              "infrastructure", "high",
              f"{stalled.get('value')} task(s) hold a non-final state past the "
              f"{_dur(stalled.get('baseline'))} threshold for this job type.")
        cause("The module that owned these tasks died mid-translation and left no end state",
              "infrastructure", "medium",
              "The tasks have a start transition but no COMPLETE/TERMINAL transition.")
        cause("Tasks are blocked on a lock or an unavailable target and will never self-clear",
              "dependency", "medium",
              f"{m.get('in_queue')} are still SCHEDULED and {m.get('active')} are mid-flight.")
        action("List the stalled task IDs and reconcile their state, then requeue or cancel them "
               "explicitly", "immediate", "critical",
               "They occupy slots and skew every average until cleared.",
               "Frees capacity and makes the metrics trustworthy.")
        action("Check whether the dispatcher services restarted around these tasks' start times",
               "immediate", "high", "A restart is the usual origin of orphaned tasks.",
               "Confirms the cause in one look.")
        action("Enable a stale-task timeout so tasks fail loudly instead of hanging forever",
               "preventive", "high", "Nothing currently ages these out.",
               "Bounds the damage of the next crash.")
        check("Dispatcher service start/stop times against the stalled tasks' INITIAL timestamps",
              "DispatcherClient / Scheduler / Module logs")
        check("Whether the stalled task IDs still exist as live tasks in the dispatcher queue",
              "Dispatcher admin/queue listing")

    def _high_variance(a):
        if causes:
            return  # something more specific already explains this job type
        cause("This job type mixes very different workload sizes under one name",
              "input_data", "medium",
              f"p95 {_dur(m.get('p95_total'))} vs p50 {_dur(m.get('p50_total'))}.")
        action("Split or tag this job type by workload size so its SLA is meaningful",
               "short_term", "medium", "One average cannot describe a 10x spread.",
               "Makes per-job alerting possible.")
        check("Input sizes of the slowest tasks against the median task", "Dispatcher staging area")

    playbook = {
        "terminal_spike": _terminal_spike,
        "queue_backlog": _queue_backlog,
        "latency_drift": _latency_drift,
        "latency_outlier": _latency_outlier,
        "stalled_tasks": _stalled_tasks,
        "high_variance": _high_variance,
    }
    for anomaly in anomalies:
        handler = playbook.get(anomaly.get("type"))
        if handler:
            handler(anomaly)

    p50 = _timed(m.get("p50_total"))
    timing_note = (f"median completion {p50} vs a {_dur(b.get('p50_total'))} system median"
                   if p50 else "no task in this window recorded a completion time")

    if not anomalies:
        return RCAResponse(
            status="heuristic", source="rules", job=job,
            severity=detail.get("severity", "ok"), anomaly_types=[],
            summary=f"{job} shows no anomalies in this window: "
                    f"{m.get('completed_pct')}% completed, {m.get('terminal_pct')}% terminal "
                    f"(baseline {b.get('terminal_pct')}%), {timing_note}.",
            primary_cause="No anomaly detected — nothing to attribute.",
            impact="None observed for this job type in the selected window.",
            confidence=0.4,
        )

    worst = anomalies[0]
    return RCAResponse(
        status="heuristic", source="rules", job=job,
        severity=detail.get("severity", "ok"), anomaly_types=[t for t in types if t],
        summary=f"{job}: {len(anomalies)} anomaly signal(s) — "
                + "; ".join(a.get("title", "") for a in anomalies[:3])
                + ". "
                + (f"Of the elapsed time, {m.get('queue_share')}% is queue wait and the rest is "
                   f"translation, so the evidence points "
                   + ("at scheduling/capacity rather than the translator."
                      if queue_bound else "at the translation step rather than the queue.")
                   if p50 else
                   "No task recorded a full INITIAL-to-end lifecycle, so the split between "
                   "queueing and translation cannot be attributed from this window."),
        primary_cause=causes[0].cause if causes else worst.get("title", ""),
        impact=f"{m.get('terminal')} failed and {m.get('in_queue') + m.get('active')} unfinished "
               f"tasks out of {m.get('count')} for this job type; {timing_note}.",
        causes=causes, actions=actions, checks=checks,
        confidence=0.55,
    )
