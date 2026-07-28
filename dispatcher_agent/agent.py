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

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MIN_TASKS_FOR_HEALTH_SCORE = 5

SYSTEM_PROMPT = (
    "You are a senior site reliability engineer analyzing telemetry from a Teamcenter "
    "Dispatcher job-processing system. Use ONLY the data given below -- never invent metrics, "
    "jobs, users, or timestamps. If something needed is missing, say so instead of guessing. "
    "Respond only via the requested structured schema, no extra text.\n\n"
    "Produce, in this single response:\n"
    "1. detailed_analysis: a multi-paragraph technical analysis covering throughput, "
    "queue/processing latency, failure and stuck-task rates, and load concentration across "
    "jobs/groups/users.\n"
    "2. health_assessment: a concise statement of overall system health.\n"
    "3. anomalies: notable deviations or warning signs worth investigating, each with a severity.\n"
    "4. recommendations: 3-6 concrete, prioritized actions, each traceable to the analysis or an "
    "anomaly above.\n"
    "5. executive_summary: STRICT LIMIT of 1-2 short sentences (max ~40 words total). A "
    "plain-language headline naming the overall health and, if unhealthy, only the single "
    "biggest issue. Do not enumerate metrics, anomalies, or recommendations here -- that detail "
    "belongs in detailed_analysis/anomalies/recommendations, not the summary.\n"
    "6. reasoning: 2-4 sentences on how you reached these conclusions.\n"
    "7. confidence: 0.0-1.0 based on how complete the input data was.\n"
    "8. health_score: 0.0-100.0 overall score, or null if the sample size is too small to be "
    "told sufficient below."
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

    detailed_analysis: str
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
    detailed_analysis: str = ""
    health_assessment: str = ""
    anomalies: List[Anomaly] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    confidence: float = 0.0
    health_score: Optional[float] = None
    reasoning: str = ""
    error: Optional[str] = None


class GraphState(TypedDict, total=False):
    stats: dict
    is_valid: bool
    error: Optional[str]
    result: Optional[AnalysisResult]
    response: AnalysisResponse


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
class DispatcherGraph:
    """LangGraph-based AI analysis engine for dispatcher system statistics."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = ChatBedrockConverse(model=model_name)
        self._graph = self._build_graph()

    def invoke(self, stats: dict) -> AnalysisResponse:
        """Run the full workflow over a `/api/stats`-shaped payload."""
        state: GraphState = {"stats": stats or {}}
        result = self._graph.invoke(state)
        return result["response"]

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
                detailed_analysis=result.detailed_analysis,
                health_assessment=result.health_assessment,
                anomalies=result.anomalies,
                recommendations=result.recommendations,
                confidence=result.confidence,
                health_score=result.health_score,
                reasoning=result.reasoning,
            )
        }

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
            f"Stuck: {k.get('stuck', 0)} ({k.get('stuck_pct', 0)}%)",
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
        stuck = stats.get("stuck_tasks") or []
        if stuck:
            lines.append(
                f"Stuck tasks ({len(stuck)} total, sample): "
                + ", ".join(f"{t.get('task_id')}/{t.get('status')}/{t.get('job')}" for t in stuck[:10])
            )
        return "\n".join(lines)
