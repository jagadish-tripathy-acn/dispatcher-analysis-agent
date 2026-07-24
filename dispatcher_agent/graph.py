"""
Dispatcher AI Analysis Graph
============================

A single-file LangGraph workflow that turns the `/api/stats` payload into a
structured AI analysis using AWS Bedrock (Claude, via the Converse API).

Usage:
    graph = DispatcherGraph()                       # uses DEFAULT_MODEL
    graph = DispatcherGraph("us.anthropic.claude-sonnet-4-6")  # or override
    response = graph.invoke(stats_payload)           # -> AnalysisResponse

Workflow:  validate -> analyze -> detect_anomalies -> recommend -> summarize -> assemble
Invalid/empty payloads short-circuit straight to `assemble` with status="error".
Any step that fails is caught and degrades the response to status="partial"
instead of raising -- the graph never crashes on bad input or a bad model call.
"""
from __future__ import annotations

import logging
from typing import Any, List, Literal, Optional, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MIN_TASKS_FOR_HEALTH_SCORE = 5

BASE_RULES = (
    "You are a senior site reliability engineer analyzing telemetry from a Teamcenter "
    "Dispatcher job-processing system. Use ONLY the data given below -- never invent "
    "metrics, jobs, users, or timestamps. If something needed is missing, say so instead "
    "of guessing. Respond only via the requested structured schema, no extra text."
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


class SystemAnalysis(BaseModel):
    """Result of the analysis step."""

    detailed_analysis: str
    health_assessment: str


class AnomalyList(BaseModel):
    anomalies: List[Anomaly] = Field(default_factory=list)


class RecommendationList(BaseModel):
    recommendations: List[Recommendation] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    executive_summary: str
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
    analysis: Optional[SystemAnalysis]
    anomalies: List[Anomaly]
    recommendations: List[Recommendation]
    summary: Optional[ExecutiveSummary]
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
        state: GraphState = {"stats": stats or {}, "anomalies": [], "recommendations": []}
        result = self._graph.invoke(state)
        return result["response"]

    # -- graph wiring ------------------------------------------------------
    def _build_graph(self):
        g = StateGraph(GraphState)
        g.add_node("validate", self._validate)
        g.add_node("analyze", self._analyze)
        g.add_node("detect_anomalies", self._detect_anomalies)
        g.add_node("recommend", self._recommend)
        g.add_node("summarize", self._summarize)
        g.add_node("assemble", self._assemble)

        g.add_edge(START, "validate")
        g.add_conditional_edges(
            "validate",
            lambda s: "analyze" if s.get("is_valid") else "assemble",
            {"analyze": "analyze", "assemble": "assemble"},
        )
        g.add_edge("analyze", "detect_anomalies")
        g.add_edge("detect_anomalies", "recommend")
        g.add_edge("recommend", "summarize")
        g.add_edge("summarize", "assemble")
        g.add_edge("assemble", END) 
        return g.compile()

    # -- nodes (each does exactly one step) ---------------------------------
    def _validate(self, state: GraphState) -> dict:
        stats = state.get("stats")
        if not isinstance(stats, dict) or not stats.get("has_data"):
            return {"is_valid": False, "error": "Payload is missing or has no data to analyze."}
        return {"is_valid": True}

    def _analyze(self, state: GraphState) -> dict:
        try:
            analysis = self._ask(BASE_RULES, self._analysis_prompt(state["stats"]), SystemAnalysis)
        except Exception as exc:
            logger.error("analyze failed: %s", exc)
            return {"analysis": None, "error": f"Analysis failed: {exc}"}
        return {"analysis": analysis}

    def _detect_anomalies(self, state: GraphState) -> dict:
        analysis = state.get("analysis")
        if not analysis:
            return {"anomalies": []}
        prompt = f"{self._context(state['stats'])}\n\nAnalysis:\n{analysis.detailed_analysis}"
        try:
            result = self._ask(BASE_RULES, prompt, AnomalyList)
            return {"anomalies": result.anomalies}
        except Exception as exc:
            logger.error("detect_anomalies failed: %s", exc)
            return {"anomalies": []}

    def _recommend(self, state: GraphState) -> dict:
        analysis = state.get("analysis")
        if not analysis:
            return {"recommendations": []}
        anomaly_txt = self._bullets(state.get("anomalies", []), "severity")
        prompt = (
            f"{self._context(state['stats'])}\n\nAnalysis:\n{analysis.detailed_analysis}"
            f"\n\nAnomalies:\n{anomaly_txt}"
        )
        try:
            result = self._ask(BASE_RULES, prompt, RecommendationList)
            return {"recommendations": result.recommendations}
        except Exception as exc:
            logger.error("recommend failed: %s", exc)
            return {"recommendations": []}

    def _summarize(self, state: GraphState) -> dict:
        analysis = state.get("analysis")
        if not analysis:
            return {"summary": None}
        stats = state["stats"]
        sample_size = (stats.get("kpis") or {}).get("total_tasks", 0)
        note = (
            "Sample size is sufficient for a numeric health score."
            if sample_size >= MIN_TASKS_FOR_HEALTH_SCORE
            else "Sample size is too small for a reliable health score -- return null for health_score."
        )
        prompt = (
            f"{self._context(stats)}\n\nAnalysis:\n{analysis.detailed_analysis}"
            f"\n\nAnomalies:\n{self._bullets(state.get('anomalies', []), 'severity')}"
            f"\n\nRecommendations:\n{self._bullets(state.get('recommendations', []), 'priority')}"
            f"\n\n{note}"
        )
        try:
            summary = self._ask(BASE_RULES, prompt, ExecutiveSummary)
        except Exception as exc:
            logger.error("summarize failed: %s", exc)
            summary = None
        return {"summary": summary}

    def _assemble(self, state: GraphState) -> dict:
        if not state.get("is_valid"):
            return {"response": AnalysisResponse(status="error", error=state.get("error") or "Invalid input.")}

        analysis, summary = state.get("analysis"), state.get("summary")
        if not analysis or not summary:
            return {
                "response": AnalysisResponse(
                    status="partial",
                    executive_summary="Analysis could not be fully generated.",
                    detailed_analysis=analysis.detailed_analysis if analysis else "",
                    health_assessment=analysis.health_assessment if analysis else "",
                    anomalies=state.get("anomalies", []),
                    recommendations=state.get("recommendations", []),
                    error=state.get("error") or "One or more analysis steps failed.",
                )
            }

        return {
            "response": AnalysisResponse(
                status="success",
                executive_summary=summary.executive_summary,
                detailed_analysis=analysis.detailed_analysis,
                health_assessment=analysis.health_assessment,
                anomalies=state.get("anomalies", []),
                recommendations=state.get("recommendations", []),
                confidence=summary.confidence,
                health_score=summary.health_score,
                reasoning=summary.reasoning,
            )
        }

    # -- small helpers -------------------------------------------------------
    def _ask(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> BaseModel:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        return self.model.with_structured_output(schema).invoke(messages)

    @staticmethod
    def _analysis_prompt(stats: dict) -> str:
        return (
            f"{DispatcherGraph._context(stats)}\n\n"
            "Write a detailed technical analysis (throughput, latency, failures, stuck tasks, "
            "load concentration) and an overall health assessment."
        )

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

    @staticmethod
    def _bullets(items: List[Any], attr: str) -> str:
        return "\n".join(f"- [{getattr(i, attr)}] {i.title}" for i in items) or "None."
