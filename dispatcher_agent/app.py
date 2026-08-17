"""
Dispatcher Insights web app.

Serves an interactive HTML page (the replacement for DispInsights.pbix). All the
work is done by the DispatcherLogParser in parser.py, which reads the logs in
Logs\\ directly — no Perl, no CSV. AI analysis of that same stats payload is
delegated to the DispatcherGraph LangGraph agent in agent.py.

Ingesting the logs is the expensive step (seconds), while aggregating a date
window out of the ingested tasks is fast. So the parsed log set is cached in
memory and only re-ingested when the files on disk change, which is what makes
per-job drill-down and date filtering feel instant.

Routes:
    /                 the dashboard page
    /api/stats        system + per-job-type health for a date window
    /api/job          one job type in depth (drill-down charts and evidence)
    /api/rca          root cause analysis + recommended actions for a job type
    /api/tasks        filtered, sorted, paged task rows
    /api/tasks.csv    the same selection as a CSV download
    /api/analysis     AI analysis of the whole snapshot

Run:  python app.py   ->  http://127.0.0.1:5710/
"""
import csv
import io
import logging
import threading
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request

import parser
from agent import DispatcherGraph, heuristic_rca

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PAGE = 2000          # hard cap on rows one /api/tasks call will return
STATS_CACHE_SIZE = 24    # date windows to keep aggregated

# Built lazily on first use: constructing it touches AWS/Bedrock client setup,
# which shouldn't block app startup or the (fast, AI-free) /api/stats route.
_analysis_graph: DispatcherGraph | None = None

# Ingested logs, reused across requests until the files on disk change.
_parser_lock = threading.Lock()
_parser_cache: dict = {"sig": None, "parser": None}
_stats_cache: dict = {}


def _get_analysis_graph() -> DispatcherGraph:
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = DispatcherGraph()
    return _analysis_graph


def _get_parser() -> parser.DispatcherLogParser:
    """The ingested log set, re-read only when a log file changed."""
    sig = parser.signature()
    with _parser_lock:
        if _parser_cache["sig"] != sig:
            logger.info("ingest.start files=%d", len(parser.log_files()))
            started = datetime.now()
            _parser_cache["parser"] = parser.DispatcherLogParser().ingest()
            _parser_cache["sig"] = sig
            _stats_cache.clear()
            logger.info("ingest.done seconds=%.1f", (datetime.now() - started).total_seconds())
        return _parser_cache["parser"]


def _window():
    """The (from, to) date window requested, as YYYY-MM-DD strings or None."""
    return request.args.get("from") or None, request.args.get("to") or None


def _stats(date_from, date_to) -> dict:
    """Aggregated payload for a window, memoised per (log set, window)."""
    lp = _get_parser()
    key = (_parser_cache["sig"], date_from, date_to)
    if key not in _stats_cache:
        if len(_stats_cache) >= STATS_CACHE_SIZE:
            _stats_cache.pop(next(iter(_stats_cache)))
        _stats_cache[key] = lp.run(date_from=date_from, date_to=date_to)
    return _stats_cache[key]


def _int_arg(name, default, lo, hi):
    try:
        return max(lo, min(hi, int(request.args.get(name, default))))
    except (TypeError, ValueError):
        return default


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/workflows")
def workflows():
    """Workflow (EPM) analysis. Self-contained page with its own sample data —
    it does not read the Dispatcher logs, so it needs no window arguments."""
    return render_template("workflows.html")


@app.route("/api/stats")
def api_stats():
    """System KPIs, per-job-type health boxes and the charts' source data."""
    date_from, date_to = _window()
    data = dict(_stats(date_from, date_to))
    data["generated_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return jsonify(data)


@app.route("/api/job")
def api_job():
    """One job type in depth: timing series, distributions and RCA evidence."""
    job = request.args.get("job")
    if not job:
        return jsonify({"error": "job parameter is required"}), 400
    date_from, date_to = _window()
    detail = _get_parser().job_detail(job, date_from=date_from, date_to=date_to)
    if detail is None:
        return jsonify({"error": f"No tasks for job type '{job}' in this window."}), 404
    return jsonify(detail)


@app.route("/api/rca")
def api_rca():
    """Root cause analysis and recommended actions for one job type.

    Tries the Bedrock-backed RCA graph first and falls back to the rules-based
    playbook in agent.heuristic_rca, so the dashboard's "why is this flagged?"
    action always returns something usable even with no AWS credentials.
    """
    job = request.args.get("job")
    if not job:
        return jsonify({"error": "job parameter is required"}), 400
    date_from, date_to = _window()
    detail = _get_parser().job_detail(job, date_from=date_from, date_to=date_to)
    if detail is None:
        return jsonify({"error": f"No tasks for job type '{job}' in this window."}), 404

    if request.args.get("mode") == "rules":
        return jsonify(heuristic_rca(detail).model_dump(mode="json"))
    try:
        response = _get_analysis_graph().rca(detail)
    except Exception as exc:
        logger.warning("api_rca.ai_unavailable job=%s err=%s", job, exc)
        fallback = heuristic_rca(detail)
        fallback.error = f"AI root cause analysis unavailable: {exc}"
        response = fallback
    return jsonify(response.model_dump(mode="json"))


@app.route("/api/tasks")
def api_tasks():
    """Filtered, sorted, paged task rows for the task-detail table."""
    date_from, date_to = _window()
    result = _get_parser().query_tasks(
        date_from=date_from, date_to=date_to,
        job=request.args.get("job") or None,
        status=request.args.get("status") or None,
        q=request.args.get("q") or None,
        sort=request.args.get("sort") or "slowest",
        limit=_int_arg("limit", 200, 1, MAX_PAGE),
        offset=_int_arg("offset", 0, 0, 10_000_000),
    )
    return jsonify(result)


@app.route("/api/tasks.csv")
def api_tasks_csv():
    """The current task selection as a CSV download (no row cap)."""
    date_from, date_to = _window()
    result = _get_parser().query_tasks(
        date_from=date_from, date_to=date_to,
        job=request.args.get("job") or None,
        status=request.args.get("status") or None,
        q=request.args.get("q") or None,
        sort=request.args.get("sort") or "slowest",
        limit=None,
    )
    fields = ["task_id", "job", "user", "group", "target", "provider", "site",
              "status", "end_state", "queue_min", "proc_min", "total_min",
              "initial", "translating", "end_time"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(result["rows"])
    name = f"dispatcher_tasks_{request.args.get('job') or 'all'}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.route("/api/analysis")
def api_analysis():
    """AI-generated analysis of the current stats snapshot.

    Kept as a separate, on-demand route from /api/stats so a slow/unavailable
    LLM never blocks the dashboard's core charts.

    Optional query params:
      from=YYYY-MM-DD  — include only tasks on/after this date
      to=YYYY-MM-DD    — include only tasks on/before this date
    """
    date_from, date_to = _window()
    stats = _stats(date_from, date_to)
    try:
        response = _get_analysis_graph().invoke(stats)
    except Exception:
        logger.exception("api_analysis.unexpected_failure")
        return jsonify({"status": "error", "error": "AI analysis is temporarily unavailable."}), 503
    return jsonify(response.model_dump(mode="json"))


if __name__ == "__main__":
    app.run(debug=True, port=5710)
