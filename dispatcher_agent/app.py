"""
Dispatcher Insights web app.

Serves an interactive HTML page (the replacement for DispInsights.pbix). All the
work is done by the DispatcherLogAgent in agent.py, which reads the logs in
Logs\\ directly — no Perl, no CSV. AI analysis of that same stats payload is
delegated to the DispatcherGraph LangGraph workflow in graph.py.

Run:  python app.py   ->  http://127.0.0.1:5710/
"""
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template

import agent
from graph import DispatcherGraph

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Built lazily on first use: constructing it touches AWS/Bedrock client setup,
# which shouldn't block app startup or the (fast, AI-free) /api/stats route.
_analysis_graph: DispatcherGraph | None = None


def _get_analysis_graph() -> DispatcherGraph:
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = DispatcherGraph()
    return _analysis_graph


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():
    """The full stats payload the page renders."""
    data = agent.analyse()
    data["generated_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return jsonify(data)


@app.route("/api/analysis")
def api_analysis():
    """AI-generated analysis of the current stats snapshot.

    Re-parses the logs (via agent.analyse()) and runs the result through the
    DispatcherGraph LangGraph workflow. Kept as a separate, on-demand route
    from /api/stats so a slow/unavailable LLM never blocks the dashboard's
    core charts.
    """
    stats = agent.analyse()
    try:
        response = _get_analysis_graph().invoke(stats)
    except Exception:
        logger.exception("api_analysis.unexpected_failure")
        return jsonify({"status": "error", "error": "AI analysis is temporarily unavailable."}), 503
    return jsonify(response.model_dump(mode="json"))


if __name__ == "__main__":
    app.run(debug=True, port=5710)
