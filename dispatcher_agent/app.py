"""
Dispatcher Insights web app.

Serves an interactive HTML page (the replacement for DispInsights.pbix). All the
work is done by the DispatcherLogAgent in agent.py, which reads the logs in
Logs\\ directly — no Perl, no CSV.

Run:  python app.py   ->  http://127.0.0.1:5710/
"""
from datetime import datetime

from flask import Flask, jsonify, render_template

import agent

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():
    """The full stats payload the page renders."""
    data = agent.analyse()
    data["generated_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, port=5710)
