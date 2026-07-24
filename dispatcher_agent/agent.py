"""
Dispatcher Log Agent
====================

A self-contained agent that reads Teamcenter Dispatcher client history logs
straight from the Logs\\ folder, reconstructs each task's lifecycle, and
computes the statistics that the DispInsights.pbix report used to visualise.

It replaces BOTH Perl scripts (parse_dispatcher_data.pl +
generate_complete_task_analysis.pl) with one Python component, so there is no
CSV middleman. The web layer (app.py) just calls DispatcherLogAgent().run().

Log line format (comma-separated payload after a log prefix):
    <logtime>,<ms> INFO  - <time> - TaskID,Status,Site,JobName,0,User,Group,
        Provider,Date,1,Target,0,3, <State,Timestamp> <State,Timestamp> ...
The trailing (State, Timestamp) pairs (from index 13 on) are the lifecycle
transitions, e.g. INITIAL/... PREPARING/... SCHEDULED/... TRANSLATING/...
LOADING/... COMPLETE/...
"""
import glob
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

# ---- config --------------------------------------------------------------
# dispatcher_agent/  ->  parent is the Disp_Analysis_PERL root that holds Logs\
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(ROOT, "Logs")

# Valid dispatcher lifecycle states.
STATES = {
    "INITIAL", "PREPARING", "SCHEDULED", "TRANSLATING", "LOADING",
    "COMPLETE", "TERMINAL", "DELETE", "DUPLICATE",
}
TERMINAL_STATES = {"COMPLETE", "TERMINAL", "DELETE", "DUPLICATE"}

# Field positions inside the comma-split payload.
IDX_TASK_ID, IDX_STATUS, IDX_SITE, IDX_JOB = 0, 1, 2, 3
IDX_USER, IDX_GROUP, IDX_PROVIDER = 5, 6, 7
IDX_TARGET = 10
IDX_STATES_START = 13  # trailing (state, timestamp) pairs begin here

# Pull the payload (everything after "<time> - ") off the log prefix.
_LINE_RE = re.compile(
    r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\s+\w+\s+-\s+"
    r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+-\s+(?P<payload>.+)$"
)

QUEUE_BUCKETS = ["<10", "10-30", "30-60", ">60"]


def _parse_dt(value):
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _bucket(minutes):
    if minutes < 10:
        return "<10"
    if minutes < 30:
        return "10-30"
    if minutes < 60:
        return "30-60"
    return ">60"


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


class DispatcherLogAgent:
    """Parses dispatcher logs and produces a stats dict for the dashboard."""

    def __init__(self, log_dir=DEFAULT_LOG_DIR):
        self.log_dir = log_dir
        # Per-task accumulator.
        self.tasks = {}          # task_id -> meta dict
        self.state_times = defaultdict(dict)  # task_id -> {state: earliest datetime}
        self.files_read = []

    # -- ingest -----------------------------------------------------------
    def _ingest_line(self, line):
        m = _LINE_RE.match(line.strip())
        if not m:
            return
        parts = [p.strip() for p in m.group("payload").split(",")]
        if len(parts) <= IDX_TARGET:
            return

        task_id = parts[IDX_TASK_ID]
        if not task_id:
            return

        # Static metadata (first non-empty wins).
        meta = self.tasks.setdefault(task_id, {
            "task_id": task_id, "job": "", "user": "", "group": "",
            "target": "", "provider": "", "site": "", "current_status": "",
            "last_seen": None,
        })
        for key, idx in (("job", IDX_JOB), ("user", IDX_USER), ("group", IDX_GROUP),
                         ("target", IDX_TARGET), ("provider", IDX_PROVIDER), ("site", IDX_SITE)):
            if not meta[key] and idx < len(parts):
                meta[key] = parts[idx]

        # Current status = status on the most recent line for this task.
        line_time = _parse_dt(parts[8]) if len(parts) > 8 else None
        cur_status = parts[IDX_STATUS] if IDX_STATUS < len(parts) else ""
        if cur_status in STATES and (meta["last_seen"] is None or (line_time and line_time >= meta["last_seen"])):
            meta["current_status"] = cur_status
            if line_time:
                meta["last_seen"] = line_time

        # Trailing (state, timestamp) pairs -> keep earliest time per state.
        i = IDX_STATES_START
        while i + 1 < len(parts):
            state, ts = parts[i], parts[i + 1]
            if state in STATES:
                dt = _parse_dt(ts)
                if dt:
                    existing = self.state_times[task_id].get(state)
                    if existing is None or dt < existing:
                        self.state_times[task_id][state] = dt
                i += 2
            else:
                i += 1

    def ingest(self):
        pattern = os.path.join(self.log_dir, "*.log")
        for path in sorted(glob.glob(pattern)):
            self.files_read.append(os.path.basename(path))
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    self._ingest_line(line)
        return self

    # -- derive per-task timings -----------------------------------------
    def _task_timings(self):
        rows = []
        for task_id, meta in self.tasks.items():
            states = self.state_times.get(task_id, {})
            initial = states.get("INITIAL")
            translating = states.get("TRANSLATING")
            end_state = next((s for s in ("COMPLETE", "TERMINAL", "DELETE", "DUPLICATE") if s in states), None)
            end_time = states.get(end_state) if end_state else None

            queue = proc = total = None
            if initial and translating:
                queue = max(0, int((translating - initial).total_seconds() // 60))
            if translating and end_time:
                proc = max(0, int((end_time - translating).total_seconds() // 60))
            if initial and end_time:
                total = max(0, int((end_time - initial).total_seconds() // 60))

            rows.append({
                **meta,
                "initial": initial, "translating": translating,
                "end_state": end_state, "end_time": end_time,
                "queue_min": queue, "proc_min": proc, "total_min": total,
                "completed": end_state is not None,
            })
        return rows

    # -- public: compute the full stats payload --------------------------
    def run(self):
        rows = self._task_timings()
        total_tasks = len(rows)
        # "ended" = reached any lifecycle end state (COMPLETE/TERMINAL/DELETE/DUPLICATE);
        # used for timing stats since queue/proc/total times are meaningful regardless of outcome.
        ended = [r for r in rows if r["completed"]]
        stuck = [r for r in rows if not r["completed"]]
        # "Completed" KPI = successful completions only; "Terminal jobs" = failed/aborted only.
        # These two are mutually exclusive (unlike DELETE/DUPLICATE, which aren't broken out).
        succeeded = [r for r in ended if r["current_status"] == "COMPLETE"]
        terminal = [r for r in ended if r["current_status"] == "TERMINAL"]

        # KPIs
        queue_vals = [r["queue_min"] for r in ended]
        proc_vals = [r["proc_min"] for r in ended]
        total_vals = [r["total_min"] for r in ended]

        # Status distribution (by each task's current status)
        status_counts = Counter(r["current_status"] or "UNKNOWN" for r in rows)

        # Queue-time buckets
        bucket_counts = Counter(_bucket(r["queue_min"]) for r in ended if r["queue_min"] is not None)
        queue_distribution = [{"range": b, "count": bucket_counts.get(b, 0)} for b in QUEUE_BUCKETS]

        # By job type
        by_job = defaultdict(lambda: {"count": 0, "queue": [], "proc": [], "total": []})
        for r in rows:
            j = by_job[r["job"] or "unknown"]
            j["count"] += 1
            j["queue"].append(r["queue_min"])
            j["proc"].append(r["proc_min"])
            j["total"].append(r["total_min"])
        job_breakdown = sorted(
            ({"job": name, "count": d["count"], "avg_queue": _avg(d["queue"]),
              "avg_proc": _avg(d["proc"]), "avg_total": _avg(d["total"])}
             for name, d in by_job.items()),
            key=lambda x: x["count"], reverse=True,
        )

        # By group and by user (top contributors)
        group_counts = Counter(r["group"] or "unknown" for r in rows)
        user_counts = Counter(r["user"] or "unknown" for r in rows)

        # Throughput: completions per minute (end_time)
        per_min = Counter()
        for r in ended:
            if r["end_time"]:
                per_min[r["end_time"].strftime("%Y-%m-%d %H:%M")] += 1
        throughput = [{"t": t, "count": per_min[t]} for t in sorted(per_min)]

        # Task detail table (cap for payload size; page can paginate later)
        def _fmt(dt):
            return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
        task_rows = [{
            "task_id": r["task_id"], "job": r["job"], "user": r["user"],
            "group": r["group"], "target": r["target"], "provider": r["provider"],
            "status": r["current_status"], "initial": _fmt(r["initial"]),
            "translating": _fmt(r["translating"]), "end_state": r["end_state"] or "",
            "end_time": _fmt(r["end_time"]), "queue_min": r["queue_min"],
            "proc_min": r["proc_min"], "total_min": r["total_min"],
            "completed": r["completed"],
        } for r in sorted(rows, key=lambda x: (x["total_min"] is None, -(x["total_min"] or 0)))]

        return {
            "files_read": self.files_read,
            "has_data": total_tasks > 0,
            "kpis": {
                "total_tasks": total_tasks,
                "completed": len(succeeded),
                "stuck": len(stuck),
                "stuck_pct": round(100 * len(stuck) / total_tasks, 1) if total_tasks else 0.0,
                "terminal_jobs": len(terminal),
                "terminal_pct": round(100 * len(terminal) / total_tasks, 1) if total_tasks else 0.0,
                "avg_queue_min": _avg(queue_vals),
                "max_queue_min": max([q for q in queue_vals if q is not None], default=0),
                "avg_proc_min": _avg(proc_vals),
                "avg_total_min": _avg(total_vals),
            },
            "status_distribution": [{"status": s, "count": c} for s, c in status_counts.most_common()],
            "queue_distribution": queue_distribution,
            "job_breakdown": job_breakdown,
            "group_distribution": [{"group": g, "count": c} for g, c in group_counts.most_common(10)],
            "user_distribution": [{"user": u, "count": c} for u, c in user_counts.most_common(10)],
            "throughput": throughput,
            "stuck_tasks": [{"task_id": r["task_id"], "job": r["job"],
                             "status": r["current_status"], "user": r["user"]} for r in stuck],
            "tasks": task_rows,
        }


def analyse(log_dir=DEFAULT_LOG_DIR):
    """Convenience entry point used by the web layer."""
    return DispatcherLogAgent(log_dir).ingest().run()


if __name__ == "__main__":
    import json
    print(json.dumps(analyse(), indent=2))
