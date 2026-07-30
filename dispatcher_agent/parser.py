"""
Dispatcher Log Parser
=====================

A self-contained parser that reads Teamcenter Dispatcher client history logs
straight from the Logs\\ folder, reconstructs each task's lifecycle, and
computes the statistics that the DispInsights.pbix report used to visualise.

It replaces BOTH Perl scripts (parse_dispatcher_data.pl +
generate_complete_task_analysis.pl) with one Python component, so there is no
CSV middleman. The web layer (app.py) just calls DispatcherLogParser().run().

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
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_HERE, "config.json")

def _load_log_dirs():
    """Return list of resolved log directory paths from config.json."""
    import json
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        dirs = cfg.get("log_dirs") or []
        resolved = []
        for d in dirs:
            # Relative paths are resolved from the config file's directory.
            p = d if os.path.isabs(d) else os.path.normpath(os.path.join(_HERE, d))
            resolved.append(p)
        if resolved:
            return resolved
    except (FileNotFoundError, ValueError):
        pass
    # Fallback: sibling Logs\ folder next to the project root
    ROOT = os.path.dirname(_HERE)
    return [os.path.join(ROOT, "Logs")]

LOG_DIRS = _load_log_dirs()

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


class DispatcherLogParser:
    """Parses dispatcher logs and produces a stats dict for the dashboard."""

    def __init__(self, log_dirs=None):
        self.log_dirs = log_dirs if log_dirs is not None else LOG_DIRS
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
        paths = sorted(p for d in self.log_dirs for p in glob.glob(os.path.join(d, "*.log")))
        for path in paths:
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
    def run(self, date_from=None, date_to=None):
        rows = self._task_timings()

        # Apply optional date window based on the task's initial (start) timestamp.
        if date_from or date_to:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
            dt_to   = datetime.strptime(date_to,   "%Y-%m-%d").replace(hour=23, minute=59, second=59) if date_to else None
            def _in_window(r):
                dt = r["initial"] or r["end_time"]
                if not dt:
                    return dt_from is None  # undated: include only when no lower bound
                if dt_from and dt < dt_from: return False
                if dt_to   and dt > dt_to:   return False
                return True
            rows = [r for r in rows if _in_window(r)]

        total_tasks = len(rows)
        # "ended" = reached any lifecycle end state (COMPLETE/TERMINAL/DELETE/DUPLICATE);
        # used for timing stats since queue/proc/total times are meaningful regardless of outcome.
        OTHER_STATES = {"INITIAL", "PREPARING", "TRANSLATING", "LOADING"}
        ended     = [r for r in rows if r["completed"]]
        in_queue  = [r for r in rows if r["current_status"] == "SCHEDULED"]
        succeeded = [r for r in rows if r["end_state"] == "COMPLETE"]
        terminal  = [r for r in rows if r["end_state"] == "TERMINAL"]
        deleted   = [r for r in rows if r["end_state"] == "DELETE"]
        duplicate = [r for r in rows if r["end_state"] == "DUPLICATE"]
        other     = [r for r in rows if r["current_status"] in OTHER_STATES]

        # KPIs
        queue_vals = [r["queue_min"] for r in ended]
        proc_vals = [r["proc_min"] for r in ended]
        total_vals = [r["total_min"] for r in succeeded]

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
            if r["end_state"] == "COMPLETE":
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
                "in_queue": len(in_queue),
                "in_queue_pct": round(100 * len(in_queue) / total_tasks, 1) if total_tasks else 0.0,
                "terminal_jobs": len(terminal),
                "terminal_pct": round(100 * len(terminal) / total_tasks, 1) if total_tasks else 0.0,
                "deleted": len(deleted),
                "deleted_pct": round(100 * len(deleted) / total_tasks, 1) if total_tasks else 0.0,
                "duplicate": len(duplicate),
                "duplicate_pct": round(100 * len(duplicate) / total_tasks, 1) if total_tasks else 0.0,
                "other": len(other),
                "other_pct": round(100 * len(other) / total_tasks, 1) if total_tasks else 0.0,
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
            "in_queue_tasks": [{"task_id": r["task_id"], "job": r["job"],
                                   "status": r["current_status"], "user": r["user"]} for r in in_queue],
            "tasks": task_rows,
        }


def analyse(log_dirs=None, date_from=None, date_to=None):
    """Convenience entry point used by the web layer."""
    return DispatcherLogParser(log_dirs).ingest().run(date_from=date_from, date_to=date_to)


if __name__ == "__main__":
    import json
    print(json.dumps(analyse(), indent=2))
