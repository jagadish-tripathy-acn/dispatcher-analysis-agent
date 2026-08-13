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
    <prefix> <time> - TaskID,Status,Site,JobName,0,User,Group,
        Provider,Date,...,<State,Timestamp>,<State,Timestamp>,...
The prefix varies between dispatcher versions, e.g.
    2026-08-07 08:51:44,246 INFO  - 2026-08-07 08:51:44 - ...
    2026/08/07-08:51:44.246 UTC - INFO  - - 0DAE2E3C5 - - - 2026-08-07 08:51:44 - ...
so it is matched loosely: everything up to the last "<yyyy-mm-dd hh:mm:ss> - "
is discarded. The trailing (State, Timestamp) pairs are the lifecycle
transitions, e.g. INITIAL/... PREPARING/... SCHEDULED/... TRANSLATING/...
LOADING/... COMPLETE/... Their starting index also varies, so it is located by
scanning for the first known state name rather than hardcoded.
"""
import glob
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

# ---- config --------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_FILE = os.path.join(_HERE, "config.json")

# Only dispatcher client history logs are ingested.
DEFAULT_LOG_GLOB = "History_DispatcherClient*.log"

def _load_config():
    import json
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}

_CONFIG = _load_config()

def _load_log_dirs():
    """Return list of resolved log directory paths from config.json."""
    dirs = _CONFIG.get("log_dirs") or []
    resolved = []
    for d in dirs:
        # Relative paths are resolved from the config file's directory.
        p = d if os.path.isabs(d) else os.path.normpath(os.path.join(_HERE, d))
        resolved.append(p)
    if resolved:
        return resolved
    # Fallback: sibling Logs\ folder next to the project root
    ROOT = os.path.dirname(_HERE)
    return [os.path.join(ROOT, "Logs")]

LOG_DIRS = _load_log_dirs()
LOG_GLOB = _CONFIG.get("log_file_pattern") or DEFAULT_LOG_GLOB

# Valid dispatcher lifecycle states.
STATES = {
    "INITIAL", "PREPARING", "SCHEDULED", "TRANSLATING", "LOADING",
    "COMPLETE", "TERMINAL", "DELETE", "DUPLICATE",
}
TERMINAL_STATES = {"COMPLETE", "TERMINAL", "DELETE", "DUPLICATE"}

# Field positions inside the comma-split payload.
IDX_TASK_ID, IDX_STATUS, IDX_SITE, IDX_JOB = 0, 1, 2, 3
IDX_USER, IDX_GROUP, IDX_PROVIDER = 5, 6, 7
IDX_DATE = 8
IDX_TARGET = 10
MIN_FIELDS = 9  # task id .. date must be present

# Pull the payload off the log prefix: skip everything up to the first
# "<yyyy-mm-dd hh:mm:ss> - ", which is where the payload starts in every
# dispatcher prefix variant.
_LINE_RE = re.compile(
    r"^.*?(?P<logtime>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+-\s+(?P<payload>.+)$"
)

QUEUE_BUCKETS = ["<10", "10-30", "30-60", ">60"]

# States a task can sit in while still being worked on (i.e. not an end state).
ACTIVE_STATES = {"INITIAL", "PREPARING", "TRANSLATING", "LOADING"}

# ---- anomaly detection thresholds ---------------------------------------
# Every job type is scored against these; all of them are overridable from
# config.json under "anomaly_thresholds" so the sensitivity can be tuned per
# site without touching code.
DEFAULT_THRESHOLDS = {
    # A job type needs at least this many tasks before rate-based comparisons
    # against the system baseline are considered statistically meaningful.
    "min_tasks": 10,
    # Terminal (failure) rate spike vs the system-wide terminal rate.
    "terminal_z": 3.0,            # two-proportion z-score cut-off
    "terminal_min_count": 5,      # ignore spikes built on <5 failures
    "terminal_ratio": 1.25,       # and require a 25%+ relative excess
    "terminal_crit_pct": 40.0,    # absolute failure rate that is critical on its own
    # Queue build-up: tasks parked in SCHEDULED waiting for a translator.
    "queue_min_count": 5,
    "queue_ratio": 2.0,           # x the system in-queue rate
    "queue_abs_pct": 5.0,         # or this much of the job's own volume
    # Completion time taking longer "than usual".
    "latency_drift_ratio": 1.5,   # recent p50 vs the job's own earlier p50
    "latency_drift_min": 5,       # tasks needed in each half of the split
    "latency_peer_ratio": 3.0,    # job p50 vs the system p50
    # Spread between typical and worst case within one job type.
    "variance_ratio": 10.0,       # p95 / p50
    "variance_min_tasks": 20,
    # A non-terminal task older than this multiple of the job's p95 is stalled.
    "stall_factor": 3.0,
    "stall_floor_min": 60,        # never call anything under an hour stalled
}
THRESHOLDS = {**DEFAULT_THRESHOLDS, **(_CONFIG.get("anomaly_thresholds") or {})}

# Severity ladder, weakest first — used to roll per-anomaly severities up into
# one badge colour per job box.
SEVERITY_ORDER = ["ok", "low", "medium", "high", "critical"]

# Bucket widths for time series, picked so a chart never gets more points than
# it can usefully draw: (span in minutes, key format, label).
_SERIES_STEPS = [
    (6 * 60, "%Y-%m-%d %H:%M", "minute"),
    (10 * 24 * 60, "%Y-%m-%d %H:00", "hour"),
    (400 * 24 * 60, "%Y-%m-%d", "day"),
    (None, "%Y-%m", "month"),
]


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


def fmt_duration(minutes):
    """A duration in minutes as prose humans read: '18m', '3h 40m', '5d 12h'.

    Durations here span seconds to weeks, so a raw minute count ("706622 min")
    is unreadable in the anomaly/RCA text even though it is the right unit to
    keep in the numeric value/baseline fields the charts plot.
    """
    if minutes is None:
        return "n/a"
    m = float(minutes)
    if m < 1:
        return f"{round(m * 60)}s"
    if m < 10:
        return f"{m:.1f}m".replace(".0m", "m")
    # Round to whole minutes first so 59.9 reads "1h", not "60m".
    total = round(m)
    if total < 60:
        return f"{total}m"
    if total < 1440:
        h, rem = divmod(total, 60)
        return f"{h}h {rem}m" if rem else f"{h}h"
    d, rem = divmod(total, 1440)
    h = rem // 60
    return f"{d}d {h}h" if h else f"{d}d"


def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _pctl(values, q):
    """Linear-interpolated percentile (q in 0..1); 0.0 when there is no data."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return round(float(vals[0]), 1)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return round(vals[lo] * (1 - frac) + vals[hi] * frac, 1)


def _rate(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def _worst(severities):
    """Highest severity in the list, 'ok' when empty."""
    return max(severities, key=SEVERITY_ORDER.index) if severities else "ok"


def _prop_z(p_job, p_sys, n_job):
    """Two-proportion z-score of a job's rate against the system rate.

    Answers "is this job's failure rate above the system's by more than sample
    noise explains?" — so a 100%-failing job type with 3 tasks doesn't outrank a
    45%-failing one with 4000.
    """
    p_job, p_sys = p_job / 100.0, p_sys / 100.0
    if n_job <= 0 or not 0 < p_sys < 1:
        return 0.0
    se = (p_sys * (1 - p_sys) / n_job) ** 0.5
    return round((p_job - p_sys) / se, 2) if se else 0.0


def _series_mode(dts):
    """Pick a time-bucket width from the span of the timestamps given."""
    dts = [d for d in dts if d]
    if not dts:
        return "%Y-%m-%d", "day"
    span = (max(dts) - min(dts)).total_seconds() / 60.0
    for limit, fmt, label in _SERIES_STEPS:
        if limit is None or span <= limit:
            return fmt, label
    return "%Y-%m", "month"


class DispatcherLogParser:
    """Parses dispatcher logs and produces a stats dict for the dashboard."""

    def __init__(self, log_dirs=None):
        self.log_dirs = log_dirs if log_dirs is not None else LOG_DIRS
        # Per-task accumulator.
        self.tasks = {}          # task_id -> meta dict
        self.state_times = defaultdict(dict)  # task_id -> {state: earliest datetime}
        self.files_read = []
        self._rows = None        # memoised _task_timings() output

    # -- ingest -----------------------------------------------------------
    def _ingest_line(self, line):
        m = _LINE_RE.match(line.strip())
        if not m:
            return
        parts = [p.strip() for p in m.group("payload").split(",")]
        if len(parts) < MIN_FIELDS:
            return

        task_id = parts[IDX_TASK_ID]
        if not task_id:
            return

        # Locate the lifecycle block: the first known state name at or after the
        # date field, so a status value earlier in the row is never mistaken for it.
        states_start = next(
            (i for i in range(IDX_DATE + 1, len(parts) - 1) if parts[i] in STATES),
            len(parts),
        )

        # Static metadata (first non-empty wins).
        meta = self.tasks.setdefault(task_id, {
            "task_id": task_id, "job": "", "user": "", "group": "",
            "target": "", "provider": "", "site": "", "current_status": "",
            "last_seen": None,
        })
        for key, idx in (("job", IDX_JOB), ("user", IDX_USER), ("group", IDX_GROUP),
                         ("target", IDX_TARGET), ("provider", IDX_PROVIDER), ("site", IDX_SITE)):
            # Never read a metadata field out of the lifecycle block: log variants
            # that carry fewer columns would otherwise pick up a state/timestamp.
            if not meta[key] and idx < min(len(parts), states_start):
                meta[key] = parts[idx]

        # Current status = status on the most recent line for this task.
        line_time = _parse_dt(parts[IDX_DATE])
        cur_status = parts[IDX_STATUS] if IDX_STATUS < len(parts) else ""
        if cur_status in STATES and (meta["last_seen"] is None or (line_time and line_time >= meta["last_seen"])):
            meta["current_status"] = cur_status
            if line_time:
                meta["last_seen"] = line_time

        # Trailing (state, timestamp) pairs -> keep earliest time per state.
        i = states_start
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
        paths = sorted(p for d in self.log_dirs for p in glob.glob(os.path.join(d, LOG_GLOB)))
        for path in paths:
            self.files_read.append(os.path.basename(path))
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    self._ingest_line(line)
        return self

    # -- derive per-task timings -----------------------------------------
    def _task_timings(self):
        if self._rows is not None:
            return self._rows
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
        self._rows = rows
        return rows

    # -- windowing --------------------------------------------------------
    def _window(self, date_from=None, date_to=None):
        """Rows inside the given date window, based on each task's start time."""
        rows = self._task_timings()
        if not (date_from or date_to):
            return rows
        dt_from = datetime.strptime(date_from, "%Y-%m-%d") if date_from else None
        dt_to   = datetime.strptime(date_to,   "%Y-%m-%d").replace(hour=23, minute=59, second=59) if date_to else None

        def _in_window(r):
            dt = r["initial"] or r["end_time"]
            if not dt:
                return dt_from is None  # undated: include only when no lower bound
            if dt_from and dt < dt_from: return False
            if dt_to   and dt > dt_to:   return False
            return True

        return [r for r in rows if _in_window(r)]

    def _dataset_now(self):
        """Latest timestamp anywhere in the logs — the 'now' used to age tasks.

        The logs are historical, so wall-clock time would make every unfinished
        task look infinitely stalled.
        """
        stamps = [t for r in self._task_timings() for t in (r["end_time"], r["translating"], r["initial"]) if t]
        return max(stamps) if stamps else datetime.now()

    def _data_span(self):
        """First and last day covered by the logs, as YYYY-MM-DD strings."""
        stamps = [t for r in self._task_timings() for t in (r["initial"], r["end_time"]) if t]
        if not stamps:
            return {"from": None, "to": None}
        return {"from": min(stamps).strftime("%Y-%m-%d"), "to": max(stamps).strftime("%Y-%m-%d")}

    # -- time series ------------------------------------------------------
    @staticmethod
    def _throughput(rows):
        """Completions over time with the timing profile of each bucket.

        Bucket width adapts to the window's span (minute -> hour -> day ->
        month), so the series stays chart-sized whether it covers an hour or
        four years.
        """
        ended = [r for r in rows if r["end_time"]]
        fmt, mode = _series_mode([r["end_time"] for r in ended])
        agg = defaultdict(lambda: {"count": 0, "ok": 0, "fail": 0, "queue": [], "proc": [], "total": []})
        for r in ended:
            b = agg[r["end_time"].strftime(fmt)]
            b["count"] += 1
            if r["end_state"] == "COMPLETE":
                b["ok"] += 1
            elif r["end_state"] == "TERMINAL":
                b["fail"] += 1
            b["queue"].append(r["queue_min"])
            b["proc"].append(r["proc_min"])
            if r["end_state"] == "COMPLETE":
                b["total"].append(r["total_min"])
        points = [{
            "t": key, "count": v["count"], "ok": v["ok"], "fail": v["fail"],
            "avg_queue": _avg(v["queue"]), "avg_proc": _avg(v["proc"]),
            "avg_total": _avg(v["total"]), "p95_total": _pctl(v["total"], 0.95),
        } for key, v in sorted(agg.items())]
        return {"bucket": mode, "points": points}

    @staticmethod
    def _spark(rows, t0, t1, buckets=24):
        """Arrival volume (and how much of it failed) across a fixed number of
        equal-width buckets, so every job box's sparkline shares one x axis."""
        bars = [{"n": 0, "fail": 0} for _ in range(buckets)]
        if not (t0 and t1):
            return bars
        span = (t1 - t0).total_seconds() or 1.0
        for r in rows:
            dt = r["initial"] or r["end_time"]
            if not dt:
                continue
            i = int(buckets * (dt - t0).total_seconds() / span)
            i = min(max(i, 0), buckets - 1)
            bars[i]["n"] += 1
            if r["end_state"] == "TERMINAL":
                bars[i]["fail"] += 1
        return bars

    # -- per-job health & anomalies ---------------------------------------
    @staticmethod
    def _baselines(rows):
        """System-wide reference values every job type is scored against."""
        total = len(rows)
        ended = [r for r in rows if r["completed"]]
        return {
            "terminal_pct": _rate(sum(1 for r in rows if r["end_state"] == "TERMINAL"), total),
            "in_queue_pct": _rate(sum(1 for r in rows
                                      if r["current_status"] == "SCHEDULED" and not r["completed"]), total),
            "completed_pct": _rate(sum(1 for r in rows if r["end_state"] == "COMPLETE"), total),
            "p50_total": _pctl([r["total_min"] for r in rows if r["end_state"] == "COMPLETE"], 0.5),
            "p95_total": _pctl([r["total_min"] for r in rows if r["end_state"] == "COMPLETE"], 0.95),
            "p50_queue": _pctl([r["queue_min"] for r in ended], 0.5),
            "avg_queue": _avg([r["queue_min"] for r in ended]),
            "avg_proc": _avg([r["proc_min"] for r in ended]),
            "job_types": len({r["job"] or "unknown" for r in rows}),
        }

    @staticmethod
    def _job_metrics(name, jrows, now):
        """Counts, timing percentiles and drift for one job type."""
        ended     = [r for r in jrows if r["completed"]]
        completed = [r for r in jrows if r["end_state"] == "COMPLETE"]
        terminal  = [r for r in jrows if r["end_state"] == "TERMINAL"]
        deleted   = [r for r in jrows if r["end_state"] == "DELETE"]
        duplicate = [r for r in jrows if r["end_state"] == "DUPLICATE"]
        # "Still waiting" means SCHEDULED *and* never reached an end state: a
        # task whose last line says SCHEDULED but which has a TERMINAL
        # transition recorded is finished, not queued.
        in_queue  = [r for r in jrows if r["current_status"] == "SCHEDULED" and not r["completed"]]
        active    = [r for r in jrows if not r["completed"] and r["current_status"] in ACTIVE_STATES]

        totals = [r["total_min"] for r in completed if r["total_min"] is not None]
        queues = [r["queue_min"] for r in ended if r["queue_min"] is not None]
        procs  = [r["proc_min"] for r in ended if r["proc_min"] is not None]

        # "Longer than usual" is measured against the job's own past: split its
        # completions chronologically 70/30 and compare the two medians.
        chron = sorted((r for r in completed if r["total_min"] is not None and r["end_time"]),
                       key=lambda r: r["end_time"])
        split = int(len(chron) * 0.7)
        base_part, recent_part = chron[:split], chron[split:]
        baseline_p50 = _pctl([r["total_min"] for r in base_part], 0.5) if base_part else 0.0
        recent_p50   = _pctl([r["total_min"] for r in recent_part], 0.5) if recent_part else 0.0

        stamps = [r["initial"] or r["end_time"] for r in jrows if (r["initial"] or r["end_time"])]
        avg_q, avg_p = _avg(queues), _avg(procs)

        return {
            "job": name,
            "count": len(jrows),
            "ended": len(ended),
            "completed": len(completed),
            "terminal": len(terminal),
            "deleted": len(deleted),
            "duplicate": len(duplicate),
            "in_queue": len(in_queue),
            "active": len(active),
            # Tasks with a measurable INITIAL -> end elapsed time. Zero means
            # the percentiles below are "no data", not "instant".
            "timed": len(totals),
            "completed_pct": _rate(len(completed), len(jrows)),
            "terminal_pct": _rate(len(terminal), len(jrows)),
            "in_queue_pct": _rate(len(in_queue), len(jrows)),
            "avg_total": _avg(totals), "p50_total": _pctl(totals, 0.5),
            "p95_total": _pctl(totals, 0.95), "max_total": max(totals, default=0),
            "avg_queue": avg_q, "p95_queue": _pctl(queues, 0.95),
            "avg_proc": avg_p, "p95_proc": _pctl(procs, 0.95),
            # Where the time goes: queue-heavy = starved of translators,
            # proc-heavy = the translation itself is slow.
            "queue_share": _rate(avg_q, avg_q + avg_p),
            "baseline_p50": baseline_p50, "recent_p50": recent_p50,
            "drift_pct": _rate(recent_p50 - baseline_p50, baseline_p50) if baseline_p50 else 0.0,
            "first_seen": min(stamps).strftime("%Y-%m-%d %H:%M") if stamps else "",
            "last_seen": max(stamps).strftime("%Y-%m-%d %H:%M") if stamps else "",
            "_rows": jrows, "_completed": completed, "_terminal": terminal,
            "_in_queue": in_queue, "_active": active, "_chron": chron,
            "_base_part": base_part, "_recent_part": recent_part, "_now": now,
        }

    @staticmethod
    def _job_anomalies(m, base):
        """Score one job type against the system baseline and its own history.

        Returns a list of anomaly dicts, each carrying the metric, the value,
        the baseline it was judged against and sample task IDs — everything the
        UI needs to explain itself and the RCA needs as evidence.
        """
        T = THRESHOLDS
        out = []
        n = m["count"]
        big_enough = n >= T["min_tasks"]

        def add(kind, severity, title, detail, metric, value, baseline, unit="", sample=(), z=None):
            out.append({
                "type": kind, "severity": severity, "title": title, "detail": detail,
                "metric": metric, "value": value, "baseline": baseline, "unit": unit,
                "z": z, "sample": [r["task_id"] for r in list(sample)[:5]],
            })

        # 1. Terminal (failure) rate above the system baseline.
        z = _prop_z(m["terminal_pct"], base["terminal_pct"], n)
        if (m["terminal"] >= T["terminal_min_count"]
                and m["terminal_pct"] >= base["terminal_pct"] * T["terminal_ratio"]
                and (z >= T["terminal_z"] or m["terminal_pct"] >= T["terminal_crit_pct"])):
            sev = ("critical" if m["terminal_pct"] >= T["terminal_crit_pct"] and z >= T["terminal_z"]
                   else "high" if z >= T["terminal_z"] * 2 or m["terminal_pct"] >= T["terminal_crit_pct"]
                   else "medium")
            add("terminal_spike", sev, "Elevated terminal (failure) rate",
                f"{m['terminal']} of {n} tasks ended TERMINAL ({m['terminal_pct']}%) against a "
                f"system baseline of {base['terminal_pct']}% (z={z}).",
                "terminal_pct", m["terminal_pct"], base["terminal_pct"], "%", m["_terminal"], z)

        # 2. Tasks piling up in SCHEDULED (waiting for a translator).
        if m["in_queue"] >= T["queue_min_count"] and (
                m["in_queue_pct"] >= max(base["in_queue_pct"] * T["queue_ratio"], T["queue_abs_pct"])):
            sev = "high" if m["in_queue_pct"] >= T["queue_abs_pct"] * 3 else "medium"
            add("queue_backlog", sev, "Queue backlog building up",
                f"{m['in_queue']} tasks ({m['in_queue_pct']}%) are still SCHEDULED, "
                f"vs {base['in_queue_pct']}% system-wide.",
                "in_queue_pct", m["in_queue_pct"], base["in_queue_pct"], "%", m["_in_queue"])

        # 3. Completion time drifting upward against this job's own past.
        if (len(m["_base_part"]) >= T["latency_drift_min"] and len(m["_recent_part"]) >= T["latency_drift_min"]
                and m["baseline_p50"] > 0 and m["recent_p50"] >= m["baseline_p50"] * T["latency_drift_ratio"]):
            ratio = m["recent_p50"] / m["baseline_p50"]
            sev = "critical" if ratio >= 4 else "high" if ratio >= 2.5 else "medium"
            add("latency_drift", sev, "Taking longer than usual to complete",
                f"Median completion time of the most recent {len(m['_recent_part'])} tasks is "
                f"{fmt_duration(m['recent_p50'])} vs {fmt_duration(m['baseline_p50'])} for the earlier "
                f"{len(m['_base_part'])} — {round((ratio - 1) * 100)}% slower.",
                "p50_total", m["recent_p50"], m["baseline_p50"], "min",
                sorted(m["_recent_part"], key=lambda r: -(r["total_min"] or 0)))

        # 4. Chronically slower than the rest of the system.
        if big_enough and base["p50_total"] > 0 and m["p50_total"] >= base["p50_total"] * T["latency_peer_ratio"]:
            ratio = m["p50_total"] / base["p50_total"]
            sev = "high" if ratio >= T["latency_peer_ratio"] * 2 else "medium"
            add("latency_outlier", sev, "Much slower than other job types",
                f"Median completion {fmt_duration(m['p50_total'])} is {round(ratio, 1)}x the system median "
                f"of {fmt_duration(base['p50_total'])} ({m['queue_share']}% of it spent queueing).",
                "p50_total", m["p50_total"], base["p50_total"], "min",
                sorted(m["_completed"], key=lambda r: -(r["total_min"] or 0)))

        # 5. Unpredictable runtimes — the tail is far from typical.
        if (m["count"] >= T["variance_min_tasks"] and m["p50_total"] > 0
                and m["p95_total"] >= m["p50_total"] * T["variance_ratio"]):
            add("high_variance", "low", "Highly variable runtime",
                f"p95 is {fmt_duration(m['p95_total'])} against a p50 of {fmt_duration(m['p50_total'])} "
                f"({round(m['p95_total'] / m['p50_total'], 1)}x spread), so completion time is unpredictable.",
                "p95_total", m["p95_total"], m["p50_total"], "min",
                sorted(m["_completed"], key=lambda r: -(r["total_min"] or 0)))

        # 6. Individual tasks that never reached an end state and are now far
        #    older than this job type's own worst normal case.
        floor = max(m["p95_total"] * T["stall_factor"], T["stall_floor_min"])
        stalled = []
        for r in m["_rows"]:
            if r["completed"] or not r["initial"]:
                continue
            age = (m["_now"] - r["initial"]).total_seconds() / 60.0
            if age >= floor:
                stalled.append((age, r))
        if stalled:
            stalled.sort(key=lambda x: -x[0])
            oldest = round(stalled[0][0])
            sev = "critical" if len(stalled) >= 20 else "high" if len(stalled) >= 5 else "medium"
            add("stalled_tasks", sev, "Tasks stalled mid-lifecycle",
                f"{len(stalled)} task(s) never reached an end state; the oldest has been running "
                f"{fmt_duration(oldest)}, past the {fmt_duration(floor)} stall threshold for this job type.",
                "stalled", len(stalled), round(floor), "min", [r for _, r in stalled])

        return sorted(out, key=lambda a: -SEVERITY_ORDER.index(a["severity"]))

    def _job_health(self, rows, base):
        """Per-job-type box payload: counts, timings, sparkline and anomalies."""
        now = self._dataset_now()
        by_job = defaultdict(list)
        for r in rows:
            by_job[r["job"] or "unknown"].append(r)

        stamps = [r["initial"] or r["end_time"] for r in rows if (r["initial"] or r["end_time"])]
        t0, t1 = (min(stamps), max(stamps)) if stamps else (None, None)

        out = []
        for name, jrows in by_job.items():
            m = self._job_metrics(name, jrows, now)
            anomalies = self._job_anomalies(m, base)
            public = {k: v for k, v in m.items() if not k.startswith("_")}
            public.update({
                "anomalies": anomalies,
                "anomaly_count": len(anomalies),
                "severity": _worst([a["severity"] for a in anomalies]),
                "spark": self._spark(jrows, t0, t1),
            })
            out.append(public)

        # Anomalous job types float to the top, then by volume.
        out.sort(key=lambda j: (-SEVERITY_ORDER.index(j["severity"]), -j["count"]))
        return out

    # -- public: compute the full stats payload --------------------------
    def run(self, date_from=None, date_to=None, include_tasks=False):
        rows = self._window(date_from, date_to)
        total_tasks = len(rows)
        # "ended" = reached any lifecycle end state (COMPLETE/TERMINAL/DELETE/DUPLICATE);
        # used for timing stats since queue/proc/total times are meaningful regardless of outcome.
        ended     = [r for r in rows if r["completed"]]
        in_queue  = [r for r in rows if r["current_status"] == "SCHEDULED" and not r["completed"]]
        succeeded = [r for r in rows if r["end_state"] == "COMPLETE"]
        terminal  = [r for r in rows if r["end_state"] == "TERMINAL"]
        deleted   = [r for r in rows if r["end_state"] == "DELETE"]
        duplicate = [r for r in rows if r["end_state"] == "DUPLICATE"]
        other     = [r for r in rows if not r["completed"] and r["current_status"] in ACTIVE_STATES]

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

        # Throughput: completions per adaptive time bucket, with the timing
        # profile of each bucket (drives the system throughput chart).
        throughput = self._throughput(rows)

        # System baselines every job type is scored against.
        baselines = self._baselines(rows)
        job_health = self._job_health(rows, baselines)
        anomaly_total = sum(j["anomaly_count"] for j in job_health)

        payload = {
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
            # Span of the whole log set, not of the selected window: the UI
            # anchors its "last N days" presets to the newest data rather than
            # to today, because these logs are historical.
            "data_span": self._data_span(),
            "baselines": baselines,
            "job_health": job_health,
            "anomaly_total": anomaly_total,
            "anomalous_jobs": sum(1 for j in job_health if j["severity"] != "ok"),
            "worst_severity": _worst([j["severity"] for j in job_health]),
            "window": {"from": date_from or "", "to": date_to or ""},
            "status_distribution": [{"status": s, "count": c} for s, c in status_counts.most_common()],
            "queue_distribution": queue_distribution,
            "job_breakdown": job_breakdown,
            "group_distribution": [{"group": g, "count": c} for g, c in group_counts.most_common(10)],
            "user_distribution": [{"user": u, "count": c} for u, c in user_counts.most_common(10)],
            "throughput": throughput,
            "in_queue_tasks": [{"task_id": r["task_id"], "job": r["job"],
                                   "status": r["current_status"], "user": r["user"]} for r in in_queue],
        }
        if include_tasks:
            payload["tasks"] = [_task_row(r) for r in _sort_rows(rows, "slowest")]
        return payload

    # -- public: task table queries ---------------------------------------
    def query_tasks(self, date_from=None, date_to=None, job=None, status=None,
                    q=None, sort="slowest", limit=200, offset=0):
        """Server-side task table: filter, sort, page.

        Keeps the browser out of the business of holding every task row — the
        drill-down table asks for the slice it is showing.
        """
        rows = self._window(date_from, date_to)
        if job:
            rows = [r for r in rows if (r["job"] or "unknown") == job]
        if status and status != "all":
            rows = [r for r in rows if r["current_status"] == status]
        if q:
            needle = q.lower()
            rows = [r for r in rows if needle in
                    f"{r['task_id']}{r['job']}{r['user']}{r['group']}{r['target']}"
                    f"{r['provider']}{r['current_status']}".lower()]
        rows = _sort_rows(rows, sort)
        total = len(rows)
        page = rows if limit is None else rows[offset:offset + limit]
        return {"total": total, "offset": offset, "returned": len(page),
                "rows": [_task_row(r) for r in page]}

    # -- public: one job type in depth ------------------------------------
    def job_detail(self, job, date_from=None, date_to=None):
        """Everything the drill-down panel and the RCA need about one job type.

        Beyond the box metrics this adds the shape of the failures — who ran
        them, on which provider, at what hour of day, and which tasks were the
        slowest — because that is what turns "this job fails a lot" into a
        cause.
        """
        window = self._window(date_from, date_to)
        rows = [r for r in window if (r["job"] or "unknown") == job]
        if not rows:
            return None

        base = self._baselines(window)
        m = self._job_metrics(job, rows, self._dataset_now())
        anomalies = self._job_anomalies(m, base)
        metrics = {k: v for k, v in m.items() if not k.startswith("_")}

        terminal, completed = m["_terminal"], m["_completed"]
        stuck = sorted(m["_in_queue"] + m["_active"],
                       key=lambda r: r["initial"] or datetime.max)

        def top(rws, key, n=6):
            return [{"name": name or "unknown", "count": c}
                    for name, c in Counter(r[key] or "unknown" for r in rws).most_common(n)]

        # Failure clustering: a spike confined to one hour of the day, one
        # provider or one user points somewhere very different than a flat one.
        fail_hours = Counter(r["initial"].hour for r in terminal if r["initial"])
        all_hours = Counter(r["initial"].hour for r in rows if r["initial"])

        # Queue-time buckets for this job type.
        jb = Counter(_bucket(r["queue_min"]) for r in rows if r["queue_min"] is not None)

        # Processing-time histogram (log-ish buckets, minutes).
        edges = [(0, 1), (1, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, None)]
        proc_hist = []
        procs = [r["proc_min"] for r in rows if r["proc_min"] is not None]
        for lo, hi in edges:
            label = f">{lo}" if hi is None else f"{lo}-{hi}"
            proc_hist.append({"range": label,
                              "count": sum(1 for v in procs if v >= lo and (hi is None or v < hi))})

        return {
            "job": job,
            "window": {"from": date_from or "", "to": date_to or ""},
            "metrics": metrics,
            "anomalies": anomalies,
            "severity": _worst([a["severity"] for a in anomalies]),
            "baselines": base,
            "throughput": self._throughput(rows),
            "status_distribution": [{"status": s or "UNKNOWN", "count": c}
                                    for s, c in Counter(r["current_status"] for r in rows).most_common()],
            "queue_distribution": [{"range": b, "count": jb.get(b, 0)} for b in QUEUE_BUCKETS],
            "proc_histogram": proc_hist,
            "top_users": top(rows, "user"), "top_groups": top(rows, "group"),
            "top_providers": top(rows, "provider"),
            "failing_users": top(terminal, "user"), "failing_groups": top(terminal, "group"),
            "failing_providers": top(terminal, "provider"),
            "failure_by_hour": [{"hour": h, "fail": fail_hours.get(h, 0), "total": all_hours.get(h, 0)}
                                for h in range(24)],
            "slowest_tasks": [_task_row(r) for r in sorted(
                completed, key=lambda r: -(r["total_min"] or 0))[:10]],
            "stuck_tasks": [_task_row(r) for r in stuck[:10]],
            "failed_tasks": [_task_row(r) for r in sorted(
                terminal, key=lambda r: (r["initial"] or datetime.min), reverse=True)[:10]],
        }


def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""


def _task_row(r):
    """One task as the API/table sees it."""
    return {
        "task_id": r["task_id"], "job": r["job"], "user": r["user"],
        "group": r["group"], "target": r["target"], "provider": r["provider"],
        "site": r["site"], "status": r["current_status"],
        "initial": _fmt_dt(r["initial"]), "translating": _fmt_dt(r["translating"]),
        "end_state": r["end_state"] or "", "end_time": _fmt_dt(r["end_time"]),
        "queue_min": r["queue_min"], "proc_min": r["proc_min"],
        "total_min": r["total_min"], "completed": r["completed"],
    }


_SORTS = {
    "slowest":  lambda r: (r["total_min"] is None, -(r["total_min"] or 0)),
    "fastest":  lambda r: (r["total_min"] is None, r["total_min"] or 0),
    "queue":    lambda r: (r["queue_min"] is None, -(r["queue_min"] or 0)),
    "proc":     lambda r: (r["proc_min"] is None, -(r["proc_min"] or 0)),
    "newest":   lambda r: (r["initial"] is None, -(r["initial"] or datetime.min).timestamp()),
    "oldest":   lambda r: (r["initial"] is None, (r["initial"] or datetime.max).timestamp()),
    "task_id":  lambda r: r["task_id"],
}


def _sort_rows(rows, sort):
    return sorted(rows, key=_SORTS.get(sort or "slowest", _SORTS["slowest"]))


def log_files(log_dirs=None):
    """Log files currently on disk, newest state first — used for cache keys."""
    dirs = log_dirs if log_dirs is not None else LOG_DIRS
    return sorted(p for d in dirs for p in glob.glob(os.path.join(d, LOG_GLOB)))


def signature(log_dirs=None):
    """Fingerprint of the log set: re-ingest only when this changes."""
    sig = []
    for p in log_files(log_dirs):
        try:
            st = os.stat(p)
            sig.append((p, int(st.st_mtime), st.st_size))
        except OSError:
            sig.append((p, 0, 0))
    return tuple(sig)


def analyse(log_dirs=None, date_from=None, date_to=None, include_tasks=False):
    """Convenience entry point used by the web layer."""
    return DispatcherLogParser(log_dirs).ingest().run(
        date_from=date_from, date_to=date_to, include_tasks=include_tasks)


if __name__ == "__main__":
    import json
    print(json.dumps(analyse(), indent=2))
