"""
OperaMind Client Portal
=======================
Self-service dashboard where paying clients see their agent performance.

Endpoints:
  GET /portal?email=client@example.com   → HTML dashboard
  GET /api/portal/stats?email=...        → JSON stats
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import os
import datetime
import html
from collections import OrderedDict

from api.database import get_agent_runs, get_payments

portal_router = APIRouter()

SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "hello@operamind.ai")
BOOKING_URL = os.getenv("BOOKING_URL", "https://calendly.com/operamind/ai-assessment")

# Pretty labels for the raw agent_type values stored in the DB
_AGENT_LABELS = {
    "sales": "Sales Agent",
    "support": "Customer Success Agent",
    "customer_success": "Customer Success Agent",
    "recruiting": "Recruiting Agent",
    "finance": "Finance Agent",
    "operations": "Operations Agent",
    "ops": "Operations Agent",
    "executive": "Executive Assistant Agent",
    "exec": "Executive Assistant Agent",
    "knowledge": "Knowledge Agent",
}


def _label(agent_type):
    if not agent_type:
        return "AI Agent"
    return _AGENT_LABELS.get(agent_type.lower(), agent_type.replace("_", " ").title())


def _runs_for(email):
    """Return all agent runs belonging to a client email (case-insensitive)."""
    if not email:
        return []
    email = email.strip().lower()
    # get_agent_runs has no email filter; pull a generous slice and filter here.
    runs = get_agent_runs(limit=1000)
    return [r for r in runs if (r.get("client_email") or "").strip().lower() == email]


def _payments_for(email):
    if not email:
        return []
    email = email.strip().lower()
    pays = get_payments(limit=1000)
    return [p for p in pays if (p.get("customer_email") or "").strip().lower() == email]


def _compute_stats(email):
    """Compute the portal stats dict for a client."""
    runs = _runs_for(email)
    pays = _payments_for(email)

    total_tasks = len(runs)
    durations = [float(r.get("duration_seconds") or 0) for r in runs]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0.0
    total_tokens = sum(int(r.get("tokens_used") or 0) for r in runs)

    # This month's activity
    month_prefix = datetime.datetime.utcnow().strftime("%Y-%m")
    this_month = sum(
        1 for r in runs if (r.get("created_at") or "").startswith(month_prefix)
    )

    active_agents = []
    for r in runs:
        lbl = _label(r.get("agent_type"))
        if lbl not in active_agents:
            active_agents.append(lbl)

    # Most recent active plan
    plan = None
    plan_status = None
    if pays:
        latest = pays[0]  # get_payments returns most-recent-first
        plan = latest.get("plan")
        plan_status = latest.get("status")

    return {
        "email": email,
        "tasks_completed": total_tasks,
        "agents_active": len(active_agents),
        "active_agents": active_agents,
        "avg_duration_seconds": avg_duration,
        "tasks_this_month": this_month,
        "total_tokens": total_tokens,
        "plan": plan,
        "plan_status": plan_status,
    }


# ── JSON stats endpoint ────────────────────────────────────────────────────

@portal_router.get("/api/portal/stats")
def portal_stats(email: str = ""):
    """Return JSON stats for a client."""
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid 'email' query parameter is required.")
    try:
        return {"success": True, **_compute_stats(email)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── HTML dashboard ─────────────────────────────────────────────────────────

def _esc(v):
    return html.escape(str(v if v is not None else ""))


def _fmt_dt(iso):
    if not iso:
        return "—"
    try:
        dt = datetime.datetime.fromisoformat(iso)
        return dt.strftime("%b %d, %Y · %H:%M")
    except Exception:
        return _esc(iso)


def _empty_state(message, sub):
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OperaMind · Client Portal</title>{_STYLE}</head>
<body><div class="wrap"><div class="empty">
<div class="logo">OperaMind</div>
<h1>{_esc(message)}</h1>
<p>{_esc(sub)}</p>
<form class="lookup" method="get" action="/portal">
  <input type="email" name="email" placeholder="you@company.com" required>
  <button type="submit">View my dashboard</button>
</form>
<p class="muted">Need access? Contact <a href="mailto:{_esc(SUPPORT_EMAIL)}">{_esc(SUPPORT_EMAIL)}</a></p>
</div></div></body></html>"""


_STYLE = """<style>
:root{--bg:#080A18;--bg1:#0d1024;--bg2:#12162e;--blue:#3B82F6;--bluel:#60a5fa;
--cyan:#22d3ee;--violet:#8b5cf6;--t1:#f1f5f9;--t2:#94a3b8;--t3:#64748b;
--display:'Sora',system-ui,sans-serif;--body:'Inter',system-ui,sans-serif;
--border:rgba(255,255,255,.08);}
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t1);font-family:var(--body);line-height:1.5;
-webkit-font-smoothing:antialiased;min-height:100vh}
.wrap{max-width:1100px;margin:0 auto;padding:40px 24px 80px}
.logo{font-family:var(--display);font-weight:800;font-size:20px;letter-spacing:-.5px;
background:linear-gradient(90deg,var(--blue),var(--violet));-webkit-background-clip:text;
background-clip:text;-webkit-text-fill-color:transparent;display:inline-block}
.head{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;
gap:16px;border-bottom:1px solid var(--border);padding-bottom:28px;margin-bottom:32px}
h1{font-family:var(--display);font-weight:700;font-size:28px;letter-spacing:-.5px;margin-top:8px}
.sub{color:var(--t2);font-size:14px;margin-top:4px}
.plan-badge{font-family:var(--display);font-weight:600;font-size:13px;padding:8px 16px;
border-radius:999px;background:linear-gradient(90deg,rgba(59,130,246,.2),rgba(139,92,246,.2));
border:1px solid rgba(59,130,246,.35);color:var(--bluel);white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:40px}
.stat{background:var(--bg1);border:1px solid var(--border);border-radius:14px;padding:22px}
.stat .n{font-family:var(--display);font-weight:800;font-size:34px;letter-spacing:-1px;
background:linear-gradient(90deg,var(--blue),var(--cyan));-webkit-background-clip:text;
background-clip:text;-webkit-text-fill-color:transparent}
.stat .l{color:var(--t2);font-size:13px;margin-top:6px}
h2{font-family:var(--display);font-weight:600;font-size:18px;margin:36px 0 16px}
.agents{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px}
.acard{background:var(--bg1);border:1px solid var(--border);border-radius:12px;
padding:16px 18px;min-width:200px;flex:1}
.acard .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;
margin-right:8px;box-shadow:0 0 8px #22c55e}
.acard .an{font-family:var(--display);font-weight:600;font-size:15px}
.acard .as{color:var(--t3);font-size:12px;margin-top:4px}
.card{background:var(--bg1);border:1px solid var(--border);border-radius:14px;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--t3);font-weight:500;font-size:12px;text-transform:uppercase;
letter-spacing:.5px;padding:14px 18px;border-bottom:1px solid var(--border)}
td{padding:14px 18px;border-bottom:1px solid var(--border);color:var(--t1)}
tr:last-child td{border-bottom:none}
td .agent{font-family:var(--display);font-weight:600;font-size:13px;color:var(--bluel)}
td .muted{color:var(--t3)}
.support{margin-top:48px;background:var(--bg1);border:1px solid var(--border);
border-radius:14px;padding:28px;display:flex;justify-content:space-between;
align-items:center;flex-wrap:wrap;gap:16px}
.support h3{font-family:var(--display);font-weight:600;font-size:18px}
.support p{color:var(--t2);font-size:14px;margin-top:4px}
.btn{display:inline-block;font-family:var(--display);font-weight:600;font-size:14px;
padding:12px 22px;border-radius:10px;text-decoration:none;color:#fff;
background:linear-gradient(90deg,var(--blue),var(--violet));white-space:nowrap}
a{color:var(--bluel)}
.empty{max-width:460px;margin:80px auto;text-align:center}
.empty h1{margin:18px 0 8px}.empty p{color:var(--t2);margin-bottom:24px}
.lookup{display:flex;gap:10px;margin:24px 0}
.lookup input{flex:1;padding:12px 16px;border-radius:10px;background:var(--bg1);
border:1px solid var(--border);color:var(--t1);font-family:var(--body);font-size:14px}
.lookup button{padding:12px 20px;border:none;border-radius:10px;cursor:pointer;
font-family:var(--display);font-weight:600;font-size:14px;color:#fff;
background:linear-gradient(90deg,var(--blue),var(--violet))}
.muted{color:var(--t3);font-size:13px}
@media(max-width:860px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.grid{grid-template-columns:1fr}.wrap{padding:28px 16px 60px}
h1{font-size:23px}.lookup{flex-direction:column}.support{flex-direction:column;align-items:flex-start}}
</style>"""


@portal_router.get("/portal", response_class=HTMLResponse)
def portal(email: str = ""):
    """Render the client portal HTML dashboard."""
    if not email or "@" not in email:
        return HTMLResponse(_empty_state(
            "Welcome to your client portal",
            "Enter the email associated with your OperaMind subscription to view your dashboard.",
        ))

    try:
        stats = _compute_stats(email)
        runs = _runs_for(email)
    except Exception as e:
        return HTMLResponse(_empty_state(
            "Something went wrong",
            f"We couldn't load your dashboard right now ({e}). Please try again.",
        ), status_code=500)

    if not runs and not _payments_for(email):
        return HTMLResponse(_empty_state(
            "No activity yet",
            f"We don't have any records for {email} yet. Once your agents start working, "
            f"their performance will appear here. Questions? Contact {SUPPORT_EMAIL}.",
        ))

    # ── Plan badge ──
    if stats["plan"]:
        plan_html = (f'<div class="plan-badge">{_esc(stats["plan"])} · '
                     f'{_esc((stats["plan_status"] or "active").title())}</div>')
    else:
        plan_html = '<div class="plan-badge">No active plan</div>'

    # ── Stat cards ──
    avg = stats["avg_duration_seconds"]
    avg_str = f"{avg:g}s" if avg else "—"
    stat_cards = "".join([
        f'<div class="stat"><div class="n">{stats["tasks_completed"]}</div><div class="l">Tasks completed</div></div>',
        f'<div class="stat"><div class="n">{stats["agents_active"]}</div><div class="l">Active agents</div></div>',
        f'<div class="stat"><div class="n">{avg_str}</div><div class="l">Avg task duration</div></div>',
        f'<div class="stat"><div class="n">{stats["tasks_this_month"]}</div><div class="l">Tasks this month</div></div>',
    ])

    # ── Agent cards ──
    if stats["active_agents"]:
        agent_cards = "".join(
            f'<div class="acard"><div class="an"><span class="dot"></span>{_esc(a)}</div>'
            f'<div class="as">Active · running on your workspace</div></div>'
            for a in stats["active_agents"]
        )
    else:
        agent_cards = '<div class="acard"><div class="as">No agents active yet.</div></div>'

    # ── Activity table (most recent 25) ──
    rows = []
    for r in runs[:25]:
        dur = float(r.get("duration_seconds") or 0)
        agent_lbl = _label(r.get("agent_type"))
        task = r.get("input_summary") or "—"
        when = _fmt_dt(r.get("created_at"))
        rows.append(
            "<tr>"
            f'<td><span class="agent">{_esc(agent_lbl)}</span></td>'
            f'<td>{_esc(task)}</td>'
            f'<td><span class="muted">{when}</span></td>'
            f'<td>{dur:g}s</td>'
            "</tr>"
        )
    table_rows = "".join(rows) or (
        '<tr><td colspan="4"><span class="muted">No runs recorded yet.</span></td></tr>'
    )

    body = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OperaMind · Client Portal</title>{_STYLE}</head>
<body><div class="wrap">
  <div class="head">
    <div>
      <div class="logo">OperaMind</div>
      <h1>Welcome back</h1>
      <div class="sub">{_esc(email)}</div>
    </div>
    {plan_html}
  </div>

  <div class="grid">{stat_cards}</div>

  <h2>Your active agents</h2>
  <div class="agents">{agent_cards}</div>

  <h2>Recent agent activity</h2>
  <div class="card">
    <table>
      <thead><tr><th>Agent</th><th>Task</th><th>When</th><th>Duration</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <div class="support">
    <div>
      <h3>Need help?</h3>
      <p>Our team is here for you. Reach out anytime at
         <a href="mailto:{_esc(SUPPORT_EMAIL)}">{_esc(SUPPORT_EMAIL)}</a>.</p>
    </div>
    <a class="btn" href="{_esc(BOOKING_URL)}" target="_blank" rel="noopener">Book a call</a>
  </div>
</div></body></html>"""

    return HTMLResponse(body)
