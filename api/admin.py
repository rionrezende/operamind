"""
OperaMind Admin Dashboard
=========================
Inline HTML admin panel served via FastAPI.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import os
from api.database import get_leads, get_payments, get_agent_runs, get_stats

admin_router = APIRouter()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "operamind2026")


@admin_router.get("/admin", response_class=HTMLResponse)
def admin_dashboard(key: str = ""):
    if key != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    stats = get_stats()
    leads = get_leads(limit=50)
    payments = get_payments(limit=50)
    agent_runs = get_agent_runs(limit=50)

    leads_rows = ""
    for l in leads:
        leads_rows += f"""<tr>
            <td>{l.get('id','')}</td><td>{l.get('name','')}</td><td>{l.get('email','')}</td>
            <td>{l.get('company','')}</td><td>{l.get('industry','')}</td><td>{l.get('company_size','')}</td>
            <td>{l.get('source','')}</td><td>{l.get('status','')}</td><td>{l.get('created_at','')}</td>
        </tr>"""

    payments_rows = ""
    for p in payments:
        payments_rows += f"""<tr>
            <td>{p.get('id','')}</td><td>{p.get('customer_email','')}</td><td>{p.get('customer_name','')}</td>
            <td>{p.get('plan','')}</td><td>${p.get('amount',0):,.2f}</td><td>{p.get('currency','')}</td>
            <td>{p.get('status','')}</td><td>{p.get('created_at','')}</td>
        </tr>"""

    runs_rows = ""
    for r in agent_runs:
        runs_rows += f"""<tr>
            <td>{r.get('id','')}</td><td>{r.get('agent_type','')}</td><td>{r.get('client_email','')}</td>
            <td>{r.get('input_summary','')[:80] if r.get('input_summary') else ''}</td>
            <td>{r.get('tokens_used',0)}</td><td>{r.get('duration_seconds',0):.1f}s</td>
            <td>{r.get('created_at','')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OperaMind Admin</title>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#080A18; color:#EEF0FF; font-family:'Inter',sans-serif; padding:24px; }}
h1 {{ font-family:'Sora',sans-serif; font-size:28px; margin-bottom:8px; }}
h2 {{ font-family:'Sora',sans-serif; font-size:20px; margin:32px 0 12px; }}
.subtitle {{ color:#8B8FA3; margin-bottom:24px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:32px; }}
.stat {{ background:#10132A; border:1px solid #1E2140; border-radius:12px; padding:20px; }}
.stat .label {{ color:#8B8FA3; font-size:13px; text-transform:uppercase; letter-spacing:0.5px; }}
.stat .value {{ font-family:'Sora',sans-serif; font-size:32px; font-weight:700; color:#3B82F6; margin-top:4px; }}
.table-wrap {{ overflow-x:auto; margin-bottom:32px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ background:#3B82F6; color:#fff; padding:10px 12px; text-align:left; font-weight:600; white-space:nowrap; }}
td {{ padding:8px 12px; border-bottom:1px solid #1E2140; }}
tr:hover td {{ background:#10132A; }}
@media(max-width:600px) {{
  body {{ padding:12px; }}
  .stat .value {{ font-size:24px; }}
  table {{ font-size:12px; }}
}}
</style>
</head>
<body>
<h1>OperaMind Admin Dashboard</h1>
<p class="subtitle">Real-time overview of leads, payments, and agent activity.</p>

<div class="stats">
  <div class="stat"><div class="label">Total Leads</div><div class="value">{stats['total_leads']}</div></div>
  <div class="stat"><div class="label">Total Payments</div><div class="value">{stats['total_payments']}</div></div>
  <div class="stat"><div class="label">Revenue</div><div class="value">${stats['revenue_total']:,.2f}</div></div>
  <div class="stat"><div class="label">Agent Runs</div><div class="value">{stats['total_agent_runs']}</div></div>
</div>

<h2>Leads</h2>
<div class="table-wrap">
<table>
<tr><th>ID</th><th>Name</th><th>Email</th><th>Company</th><th>Industry</th><th>Size</th><th>Source</th><th>Status</th><th>Created</th></tr>
{leads_rows if leads_rows else '<tr><td colspan="9" style="text-align:center;color:#8B8FA3;">No leads yet</td></tr>'}
</table>
</div>

<h2>Payments</h2>
<div class="table-wrap">
<table>
<tr><th>ID</th><th>Email</th><th>Name</th><th>Plan</th><th>Amount</th><th>Currency</th><th>Status</th><th>Created</th></tr>
{payments_rows if payments_rows else '<tr><td colspan="8" style="text-align:center;color:#8B8FA3;">No payments yet</td></tr>'}
</table>
</div>

<h2>Agent Runs</h2>
<div class="table-wrap">
<table>
<tr><th>ID</th><th>Agent</th><th>Client</th><th>Input</th><th>Tokens</th><th>Duration</th><th>Created</th></tr>
{runs_rows if runs_rows else '<tr><td colspan="7" style="text-align:center;color:#8B8FA3;">No agent runs yet</td></tr>'}
</table>
</div>
</body>
</html>"""

    return HTMLResponse(content=html)
