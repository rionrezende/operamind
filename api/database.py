"""
OperaMind Database Module
=========================
SQLite database for leads, payments, and agent run tracking.
"""
import sqlite3, os, datetime, logging
from pathlib import Path

logger = logging.getLogger("operamind.db")
DB_PATH = os.getenv("DATABASE_PATH", "operamind.db")


def _get_conn():
    """Get a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                company TEXT,
                industry TEXT,
                company_size TEXT,
                pain_point TEXT,
                opportunities_count INTEGER,
                source TEXT DEFAULT 'scanner',
                language TEXT DEFAULT 'en',
                created_at TEXT,
                status TEXT DEFAULT 'new'
            );

            CREATE TABLE IF NOT EXISTS payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stripe_session_id TEXT,
                customer_email TEXT,
                customer_name TEXT,
                plan TEXT,
                amount REAL,
                currency TEXT DEFAULT 'USD',
                status TEXT DEFAULT 'completed',
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT,
                client_email TEXT,
                input_summary TEXT,
                output_summary TEXT,
                tokens_used INTEGER DEFAULT 0,
                duration_seconds REAL,
                created_at TEXT
            );
        """)
    logger.info(f"Database initialized at {DB_PATH}")


# ── Leads ─────────────────────────────────────────────────────────────────

def save_lead(name, email, company, industry, company_size, pain_point,
              opportunities_count, source="scanner", language="en"):
    """Save a lead and return its ID."""
    now = datetime.datetime.utcnow().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO leads
               (name, email, company, industry, company_size, pain_point,
                opportunities_count, source, language, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, email, company, industry, company_size, pain_point,
             opportunities_count, source, language, now),
        )
        return cur.lastrowid


def get_leads(limit=100, offset=0):
    """Return leads ordered by most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM leads ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def get_lead_count():
    """Return total number of leads."""
    with _get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]


# ── Payments ──────────────────────────────────────────────────────────────

def save_payment(stripe_session_id, customer_email, customer_name, plan,
                 amount, currency="USD", status="completed"):
    """Save a payment record and return its ID."""
    now = datetime.datetime.utcnow().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO payments
               (stripe_session_id, customer_email, customer_name, plan,
                amount, currency, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (stripe_session_id, customer_email, customer_name, plan,
             amount, currency, status, now),
        )
        return cur.lastrowid


def get_payments(limit=100):
    """Return payments ordered by most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM payments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_payment_stats():
    """Return payment statistics: total_revenue, total_payments, mrr estimate."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) as total_revenue, COUNT(*) as total_payments FROM payments"
        ).fetchone()
        total_revenue = row["total_revenue"]
        total_payments = row["total_payments"]
        # MRR estimate: revenue from last 30 days
        thirty_days_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
        mrr = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM payments WHERE created_at >= ?",
            (thirty_days_ago,),
        ).fetchone()[0]
        return {
            "total_revenue": total_revenue,
            "total_payments": total_payments,
            "mrr": mrr,
        }


# ── Agent Runs ────────────────────────────────────────────────────────────

def save_agent_run(agent_type, client_email, input_summary, output_summary,
                   tokens_used=0, duration_seconds=0.0):
    """Save an agent run record and return its ID."""
    now = datetime.datetime.utcnow().isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO agent_runs
               (agent_type, client_email, input_summary, output_summary,
                tokens_used, duration_seconds, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (agent_type, client_email, input_summary, output_summary,
             tokens_used, duration_seconds, now),
        )
        return cur.lastrowid


def get_agent_runs(limit=100):
    """Return agent runs ordered by most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Aggregate Stats ──────────────────────────────────────────────────────

def get_stats():
    """Return dashboard-level statistics."""
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    with _get_conn() as conn:
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        total_payments = conn.execute("SELECT COUNT(*) FROM payments").fetchone()[0]
        total_agent_runs = conn.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0]
        leads_today = conn.execute(
            "SELECT COUNT(*) FROM leads WHERE created_at LIKE ?", (f"{today}%",)
        ).fetchone()[0]
        revenue_total = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM payments"
        ).fetchone()[0]
        return {
            "total_leads": total_leads,
            "total_payments": total_payments,
            "total_agent_runs": total_agent_runs,
            "leads_today": leads_today,
            "revenue_total": revenue_total,
        }


# Initialize tables on import
init_db()
