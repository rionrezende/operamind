"""
OperaMind API
=============
FastAPI backend that exposes all 7 AI agents as REST endpoints.

Start: uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional
from dotenv import load_dotenv
import datetime, logging
from api.database import init_db, save_lead

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("operamind.api")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OperaMind AI Workforce API",
    description="API endpoints for all 7 OperaMind AI Employees",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow website to call this API ──────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to CORS_ORIGINS in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy agent imports (only load when endpoint is called) ─────────────────
# This prevents startup failures if a dep isn't installed yet

def get_sales_agent():
    from agents.sales_agent import process_lead
    return process_lead

def get_support_agent():
    from agents.customer_success_agent import handle_ticket
    return handle_ticket

def get_recruiting_agent():
    from agents.recruiting_agent import process_candidate, Candidate, JobRequirements
    return process_candidate, Candidate, JobRequirements

def get_finance_agent():
    from agents.finance_agent import process_invoice, audit_expenses
    return process_invoice, audit_expenses

def get_ops_agent():
    from agents.operations_agent import run_morning_check, trigger_alert
    return run_morning_check, trigger_alert

def get_exec_agent():
    from agents.executive_assistant_agent import run_morning_brief, process_request
    return run_morning_brief, process_request

def get_knowledge_agent():
    from agents.knowledge_agent import ask, add_document
    return ask, add_document

# ══════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════

class LeadRequest(BaseModel):
    name: str
    email: str
    company: str
    company_size: str = "11-50"
    industry: str = "Technology"
    title: str = "CEO"
    message: str = ""
    thread_id: Optional[str] = None

class TicketRequest(BaseModel):
    id: Optional[str] = None
    customer_email: str
    subject: str
    message: str
    channel: str = "email"
    language: str = "en"

class CandidateRequest(BaseModel):
    name: str
    email: str
    resume_text: str
    role_applied: str
    years_experience: int = 0
    linkedin_url: str = ""

class JobRequest(BaseModel):
    role: str
    must_have_skills: list[str]
    nice_to_have_skills: list[str] = []
    min_years_experience: int = 2
    team: str = "Engineering"
    hiring_manager: str = "Hiring Manager"

class RecruitingRequest(BaseModel):
    candidate: CandidateRequest
    job: JobRequest

class InvoiceRequest(BaseModel):
    raw_invoice_text: str

class ExpenseAuditRequest(BaseModel):
    expenses: list[dict]

class OpsCheckRequest(BaseModel):
    scope: str = "all"

class AlertRequest(BaseModel):
    alert_type: str
    details: str

class ExecBriefRequest(BaseModel):
    executive_name: str
    executive_email: str

class ExecTaskRequest(BaseModel):
    executive_name: str
    request: str
    thread_id: str = "default"

class KnowledgeAskRequest(BaseModel):
    question: str
    company_name: str
    employee_role: str = "employee"
    session_id: Optional[str] = None

class KnowledgeIngestRequest(BaseModel):
    document_url: str
    document_type: str
    category: str
    added_by: str
    company_name: str

# Lead magnet scanner (from website)
class ScannerRequest(BaseModel):
    industry: str
    company_size: str
    main_pain: str
    email: str

# ROI Calculator data (for CRM capture)
class ROIRequest(BaseModel):
    employees: int
    hourly_cost: float
    hours_on_tasks: float
    agents_count: int
    annual_savings: float
    email: Optional[str] = None
    company: Optional[str] = None

class AgentResponse(BaseModel):
    success: bool
    agent: str
    result: dict
    timestamp: str = ""

    def __init__(self, **data):
        if not data.get("timestamp"):
            data["timestamp"] = datetime.datetime.utcnow().isoformat()
        super().__init__(**data)

# ══════════════════════════════════════════════════════════════════════════
# HEALTH & INFO
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "agents": 7,
        "agents_list": [
            "Sales Agent", "Customer Success Agent", "Recruiting Agent",
            "Finance Agent", "Operations Agent", "Executive Assistant Agent",
            "Knowledge Agent",
        ],
        "version": "1.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

@app.get("/")
def root():
    return {
        "message": "OperaMind AI Workforce API",
        "docs": "/docs",
        "health": "/api/health",
    }

# ══════════════════════════════════════════════════════════════════════════
# SALES AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/sales/lead", response_model=AgentResponse)
async def process_lead(req: LeadRequest):
    """
    Submit an inbound lead for qualification.
    The Sales Agent will qualify, update CRM, and book a meeting if qualified.
    """
    try:
        process_lead_fn = get_sales_agent()
        result = process_lead_fn(
            lead=req.model_dump(exclude={"thread_id"}),
            thread_id=req.thread_id,
        )
        return AgentResponse(success=True, agent="Sales Agent", result=result)
    except Exception as e:
        logger.error(f"Sales Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/sales/followup")
async def sales_followup(thread_id: str, message: str):
    """Continue a conversation with the Sales Agent for an existing lead."""
    try:
        from agents.sales_agent import followup
        result = followup(thread_id=thread_id, message=message)
        return AgentResponse(success=True, agent="Sales Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# CUSTOMER SUCCESS AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/support/ticket", response_model=AgentResponse)
async def handle_support_ticket(req: TicketRequest):
    """
    Submit a support ticket for autonomous resolution.
    Agent will look up order, process refund if applicable, create ticket, escalate if needed.
    """
    try:
        handle_ticket_fn = get_support_agent()
        result = handle_ticket_fn(ticket=req.model_dump())
        return AgentResponse(success=True, agent="Customer Success Agent", result=result)
    except Exception as e:
        logger.error(f"Support Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# RECRUITING AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/recruiting/screen", response_model=AgentResponse)
async def screen_candidate(req: RecruitingRequest):
    """
    Screen a candidate for a role. Returns scorecard and next steps.
    """
    try:
        process_candidate_fn, Candidate, JobRequirements = get_recruiting_agent()
        candidate = Candidate(**req.candidate.model_dump())
        job = JobRequirements(**req.job.model_dump())
        result = process_candidate_fn(candidate=candidate, job=job)
        return AgentResponse(success=True, agent="Recruiting Agent", result=result)
    except Exception as e:
        logger.error(f"Recruiting Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# FINANCE AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/finance/invoice", response_model=AgentResponse)
async def process_invoice(req: InvoiceRequest):
    """
    Process an invoice — extract, classify, check for anomalies, post to accounting.
    """
    try:
        process_invoice_fn, _ = get_finance_agent()
        result = process_invoice_fn(raw_invoice_text=req.raw_invoice_text)
        return AgentResponse(success=True, agent="Finance Agent", result=result)
    except Exception as e:
        logger.error(f"Finance Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/finance/expenses", response_model=AgentResponse)
async def audit_expenses(req: ExpenseAuditRequest):
    """Audit a batch of expense records for classification and policy violations."""
    try:
        _, audit_expenses_fn = get_finance_agent()
        result = audit_expenses_fn(expenses=req.expenses)
        return AgentResponse(success=True, agent="Finance Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/finance/monthly-close", response_model=AgentResponse)
async def monthly_close(period: Optional[str] = None):
    """Run the monthly financial close: reconcile + generate reports."""
    try:
        from agents.finance_agent import run_monthly_close
        result = run_monthly_close(period=period)
        return AgentResponse(success=True, agent="Finance Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# OPERATIONS AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/ops/morning-check", response_model=AgentResponse)
async def morning_check(req: OpsCheckRequest, background_tasks: BackgroundTasks):
    """
    Run the daily operations morning check.
    Fetches KPIs, checks SLA compliance, sends Slack alerts, generates digest.
    """
    try:
        run_morning_check_fn, _ = get_ops_agent()
        # Run in background so the API responds immediately
        result = run_morning_check_fn(scope=req.scope)
        return AgentResponse(success=True, agent="Operations Agent", result=result)
    except Exception as e:
        logger.error(f"Ops Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/ops/alert", response_model=AgentResponse)
async def trigger_ops_alert(req: AlertRequest):
    """Trigger an operations alert — agent will take immediate action."""
    try:
        _, trigger_alert_fn = get_ops_agent()
        result = trigger_alert_fn(alert_type=req.alert_type, details=req.details)
        return AgentResponse(success=True, agent="Operations Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# EXECUTIVE ASSISTANT AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/exec/morning-brief", response_model=AgentResponse)
async def exec_morning_brief(req: ExecBriefRequest):
    """Run the executive morning brief — email triage + calendar + priorities."""
    try:
        run_morning_brief_fn, _ = get_exec_agent()
        result = run_morning_brief_fn(
            executive_name=req.executive_name,
            executive_email=req.executive_email,
        )
        return AgentResponse(success=True, agent="Executive Assistant Agent", result=result)
    except Exception as e:
        logger.error(f"Exec Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/exec/task", response_model=AgentResponse)
async def exec_task(req: ExecTaskRequest):
    """Handle an ad-hoc executive assistant request."""
    try:
        _, process_request_fn = get_exec_agent()
        result = process_request_fn(
            executive_name=req.executive_name,
            request=req.request,
            thread_id=req.thread_id,
        )
        return AgentResponse(success=True, agent="Executive Assistant Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE AGENT
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/agents/knowledge/ask", response_model=AgentResponse)
async def knowledge_ask(req: KnowledgeAskRequest):
    """Ask the Knowledge Agent a question. Returns answer with source citations."""
    try:
        ask_fn, _ = get_knowledge_agent()
        result = ask_fn(
            question=req.question,
            company_name=req.company_name,
            employee_role=req.employee_role,
            session_id=req.session_id,
        )
        return AgentResponse(success=True, agent="Knowledge Agent", result=result)
    except Exception as e:
        logger.error(f"Knowledge Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/knowledge/ingest", response_model=AgentResponse)
async def knowledge_ingest(req: KnowledgeIngestRequest):
    """Add a new document to the company knowledge base."""
    try:
        _, add_document_fn = get_knowledge_agent()
        result = add_document_fn(
            document_url=req.document_url,
            document_type=req.document_type,
            category=req.category,
            added_by=req.added_by,
            company_name=req.company_name,
        )
        return AgentResponse(success=True, agent="Knowledge Agent", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ══════════════════════════════════════════════════════════════════════════
# WEBSITE UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/scanner")
async def opportunity_scanner(req: ScannerRequest):
    """
    AI Opportunity Scanner from the website lead magnet.
    Takes industry/size/pain/email → returns personalised automation opportunities.
    Also captures the lead into CRM if HUBSPOT_API_KEY is set.
    """
    opp_map = {
        "Customer support volume": [
            "Customer Success Agent resolves 70-80% of tickets autonomously",
            "Automated ticket triage saves 15+ hours/week",
            "24/7 support coverage without adding headcount",
        ],
        "Sales follow-up and qualification": [
            "Sales Agent responds to every lead in under 30 seconds",
            "Automated follow-up sequences increase conversion by 35%",
            "CRM auto-updated after every interaction",
        ],
        "Admin and data entry": [
            "Finance Agent processes invoices 10x faster",
            "Data entry errors eliminated across all workflows",
            "Monthly reports generated automatically",
        ],
        "Hiring and recruiting": [
            "Recruiting Agent screens 100% of applicants instantly",
            "Interview scheduling fully automated — zero back-and-forth",
            "Candidate scorecards generated objectively for every role",
        ],
        "Reporting and analytics": [
            "Operations Agent generates all KPI reports automatically",
            "Real-time anomaly detection — issues flagged before they escalate",
            "Stakeholder digests delivered without analyst hours",
        ],
        "Internal coordination": [
            "Zero-touch task routing across all departments",
            "SLA monitoring with proactive escalation alerts",
            "Automated standup digests replace 3 status meetings/week",
        ],
    }

    base_opps = opp_map.get(req.main_pain, ["AI automation opportunities identified"])
    all_opps = base_opps + [
        "Knowledge Agent cuts onboarding time by 60%",
        "Executive Assistant saves leadership 10+ hours/week",
        f"Industry-specific workflows for {req.industry} automated",
        "Cross-department coordination overhead reduced by 65%",
    ]

    # Capture lead (mock — replace with real CRM call)
    logger.info(f"Scanner lead captured: {req.email} | {req.industry} | {req.company_size}")

    try:
        save_lead(name=req.email.split('@')[0], email=req.email, company='', industry=req.industry, company_size=req.company_size, pain_point=req.main_pain, opportunities_count=len(all_opps), source='scanner', language='en')
    except Exception:
        pass

    return {
        "success": True,
        "email": req.email,
        "opportunities_count": len(all_opps),
        "opportunities": all_opps,
        "recommended_first_agent": "Sales Agent" if "sales" in req.main_pain.lower()
                                   else "Customer Success Agent" if "support" in req.main_pain.lower()
                                   else "Operations Agent",
        "next_step": "Book a free 45-minute AI Assessment to get your full automation roadmap.",
    }

@app.post("/api/roi-capture")
async def capture_roi(req: ROIRequest):
    """
    Capture ROI Calculator data from the website.
    Stores the lead and their calculated savings for the sales team.
    """
    logger.info(f"ROI capture: {req.email} | {req.company} | ${req.annual_savings:,.0f}/yr savings")
    # Production: push to HubSpot/Salesforce with the savings data as a custom property

    try:
        save_lead(name='ROI User', email=req.email or 'anonymous', company=req.company or '', industry='', company_size='', pain_point='', opportunities_count=0, source='roi_calculator', language='en')
    except Exception:
        pass

    return {
        "success": True,
        "message": "ROI data captured. Book your free assessment to see this in action.",
        "calculated_savings": req.annual_savings,
        "booking_url": "https://calendly.com/operamind/ai-assessment",
    }

# ══════════════════════════════════════════════════════════════════════════
# ROUTER REGISTRATIONS
# ══════════════════════════════════════════════════════════════════════════

try:
    from api.stripe_handler import stripe_router
    app.include_router(stripe_router)
except Exception as e:
    logger.warning(f"Stripe handler not loaded: {e}")

try:
    from api.admin import admin_router
    app.include_router(admin_router)
except Exception as e:
    logger.warning(f"Admin handler not loaded: {e}")

# ══════════════════════════════════════════════════════════════════════════
# STATUS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
def detailed_status():
    status = {"api": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}
    try:
        from api.database import get_stats
        status["database"] = "ok"
        status["stats"] = get_stats()
    except Exception as e:
        status["database"] = f"error: {str(e)}"
    status["keys"] = {
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY", "")),
        "openai": bool(os.getenv("OPENAI_API_KEY", "")),
        "hubspot": bool(os.getenv("HUBSPOT_API_KEY", "")),
        "sendgrid": bool(os.getenv("SENDGRID_API_KEY", "")),
        "stripe": bool(os.getenv("STRIPE_SECRET_KEY", "")),
    }
    agents_ok = {}
    for name, mod in [("sales","agents.sales_agent"),("support","agents.customer_success_agent"),("recruiting","agents.recruiting_agent"),("finance","agents.finance_agent"),("operations","agents.operations_agent"),("executive","agents.executive_assistant_agent"),("knowledge","agents.knowledge_agent")]:
        try:
            __import__(mod)
            agents_ok[name] = "ok"
        except Exception as e:
            agents_ok[name] = f"error: {str(e)[:80]}"
    status["agents"] = agents_ok
    return status

# ══════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════

@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "docs": "/docs"})

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error. Check logs."})
