"""
OperaMind — Sales Agent
=======================
Framework : LangGraph 0.4+ (stateful graph, HITL, durable execution)
Model     : claude-sonnet-4-6  (Anthropic)
Purpose   : Qualifies inbound leads, follows up automatically,
            schedules meetings, updates CRM, writes proposals.

Production dependencies
-----------------------
pip install langgraph langchain-anthropic langchain-community \
            anthropic python-dotenv pydantic

Environment variables (.env)
-----------------------------
ANTHROPIC_API_KEY=sk-ant-...
HUBSPOT_API_KEY=...          # or SALESFORCE_CLIENT_ID/SECRET
CALENDLY_API_KEY=...
"""

from __future__ import annotations
import os, json, datetime
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

load_dotenv()

# ── Model ──────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    temperature=0.2,        # low temp for consistent qualification decisions
)

# ── Shared state ────────────────────────────────────────────────────────────
class SalesState(TypedDict):
    messages: Annotated[list, add_messages]
    lead: dict                  # raw lead data
    score: float                # 0–1 ICP score
    qualified: bool | None      # None = not yet decided
    crm_updated: bool
    meeting_booked: bool
    proposal_sent: bool
    next_action: str

# ── ICP definition (customise per client) ──────────────────────────────────
ICP = {
    "target_company_sizes": ["11-50", "51-200", "201-500"],
    "target_industries": ["SaaS", "E-commerce", "Healthcare", "Financial Services", "Logistics"],
    "target_titles": ["CEO", "COO", "CTO", "VP Operations", "VP Sales", "Head of", "Director of"],
    "budget_indicators": ["looking to automate", "reduce costs", "scale operations", "efficiency"],
    "disqualifiers": ["student", "personal project", "no budget", "just browsing"],
}

# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def qualify_lead(
    name: str,
    email: str,
    company: str,
    company_size: str,
    industry: str,
    title: str,
    message: str,
) -> dict:
    """
    Score a lead against our ICP (Ideal Customer Profile).
    Returns a score 0-100 and a qualification decision.
    """
    score = 0

    # Company size match (30 pts)
    if any(s in company_size for s in ICP["target_company_sizes"]):
        score += 30

    # Industry match (25 pts)
    if any(ind.lower() in industry.lower() for ind in ICP["target_industries"]):
        score += 25

    # Title / seniority match (25 pts)
    if any(t.lower() in title.lower() for t in ICP["target_titles"]):
        score += 25

    # Intent signals in message (20 pts)
    if any(kw.lower() in message.lower() for kw in ICP["budget_indicators"]):
        score += 20

    # Disqualifiers
    if any(d.lower() in message.lower() for d in ICP["disqualifiers"]):
        score = max(0, score - 50)

    qualified = score >= 60
    return {
        "score": score,
        "qualified": qualified,
        "reason": (
            f"Score {score}/100. "
            + ("Meets ICP criteria." if qualified else "Does not meet minimum ICP threshold (60).")
        ),
    }

@tool
def update_crm(
    lead_email: str,
    stage: str,
    score: float,
    notes: str,
    next_followup: str | None = None,
) -> dict:
    """
    Update the CRM record for a lead (HubSpot / Salesforce).
    stage: 'New Lead' | 'Qualified' | 'Meeting Booked' | 'Proposal Sent' | 'Disqualified'
    """
    # In production: call HubSpot/Salesforce API here
    # import hubspot; client = hubspot.Client.create(api_key=os.environ["HUBSPOT_API_KEY"])
    print(f"[CRM] Updating {lead_email}: stage={stage}, score={score}, notes={notes}")
    return {
        "success": True,
        "crm_record_id": f"crm_{lead_email.replace('@','_')}",
        "stage": stage,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def book_meeting(
    lead_name: str,
    lead_email: str,
    preferred_time: str,
    meeting_type: str = "Discovery Call",
) -> dict:
    """
    Book a meeting via Calendly or Google Calendar.
    Returns confirmation details and meeting link.
    """
    # In production: call Calendly API
    # import requests; r = requests.post("https://api.calendly.com/scheduled_events", ...)
    print(f"[CALENDAR] Booking {meeting_type} with {lead_name} ({lead_email}) at {preferred_time}")
    return {
        "success": True,
        "meeting_link": f"https://calendly.com/operamind/discovery?email={lead_email}",
        "meeting_time": preferred_time,
        "meeting_type": meeting_type,
        "confirmation_sent": True,
    }

@tool
def send_email(
    to_email: str,
    to_name: str,
    subject: str,
    body: str,
    email_type: str = "outreach",
) -> dict:
    """
    Send a personalised email via SendGrid / Mailgun.
    email_type: 'outreach' | 'followup' | 'proposal' | 'disqualify'
    """
    # In production: use SendGrid / Mailgun SDK
    # import sendgrid; sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    print(f"[EMAIL] Sending '{email_type}' email to {to_name} <{to_email}>")
    print(f"  Subject: {subject}")
    return {
        "success": True,
        "message_id": f"msg_{datetime.datetime.utcnow().timestamp():.0f}",
        "delivered_at": datetime.datetime.utcnow().isoformat(),
    }

@tool
def write_proposal(
    company: str,
    pain_points: str,
    recommended_agents: str,
    estimated_roi: str,
) -> str:
    """
    Generate a personalised proposal document (returns Markdown).
    """
    return f"""
# OperaMind Proposal for {company}

## Executive Summary
Based on our discovery call, we've identified the following automation opportunities
that align directly with {company}'s operational goals.

## Pain Points Identified
{pain_points}

## Recommended AI Employees
{recommended_agents}

## Expected ROI
{estimated_roi}

## Investment
- **Setup:** from $4,997 (one-time)
- **Monthly:** from $997/month
- **Payback period:** Typically 30–60 days

## Next Steps
1. Sign SOW (Statement of Work) — 2 business days
2. Integration access setup — Day 3
3. Agent deployment — Days 4–14
4. Go-live + monitoring — Day 15+

*Prepared by OperaMind AI Sales Agent · {datetime.date.today()}*
"""

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are OperaMind's AI Sales Employee. Your job is to:

1. Qualify every inbound lead against our ICP using the qualify_lead tool
2. If qualified (score ≥ 60): book a discovery call, update CRM to 'Qualified', send a warm personalised email
3. If not qualified (score < 60): update CRM to 'Disqualified', send a polite nurture email
4. After a discovery call: write a personalised proposal using write_proposal
5. Always update the CRM after every action

Tone: Professional, warm, concise. Never pushy. Focus on business outcomes, not technology.
Always respond in under 30 seconds. Never miss a lead.

Current date: {date}
"""

# ── Graph nodes ─────────────────────────────────────────────────────────────
tools = [qualify_lead, update_crm, book_meeting, send_email, write_proposal]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)

def sales_agent_node(state: SalesState) -> SalesState:
    """Main agent reasoning node."""
    system = SYSTEM_PROMPT.format(date=datetime.date.today())
    messages = [SystemMessage(content=system)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: SalesState) -> Literal["tools", "end"]:
    """Route: if the model called tools → run them; otherwise → done."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

# ── Build graph ──────────────────────────────────────────────────────────────
checkpointer = MemorySaver()   # swap for SqliteSaver / PostgresSaver in production

builder = StateGraph(SalesState)
builder.add_node("agent", sales_agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=checkpointer)

# ── Public API ───────────────────────────────────────────────────────────────

def process_lead(lead: dict, thread_id: str | None = None) -> dict:
    """
    Entry point: process a new inbound lead.

    lead = {
        "name": "James Rodrigues",
        "email": "james@terraops.com",
        "company": "TerraOps B2B",
        "company_size": "51-200",
        "industry": "SaaS",
        "title": "VP Sales",
        "message": "We're looking to automate our sales qualification process."
    }
    """
    tid = thread_id or f"lead_{lead.get('email', 'unknown').replace('@','_')}"
    config = {"configurable": {"thread_id": tid}}

    user_message = (
        f"New inbound lead received:\n"
        f"Name: {lead.get('name')}\n"
        f"Email: {lead.get('email')}\n"
        f"Company: {lead.get('company')} ({lead.get('company_size')} employees)\n"
        f"Industry: {lead.get('industry')}\n"
        f"Title: {lead.get('title')}\n"
        f"Message: {lead.get('message')}\n\n"
        f"Please qualify this lead and take the appropriate next action."
    )

    initial_state: SalesState = {
        "messages": [HumanMessage(content=user_message)],
        "lead": lead,
        "score": 0.0,
        "qualified": None,
        "crm_updated": False,
        "meeting_booked": False,
        "proposal_sent": False,
        "next_action": "qualify",
    }

    result = graph.invoke(initial_state, config=config)

    # Extract final agent message
    final_msg = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    return {
        "thread_id": tid,
        "lead_email": lead.get("email"),
        "agent_response": final_msg.content if final_msg else "Processing complete.",
        "messages": [m.content if hasattr(m, "content") else str(m) for m in result["messages"]],
    }

def followup(thread_id: str, message: str) -> dict:
    """Continue an existing lead conversation (e.g. after discovery call)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    final_msg = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    return {"agent_response": final_msg.content if final_msg else "Done."}


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_lead = {
        "name": "James Rodrigues",
        "email": "james@terraops.com",
        "company": "TerraOps B2B",
        "company_size": "51-200",
        "industry": "SaaS",
        "title": "VP Sales",
        "message": "We're struggling to qualify leads fast enough. Looking to automate our sales ops.",
    }
    result = process_lead(sample_lead)
    print("\n" + "="*60)
    print("SALES AGENT RESPONSE:")
    print(result["agent_response"])
