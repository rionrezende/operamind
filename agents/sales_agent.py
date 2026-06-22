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

@tool
def score_lead_bant(
    name: str,
    budget_signals: str,
    authority_level: str,
    need_urgency: str,
    timeline: str,
) -> dict:
    """
    Score a lead using BANT framework (Budget, Authority, Need, Timeline).
    Returns a weighted score out of 100 with per-dimension breakdown.
    """
    # Budget (30 pts)
    budget_keywords = {"confirmed": 30, "exploring": 20, "limited": 10, "unknown": 5}
    budget_score = next(
        (v for k, v in budget_keywords.items() if k in budget_signals.lower()), 5
    )

    # Authority (25 pts)
    authority_keywords = {"decision_maker": 25, "influencer": 18, "champion": 15, "end_user": 8}
    authority_score = next(
        (v for k, v in authority_keywords.items() if k in authority_level.lower()), 8
    )

    # Need (25 pts)
    need_keywords = {"critical": 25, "high": 20, "moderate": 12, "low": 5}
    need_score = next(
        (v for k, v in need_keywords.items() if k in need_urgency.lower()), 5
    )

    # Timeline (20 pts)
    timeline_keywords = {"immediate": 20, "this_quarter": 16, "next_quarter": 10, "no_timeline": 4}
    timeline_score = next(
        (v for k, v in timeline_keywords.items() if k in timeline.lower()), 4
    )

    total = budget_score + authority_score + need_score + timeline_score
    return {
        "lead_name": name,
        "total_score": total,
        "breakdown": {
            "budget": {"score": budget_score, "max": 30},
            "authority": {"score": authority_score, "max": 25},
            "need": {"score": need_score, "max": 25},
            "timeline": {"score": timeline_score, "max": 20},
        },
        "rating": "Hot" if total >= 75 else "Warm" if total >= 50 else "Cold",
    }

@tool
def handle_objection(
    objection_type: str,
    lead_context: str,
) -> dict:
    """
    Generate a value-based counter-response for common sales objections.
    objection_type: 'too_expensive' | 'not_now' | 'using_competitor' | 'no_budget' | 'need_approval'
    """
    responses = {
        "too_expensive": {
            "counter": "I understand cost is a concern. Let's look at the ROI — most clients see payback within 30–60 days.",
            "talking_points": [
                "Average client saves 40+ hours/week on manual tasks",
                "Cost of inaction: what does each missed lead cost you today?",
                "Flexible pricing — we can start with 1 agent and scale",
                "Compare against hiring a full-time employee ($60K–$120K/yr)",
            ],
        },
        "not_now": {
            "counter": "Timing matters. Would it help to see how similar companies prepared 1–2 months ahead of their busy season?",
            "talking_points": [
                "Setup takes 2 weeks — starting now means you're ready before Q-end",
                "We can lock in current pricing with a delayed start",
                "A quick discovery call now costs 30 minutes, not a commitment",
            ],
        },
        "using_competitor": {
            "counter": "Great that you're already automating. Many clients switch to us for deeper AI capabilities and white-glove onboarding.",
            "talking_points": [
                "OperaMind agents learn and adapt — not just rule-based workflows",
                "Dedicated onboarding specialist (no DIY setup)",
                "We integrate with your existing stack, not replace it",
                "Ask about our migration credit for switching",
            ],
        },
        "no_budget": {
            "counter": "I hear you. Let me show you a cost-neutral scenario where savings from automation fund the subscription.",
            "talking_points": [
                "ROI calculator shows breakeven in most cases within 30 days",
                "Start with a single agent at $297/month",
                "We offer a 14-day pilot with no long-term contract",
            ],
        },
        "need_approval": {
            "counter": "Absolutely — let me prepare a business case you can share with your team.",
            "talking_points": [
                "I'll send a 1-page ROI summary tailored to your numbers",
                "Happy to join a call with your decision-maker",
                "Case studies from similar companies in your industry",
                "We can do a live demo for the broader team",
            ],
        },
    }
    result = responses.get(objection_type, {
        "counter": "Let me address that concern directly.",
        "talking_points": ["Let's schedule a call to discuss your specific situation."],
    })
    result["objection_type"] = objection_type
    result["lead_context"] = lead_context
    return result

@tool
def detect_competitor(message_text: str) -> dict:
    """
    Scan a message for competitor mentions and return competitive positioning points.
    """
    competitors = {
        "Zapier": "Rule-based automation only; no AI reasoning. OperaMind agents think and adapt.",
        "Make": "Visual workflow builder — powerful but requires manual setup per scenario. OperaMind is autonomous.",
        "n8n": "Open-source, self-hosted complexity. OperaMind is fully managed with zero DevOps.",
        "UiPath": "Enterprise RPA — heavy, expensive, long implementation. OperaMind deploys in days, not months.",
        "Automation Anywhere": "Legacy RPA focused on desktop bots. OperaMind is cloud-native AI-first.",
        "Workato": "Integration-focused iPaaS. OperaMind goes beyond integration to autonomous decision-making.",
        "Relevance AI": "AI tool builder — requires technical setup. OperaMind is turnkey with white-glove onboarding.",
        "Lindy AI": "AI assistant platform. OperaMind offers deeper vertical specialization with 7 dedicated agent roles.",
    }

    detected = []
    for name, positioning in competitors.items():
        if name.lower() in message_text.lower():
            detected.append({"competitor": name, "positioning": positioning})

    return {
        "competitors_found": len(detected),
        "detected": detected,
        "general_positioning": (
            "OperaMind combines autonomous AI agents with white-glove onboarding — "
            "not just workflows, but a digital workforce that learns and scales with you."
        ) if detected else "No competitors detected.",
    }

@tool
def schedule_followup_sequence(
    lead_email: str,
    lead_name: str,
    sequence_type: str,
) -> dict:
    """
    Create a 3-touch follow-up email sequence plan.
    sequence_type: 'warm' | 'cold' | 're-engage'
    """
    sequences = {
        "warm": [
            {"day": 1, "type": "intro", "subject": f"Great connecting, {lead_name}!",
             "focus": "Recap conversation, share relevant case study link", "send_time": "10:00 AM"},
            {"day": 3, "type": "value", "subject": f"Quick ROI breakdown for {lead_name}'s team",
             "focus": "Personalised ROI estimate based on their company size and pain points", "send_time": "2:00 PM"},
            {"day": 7, "type": "case_study", "subject": "How [similar company] saved 40 hrs/week",
             "focus": "Industry-matched case study with concrete metrics", "send_time": "9:00 AM"},
        ],
        "cold": [
            {"day": 1, "type": "intro", "subject": f"{lead_name}, a question about your operations",
             "focus": "Pattern interrupt — ask about their biggest operational bottleneck", "send_time": "8:00 AM"},
            {"day": 3, "type": "value", "subject": "The hidden cost of manual processes",
             "focus": "Industry stats on time lost to manual work, bridge to OperaMind", "send_time": "11:00 AM"},
            {"day": 7, "type": "case_study", "subject": "This CEO automated 80% of back-office ops",
             "focus": "Compelling transformation story with before/after metrics", "send_time": "10:00 AM"},
        ],
        "re-engage": [
            {"day": 1, "type": "intro", "subject": f"{lead_name}, things have changed at OperaMind",
             "focus": "New features or pricing update as a reason to reconnect", "send_time": "9:00 AM"},
            {"day": 3, "type": "value", "subject": "3 automations your competitors are using right now",
             "focus": "Competitive urgency — what their industry peers are automating", "send_time": "1:00 PM"},
            {"day": 7, "type": "case_study", "subject": "Still thinking about automation?",
             "focus": "Low-pressure check-in with a limited-time offer or pilot program", "send_time": "10:00 AM"},
        ],
    }

    sequence = sequences.get(sequence_type, sequences["warm"])
    return {
        "lead_email": lead_email,
        "lead_name": lead_name,
        "sequence_type": sequence_type,
        "touches": sequence,
        "total_duration_days": 7,
        "note": "All times are in the lead's local timezone. Adjust for holidays.",
    }

@tool
def prepare_meeting_brief(
    company: str,
    attendees: str,
    meeting_type: str,
) -> dict:
    """
    Generate a pre-meeting briefing document.
    meeting_type: 'discovery' | 'demo' | 'proposal_review'
    """
    type_config = {
        "discovery": {
            "objective": "Understand pain points, confirm ICP fit, identify decision-making process",
            "talking_points": [
                "What does your current workflow look like end-to-end?",
                "Where do bottlenecks or manual steps slow you down?",
                "What have you tried before to solve this?",
                "Who else is involved in evaluating solutions?",
                "What does your ideal timeline for implementation look like?",
            ],
            "objections_to_prepare": ["not_now", "no_budget", "need_approval"],
            "demo_notes": None,
        },
        "demo": {
            "objective": "Show tailored demo addressing their specific pain points, build urgency",
            "talking_points": [
                "Recap pain points from discovery (confirm they're still accurate)",
                "Live walkthrough of the most relevant agent(s)",
                "Show integration with their existing tools",
                "Share metrics from similar companies",
                "Discuss implementation timeline and next steps",
            ],
            "objections_to_prepare": ["too_expensive", "using_competitor", "need_approval"],
            "demo_notes": f"Customise the demo environment to reflect {company}'s branding and use case. Pre-load sample data relevant to their industry.",
        },
        "proposal_review": {
            "objective": "Walk through proposal, handle objections, close or define next steps",
            "talking_points": [
                "Review each section of the proposal together",
                "Confirm the ROI assumptions are accurate",
                "Address any concerns about implementation",
                "Discuss contract terms and timeline",
                "Define clear next steps and decision date",
            ],
            "objections_to_prepare": ["too_expensive", "not_now", "need_approval"],
            "demo_notes": None,
        },
    }

    config = type_config.get(meeting_type, type_config["discovery"])
    return {
        "company": company,
        "attendees": attendees,
        "meeting_type": meeting_type,
        "objective": config["objective"],
        "talking_points": config["talking_points"],
        "objections_to_prepare": config["objections_to_prepare"],
        "demo_customization": config["demo_notes"],
        "research_summary": f"Review {company}'s website, recent news, LinkedIn profiles of attendees before the call.",
        "prepared_at": datetime.datetime.utcnow().isoformat(),
    }

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are OperaMind's AI Sales Employee. Your job is to:

1. Qualify every inbound lead against our ICP using the qualify_lead tool
2. Use score_lead_bant for deeper BANT analysis on promising leads
3. If qualified (score ≥ 60): book a discovery call, update CRM to 'Qualified', send a warm personalised email
4. If not qualified (score < 60): update CRM to 'Disqualified', send a polite nurture email
5. After a discovery call: write a personalised proposal using write_proposal
6. Use detect_competitor to scan lead messages for competitor mentions and position OperaMind accordingly
7. When a lead raises objections, use handle_objection to craft value-based counter-responses
8. Use schedule_followup_sequence to create multi-touch email sequences (warm, cold, or re-engage)
9. Before any meeting, use prepare_meeting_brief to generate a briefing doc with talking points and objection prep
10. Always update the CRM after every action

Tone: Professional, warm, concise. Never pushy. Focus on business outcomes, not technology.
Always respond in under 30 seconds. Never miss a lead.

Current date: {date}
"""

# ── Graph nodes ─────────────────────────────────────────────────────────────
tools = [qualify_lead, update_crm, book_meeting, send_email, write_proposal,
         score_lead_bant, handle_objection, detect_competitor,
         schedule_followup_sequence, prepare_meeting_brief]
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
